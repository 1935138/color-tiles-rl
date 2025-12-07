"""
Training script for Color Tiles RL agent using PPO

This script handles:
- Creating parallel training environments
- Initializing PPO model with hyperparameters from RL plan
- Setting up checkpointing and evaluation callbacks
- Running the training loop
- Saving the final model

Usage:
    # New training
    python -m rl.training.train --total-timesteps 1000000 --n-envs 8

    # Resume from checkpoint
    python -m rl.training.train --checkpoint checkpoints/ppo_colortiles_step_50000.zip

    # Short test run
    python -m rl.training.train --total-timesteps 10000 --n-envs 2
"""

import argparse
from pathlib import Path
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not installed. Install with:")
    print("pip install stable-baselines3 torch tensorboard")

from rl.env.color_tiles_env import ColorTilesEnv
try:
    from rl.training.callbacks import MetricsCallback
    CALLBACKS_AVAILABLE = True
except ImportError:
    CALLBACKS_AVAILABLE = False


# ============================================================================
# GPU OPTIMIZATION UTILITIES
# ============================================================================

def detect_and_configure_gpu() -> dict:
    """
    Detect GPU, enable optimizations, and return device configuration.

    Optimizations enabled:
    - TF32 precision for RTX 3000+ (8x speedup for matmul)
    - cuDNN benchmark mode (faster convolutions)
    - Explicit CUDA device selection

    Returns:
        config: Dictionary with device info and settings
    """
    import torch

    config = {
        'device': 'cpu',
        'device_name': 'CPU',
        'gpu_available': False,
        'cuda_version': None,
        'gpu_memory_gb': 0.0,
        'tf32_enabled': False,
        'cudnn_benchmark': False,
        'compute_capability': None,
    }

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. Training on CPU will be slow.")
        print("   Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/\n")
        return config

    # GPU detected - configure optimizations
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(0)

    config.update({
        'device': 'cuda:0',
        'device_name': props.name,
        'gpu_available': True,
        'cuda_version': torch.version.cuda,
        'gpu_memory_gb': props.total_memory / (1024**3),
        'compute_capability': f"{props.major}.{props.minor}",
    })

    # Enable TF32 for Ampere+ GPUs (compute capability >= 8.0)
    # RTX 3000/4000/5000 series support TF32 (8x speedup)
    if props.major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        config['tf32_enabled'] = True
        print("✓ TF32 precision enabled (8x matmul speedup)")

    # Enable cuDNN benchmark mode for faster convolutions
    # Only enable if using CnnPolicy (fixed input size)
    torch.backends.cudnn.benchmark = True
    config['cudnn_benchmark'] = True

    return config


def calculate_optimal_batch_size(gpu_memory_gb: float, manual_override: int = None) -> int:
    """
    Calculate optimal batch size based on available GPU memory.

    Conservative estimates to prevent OOM:
    - RTX 4060/5060 (8GB): 256
    - RTX 4070/5070 (12GB): 384
    - RTX 4080/5080 (16GB+): 512

    Args:
        gpu_memory_gb: Available GPU memory in GB
        manual_override: Manual batch size (bypasses auto-calculation)

    Returns:
        batch_size: Recommended batch size
    """
    if manual_override is not None:
        return manual_override

    # Conservative batch size mapping (leaves 2GB buffer for system)
    if gpu_memory_gb < 6.0:
        return 128  # Small GPU or shared memory
    elif gpu_memory_gb < 10.0:
        return 256  # RTX 4060/5060 (8GB)
    elif gpu_memory_gb < 14.0:
        return 384  # RTX 4070/5070 (12GB)
    else:
        return 512  # RTX 4080/5080/4090 (16GB+)


def print_gpu_info(config: dict):
    """Print GPU configuration summary."""
    print(f"\n{'='*70}")
    print("GPU CONFIGURATION")
    print(f"{'='*70}")

    if config['gpu_available']:
        print(f"Device: {config['device_name']}")
        print(f"Memory: {config['gpu_memory_gb']:.2f} GB VRAM")
        print(f"CUDA Version: {config['cuda_version']}")
        print(f"Compute Capability: {config['compute_capability']}")
        print(f"TF32 Precision: {'Enabled ✓' if config['tf32_enabled'] else 'Disabled'}")
        print(f"cuDNN Benchmark: {'Enabled ✓' if config['cudnn_benchmark'] else 'Disabled'}")
    else:
        print("Device: CPU (No CUDA GPU detected)")
        print("⚠️  Training will be significantly slower on CPU")

    print(f"{'='*70}\n")


# ============================================================================
# CUSTOM CNN FOR SMALL BOARD
# ============================================================================

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

class ColorTilesCNN(BaseFeaturesExtractor):
    """
    Custom CNN optimized for Color Tiles board (15x23).

    Uses smaller kernels and strides suitable for small board sizes.
    Default NatureCNN uses 8x8 kernels which are too large for our board.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[2]  # Should be 1

        # Custom CNN layers for small board (15, 23)
        self.cnn = nn.Sequential(
            # First conv layer: 1 -> 32 channels, 3x3 kernel
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second conv layer: 32 -> 64 channels, 3x3 kernel
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Flatten
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape).permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (batch, height, width, channels)
        # Conv2d expects: (batch, channels, height, width)
        observations = observations.permute(0, 3, 1, 2).float()
        return self.linear(self.cnn(observations))


# ============================================================================
# ENVIRONMENT WRAPPER FOR CNN POLICY
# ============================================================================

class CnnObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper to add channel dimension for CnnPolicy compatibility.

    Transforms observation from (H, W) → (H, W, 1) for CNN processing.
    ColorTilesEnv outputs (15, 23) but CnnPolicy expects (15, 23, 1).

    This is a zero-copy operation (uses view/reshape, not copy).
    """

    def __init__(self, env):
        super().__init__(env)

        # Update observation space to 3D (H, W, C)
        old_shape = env.observation_space.shape
        new_shape = (*old_shape, 1)  # Add channel dimension

        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        """Add channel dimension: (H, W) → (H, W, 1)."""
        return obs[..., np.newaxis]  # Zero-copy reshape


def make_env(rank: int, seed: int = 0, disable_time_limit: bool = True, use_cnn: bool = True):
    """
    Factory function for creating environment instances.

    Args:
        rank: Environment index (for seeding)
        seed: Base random seed
        disable_time_limit: Whether to disable time limit (True for training)
        use_cnn: Whether to wrap with CnnObservationWrapper (True for CnnPolicy)

    Returns:
        _init: Function that creates and returns the environment
    """
    def _init():
        env = ColorTilesEnv(disable_time_limit=disable_time_limit)
        env.reset(seed=seed + rank)

        # Wrap with CNN observation wrapper if using CnnPolicy
        if use_cnn:
            env = CnnObservationWrapper(env)

        env = Monitor(env)  # Wrap with Monitor for automatic logging
        return env
    return _init


def create_model(
    env,
    learning_rate: float = 3e-4,
    batch_size: int = 256,
    n_steps: int = 4096,
    device: str = 'cuda:0',
    use_cnn: bool = True,
    checkpoint_path: str = None
):
    """
    Create or load PPO model with GPU-optimized hyperparameters.

    GPU Optimizations:
    - CnnPolicy for spatial feature extraction (vs MlpPolicy)
    - Large batch sizes (256-512) for GPU parallelism
    - Increased n_steps (4096) to reduce environment resets
    - Explicit device selection (cuda:0 vs auto)
    - Custom CNN architecture optimized for (15, 23, 1) input

    Args:
        env: Training environment (can be vectorized)
        learning_rate: Learning rate for optimizer
        batch_size: Minibatch size (auto-adjusted based on GPU memory)
        n_steps: Rollout buffer size per environment
        device: Device string ('cuda:0', 'cpu', etc.)
        use_cnn: Use CnnPolicy (True) or MlpPolicy (False)
        checkpoint_path: Path to checkpoint file to resume from (optional)

    Returns:
        model: PPO model instance
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3 torch")

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env, device=device)
        print(f"Resumed from checkpoint. Continuing training...")
        print(f"  Device: {model.device}")
        return model

    # Select policy type
    policy = "CnnPolicy" if use_cnn else "MlpPolicy"

    print(f"Creating new PPO model...")
    print(f"  Policy: {policy}")
    print(f"  Batch size: {batch_size}")
    print(f"  Rollout steps (n_steps): {n_steps}")
    print(f"  Device: {device}")

    # Custom policy kwargs for CnnPolicy
    # Use custom CNN architecture optimized for small board (15x23)
    policy_kwargs = None
    if use_cnn:
        policy_kwargs = dict(
            features_extractor_class=ColorTilesCNN,
            features_extractor_kwargs=dict(features_dim=256),
            normalize_images=False,  # Our input is already in [0, 10] range
        )

    # Create model with GPU-optimized hyperparameters
    model = PPO(
        policy=policy,
        env=env,
        learning_rate=learning_rate,

        # GPU-optimized parameters
        n_steps=n_steps,          # 2048 → 4096 (fewer env resets, better GPU utilization)
        batch_size=batch_size,    # 64 → 256-512 (maximize GPU parallelism)

        # Keep proven hyperparameters from original config
        n_epochs=10,              # Number of epochs for optimization
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE parameter
        clip_range=0.2,           # PPO clipping epsilon
        ent_coef=0.01,            # Entropy coefficient (exploration)
        vf_coef=0.5,              # Value function coefficient
        max_grad_norm=0.5,        # Gradient clipping

        # Logging and device
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device=device,            # Explicit device (not "auto")
        policy_kwargs=policy_kwargs,
    )

    return model


def setup_callbacks(save_dir: Path, eval_env):
    """
    Setup training callbacks for checkpointing, evaluation, and metrics.

    Args:
        save_dir: Directory to save checkpoints
        eval_env: Environment for periodic evaluation

    Returns:
        callbacks: CallbackList with all callbacks
    """
    if not SB3_AVAILABLE:
        return None

    save_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint callback - save every 10k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save every 10k steps
        save_path=str(save_dir),
        name_prefix="ppo_colortiles_step",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Evaluation callback - evaluate every 5k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval_logs"),
        eval_freq=5000,  # Evaluate every 5k steps
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    callbacks = [checkpoint_callback, eval_callback]

    # Add custom metrics callback if available
    if CALLBACKS_AVAILABLE:
        metrics_callback = MetricsCallback(log_dir=save_dir, verbose=1)
        callbacks.append(metrics_callback)

    return CallbackList(callbacks)


def main():
    """Main training function with GPU optimization."""
    parser = argparse.ArgumentParser(
        description="Train Color Tiles RL agent with PPO (GPU-optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GPU-optimized training (auto batch size)
  python -m rl.training.train --total-timesteps 1000000 --n-envs 8

  # Override batch size manually
  python -m rl.training.train --batch-size 512 --n-envs 8

  # CPU fallback (no GPU)
  python -m rl.training.train --device cpu --no-cnn

  # Resume from checkpoint
  python -m rl.training.train --checkpoint checkpoints/ppo_colortiles_step_50000.zip
        """
    )

    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                       help="Total number of training timesteps (default: 1M)")
    parser.add_argument("--n-envs", type=int, default=8,
                       help="Number of parallel environments (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")

    # GPU optimization parameters
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size (default: auto-adjust based on GPU memory)")
    parser.add_argument("--n-steps", type=int, default=4096,
                       help="Rollout buffer size per env (default: 4096)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda:0, cpu, etc.). Default: auto-detect")
    parser.add_argument("--no-cnn", action="store_true",
                       help="Use MlpPolicy instead of CnnPolicy (not recommended)")
    parser.add_argument("--no-tf32", action="store_true",
                       help="Disable TF32 precision (for debugging)")

    # Checkpoint and logging
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file to resume from")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints (default: checkpoints)")

    args = parser.parse_args()

    if not SB3_AVAILABLE:
        print("\nERROR: stable-baselines3 is not installed.")
        print("Please install required packages:")
        print("  pip install gymnasium stable-baselines3 torch tensorboard")
        return

    # ========================================================================
    # GPU DETECTION AND CONFIGURATION
    # ========================================================================

    import torch

    print(f"\n{'='*70}")
    print("COLOR TILES RL TRAINING (GPU-OPTIMIZED)")
    print(f"{'='*70}\n")

    # Detect and configure GPU
    gpu_config = detect_and_configure_gpu()

    # Disable TF32 if requested
    if args.no_tf32 and gpu_config['gpu_available']:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        gpu_config['tf32_enabled'] = False
        print("⚠️  TF32 disabled via --no-tf32 flag")

    # Print GPU info
    print_gpu_info(gpu_config)

    # Determine device
    if args.device:
        device = args.device
    else:
        device = gpu_config['device']

    # Calculate optimal batch size
    batch_size = calculate_optimal_batch_size(
        gpu_memory_gb=gpu_config['gpu_memory_gb'],
        manual_override=args.batch_size
    )

    # Determine policy type
    use_cnn = not args.no_cnn

    # ========================================================================
    # TRAINING CONFIGURATION SUMMARY
    # ========================================================================

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*70}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Policy: {'CnnPolicy' if use_cnn else 'MlpPolicy'}")
    print(f"Batch size: {batch_size} {'(auto)' if args.batch_size is None else '(manual)'}")
    print(f"Rollout steps (n_steps): {args.n_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {device}")
    print(f"Save directory: {save_dir}")
    print(f"Random seed: {args.seed}")
    if args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
    print(f"{'='*70}\n")

    # ========================================================================
    # ENVIRONMENT SETUP
    # ========================================================================

    print("Setting up environments...")

    # Create parallel training environments
    if args.n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, args.seed, disable_time_limit=True, use_cnn=use_cnn)
            for i in range(args.n_envs)
        ])
        print(f"✓ Created {args.n_envs} parallel training environments (SubprocVecEnv)")
    else:
        env = DummyVecEnv([make_env(0, args.seed, disable_time_limit=True, use_cnn=use_cnn)])
        print(f"✓ Created 1 training environment (DummyVecEnv)")

    # Create evaluation environment (single env, deterministic)
    eval_env = DummyVecEnv([make_env(args.n_envs, args.seed, disable_time_limit=True, use_cnn=use_cnn)])
    print("✓ Created evaluation environment")

    # Print observation space info
    sample_obs = env.reset()
    print(f"✓ Observation shape: {sample_obs.shape}")
    if use_cnn:
        print(f"  CnnPolicy expects: (batch, height, width, channels)")
        print(f"  Actual: {sample_obs.shape} = ({args.n_envs}, 15, 23, 1)")
    print()

    # ========================================================================
    # MODEL CREATION
    # ========================================================================

    print("Creating PPO model...")
    model = create_model(
        env=env,
        learning_rate=args.learning_rate,
        batch_size=batch_size,
        n_steps=args.n_steps,
        device=device,
        use_cnn=use_cnn,
        checkpoint_path=args.checkpoint
    )
    print(f"✓ Model ready on device: {model.device}")

    # Print model summary
    if use_cnn and not args.checkpoint:
        print("\nModel Architecture:")
        print(f"  Policy: CnnPolicy")
        print(f"  Feature Extractor: ColorTilesCNN (custom 3x3 kernels)")
        print(f"  Input: (15, 23, 1) board state")
        print(f"  Output: 345 actions (Discrete)")
        print(f"  Features: 256-dim after CNN layers")
    print()

    # ========================================================================
    # CALLBACKS SETUP
    # ========================================================================

    print("Setting up callbacks...")
    callbacks = setup_callbacks(save_dir, eval_env)
    if callbacks:
        print("✓ Callbacks configured:")
        print("  - Checkpoint: every 10k steps")
        print("  - Evaluation: every 5k steps")
        if CALLBACKS_AVAILABLE:
            print("  - Custom metrics: every 100 episodes")
    print()

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================

    print(f"{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Expected speedup: 5-10x faster than CPU baseline")
    print(f"\nMonitor training:")
    print(f"  TensorBoard: tensorboard --logdir ./logs/tensorboard/")
    print(f"  Checkpoints: {save_dir}/")
    print(f"{'='*70}\n")

    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    end_time = datetime.now()
    duration = end_time - start_time

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================

    # Save final model
    final_path = save_dir / "ppo_colortiles_final.zip"
    model.save(final_path)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    print(f"Duration: {duration}")
    print(f"Timesteps/second: {args.total_timesteps / duration.total_seconds():.1f}")
    print(f"Final model saved to: {final_path}")
    print(f"Checkpoints saved in: {save_dir}")
    print(f"\nNext steps:")
    print(f"  1. Evaluate model:")
    print(f"     python -m rl.training.evaluate --checkpoint {final_path}")
    print(f"  2. View training logs:")
    print(f"     tensorboard --logdir ./logs/tensorboard/")
    print(f"  3. Play in GUI:")
    print(f"     python main.py  # Select AI vs Human mode")
    print(f"{'='*70}\n")

    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

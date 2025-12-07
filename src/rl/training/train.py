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


def make_env(rank: int, seed: int = 0, disable_time_limit: bool = True):
    """
    Factory function for creating environment instances.

    Args:
        rank: Environment index (for seeding)
        seed: Base random seed
        disable_time_limit: Whether to disable time limit (True for training)

    Returns:
        _init: Function that creates and returns the environment
    """
    def _init():
        env = ColorTilesEnv(disable_time_limit=disable_time_limit)
        env.reset(seed=seed + rank)
        env = Monitor(env)  # Wrap with Monitor for automatic logging
        return env
    return _init


def create_model(env, learning_rate: float = 3e-4, checkpoint_path: str = None):
    """
    Create or load PPO model with hyperparameters from RL plan (section 9).

    Args:
        env: Training environment (can be vectorized)
        learning_rate: Learning rate for optimizer
        checkpoint_path: Path to checkpoint file to resume from (optional)

    Returns:
        model: PPO model instance
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 is required. Install with: pip install stable-baselines3 torch")

    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env)
        print(f"Resumed from checkpoint. Continuing training...")
        return model

    # Create new model with hyperparameters from RL plan section 9
    print("Creating new PPO model with plan hyperparameters...")
    model = PPO(
        policy="MlpPolicy",  # Multi-layer perceptron policy
        env=env,
        learning_rate=learning_rate,
        n_steps=2048,           # Number of steps per update
        batch_size=64,          # Minibatch size
        n_epochs=10,            # Number of epochs for optimization
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # GAE parameter
        clip_range=0.2,         # PPO clipping epsilon
        ent_coef=0.01,          # Entropy coefficient (exploration)
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Gradient clipping
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device="auto"           # Use GPU if available
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
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Color Tiles RL agent with PPO")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                       help="Total number of training timesteps (default: 1M)")
    parser.add_argument("--n-envs", type=int, default=8,
                       help="Number of parallel environments (default: 8)")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint file to resume from")
    parser.add_argument("--save-dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")
    args = parser.parse_args()

    if not SB3_AVAILABLE:
        print("\nERROR: stable-baselines3 is not installed.")
        print("Please install required packages:")
        print("  pip install gymnasium stable-baselines3 torch tensorboard")
        return

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Color Tiles PPO Training")
    print(f"{'='*60}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save directory: {save_dir}")
    print(f"Random seed: {args.seed}")
    if args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
    print(f"{'='*60}\n")

    # Create parallel training environments
    print(f"Creating {args.n_envs} parallel environments...")
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
        print(f"✓ Created {args.n_envs} parallel environments (SubprocVecEnv)")
    else:
        env = DummyVecEnv([make_env(0, args.seed)])
        print(f"✓ Created 1 environment (DummyVecEnv)")

    # Create evaluation environment (single env, deterministic)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(args.n_envs, args.seed)])
    print("✓ Evaluation environment created\n")

    # Create or load model
    print("Setting up PPO model...")
    model = create_model(env, args.learning_rate, args.checkpoint)
    print(f"✓ Model ready on device: {model.device}\n")

    # Setup callbacks
    print("Setting up callbacks...")
    callbacks = setup_callbacks(save_dir, eval_env)
    if callbacks:
        print("✓ Callbacks configured:")
        print("  - Checkpoint: every 10k steps")
        print("  - Evaluation: every 5k steps")
        if CALLBACKS_AVAILABLE:
            print("  - Custom metrics: every 100 episodes")
    print()

    # Train
    print(f"Starting training for {args.total_timesteps:,} timesteps...")
    print(f"TensorBoard: tensorboard --logdir ./logs/tensorboard/")
    print(f"{'='*60}\n")

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

    # Save final model
    final_path = save_dir / "ppo_colortiles_final.zip"
    model.save(final_path)

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}")
    print(f"Duration: {duration}")
    print(f"Final model saved to: {final_path}")
    print(f"Checkpoints saved in: {save_dir}")
    print(f"\nTo evaluate:")
    print(f"  python -m rl.training.evaluate --checkpoint {final_path}")
    print(f"\nTo view training logs:")
    print(f"  tensorboard --logdir ./logs/tensorboard/")
    print(f"{'='*60}\n")

    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

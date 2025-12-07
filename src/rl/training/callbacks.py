"""
Custom callbacks for Color Tiles RL training

Provides additional metrics tracking and logging specific to the Color Tiles game:
- Win rate (percentage of episodes won)
- Average tiles cleared per episode
- Average episode length
- Invalid move rate
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    # Define dummy BaseCallback for when SB3 is not installed
    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.n_calls = 0
            self.locals = {}
            self.logger = None


class MetricsCallback(BaseCallback):
    """
    Custom callback for tracking Color Tiles specific metrics.

    Tracks rolling averages over the last N episodes:
    - Win rate: Percentage of episodes that ended with victory
    - Mean tiles cleared: Average number of tiles removed per episode
    - Mean episode length: Average number of steps per episode
    - Invalid move rate: Percentage of actions that were invalid

    Logs these metrics to TensorBoard every 100 episodes.
    """

    def __init__(self, log_dir: Path, window_size: int = 100, verbose: int = 0):
        """
        Initialize MetricsCallback.

        Args:
            log_dir: Directory for saving logs
            window_size: Number of episodes for rolling average (default: 100)
            verbose: Verbosity level (0: no output, 1: print metrics)
        """
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size

        # Metrics storage
        self.episode_metrics = defaultdict(list)
        self.episode_count = 0

        # For tracking current episode
        self.current_episode_invalid_moves = 0
        self.current_episode_steps = 0

    def _on_step(self) -> bool:
        """
        Called after each environment step.

        Returns:
            True to continue training, False to stop
        """
        if not SB3_AVAILABLE:
            return True

        # Check if any episode ended in any of the parallel envs
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])

        for i, done in enumerate(dones):
            if done and i < len(infos):
                info = infos[i]

                # Extract episode info from Monitor wrapper
                if 'episode' in info:
                    ep_info = info['episode']

                    # Store basic metrics from Monitor
                    self.episode_metrics['reward'].append(ep_info['r'])
                    self.episode_metrics['length'].append(ep_info['l'])

                    # Extract custom info if available
                    game_state = info.get('game_state', '')
                    remaining_tiles = info.get('remaining_tiles', 0)

                    # Win if game state is 'WON'
                    won = (game_state == 'WON')
                    self.episode_metrics['win'].append(1 if won else 0)

                    # Tiles cleared = 200 - remaining
                    tiles_cleared = 200 - remaining_tiles
                    self.episode_metrics['tiles_cleared'].append(tiles_cleared)

                    self.episode_count += 1

                    # Log every window_size episodes
                    if self.episode_count % self.window_size == 0:
                        self._log_metrics()

        return True

    def _log_metrics(self):
        """
        Log rolling statistics to tensorboard and console.

        Calculates and logs metrics over the last window_size episodes.
        """
        if not self.episode_metrics['reward']:
            return

        n = self.window_size

        # Calculate rolling averages
        recent_rewards = self.episode_metrics['reward'][-n:]
        recent_wins = self.episode_metrics['win'][-n:] if self.episode_metrics['win'] else []
        recent_tiles = self.episode_metrics['tiles_cleared'][-n:] if self.episode_metrics['tiles_cleared'] else []
        recent_lengths = self.episode_metrics['length'][-n:]

        mean_reward = np.mean(recent_rewards)
        win_rate = np.mean(recent_wins) if recent_wins else 0
        mean_tiles_cleared = np.mean(recent_tiles) if recent_tiles else 0
        mean_episode_length = np.mean(recent_lengths)

        # Log to tensorboard (if logger is available)
        if self.logger:
            self.logger.record("custom/mean_reward", mean_reward)
            self.logger.record("custom/win_rate", win_rate)
            self.logger.record("custom/mean_tiles_cleared", mean_tiles_cleared)
            self.logger.record("custom/mean_episode_length", mean_episode_length)

        # Print to console if verbose
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Episode {self.episode_count} (last {n} episodes)")
            print(f"{'='*60}")
            print(f"  Win Rate: {win_rate*100:.1f}%")
            print(f"  Mean Reward: {mean_reward:.2f}")
            print(f"  Mean Tiles Cleared: {mean_tiles_cleared:.1f}/200")
            print(f"  Mean Episode Length: {mean_episode_length:.1f} steps")
            print(f"{'='*60}\n")

    def _on_training_end(self):
        """Called at the end of training."""
        if self.verbose > 0:
            print(f"\nTraining completed! Total episodes: {self.episode_count}")


class ProgressCallback(BaseCallback):
    """
    Callback for tracking and displaying training progress.

    Displays progress updates at regular intervals during training.
    """

    def __init__(self, total_timesteps: int, update_freq: int = 1000, verbose: int = 1):
        """
        Initialize ProgressCallback.

        Args:
            total_timesteps: Total training timesteps
            update_freq: Frequency of progress updates in timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq

    def _on_step(self) -> bool:
        """Called after each environment step."""
        if self.n_calls % self.update_freq == 0 and self.verbose > 0:
            progress = (self.n_calls / self.total_timesteps) * 100
            print(f"Progress: {self.n_calls:,}/{self.total_timesteps:,} ({progress:.1f}%)")

        return True


class TensorBoardCallback(BaseCallback):
    """
    Enhanced TensorBoard logging callback.

    Logs additional training information to TensorBoard at regular intervals.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        """
        Initialize TensorBoardCallback.

        Args:
            log_freq: Frequency of logging in timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """Called after each environment step."""
        if not SB3_AVAILABLE:
            return True

        # Log additional training info periodically
        if self.n_calls % self.log_freq == 0 and self.logger:
            # Log learning rate (if available)
            if hasattr(self.model, 'learning_rate'):
                if callable(self.model.learning_rate):
                    lr = self.model.learning_rate(1.0)  # Current LR
                else:
                    lr = self.model.learning_rate
                self.logger.record("train/learning_rate", lr)

        return True

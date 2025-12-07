"""
AI Player for GUI integration

Provides a wrapper around trained PPO models to play Color Tiles in the GUI.
Handles model loading, state encoding, action prediction, and value extraction.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

try:
    import torch
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Warning: stable-baselines3 not available")

from color_tiles.domain.models import Position
from color_tiles.domain.constants import BOARD_WIDTH, BOARD_HEIGHT


class AIPlayer:
    """
    Wrapper for trained RL agent to play Color Tiles in GUI.

    Loads a trained PPO model and provides methods to:
    - Get actions from board state
    - Extract value estimates and action probabilities
    - Convert actions to board positions

    Usage:
        ai = AIPlayer("checkpoints/ppo_colortiles_best.zip")
        action, value, action_probs = ai.get_action(board)
        position = ai.action_to_position(action)
    """

    def __init__(self, checkpoint_path: str):
        """
        Load trained model from checkpoint.

        Args:
            checkpoint_path: Path to .zip checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ImportError: If stable-baselines3 is not installed
        """
        if not SB3_AVAILABLE:
            raise ImportError(
                "stable-baselines3 is required to use AIPlayer. "
                "Install with: pip install stable-baselines3 torch"
            )

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load model
        print(f"Loading AI model from: {checkpoint_path}")
        self.model = PPO.load(checkpoint_path)
        self.model.policy.eval()  # Set to evaluation mode

        print(f"✓ AI model loaded successfully")
        print(f"  Device: {self.model.device}")

    def get_action(
        self,
        board,
        deterministic: bool = True
    ) -> Tuple[int, float, Optional[np.ndarray]]:
        """
        Get action from current board state.

        Args:
            board: Board object from game engine
            deterministic: If True, select action with highest probability

        Returns:
            action: Action index (0-344)
            value: State value estimate from critic network
            action_probs: Probability distribution over all actions (345,)

        Example:
            action, value, probs = ai.get_action(game.get_board())
            print(f"AI selects action {action} with value {value:.2f}")
            print(f"Confidence: {probs[action]:.1%}")
        """
        # Encode board to observation
        state = self._encode_board(board)

        # Get action from policy
        action, _states = self.model.predict(state, deterministic=deterministic)

        # Extract value and action probabilities for visualization
        try:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(state).unsqueeze(0).to(self.model.device)

                # Get value estimate from critic
                value = self.model.policy.predict_values(obs_tensor).cpu().numpy()[0, 0]

                # Get action probabilities from actor
                distribution = self.model.policy.get_distribution(obs_tensor)
                action_probs = distribution.distribution.probs.cpu().numpy()[0]
        except Exception as e:
            print(f"Warning: Could not extract value/probs: {e}")
            value = 0.0
            action_probs = None

        return int(action), float(value), action_probs

    def action_to_position(self, action: int) -> Position:
        """
        Convert action index to board position.

        Uses row-major indexing: action = row * 23 + col

        Args:
            action: Action index (0-344)

        Returns:
            position: Position(row, col)

        Example:
            pos = ai.action_to_position(100)
            print(f"Action 100 → Position({pos.row}, {pos.col})")
        """
        row = action // BOARD_WIDTH
        col = action % BOARD_WIDTH
        return Position(row, col)

    def _encode_board(self, board) -> np.ndarray:
        """
        Encode Board object to observation array.

        Uses the same encoding as ColorTilesEnv:
        - 0 = empty cell
        - 1-10 = Color enum value + 1

        Args:
            board: Board object from game engine

        Returns:
            state: Numpy array (15, 23) with int8 values 0-10
        """
        state = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)

        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                pos = Position(row, col)
                cell = board.get_cell(pos)
                if not cell.is_empty:
                    # Offset Color enum by 1 (same as ColorTilesEnv)
                    state[row, col] = cell.color.value + 1
                # else: already 0 (empty)

        return state

    def get_checkpoint_info(self) -> dict:
        """
        Get metadata about loaded checkpoint.

        Returns:
            info: Dictionary with checkpoint information
        """
        return {
            'path': str(self.checkpoint_path),
            'name': self.checkpoint_path.stem,
            'size_mb': self.checkpoint_path.stat().st_size / (1024 * 1024),
            'device': str(self.model.device) if hasattr(self.model, 'device') else 'unknown',
        }

    def __str__(self) -> str:
        """String representation of AIPlayer."""
        info = self.get_checkpoint_info()
        return (
            f"AIPlayer(\n"
            f"  checkpoint='{info['name']}',\n"
            f"  size={info['size_mb']:.1f}MB,\n"
            f"  device={info['device']}\n"
            f")"
        )


def main():
    """Demo/test function for AIPlayer."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rl.inference.ai_player <checkpoint_path>")
        print("\nExample:")
        print("  python -m rl.inference.ai_player checkpoints/ppo_colortiles_best.zip")
        return

    checkpoint_path = sys.argv[1]

    try:
        # Load AI
        ai = AIPlayer(checkpoint_path)
        print(ai)

        # Test with dummy board
        from color_tiles.utils.board_generator import BoardGenerator

        print("\nGenerating test board...")
        board = BoardGenerator.generate_random_board()

        print("Getting AI action...")
        action, value, probs = ai.get_action(board)

        position = ai.action_to_position(action)

        print(f"\nAI Decision:")
        print(f"  Action: {action}")
        print(f"  Position: ({position.row}, {position.col})")
        print(f"  Value Estimate: {value:.3f}")
        if probs is not None:
            print(f"  Confidence: {probs[action]:.1%}")
            print(f"  Top 5 actions:")
            top_5 = np.argsort(probs)[-5:][::-1]
            for rank, a in enumerate(top_5, 1):
                p = ai.action_to_position(a)
                print(f"    {rank}. Action {a} @ ({p.row},{p.col}): {probs[a]:.1%}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

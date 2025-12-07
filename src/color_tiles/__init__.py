"""Color Tiles Game Engine.

A pure Python game logic engine for the Color Tiles puzzle game.
Completely separated from GUI for easy integration with PyQt6 or other frameworks.
"""

from color_tiles.domain.models import Color, Position, Cell, GameState, MoveResult
from color_tiles.domain.constants import (
    BOARD_WIDTH,
    BOARD_HEIGHT,
    TOTAL_CELLS,
    NUM_COLORS,
    TILES_PER_COLOR,
    TOTAL_TILES,
    EMPTY_CELLS,
    INITIAL_TIME,
    PENALTY_TIME,
    POINTS_PER_TILE,
    MIN_TILES_FOR_REMOVAL
)
from color_tiles.domain.exceptions import (
    ColorTilesException,
    InvalidPositionError,
    InvalidMoveError,
    GameNotStartedError
)

__version__ = "0.1.0"

__all__ = [
    # Models
    "Color",
    "Position",
    "Cell",
    "GameState",
    "MoveResult",
    # Constants
    "BOARD_WIDTH",
    "BOARD_HEIGHT",
    "TOTAL_CELLS",
    "NUM_COLORS",
    "TILES_PER_COLOR",
    "TOTAL_TILES",
    "EMPTY_CELLS",
    "INITIAL_TIME",
    "PENALTY_TIME",
    "POINTS_PER_TILE",
    "MIN_TILES_FOR_REMOVAL",
    # Exceptions
    "ColorTilesException",
    "InvalidPositionError",
    "InvalidMoveError",
    "GameNotStartedError",
]

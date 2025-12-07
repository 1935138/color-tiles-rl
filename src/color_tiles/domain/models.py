"""Domain models for Color Tiles game."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Color(Enum):
    """10가지 타일 색상."""
    WHITE = 0
    PINK = 1
    BLUE = 2
    SKY_BLUE = 3
    GREEN = 4
    ORANGE = 5
    YELLOW = 6
    PURPLE = 7
    BROWN = 8
    RED = 9


class GameState(Enum):
    """게임 상태."""
    READY = "ready"
    PLAYING = "playing"
    WON = "won"
    LOST_TIME = "lost_time"
    LOST_NO_MOVES = "lost_no_moves"


@dataclass(frozen=True)
class Position:
    """보드 상의 불변 위치."""
    row: int
    col: int

    def __repr__(self) -> str:
        return f"Position({self.row}, {self.col})"


@dataclass(frozen=True)
class Cell:
    """보드의 단일 셀을 나타냄."""
    position: Position
    color: Optional[Color]  # None = 빈칸

    @property
    def is_empty(self) -> bool:
        """셀이 비어있는지 확인."""
        return self.color is None

    def __repr__(self) -> str:
        color_str = self.color.name if self.color else "EMPTY"
        return f"Cell({self.position}, {color_str})"


@dataclass
class MoveResult:
    """이동 시도의 결과."""
    success: bool
    tiles_removed: list[Cell]
    points_earned: int
    time_penalty: float
    message: str
    game_state: GameState

    def __repr__(self) -> str:
        return (f"MoveResult(success={self.success}, "
                f"tiles_removed={len(self.tiles_removed)}, "
                f"points={self.points_earned}, "
                f"penalty={self.time_penalty}s, "
                f"state={self.game_state.value})")

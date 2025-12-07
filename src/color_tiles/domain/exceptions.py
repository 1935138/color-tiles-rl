"""Custom exceptions for Color Tiles game."""


class ColorTilesException(Exception):
    """Base exception for Color Tiles game."""
    pass


class InvalidPositionError(ColorTilesException):
    """위치가 보드 범위를 벗어남."""
    pass


class InvalidMoveError(ColorTilesException):
    """현재 게임 상태에서 허용되지 않는 이동."""
    pass


class GameNotStartedError(ColorTilesException):
    """게임 시작 전 작업 시도."""
    pass

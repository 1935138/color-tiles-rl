"""Board state management."""

import copy
from typing import Optional

from color_tiles.domain.models import Color, Position, Cell
from color_tiles.domain.constants import BOARD_WIDTH, BOARD_HEIGHT
from color_tiles.domain.exceptions import InvalidPositionError


class Board:
    """보드 상태를 관리하는 핵심 클래스.

    보드는 23×15 그리드로 구성되며, 각 셀은 타일 색상 또는 빈칸(None)을 포함합니다.
    """

    def __init__(self, cells: list[list[Optional[Color]]]):
        """보드를 2D 색상 배열로 초기화.

        Args:
            cells: BOARD_HEIGHT × BOARD_WIDTH 크기의 2D 리스트.
                   각 요소는 Color 또는 None (빈칸).

        Raises:
            ValueError: 보드 크기가 올바르지 않은 경우.
        """
        if len(cells) != BOARD_HEIGHT:
            raise ValueError(f"Board must have {BOARD_HEIGHT} rows, got {len(cells)}")

        for i, row in enumerate(cells):
            if len(row) != BOARD_WIDTH:
                raise ValueError(
                    f"Row {i} must have {BOARD_WIDTH} columns, got {len(row)}"
                )

        self._cells = cells

    def get_cell(self, position: Position) -> Cell:
        """특정 위치의 셀 조회.

        Args:
            position: 조회할 위치.

        Returns:
            해당 위치의 Cell 객체.

        Raises:
            InvalidPositionError: 위치가 보드 범위를 벗어난 경우.
        """
        if not self._is_valid_position(position):
            raise InvalidPositionError(
                f"Position {position} is out of bounds "
                f"(board size: {BOARD_HEIGHT}×{BOARD_WIDTH})"
            )

        color = self._cells[position.row][position.col]
        return Cell(position=position, color=color)

    def is_empty(self, position: Position) -> bool:
        """셀이 비어있는지 확인.

        Args:
            position: 확인할 위치.

        Returns:
            빈칸이면 True, 타일이 있으면 False.

        Raises:
            InvalidPositionError: 위치가 보드 범위를 벗어난 경우.
        """
        cell = self.get_cell(position)
        return cell.is_empty

    def remove_tiles(self, positions: list[Position]) -> int:
        """지정된 위치의 타일들을 제거 (빈칸으로 만듦).

        Args:
            positions: 제거할 타일들의 위치 리스트.

        Returns:
            제거된 타일의 개수.

        Raises:
            InvalidPositionError: 위치가 보드 범위를 벗어난 경우.
        """
        count = 0
        for pos in positions:
            if not self._is_valid_position(pos):
                raise InvalidPositionError(
                    f"Position {pos} is out of bounds"
                )

            if not self.is_empty(pos):
                self._cells[pos.row][pos.col] = None
                count += 1

        return count

    def get_all_tiles(self) -> list[Cell]:
        """보드의 모든 비어있지 않은 셀 조회.

        Returns:
            타일이 있는 모든 Cell 객체의 리스트.
        """
        tiles = []
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                pos = Position(row, col)
                cell = self.get_cell(pos)
                if not cell.is_empty:
                    tiles.append(cell)
        return tiles

    def count_tiles(self) -> int:
        """보드에 남아있는 타일 개수 계산.

        Returns:
            남은 타일의 개수.
        """
        return len(self.get_all_tiles())

    def get_neighbors(self, position: Position) -> dict[str, Position]:
        """특정 위치의 유효한 인접 위치들 (상하좌우) 조회.

        Args:
            position: 기준 위치.

        Returns:
            방향("up", "down", "left", "right")을 키로 하는 인접 Position 딕셔너리.
            보드 범위를 벗어나는 방향은 포함되지 않음.
        """
        neighbors = {}

        # 상
        if position.row > 0:
            neighbors["up"] = Position(position.row - 1, position.col)

        # 하
        if position.row < BOARD_HEIGHT - 1:
            neighbors["down"] = Position(position.row + 1, position.col)

        # 좌
        if position.col > 0:
            neighbors["left"] = Position(position.row, position.col - 1)

        # 우
        if position.col < BOARD_WIDTH - 1:
            neighbors["right"] = Position(position.row, position.col + 1)

        return neighbors

    def to_dict(self) -> dict:
        """GUI용 보드 상태를 직렬화.

        Returns:
            보드 상태를 나타내는 딕셔너리.
        """
        # Color enum을 문자열로 변환
        cells_data = []
        for row in self._cells:
            row_data = []
            for color in row:
                row_data.append(color.name if color else None)
            cells_data.append(row_data)

        return {
            "cells": cells_data,
            "width": BOARD_WIDTH,
            "height": BOARD_HEIGHT,
            "remaining_tiles": self.count_tiles()
        }

    def copy(self) -> "Board":
        """보드의 깊은 복사본 생성.

        Undo/Redo 또는 Save/Load 기능을 위해 사용.

        Returns:
            새로운 Board 객체 (독립적인 복사본).
        """
        cells_copy = copy.deepcopy(self._cells)
        return Board(cells_copy)

    def _is_valid_position(self, position: Position) -> bool:
        """위치가 보드 범위 내에 있는지 확인.

        Args:
            position: 확인할 위치.

        Returns:
            유효하면 True, 아니면 False.
        """
        return (0 <= position.row < BOARD_HEIGHT and
                0 <= position.col < BOARD_WIDTH)

    def __repr__(self) -> str:
        return f"Board({BOARD_HEIGHT}×{BOARD_WIDTH}, tiles={self.count_tiles()})"

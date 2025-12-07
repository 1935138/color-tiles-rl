"""Tile finding algorithm - core game mechanic."""

from typing import Optional

from color_tiles.domain.models import Position, Cell
from color_tiles.domain.constants import BOARD_WIDTH, BOARD_HEIGHT
from color_tiles.engine.board import Board


class TileFinder:
    """4방향 타일 찾기 알고리즘 구현.

    게임의 핵심 메커니즘:
    빈칸에서 상/하/좌/우 4방향으로 빈칸을 통과하며
    각 방향에서 첫 번째로 만나는 타일을 찾습니다.
    """

    def __init__(self, board: Board):
        """TileFinder 초기화.

        Args:
            board: 타일을 찾을 Board 객체.
        """
        self._board = board

    def find_tiles_from_position(self, position: Position) -> list[Cell]:
        """빈칸 위치에서 4방향의 타일 찾기.

        알고리즘:
        1. 클릭한 위치가 빈칸이 아니면 빈 리스트 반환
        2. 4방향(상/하/좌/우)으로 탐색
        3. 각 방향에서:
           - 빈칸을 통과하며 이동
           - 첫 번째 타일을 만나면 기록하고 중단
           - 보드 경계에 도달하면 중단
        4. 찾은 모든 타일 반환

        Args:
            position: 탐색을 시작할 빈칸 위치.

        Returns:
            4방향에서 찾은 타일들의 Cell 리스트 (0~4개).
        """
        if not self._board.is_empty(position):
            return []

        # 4방향: (dy, dx) 형식
        directions = [
            (-1, 0),   # 상 (위로)
            (1, 0),    # 하 (아래로)
            (0, -1),   # 좌 (왼쪽으로)
            (0, 1),    # 우 (오른쪽으로)
        ]

        found_tiles = []

        for direction in directions:
            tile = self._find_in_direction(position, direction)
            if tile:
                found_tiles.append(tile)

        return found_tiles

    def _find_in_direction(
        self,
        start: Position,
        direction: tuple[int, int]
    ) -> Optional[Cell]:
        """한 방향으로 타일 탐색.

        빈칸을 통과하며 이동하다가:
        - 타일을 만나면 해당 타일 반환
        - 보드 경계에 도달하면 None 반환

        Args:
            start: 시작 위치.
            direction: 이동 방향 (dy, dx).

        Returns:
            찾은 타일의 Cell 또는 None.
        """
        dy, dx = direction
        current_row = start.row + dy
        current_col = start.col + dx

        # 빈칸을 통과하며 이동
        while (0 <= current_row < BOARD_HEIGHT and
               0 <= current_col < BOARD_WIDTH):
            current_pos = Position(current_row, current_col)
            cell = self._board.get_cell(current_pos)

            if not cell.is_empty:
                # 타일 발견
                return cell

            # 다음 칸으로 이동 (같은 방향으로 계속)
            current_row += dy
            current_col += dx

        # 보드 경계에 도달, 타일 없음
        return None

    def __repr__(self) -> str:
        return f"TileFinder(board={self._board})"

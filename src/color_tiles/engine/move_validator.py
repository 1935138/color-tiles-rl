"""Move validation logic."""

from typing import Dict

from color_tiles.domain.models import Position, Cell, Color
from color_tiles.domain.constants import MIN_TILES_FOR_REMOVAL, BOARD_WIDTH, BOARD_HEIGHT
from color_tiles.engine.board import Board
from color_tiles.engine.tile_finder import TileFinder


class MoveValidator:
    """이동 유효성 검증 및 유효한 이동 탐색.

    이동이 유효하려면:
    - 클릭한 위치가 빈칸이어야 함
    - 4방향에서 찾은 타일 중 같은 색상이 2개 이상 있어야 함
    """

    def __init__(self, board: Board):
        """MoveValidator 초기화.

        Args:
            board: 검증할 Board 객체.
        """
        self._board = board
        self._tile_finder = TileFinder(board)

    def is_valid_move(self, position: Position) -> tuple[bool, list[Cell]]:
        """특정 위치 클릭이 유효한 이동인지 확인.

        알고리즘:
        1. 위치가 빈칸이 아니면 유효하지 않음
        2. TileFinder로 4방향의 타일 찾기
        3. 색상별로 그룹화
        4. 2개 이상인 색상이 있으면 유효

        Args:
            position: 클릭할 위치.

        Returns:
            (is_valid, tiles_to_remove) 튜플.
            - is_valid: 유효한 이동이면 True
            - tiles_to_remove: 제거할 타일 리스트 (유효하지 않으면 빈 리스트)
        """
        if not self._board.is_empty(position):
            return False, []

        # 4방향에서 타일 찾기
        found_tiles = self._tile_finder.find_tiles_from_position(position)

        # 색상별 그룹화
        color_groups: Dict[Color, list[Cell]] = {}
        for tile in found_tiles:
            if tile.color not in color_groups:
                color_groups[tile.color] = []
            color_groups[tile.color].append(tile)

        # 2개 이상인 모든 색상의 타일 수집
        tiles_to_remove = []
        for color, tiles in color_groups.items():
            if len(tiles) >= MIN_TILES_FOR_REMOVAL:
                tiles_to_remove.extend(tiles)

        if tiles_to_remove:
            return True, tiles_to_remove
        return False, []

    def find_all_valid_moves(self) -> list[Position]:
        """보드의 모든 유효한 이동 위치 찾기.

        게임이 여전히 플레이 가능한지 확인하는 데 사용됩니다.

        Returns:
            유효한 이동이 가능한 빈칸 Position 리스트.
        """
        valid_positions = []

        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                pos = Position(row, col)
                if self._board.is_empty(pos):
                    is_valid, _ = self.is_valid_move(pos)
                    if is_valid:
                        valid_positions.append(pos)

        return valid_positions

    def has_valid_moves(self) -> bool:
        """최소 하나의 유효한 이동이 존재하는지 확인.

        Returns:
            유효한 이동이 있으면 True, 없으면 False.
        """
        # 최적화: 첫 번째 유효한 이동을 찾으면 즉시 반환
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                pos = Position(row, col)
                if self._board.is_empty(pos):
                    is_valid, _ = self.is_valid_move(pos)
                    if is_valid:
                        return True

        return False

    def __repr__(self) -> str:
        return f"MoveValidator(board={self._board})"

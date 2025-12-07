"""Random board generation for Color Tiles."""

import random
from typing import Optional

from color_tiles.domain.models import Color
from color_tiles.domain.constants import (
    BOARD_WIDTH,
    BOARD_HEIGHT,
    TILES_PER_COLOR,
    EMPTY_CELLS
)
from color_tiles.engine.board import Board


class BoardGenerator:
    """랜덤 게임 보드 생성기."""

    @staticmethod
    def generate_random_board() -> Board:
        """완전 랜덤 보드 생성.

        알고리즘:
        1. 각 색상별로 20개씩 타일 생성 (총 200개)
        2. 145개의 빈칸(None) 추가
        3. 전체 섞기
        4. 15×23 2D 배열로 변환

        Returns:
            새로 생성된 랜덤 Board 객체.
        """
        # 1. 타일 리스트 생성: 10가지 색상 × 20개씩
        tiles: list[Optional[Color]] = []
        for color in Color:
            tiles.extend([color] * TILES_PER_COLOR)

        # 2. 빈칸 추가
        tiles.extend([None] * EMPTY_CELLS)

        # 3. 섞기
        random.shuffle(tiles)

        # 4. 2D 배열로 변환 (15 rows × 23 cols)
        cells = []
        for row in range(BOARD_HEIGHT):
            row_cells = []
            for col in range(BOARD_WIDTH):
                index = row * BOARD_WIDTH + col
                row_cells.append(tiles[index])
            cells.append(row_cells)

        return Board(cells)

    @staticmethod
    def generate_solvable_board(max_attempts: int = 100) -> Board:
        """최소 하나의 유효한 이동이 있는 보드 생성.

        랜덤 보드를 생성하되, 유효한 이동이 없으면 재시도합니다.
        이 메서드는 MoveValidator가 구현된 후에 제대로 작동합니다.

        Args:
            max_attempts: 최대 시도 횟수.

        Returns:
            유효한 이동이 있는 Board 객체.

        Note:
            현재는 단순히 랜덤 보드를 반환합니다.
            향후 MoveValidator를 사용하여 유효성 검증을 추가할 수 있습니다.
        """
        # TODO: MoveValidator 구현 후 유효한 이동 체크 추가
        return BoardGenerator.generate_random_board()

#!/usr/bin/env python3
"""CLI Demo for Color Tiles Game Engine.

간단한 터미널 인터페이스로 게임 엔진을 테스트합니다.
"""

import sys
import os

# src 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from color_tiles.domain.models import Position, GameState
from color_tiles.engine.game import GameEngine, GameObserver
from color_tiles.utils.board_generator import BoardGenerator


class CLIObserver(GameObserver):
    """CLI용 게임 옵저버."""

    def on_move_made(self, result):
        """이동 결과 출력."""
        print(f"\n{result.message}")

    def on_game_state_changed(self, state):
        """게임 상태 변경 출력."""
        if state == GameState.WON:
            print("\n" + "=" * 60)
            print("🎉 축하합니다! 모든 타일을 제거했습니다! 🎉")
            print("=" * 60)
        elif state == GameState.LOST_TIME:
            print("\n" + "=" * 60)
            print("⏰ 시간 초과! 게임 종료.")
            print("=" * 60)
        elif state == GameState.LOST_NO_MOVES:
            print("\n" + "=" * 60)
            print("❌ 유효한 이동이 없어 게임 종료.")
            print("=" * 60)

    def on_time_updated(self, remaining):
        """시간 업데이트 (CLI에서는 사용하지 않음)."""
        pass


def print_board(game: GameEngine):
    """보드를 시각적으로 출력."""
    board_state = game.get_board_state()
    cells = board_state["cells"]
    width = board_state["width"]
    height = board_state["height"]

    # 색상을 문자로 매핑
    color_map = {
        "WHITE": "W",
        "PINK": "P",
        "BLUE": "B",
        "SKY_BLUE": "S",
        "GREEN": "G",
        "ORANGE": "O",
        "YELLOW": "Y",
        "PURPLE": "U",
        "BROWN": "N",
        "RED": "R",
        None: "."
    }

    print("\n" + "=" * 60)
    print("Color Tiles - CLI Demo")
    print("=" * 60)

    # 열 번호 출력 (0-22)
    print("   ", end="")
    for col in range(width):
        if col < 10:
            print(col, end=" ")
        else:
            print(col % 10, end=" ")
    print()

    # 보드 출력
    for row in range(height):
        # 행 번호 출력
        print(f"{row:2d} ", end="")

        for col in range(width):
            color = cells[row][col]
            char = color_map.get(color, "?")
            print(char, end=" ")
        print()

    print("=" * 60)


def print_status(game: GameEngine):
    """게임 상태 정보 출력."""
    board_state = game.get_board_state()
    remaining_tiles = board_state["remaining_tiles"]
    score = game.get_score()
    remaining_time = game.get_remaining_time()

    print(f"남은 타일: {remaining_tiles:3d} | 점수: {score:3d} | 시간: {remaining_time:6.1f}초")
    print("=" * 60)


def print_help():
    """도움말 출력."""
    print("\n사용법:")
    print("  - 빈칸의 좌표(행 열)를 입력하여 타일 제거")
    print("  - 예: '5 10' (5행 10열)")
    print("  - 'help' : 도움말")
    print("  - 'quit' : 종료")
    print("\n색상 코드:")
    print("  W=White, P=Pink, B=Blue, S=SkyBlue, G=Green")
    print("  O=Orange, Y=Yellow, U=Purple, N=Brown, R=Red")
    print("  . = 빈칸")
    print()


def main():
    """메인 CLI 루프."""
    print("\n" + "=" * 60)
    print("Color Tiles Game Engine - CLI Demo")
    print("=" * 60)
    print("\n게임 보드를 생성하는 중...")

    # 보드 생성 및 게임 초기화
    board = BoardGenerator.generate_random_board()
    game = GameEngine(board)

    # 옵저버 등록
    observer = CLIObserver()
    game.add_observer(observer)

    print("보드가 생성되었습니다!")
    print_help()

    # 게임 시작
    game.start_game()

    # 게임 루프
    while True:
        # 보드 및 상태 출력
        print_board(game)
        print_status(game)

        # 게임 종료 확인
        state = game.get_game_state()
        if state != GameState.PLAYING:
            print(f"\n최종 점수: {game.get_score()}")
            print("게임을 종료합니다.")
            break

        # 시간 체크
        if game.get_remaining_time() <= 0:
            print("\n⏰ 시간 초과! 게임 종료.")
            print(f"최종 점수: {game.get_score()}")
            break

        # 사용자 입력
        try:
            user_input = input("\n빈칸 좌표 입력 (행 열): ").strip().lower()

            if user_input == "quit":
                print("게임을 종료합니다.")
                break

            if user_input == "help":
                print_help()
                continue

            # 좌표 파싱
            parts = user_input.split()
            if len(parts) != 2:
                print("잘못된 입력 형식입니다. '행 열' 형식으로 입력하세요. (예: 5 10)")
                continue

            row = int(parts[0])
            col = int(parts[1])

            # 이동 실행
            position = Position(row, col)
            result = game.make_move(position)

        except ValueError:
            print("잘못된 입력입니다. 숫자를 입력하세요.")
        except Exception as e:
            print(f"오류 발생: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n게임을 중단합니다.")
        sys.exit(0)

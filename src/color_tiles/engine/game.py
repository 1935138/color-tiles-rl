"""Main game engine - Public API for GUI integration."""

import time
from abc import ABC, abstractmethod
from typing import Optional

from color_tiles.domain.models import Position, GameState, MoveResult
from color_tiles.domain.constants import (
    INITIAL_TIME,
    PENALTY_TIME,
    POINTS_PER_TILE,
    TOTAL_TILES
)
from color_tiles.domain.exceptions import GameNotStartedError, InvalidMoveError
from color_tiles.engine.board import Board
from color_tiles.engine.move_validator import MoveValidator


class GameObserver(ABC):
    """GUI가 게임 이벤트를 관찰하기 위한 인터페이스."""

    @abstractmethod
    def on_move_made(self, result: MoveResult) -> None:
        """이동이 완료된 후 호출됨.

        Args:
            result: 이동 결과 정보.
        """
        pass

    @abstractmethod
    def on_game_state_changed(self, state: GameState) -> None:
        """게임 상태가 변경될 때 호출됨.

        Args:
            state: 새로운 게임 상태.
        """
        pass

    @abstractmethod
    def on_time_updated(self, remaining: float) -> None:
        """시간이 업데이트될 때 호출됨 (선택적).

        Args:
            remaining: 남은 시간 (초).
        """
        pass


class GameEngine:
    """메인 게임 오케스트레이터 - GUI용 Primary API.

    이 클래스는 게임의 모든 로직을 관리합니다:
    - 게임 상태 (ready, playing, won, lost)
    - 시간 추적
    - 이동 실행
    - 점수 계산
    - 옵저버 알림
    """

    def __init__(self, board: Board, time_limit: float = INITIAL_TIME):
        """GameEngine 초기화.

        Args:
            board: 게임에 사용할 Board 객체.
            time_limit: 제한 시간 (초). 기본값은 120초.
        """
        self._board = board
        self._time_limit = time_limit
        self._game_state = GameState.READY
        self._score = 0

        # 시간 관리
        self._start_time: Optional[float] = None
        self._time_penalty_accumulated = 0.0

        # 옵저버들
        self._observers: list[GameObserver] = []

        # 검증기
        self._validator = MoveValidator(board)

    # ========== 게임 생명주기 ==========

    def start_game(self) -> None:
        """게임 시작 (타이머 시작)."""
        self._start_time = time.time()
        self._game_state = GameState.PLAYING
        self._notify_state_changed(GameState.PLAYING)

    def reset_game(self, new_board: Optional[Board] = None) -> None:
        """게임 리셋.

        Args:
            new_board: 새 보드. None이면 현재 보드를 복사하여 사용.
        """
        if new_board:
            self._board = new_board
        else:
            self._board = self._board.copy()

        self._game_state = GameState.READY
        self._score = 0
        self._start_time = None
        self._time_penalty_accumulated = 0.0
        self._validator = MoveValidator(self._board)
        self._notify_state_changed(GameState.READY)

    # ========== 이동 실행 ==========

    def make_move(self, position: Position) -> MoveResult:
        """위치를 클릭하여 이동 실행.

        프로세스:
        1. 게임 상태 확인 (playing이어야 함)
        2. 시간 확인 (남은 시간이 있어야 함)
        3. 이동 유효성 검증
        4. 유효하면: 타일 제거, 점수 증가
        5. 유효하지 않으면: 시간 패널티 적용
        6. 승/패 조건 확인
        7. 옵저버 알림

        Args:
            position: 클릭한 위치.

        Returns:
            MoveResult 객체 (성공 여부, 제거된 타일 등).

        Raises:
            GameNotStartedError: 게임이 시작되지 않은 경우.
            InvalidMoveError: 게임이 이미 종료된 경우.
        """
        # 상태 확인
        if self._game_state == GameState.READY:
            raise GameNotStartedError("Game has not been started. Call start_game() first.")

        if self._game_state != GameState.PLAYING:
            raise InvalidMoveError(f"Cannot make move in state: {self._game_state.value}")

        # 시간 확인
        remaining_time = self.get_remaining_time()
        if remaining_time <= 0:
            self._game_state = GameState.LOST_TIME
            result = MoveResult(
                success=False,
                tiles_removed=[],
                points_earned=0,
                time_penalty=0.0,
                message="시간 초과! 게임 종료.",
                game_state=GameState.LOST_TIME
            )
            self._notify_state_changed(GameState.LOST_TIME)
            self._notify_move_made(result)
            return result

        # 이동 검증
        is_valid, tiles_to_remove = self._validator.is_valid_move(position)

        if is_valid:
            # 유효한 이동: 타일 제거 및 점수 증가
            num_removed = self._board.remove_tiles([t.position for t in tiles_to_remove])
            points = num_removed * POINTS_PER_TILE
            self._score += points

            result = MoveResult(
                success=True,
                tiles_removed=tiles_to_remove,
                points_earned=points,
                time_penalty=0.0,
                message=f"{num_removed}개 타일 제거! +{points}점",
                game_state=self._game_state
            )

            # 승리 조건 확인
            if self._board.count_tiles() == 0:
                self._game_state = GameState.WON
                result.game_state = GameState.WON
                result.message = f"축하합니다! 모든 타일 제거 완료! 최종 점수: {self._score}"
                self._notify_state_changed(GameState.WON)

            # 검증기 업데이트 (보드가 변경되었으므로)
            self._validator = MoveValidator(self._board)

        else:
            # 유효하지 않은 이동: 시간 패널티
            self._time_penalty_accumulated += PENALTY_TIME
            remaining_after_penalty = self.get_remaining_time()

            if remaining_after_penalty <= 0:
                # 패널티로 인해 시간 초과
                self._game_state = GameState.LOST_TIME
                result = MoveResult(
                    success=False,
                    tiles_removed=[],
                    points_earned=0,
                    time_penalty=PENALTY_TIME,
                    message=f"잘못된 이동! 시간 패널티 {PENALTY_TIME}초. 시간 초과로 게임 종료.",
                    game_state=GameState.LOST_TIME
                )
                self._notify_state_changed(GameState.LOST_TIME)
            else:
                result = MoveResult(
                    success=False,
                    tiles_removed=[],
                    points_earned=0,
                    time_penalty=PENALTY_TIME,
                    message=f"잘못된 이동! 시간 패널티 {PENALTY_TIME}초. (남은 시간: {remaining_after_penalty:.1f}초)",
                    game_state=self._game_state
                )

        # 패배 조건 확인 (유효한 이동이 없는 경우)
        if self._game_state == GameState.PLAYING:
            if not self._validator.has_valid_moves():
                self._game_state = GameState.LOST_NO_MOVES
                result.game_state = GameState.LOST_NO_MOVES
                result.message += " | 유효한 이동이 없어 게임 종료."
                self._notify_state_changed(GameState.LOST_NO_MOVES)

        self._notify_move_made(result)
        return result

    # ========== 상태 조회 ==========

    def get_game_state(self) -> GameState:
        """현재 게임 상태 조회.

        Returns:
            현재 GameState.
        """
        return self._game_state

    def get_remaining_time(self) -> float:
        """남은 시간 계산.

        Returns:
            남은 시간 (초). 음수일 수 없음 (최소 0).
        """
        if self._start_time is None:
            return self._time_limit

        elapsed = time.time() - self._start_time
        remaining = self._time_limit - elapsed - self._time_penalty_accumulated
        return max(0.0, remaining)

    def get_score(self) -> int:
        """현재 점수 조회.

        Returns:
            현재 점수.
        """
        return self._score

    def get_board_state(self) -> dict:
        """GUI용 보드 상태 조회.

        Returns:
            보드 상태를 나타내는 딕셔너리.
        """
        return self._board.to_dict()

    def get_board(self) -> Board:
        """Board 객체 직접 접근.

        Returns:
            현재 Board 객체.
        """
        return self._board

    # ========== 옵저버 패턴 ==========

    def add_observer(self, observer: GameObserver) -> None:
        """옵저버 등록.

        Args:
            observer: 등록할 GameObserver 구현체.
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: GameObserver) -> None:
        """옵저버 제거.

        Args:
            observer: 제거할 GameObserver 구현체.
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def _notify_move_made(self, result: MoveResult) -> None:
        """모든 옵저버에게 이동 완료 알림."""
        for observer in self._observers:
            observer.on_move_made(result)

    def _notify_state_changed(self, state: GameState) -> None:
        """모든 옵저버에게 상태 변경 알림."""
        for observer in self._observers:
            observer.on_game_state_changed(state)

    def _notify_time_updated(self, remaining: float) -> None:
        """모든 옵저버에게 시간 업데이트 알림."""
        for observer in self._observers:
            observer.on_time_updated(remaining)

    def __repr__(self) -> str:
        return (f"GameEngine(state={self._game_state.value}, "
                f"score={self._score}, "
                f"remaining_time={self.get_remaining_time():.1f}s)")

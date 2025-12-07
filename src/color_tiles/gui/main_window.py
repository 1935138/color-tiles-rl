"""Main window for Color Tiles GUI application."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QMessageBox
)
from PyQt6.QtCore import QTimer

from color_tiles.domain.models import Position, GameState
from color_tiles.engine.game import GameEngine, GameObserver
from color_tiles.utils.board_generator import BoardGenerator
from color_tiles.gui.board_widget import BoardWidget
from color_tiles.gui.info_panel import InfoPanel
from color_tiles.gui.control_panel import ControlPanel
from color_tiles.gui.ai_control_panel import AIControlPanel
from color_tiles.gui.ai_status_panel import AIStatusPanel

try:
    from rl.inference.ai_player import AIPlayer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class ColorTilesObserver(GameObserver):
    """게임 엔진 이벤트를 GUI에 전달하는 옵저버."""

    def __init__(self, main_window):
        """Observer 초기화.

        Args:
            main_window: MainWindow 인스턴스.
        """
        self.window = main_window

    def on_move_made(self, result):
        """이동 완료 시 호출."""
        # 보드 업데이트
        self.window.update_board_display()

        # 점수 및 타일 수 업데이트
        self.window.update_info_display()

        # 실패한 이동: 팝업 제거 (시간 패널티만 자동 적용됨)
        # game.py Line 186에서 이미 PENALTY_TIME(10초) 감소 처리

    def on_game_state_changed(self, state):
        """게임 상태 변경 시 호출."""
        if state == GameState.WON:
            self.window.on_victory()
        elif state == GameState.LOST_TIME:
            self.window.on_game_over("시간 초과")
        elif state == GameState.LOST_NO_MOVES:
            self.window.on_game_over("유효한 이동 없음")
        elif state == GameState.PLAYING:
            self.window.info_panel.set_playing_state()
        elif state == GameState.READY:
            self.window.info_panel.set_ready_state()

    def on_time_updated(self, remaining):
        """시간 업데이트 시 호출."""
        self.window.info_panel.update_time(remaining)


class MainWindow(QMainWindow):
    """Color Tiles 메인 윈도우."""

    def __init__(self):
        """MainWindow 초기화."""
        super().__init__()

        # 게임 엔진
        self.game = None
        self.observer = None

        # UI 위젯들
        self.board_widget = None
        self.info_panel = None
        self.control_panel = None
        self.ai_control_panel = None
        self.ai_status_panel = None

        # 타이머
        self.timer = None

        # AI 플레이어
        self.ai_player = None
        self.ai_timer = None
        self.ai_playing = False
        self.ai_speed = 5  # 액션/초
        self.ai_step_count = 0

        self._init_game()
        self._init_ui()
        self._connect_signals()
        self._start_timer()

    def _init_game(self):
        """게임 엔진 초기화."""
        board = BoardGenerator.generate_random_board()
        self.game = GameEngine(board)

        # 옵저버 등록
        self.observer = ColorTilesObserver(self)
        self.game.add_observer(self.observer)

    def _init_ui(self):
        """UI 초기화."""
        self.setWindowTitle("Color Tiles")
        self.setGeometry(100, 100, 1000, 600)

        # 중앙 위젯 및 레이아웃
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # 보드 위젯 (왼쪽)
        self.board_widget = BoardWidget()
        main_layout.addWidget(self.board_widget)

        # 오른쪽 패널 (정보 + 컨트롤)
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        self.info_panel = InfoPanel()
        right_layout.addWidget(self.info_panel)

        self.control_panel = ControlPanel()
        right_layout.addWidget(self.control_panel)

        # AI 패널
        if AI_AVAILABLE:
            self.ai_control_panel = AIControlPanel()
            right_layout.addWidget(self.ai_control_panel)

            self.ai_status_panel = AIStatusPanel()
            right_layout.addWidget(self.ai_status_panel)

        right_layout.addStretch()

        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 초기 보드 표시
        self.update_board_display()
        self.update_info_display()

    def _connect_signals(self):
        """시그널 연결."""
        # 보드 셀 클릭
        self.board_widget.cell_clicked.connect(self.on_cell_clicked)

        # 컨트롤 버튼
        self.control_panel.start_clicked.connect(self.on_start_game)
        self.control_panel.reset_clicked.connect(self.on_reset_game)

        # AI 컨트롤
        if AI_AVAILABLE and self.ai_control_panel:
            self.ai_control_panel.checkpoint_selected.connect(self.on_checkpoint_selected)
            self.ai_control_panel.ai_play_clicked.connect(self.on_ai_play)
            self.ai_control_panel.ai_stop_clicked.connect(self.on_ai_stop)
            self.ai_control_panel.speed_changed.connect(self.on_ai_speed_changed)

    def _start_timer(self):
        """시간 업데이트 타이머 시작."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(100)  # 100ms마다 업데이트

    def update_time(self):
        """타이머 콜백 - 남은 시간 업데이트."""
        if self.game.get_game_state() == GameState.PLAYING:
            remaining = self.game.get_remaining_time()
            self.info_panel.update_time(remaining)

    def update_board_display(self):
        """보드 표시 업데이트."""
        board_state = self.game.get_board_state()
        self.board_widget.update_board(board_state)

    def update_info_display(self):
        """정보 패널 업데이트."""
        self.info_panel.update_score(self.game.get_score())
        self.info_panel.update_time(self.game.get_remaining_time())
        board_state = self.game.get_board_state()
        self.info_panel.update_tiles(board_state['remaining_tiles'])

    def on_cell_clicked(self, row: int, col: int):
        """셀 클릭 핸들러."""
        if self.game.get_game_state() != GameState.PLAYING:
            self.show_message("게임 시작 필요", "먼저 '게임 시작' 버튼을 클릭하세요.")
            return

        position = Position(row, col)
        try:
            result = self.game.make_move(position)
            # Observer가 자동으로 UI 업데이트
        except Exception as e:
            self.show_message("오류", str(e))

    def on_start_game(self):
        """게임 시작 버튼 핸들러."""
        if self.game.get_game_state() == GameState.READY:
            self.game.start_game()
            self.board_widget.set_enabled(True)
            self.control_panel.set_start_button_enabled(False)
            self.info_panel.set_playing_state()

    def on_reset_game(self):
        """새 게임 버튼 핸들러."""
        # 확인 다이얼로그
        reply = QMessageBox.question(
            self,
            "새 게임",
            "새 게임을 시작하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # 새 보드 생성
            board = BoardGenerator.generate_random_board()
            self.game.reset_game(board)

            # UI 초기화
            self.update_board_display()
            self.update_info_display()
            self.board_widget.set_enabled(False)
            self.control_panel.set_start_button_enabled(True)
            self.info_panel.set_ready_state()

    def on_victory(self):
        """승리 시 호출."""
        self.board_widget.set_enabled(False)
        self.info_panel.set_victory_state()

        QMessageBox.information(
            self,
            "승리!",
            f"축하합니다! 모든 타일을 제거했습니다!\n\n최종 점수: {self.game.get_score()}"
        )

    def on_game_over(self, reason: str):
        """게임 오버 시 호출."""
        self.board_widget.set_enabled(False)
        self.info_panel.set_game_over_state(reason)

        QMessageBox.warning(
            self,
            "게임 오버",
            f"{reason}로 게임이 종료되었습니다.\n\n최종 점수: {self.game.get_score()}"
        )

    def show_message(self, title: str, message: str):
        """메시지 다이얼로그 표시."""
        QMessageBox.information(self, title, message)

    def on_checkpoint_selected(self, checkpoint_name: str):
        """체크포인트 선택 핸들러."""
        if not AI_AVAILABLE:
            return

        checkpoint_path = self.ai_control_panel.get_selected_checkpoint()
        if checkpoint_path and checkpoint_path != "(체크포인트 없음)":
            try:
                self.ai_player = AIPlayer(checkpoint_path)
                self.show_message("AI 로드 완료", f"체크포인트 로드 성공:\n{checkpoint_name}")
            except Exception as e:
                self.show_message("오류", f"AI 로드 실패:\n{str(e)}")
                self.ai_player = None

    def on_ai_play(self):
        """AI 플레이 시작."""
        if not AI_AVAILABLE or not self.ai_player:
            self.show_message("오류", "먼저 체크포인트를 선택하세요")
            return

        # 게임이 진행 중이 아니면 시작
        if self.game.get_game_state() != GameState.PLAYING:
            self.on_start_game()

        self.ai_playing = True
        self.ai_control_panel.set_playing_state(True)
        self.board_widget.set_enabled(False)  # 수동 클릭 비활성화

        # AI 타이머 시작
        self.ai_timer = QTimer()
        self.ai_timer.timeout.connect(self.ai_step)
        interval_ms = int(1000 / self.ai_speed)
        self.ai_timer.start(interval_ms)

    def on_ai_stop(self):
        """AI 플레이 중지."""
        self.ai_playing = False
        self.ai_control_panel.set_playing_state(False)
        self.board_widget.set_enabled(True)

        if self.ai_timer:
            self.ai_timer.stop()
            self.ai_timer = None

        self.ai_status_panel.clear()
        self.board_widget.clear_highlight()

    def on_ai_speed_changed(self, speed: int):
        """AI 속도 변경."""
        self.ai_speed = speed
        if self.ai_timer and self.ai_timer.isActive():
            interval_ms = int(1000 / speed)
            self.ai_timer.setInterval(interval_ms)

    def ai_step(self):
        """AI 한 스텝 실행 (타이머 콜백)."""
        if not self.ai_playing or self.game.get_game_state() != GameState.PLAYING:
            self.on_ai_stop()
            return

        try:
            # AI로부터 action 가져오기
            board = self.game.get_board()
            action, value, action_probs = self.ai_player.get_action(board)
            position = self.ai_player.action_to_position(action)

            # 신뢰도 계산
            confidence = action_probs[action] if action_probs is not None else 0.0

            # 상태 표시 업데이트
            self.ai_step_count += 1
            self.ai_status_panel.update_status(
                self.ai_step_count, value, confidence, position
            )

            # 다음 행동 하이라이트
            self.board_widget.highlight_cell(position, confidence)

            # 액션 실행
            result = self.game.make_move(position)

            # 게임 종료 체크
            if self.game.get_game_state() != GameState.PLAYING:
                self.on_ai_stop()
                self.ai_step_count = 0

        except Exception as e:
            print(f"AI step error: {e}")
            import traceback
            traceback.print_exc()
            self.on_ai_stop()

    def closeEvent(self, event):
        """윈도우 닫기 이벤트."""
        # AI 플레이 중이면 중지
        if self.ai_playing:
            self.on_ai_stop()

        reply = QMessageBox.question(
            self,
            "종료",
            "게임을 종료하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()

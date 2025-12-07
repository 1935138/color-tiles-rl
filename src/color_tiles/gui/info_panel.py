"""Information panel widget for game status display."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class InfoPanel(QWidget):
    """게임 상태 정보를 표시하는 패널."""

    def __init__(self, parent=None):
        """InfoPanel 초기화."""
        super().__init__(parent)

        # 레이블들
        self._score_label = None
        self._time_label = None
        self._tiles_label = None
        self._state_label = None

        self._init_ui()

    def _init_ui(self):
        """UI 초기화."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 타이틀
        title_label = QLabel("Color Tiles")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # 구분선
        separator = QLabel("─" * 30)
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(separator)

        # 점수
        self._score_label = self._create_info_label("점수: 0")
        layout.addWidget(self._score_label)

        # 남은 시간
        self._time_label = self._create_info_label("시간: 120.0초")
        layout.addWidget(self._time_label)

        # 남은 타일
        self._tiles_label = self._create_info_label("타일: 200개")
        layout.addWidget(self._tiles_label)

        # 게임 상태
        self._state_label = self._create_info_label("상태: 대기 중", is_status=True)
        layout.addWidget(self._state_label)

        # 여백 추가
        layout.addStretch()

        self.setLayout(layout)
        self.setFixedWidth(250)

    def _create_info_label(self, text: str, is_status: bool = False) -> QLabel:
        """정보 레이블 생성."""
        label = QLabel(text)
        font = QFont()
        font.setPointSize(12 if not is_status else 14)
        if is_status:
            font.setBold(True)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        if is_status:
            label.setStyleSheet("""
                QLabel {
                    padding: 10px;
                    background-color: #e0e0e0;
                    border-radius: 5px;
                }
            """)

        return label

    def update_score(self, score: int):
        """점수 업데이트."""
        self._score_label.setText(f"점수: {score}")

    def update_time(self, remaining_time: float):
        """시간 업데이트."""
        self._time_label.setText(f"시간: {remaining_time:.1f}초")

        # 시간이 얼마 안 남으면 빨간색으로 표시
        if remaining_time <= 10:
            self._time_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        elif remaining_time <= 30:
            self._time_label.setStyleSheet("QLabel { color: orange; }")
        else:
            self._time_label.setStyleSheet("")

    def update_tiles(self, remaining_tiles: int):
        """남은 타일 수 업데이트."""
        self._tiles_label.setText(f"타일: {remaining_tiles}개")

    def update_state(self, state_text: str, color: str = "#333"):
        """게임 상태 업데이트."""
        self._state_label.setText(f"상태: {state_text}")
        self._state_label.setStyleSheet(f"""
            QLabel {{
                padding: 10px;
                background-color: #e0e0e0;
                border-radius: 5px;
                color: {color};
            }}
        """)

    def set_victory_state(self):
        """승리 상태 표시."""
        self.update_state("🎉 승리!", "#00aa00")

    def set_game_over_state(self, reason: str):
        """게임 오버 상태 표시."""
        self.update_state(f"❌ {reason}", "#aa0000")

    def set_playing_state(self):
        """플레이 중 상태 표시."""
        self.update_state("플레이 중", "#0066cc")

    def set_ready_state(self):
        """대기 중 상태 표시."""
        self.update_state("대기 중", "#666")

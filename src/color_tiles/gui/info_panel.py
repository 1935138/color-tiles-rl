"""Information panel widget for game status display."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class InfoPanel(QWidget):
    """게임 상태 정보를 표시하는 패널."""

    def __init__(self, parent=None):
        """InfoPanel 초기화."""
        super().__init__(parent)

        # 레이블들
        self._score_label = None
        self._time_bar = None
        self._time_text = None
        self._tiles_label = None
        self._state_label = None

        self._init_ui()

    def _init_ui(self):
        """UI 초기화."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # 타이틀 - Windows Classic 스타일
        title_label = QLabel("Color Tiles")
        title_font = QFont("MS Sans Serif", 16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Raised 효과 테두리
        title_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                background-color: #c0c0c0;
                border-top: 2px solid #ffffff;
                border-left: 2px solid #ffffff;
                border-bottom: 2px solid #404040;
                border-right: 2px solid #404040;
            }
        """)
        layout.addWidget(title_label)

        # 구분선 - Classic groove 스타일
        separator = QLabel()
        separator.setFixedHeight(4)
        separator.setStyleSheet("""
            QLabel {
                border-top: 1px solid #808080;
                border-bottom: 1px solid #ffffff;
            }
        """)
        layout.addWidget(separator)

        # 점수
        self._score_label = self._create_info_label("점수: 0")
        layout.addWidget(self._score_label)

        # 남은 시간 - 프로그레스 바
        time_container = QWidget()
        time_layout = QVBoxLayout()
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.setSpacing(2)

        # 레이블 (상단)
        self._time_text = QLabel("시간: 120.0초")
        font = QFont("MS Sans Serif", 9)
        self._time_text.setFont(font)
        time_layout.addWidget(self._time_text)

        # 프로그레스 바 (하단)
        self._time_bar = QProgressBar()
        self._time_bar.setMinimum(0)
        self._time_bar.setMaximum(120)  # INITIAL_TIME
        self._time_bar.setValue(120)
        self._time_bar.setTextVisible(False)  # 텍스트는 위에 별도 표시
        self._time_bar.setFixedHeight(20)

        # Windows Classic 스타일 프로그레스 바
        self._time_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #808080;
                border-radius: 0px;
                background-color: #ffffff;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0000aa;  /* 파란색 */
                width: 1px;
            }
        """)

        time_layout.addWidget(self._time_bar)
        time_container.setLayout(time_layout)
        layout.addWidget(time_container)

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
        """정보 레이블 생성 - Windows Classic 스타일."""
        label = QLabel(text)
        font = QFont("MS Sans Serif", 10 if not is_status else 11)
        if is_status:
            font.setBold(True)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)

        if is_status:
            # Sunken 효과 (움푹 들어간 상태 표시 영역)
            label.setStyleSheet("""
                QLabel {
                    padding: 8px;
                    background-color: #ffffff;
                    border-top: 2px solid #808080;
                    border-left: 2px solid #808080;
                    border-bottom: 2px solid #ffffff;
                    border-right: 2px solid #ffffff;
                }
            """)

        return label

    def update_score(self, score: int):
        """점수 업데이트."""
        self._score_label.setText(f"점수: {score}")

    def update_time(self, remaining_time: float):
        """시간 업데이트 - 프로그레스 바 + 텍스트."""
        # 텍스트 업데이트
        self._time_text.setText(f"시간: {remaining_time:.1f}초")

        # 프로그레스 바 값 업데이트
        bar_value = int(remaining_time)  # 소수점 버림
        self._time_bar.setValue(bar_value)

        # 시간에 따라 색상 변경
        if remaining_time <= 10:
            # 10초 이하: 빨간색
            self._time_text.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            self._time_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #808080;
                    background-color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #ff0000;  /* 빨간색 */
                }
            """)
        elif remaining_time <= 30:
            # 30초 이하: 주황색
            self._time_text.setStyleSheet("QLabel { color: orange; }")
            self._time_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #808080;
                    background-color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #ff8800;  /* 주황색 */
                }
            """)
        else:
            # 30초 초과: 파란색 (기본)
            self._time_text.setStyleSheet("")
            self._time_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #808080;
                    background-color: #ffffff;
                }
                QProgressBar::chunk {
                    background-color: #0000aa;  /* 파란색 */
                }
            """)

    def update_tiles(self, remaining_tiles: int):
        """남은 타일 수 업데이트."""
        self._tiles_label.setText(f"타일: {remaining_tiles}개")

    def update_state(self, state_text: str, color: str = "#000000"):
        """게임 상태 업데이트 - Windows Classic 스타일."""
        self._state_label.setText(f"상태: {state_text}")
        self._state_label.setStyleSheet(f"""
            QLabel {{
                padding: 8px;
                background-color: #ffffff;
                border-top: 2px solid #808080;
                border-left: 2px solid #808080;
                border-bottom: 2px solid #ffffff;
                border-right: 2px solid #ffffff;
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
        self.update_state("대기 중", "#000000")

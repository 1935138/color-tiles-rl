"""Control panel widget with game control buttons."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont


class ControlPanel(QWidget):
    """게임 컨트롤 버튼을 포함하는 패널."""

    # 시그널
    start_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()

    def __init__(self, parent=None):
        """ControlPanel 초기화."""
        super().__init__(parent)

        self._start_button = None
        self._reset_button = None

        self._init_ui()

    def _init_ui(self):
        """UI 초기화."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Windows Classic 버튼 스타일
        classic_button_style = """
            QPushButton {
                background-color: #c0c0c0;
                color: #000000;
                border-top: 2px solid #ffffff;
                border-left: 2px solid #ffffff;
                border-bottom: 2px solid #404040;
                border-right: 2px solid #404040;
                padding: 4px 8px;
            }
            QPushButton:hover {
                background-color: #d0d0d0;
            }
            QPushButton:pressed {
                background-color: #a0a0a0;
                border-top: 2px solid #404040;
                border-left: 2px solid #404040;
                border-bottom: 2px solid #ffffff;
                border-right: 2px solid #ffffff;
            }
            QPushButton:disabled {
                color: #808080;
                background-color: #c0c0c0;
            }
        """

        # Start 버튼
        self._start_button = QPushButton("게임 시작")
        self._start_button.setFixedHeight(40)
        font = QFont("MS Sans Serif", 10)
        font.setBold(True)
        self._start_button.setFont(font)
        self._start_button.setStyleSheet(classic_button_style)
        self._start_button.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self._start_button)

        # Reset 버튼
        self._reset_button = QPushButton("새 게임")
        self._reset_button.setFixedHeight(40)
        self._reset_button.setFont(font)
        self._reset_button.setStyleSheet(classic_button_style)
        self._reset_button.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self._reset_button)

        self.setLayout(layout)

    def set_start_button_enabled(self, enabled: bool):
        """Start 버튼 활성화/비활성화."""
        self._start_button.setEnabled(enabled)

    def set_reset_button_enabled(self, enabled: bool):
        """Reset 버튼 활성화/비활성화."""
        self._reset_button.setEnabled(enabled)

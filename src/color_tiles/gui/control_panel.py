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

        # Start 버튼
        self._start_button = QPushButton("게임 시작")
        self._start_button.setFixedHeight(50)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self._start_button.setFont(font)
        self._start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self._start_button.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self._start_button)

        # Reset 버튼
        self._reset_button = QPushButton("새 게임")
        self._reset_button.setFixedHeight(50)
        self._reset_button.setFont(font)
        self._reset_button.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #e68900;
            }
            QPushButton:pressed {
                background-color: #cc7a00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self._reset_button.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self._reset_button)

        self.setLayout(layout)

    def set_start_button_enabled(self, enabled: bool):
        """Start 버튼 활성화/비활성화."""
        self._start_button.setEnabled(enabled)

    def set_reset_button_enabled(self, enabled: bool):
        """Reset 버튼 활성화/비활성화."""
        self._reset_button.setEnabled(enabled)

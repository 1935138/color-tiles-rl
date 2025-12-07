"""AI Control Panel widget for GUI"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QComboBox, QSlider, QLabel, QGroupBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from pathlib import Path


class AIControlPanel(QWidget):
    """
    AI Player control panel.

    Provides UI controls for:
    - Selecting trained model checkpoints
    - Starting/stopping AI playback
    - Adjusting playback speed

    Signals:
        checkpoint_selected(str): Emitted when checkpoint is selected
        ai_play_clicked(): Emitted when AI Play button clicked
        ai_stop_clicked(): Emitted when Stop button clicked
        speed_changed(int): Emitted when speed slider changed (1-10)
    """

    checkpoint_selected = pyqtSignal(str)
    ai_play_clicked = pyqtSignal()
    ai_stop_clicked = pyqtSignal()
    speed_changed = pyqtSignal(int)

    def __init__(self, checkpoint_dir: str = "checkpoints", parent=None):
        """
        Initialize AIControlPanel.

        Args:
            checkpoint_dir: Directory containing checkpoint files
            parent: Parent widget
        """
        super().__init__(parent)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._init_ui()
        self._refresh_checkpoints()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        # Group box for AI controls
        group = QGroupBox("AI 플레이어")
        group_layout = QVBoxLayout()

        # Checkpoint selection
        checkpoint_layout = QHBoxLayout()
        checkpoint_layout.addWidget(QLabel("체크포인트:"))

        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.currentTextChanged.connect(
            lambda text: self.checkpoint_selected.emit(text)
        )
        checkpoint_layout.addWidget(self.checkpoint_combo)

        self.refresh_btn = QPushButton("새로고침")
        self.refresh_btn.clicked.connect(self._refresh_checkpoints)
        checkpoint_layout.addWidget(self.refresh_btn)

        group_layout.addLayout(checkpoint_layout)

        # Play/Stop buttons
        button_layout = QHBoxLayout()
        self.play_btn = QPushButton("AI 시작")
        self.play_btn.clicked.connect(self.ai_play_clicked.emit)
        button_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("중지")
        self.stop_btn.clicked.connect(self.ai_stop_clicked.emit)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        group_layout.addLayout(button_layout)

        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("속도:"))

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.speed_changed.emit)
        speed_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("5")
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_label.setText(str(v))
        )
        speed_layout.addWidget(self.speed_label)

        group_layout.addLayout(speed_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)
        self.setLayout(layout)

    def _refresh_checkpoints(self):
        """Scan checkpoints directory and populate combo box."""
        self.checkpoint_combo.clear()

        # Create directory if it doesn't exist
        if not self._checkpoint_dir.exists():
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_combo.addItem("(체크포인트 없음)")
            self.play_btn.setEnabled(False)
            return

        # Find all .zip files
        checkpoints = sorted(self._checkpoint_dir.glob("*.zip"))

        if not checkpoints:
            self.checkpoint_combo.addItem("(체크포인트 없음)")
            self.play_btn.setEnabled(False)
        else:
            for cp in checkpoints:
                # Add filename as text, full path as data
                self.checkpoint_combo.addItem(cp.name, str(cp))
            self.play_btn.setEnabled(True)

    def get_selected_checkpoint(self) -> str:
        """
        Get path to currently selected checkpoint.

        Returns:
            path: Full path to selected checkpoint file, or None if none selected
        """
        return self.checkpoint_combo.currentData()

    def set_playing_state(self, playing: bool):
        """
        Update UI for playing/stopped state.

        Args:
            playing: True if AI is playing, False if stopped
        """
        self.play_btn.setEnabled(not playing)
        self.stop_btn.setEnabled(playing)
        self.checkpoint_combo.setEnabled(not playing)
        self.refresh_btn.setEnabled(not playing)

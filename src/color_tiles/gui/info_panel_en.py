"""Information panel widget for game status display (English version)."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class InfoPanelEN(QWidget):
    """Game status information panel (English)."""

    def __init__(self, parent=None):
        """Initialize InfoPanel."""
        super().__init__(parent)

        # Labels
        self._score_label = None
        self._time_label = None
        self._tiles_label = None
        self._state_label = None

        self._init_ui()

    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Title
        title_label = QLabel("Color Tiles")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Separator
        separator = QLabel("─" * 30)
        separator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(separator)

        # Score
        self._score_label = self._create_info_label("Score: 0")
        layout.addWidget(self._score_label)

        # Time
        self._time_label = self._create_info_label("Time: 120.0s")
        layout.addWidget(self._time_label)

        # Tiles
        self._tiles_label = self._create_info_label("Tiles: 200")
        layout.addWidget(self._tiles_label)

        # State
        self._state_label = self._create_info_label("Status: Ready", is_status=True)
        layout.addWidget(self._state_label)

        # Spacer
        layout.addStretch()

        self.setLayout(layout)
        self.setFixedWidth(250)

    def _create_info_label(self, text: str, is_status: bool = False) -> QLabel:
        """Create info label."""
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
        """Update score."""
        self._score_label.setText(f"Score: {score}")

    def update_time(self, remaining_time: float):
        """Update time."""
        self._time_label.setText(f"Time: {remaining_time:.1f}s")

        # Red warning when low
        if remaining_time <= 10:
            self._time_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
        elif remaining_time <= 30:
            self._time_label.setStyleSheet("QLabel { color: orange; }")
        else:
            self._time_label.setStyleSheet("")

    def update_tiles(self, remaining_tiles: int):
        """Update tiles count."""
        self._tiles_label.setText(f"Tiles: {remaining_tiles}")

    def update_state(self, state_text: str, color: str = "#333"):
        """Update game state."""
        self._state_label.setText(f"Status: {state_text}")
        self._state_label.setStyleSheet(f"""
            QLabel {{
                padding: 10px;
                background-color: #e0e0e0;
                border-radius: 5px;
                color: {color};
            }}
        """)

    def set_victory_state(self):
        """Show victory state."""
        self.update_state("Victory!", "#00aa00")

    def set_game_over_state(self, reason: str):
        """Show game over state."""
        self.update_state(f"Game Over: {reason}", "#aa0000")

    def set_playing_state(self):
        """Show playing state."""
        self.update_state("Playing", "#0066cc")

    def set_ready_state(self):
        """Show ready state."""
        self.update_state("Ready", "#666")

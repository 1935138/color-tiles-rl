"""AI Status Panel widget for GUI"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox


class AIStatusPanel(QWidget):
    """
    Display AI agent status information.

    Shows:
    - Current step count
    - State value estimate
    - Next action confidence (probability)
    - Next action position
    """

    def __init__(self, parent=None):
        """
        Initialize AIStatusPanel.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()

        group = QGroupBox("AI 상태")
        group_layout = QVBoxLayout()

        # Step count
        self.step_label = QLabel("스텝: -")
        group_layout.addWidget(self.step_label)

        # Value estimate
        self.value_label = QLabel("가치 추정: -")
        group_layout.addWidget(self.value_label)

        # Action confidence
        self.confidence_label = QLabel("행동 신뢰도: -")
        group_layout.addWidget(self.confidence_label)

        # Next action position
        self.action_label = QLabel("다음 행동: -")
        group_layout.addWidget(self.action_label)

        group.setLayout(group_layout)
        layout.addWidget(group)
        self.setLayout(layout)

    def update_status(self, step: int, value: float, confidence: float, position=None):
        """
        Update all status displays.

        Args:
            step: Current step number
            value: State value estimate
            confidence: Action confidence (probability 0-1)
            position: Next action position (row, col) or None
        """
        self.step_label.setText(f"스텝: {step}")
        self.value_label.setText(f"가치 추정: {value:.3f}")
        self.confidence_label.setText(f"행동 신뢰도: {confidence:.1%}")

        if position:
            self.action_label.setText(f"다음 행동: ({position.row}, {position.col})")
        else:
            self.action_label.setText("다음 행동: -")

    def clear(self):
        """Clear all displays."""
        self.step_label.setText("스텝: -")
        self.value_label.setText("가치 추정: -")
        self.confidence_label.setText("행동 신뢰도: -")
        self.action_label.setText("다음 행동: -")

"""Board grid widget for Color Tiles GUI."""

from PyQt6.QtWidgets import QWidget, QGridLayout, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor

from color_tiles.domain.models import Color, Position
from color_tiles.domain.constants import BOARD_WIDTH, BOARD_HEIGHT


class BoardWidget(QWidget):
    """보드 그리드를 표시하는 위젯.

    23×15 그리드의 클릭 가능한 셀들로 구성됩니다.
    """

    # 셀이 클릭되었을 때 발생하는 시그널 (row, col)
    cell_clicked = pyqtSignal(int, int)

    # 색상 매핑 (Color enum -> QColor)
    COLOR_MAP = {
        Color.WHITE: QColor(255, 255, 255),      # 흰색
        Color.PINK: QColor(255, 192, 203),       # 분홍
        Color.BLUE: QColor(0, 0, 255),           # 파랑
        Color.SKY_BLUE: QColor(135, 206, 235),   # 하늘색
        Color.GREEN: QColor(0, 200, 0),          # 초록
        Color.ORANGE: QColor(255, 165, 0),       # 주황
        Color.YELLOW: QColor(255, 255, 0),       # 노랑
        Color.PURPLE: QColor(160, 32, 240),      # 보라
        Color.BROWN: QColor(139, 69, 19),        # 갈색
        Color.RED: QColor(255, 0, 0),            # 빨강
    }

    EMPTY_COLOR = QColor(240, 240, 240)  # 빈칸 (밝은 회색)

    def __init__(self, parent=None):
        """BoardWidget 초기화."""
        super().__init__(parent)

        # 셀 버튼들을 저장할 2D 리스트
        self._cell_buttons = []

        self._init_ui()

    def _init_ui(self):
        """UI 초기화."""
        layout = QGridLayout()
        layout.setSpacing(2)  # 셀 간격
        layout.setContentsMargins(5, 5, 5, 5)

        # 23×15 그리드 생성
        for row in range(BOARD_HEIGHT):
            button_row = []
            for col in range(BOARD_WIDTH):
                button = QPushButton()
                button.setFixedSize(30, 30)  # 셀 크기
                button.setStyleSheet(self._get_cell_style(self.EMPTY_COLOR))

                # 클릭 이벤트 연결 (lambda의 기본 인자 사용으로 클로저 문제 해결)
                button.clicked.connect(
                    lambda checked=False, r=row, c=col: self._on_cell_clicked(r, c)
                )

                layout.addWidget(button, row, col)
                button_row.append(button)

            self._cell_buttons.append(button_row)

        self.setLayout(layout)

    def _on_cell_clicked(self, row: int, col: int):
        """셀 클릭 핸들러."""
        self.cell_clicked.emit(row, col)

    def update_board(self, board_state: dict):
        """보드 상태를 받아서 UI 업데이트.

        Args:
            board_state: game.get_board_state()로 받은 딕셔너리.
                        {'cells': [[color_name or None, ...], ...]}
        """
        cells = board_state['cells']

        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                color_name = cells[row][col]
                button = self._cell_buttons[row][col]

                if color_name is None:
                    # 빈칸
                    color = self.EMPTY_COLOR
                else:
                    # 타일
                    color_enum = Color[color_name]
                    color = self.COLOR_MAP[color_enum]

                button.setStyleSheet(self._get_cell_style(color))

    def _get_cell_style(self, color: QColor) -> str:
        """셀 버튼의 스타일시트 생성."""
        r, g, b = color.red(), color.green(), color.blue()

        # 호버 효과를 위한 밝은 색상
        hover_color = QColor(
            min(r + 20, 255),
            min(g + 20, 255),
            min(b + 20, 255)
        )
        hr, hg, hb = hover_color.red(), hover_color.green(), hover_color.blue()

        return f"""
            QPushButton {{
                background-color: rgb({r}, {g}, {b});
                border: 1px solid #999;
                border-radius: 3px;
            }}
            QPushButton:hover {{
                background-color: rgb({hr}, {hg}, {hb});
                border: 2px solid #333;
            }}
            QPushButton:pressed {{
                background-color: rgb({max(r-30, 0)}, {max(g-30, 0)}, {max(b-30, 0)});
            }}
        """

    def set_enabled(self, enabled: bool):
        """보드 전체 활성화/비활성화."""
        for row in self._cell_buttons:
            for button in row:
                button.setEnabled(enabled)

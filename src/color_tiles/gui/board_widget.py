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

        # AI 하이라이트 추적
        self._highlighted_button = None
        self._highlighted_position = None

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
                    # 빈칸: Sunken 스타일
                    color = self.EMPTY_COLOR
                    button.setStyleSheet(self._get_cell_style(color, is_empty=True))
                else:
                    # 타일: Raised 스타일 + 색상
                    color_enum = Color[color_name]
                    color = self.COLOR_MAP[color_enum]
                    button.setStyleSheet(self._get_cell_style(color, is_empty=False))

    def _get_cell_style(self, color: QColor, is_empty: bool = False) -> str:
        """셀 버튼의 Windows Classic 스타일시트 생성."""
        r, g, b = color.red(), color.green(), color.blue()

        if is_empty:
            # 빈 칸: Sunken 효과 (움푹 들어간)
            return """
                QPushButton {
                    background-color: #c0c0c0;
                    border-top: 2px solid #808080;
                    border-left: 2px solid #808080;
                    border-bottom: 2px solid #ffffff;
                    border-right: 2px solid #ffffff;
                }
                QPushButton:hover {
                    background-color: #d4d4d4;
                }
                QPushButton:pressed {
                    background-color: #a0a0a0;
                }
            """
        else:
            # 타일: Raised 효과 (볼록한) + 게임 색상 유지
            # 색상 기반 하이라이트/섀도우 계산
            highlight_r = min(r + 60, 255)
            highlight_g = min(g + 60, 255)
            highlight_b = min(b + 60, 255)
            shadow_r = max(r - 80, 0)
            shadow_g = max(g - 80, 0)
            shadow_b = max(b - 80, 0)

            return f"""
                QPushButton {{
                    background-color: rgb({r}, {g}, {b});
                    border-top: 2px solid rgb({highlight_r}, {highlight_g}, {highlight_b});
                    border-left: 2px solid rgb({highlight_r}, {highlight_g}, {highlight_b});
                    border-bottom: 2px solid rgb({shadow_r}, {shadow_g}, {shadow_b});
                    border-right: 2px solid rgb({shadow_r}, {shadow_g}, {shadow_b});
                }}
                QPushButton:hover {{
                    border-width: 3px;
                }}
                QPushButton:pressed {{
                    border-top: 2px solid rgb({shadow_r}, {shadow_g}, {shadow_b});
                    border-left: 2px solid rgb({shadow_r}, {shadow_g}, {shadow_b});
                    border-bottom: 2px solid rgb({highlight_r}, {highlight_g}, {highlight_b});
                    border-right: 2px solid rgb({highlight_r}, {highlight_g}, {highlight_b});
                }}
            """

    def set_enabled(self, enabled: bool):
        """보드 전체 활성화/비활성화."""
        for row in self._cell_buttons:
            for button in row:
                button.setEnabled(enabled)

    def highlight_cell(self, position: Position, confidence: float):
        """
        AI가 선택할 셀을 하이라이트.

        Args:
            position: 하이라이트할 Position
            confidence: AI의 신뢰도 (0.0-1.0)
        """
        # 이전 하이라이트 제거
        if self._highlighted_button and self._highlighted_position:
            # 보드 업데이트를 통해 원래 스타일로 복원
            # (여기서는 간단히 처리)
            pass

        # 새 하이라이트 적용
        button = self._cell_buttons[position.row][position.col]
        highlight_color = self._get_confidence_color(confidence)

        # 기존 스타일에 굵은 테두리 추가
        current_style = button.styleSheet()
        button.setStyleSheet(current_style + f"""
            QPushButton {{
                border: 4px solid {highlight_color} !important;
            }}
        """)

        self._highlighted_button = button
        self._highlighted_position = position

    def clear_highlight(self):
        """하이라이트 제거."""
        if self._highlighted_button and self._highlighted_position:
            # 원래 색상으로 복원하려면 보드를 다시 그려야 함
            # 간단하게 스타일 재설정
            row = self._highlighted_position.row
            col = self._highlighted_position.col
            button = self._cell_buttons[row][col]

            # 기본 스타일로 재설정 (빈칸으로 가정)
            button.setStyleSheet(self._get_cell_style(self.EMPTY_COLOR, is_empty=True))

            self._highlighted_button = None
            self._highlighted_position = None

    def _get_confidence_color(self, confidence: float) -> str:
        """
        신뢰도에 따른 하이라이트 색상 반환.

        Args:
            confidence: 0.0-1.0 범위의 신뢰도

        Returns:
            색상 문자열 (CSS 형식)
        """
        if confidence > 0.8:
            return "#00FF00"  # 밝은 녹색 (높은 신뢰도)
        elif confidence > 0.5:
            return "#FFFF00"  # 노란색 (중간 신뢰도)
        else:
            return "#FF8800"  # 주황색 (낮은 신뢰도)

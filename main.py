#!/usr/bin/env python3
"""Color Tiles - Main GUI Application Entry Point."""

import sys
import os

# src 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication
from color_tiles.gui.main_window import MainWindow


# Windows Classic (95/98) 전역 스타일시트
CLASSIC_STYLESHEET = """
    /* 전역 설정 */
    QWidget {
        background-color: #c0c0c0;
        font-family: "MS Sans Serif", "Tahoma", sans-serif;
        font-size: 9pt;
        color: #000000;
    }

    /* 메인 윈도우 */
    QMainWindow {
        background-color: #c0c0c0;
    }

    /* 레이블 */
    QLabel {
        background-color: transparent;
    }

    /* 메시지 박스 */
    QMessageBox {
        background-color: #c0c0c0;
    }

    QMessageBox QPushButton {
        min-width: 75px;
        min-height: 23px;
        background-color: #c0c0c0;
        border-top: 2px solid #ffffff;
        border-left: 2px solid #ffffff;
        border-bottom: 2px solid #404040;
        border-right: 2px solid #404040;
        padding: 2px 10px;
    }

    QMessageBox QPushButton:pressed {
        border-top: 2px solid #404040;
        border-left: 2px solid #404040;
        border-bottom: 2px solid #ffffff;
        border-right: 2px solid #ffffff;
    }
"""


def main():
    """메인 애플리케이션 실행."""
    app = QApplication(sys.argv)
    app.setApplicationName("Color Tiles")
    app.setOrganizationName("Color Tiles Team")

    # Windows Classic 테마 적용
    app.setStyleSheet(CLASSIC_STYLESHEET)

    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

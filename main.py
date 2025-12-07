#!/usr/bin/env python3
"""Color Tiles - Main GUI Application Entry Point."""

import sys
import os

# src 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from PyQt6.QtWidgets import QApplication
from color_tiles.gui.main_window import MainWindow


def main():
    """메인 애플리케이션 실행."""
    app = QApplication(sys.argv)
    app.setApplicationName("Color Tiles")
    app.setOrganizationName("Color Tiles Team")

    # 메인 윈도우 생성 및 표시
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

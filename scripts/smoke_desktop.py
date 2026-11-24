"""Create and render the desktop window without requiring a physical display."""

import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QLibraryInfo, QTimer
from PyQt6.QtWidgets import QApplication

from app import MainWindow


def main():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    app.processEvents()

    if not window.isVisible():
        raise RuntimeError("The desktop main window did not become visible")
    if not hasattr(window, "canvas"):
        raise RuntimeError("The desktop plot canvas was not initialized")

    QTimer.singleShot(100, app.quit)
    app.exec()
    title = window.windowTitle()
    window.close()

    print(
        "Desktop smoke passed; "
        f"architecture={platform.machine()}, "
        f"PyQt={__import__('PyQt6.QtCore', fromlist=['PYQT_VERSION_STR']).PYQT_VERSION_STR}, "
        f"Qt={__import__('PyQt6.QtCore', fromlist=['QT_VERSION_STR']).QT_VERSION_STR}, "
        f"plugins={QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)}, "
        f"window={title!r}"
    )


if __name__ == "__main__":
    main()

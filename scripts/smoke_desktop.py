"""Create and render the desktop window without requiring a physical display."""

import os
import platform
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QLibraryInfo, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from app import MainWindow
from ui.grid_interpretation import GridInterpretationDialog


def main():
    app = QApplication.instance() or QApplication([])
    # Keep unexpected warning dialogs from blocking the headless smoke run.
    QMessageBox.warning = staticmethod(lambda *args, **kwargs: None)
    QMessageBox.critical = staticmethod(lambda *args, **kwargs: None)
    window = MainWindow()
    window.show()
    app.processEvents()

    if not window.isVisible():
        raise RuntimeError("The desktop main window did not become visible")
    if not hasattr(window, "canvas"):
        raise RuntimeError("The desktop plot canvas was not initialized")

    n = 64
    spacing = 10.0
    x = np.arange(n, dtype=np.float64) * spacing
    y = np.arange(n, dtype=np.float64) * spacing
    xx, yy = np.meshgrid(x, y)
    grid_values = np.zeros_like(xx)
    for frequency in range(1, 12):
        k = 2.0 * np.pi * frequency / (n * spacing)
        grid_values += np.exp(-100.0 * k) * np.cos(2.0 * np.pi * frequency * xx / (n * spacing))
        grid_values += np.exp(-100.0 * k) * np.cos(2.0 * np.pi * frequency * yy / (n * spacing))
    grid_dialog = GridInterpretationDialog(
        window, xx.ravel(), yy.ravel(), grid_values.ravel(),
        field_inclination_deg=50.0, field_declination_deg=10.0,
    )
    grid_dialog.padding.setCurrentIndex(1)
    grid_dialog.run_spectral_transform()
    if grid_dialog.transform_result is None:
        raise RuntimeError(f"The grid transform workflow did not produce a result: {grid_dialog.status.toPlainText()}")
    grid_dialog.run_direction_transform()
    if grid_dialog.direction_result is None:
        raise RuntimeError(f"The stabilized magnetic transform workflow did not produce a result: {grid_dialog.status.toPlainText()}")
    grid_dialog.depth_k_min.setValue(0.005)
    grid_dialog.depth_k_max.setValue(0.1)
    grid_dialog.run_depth_estimate()
    if "Depth" not in grid_dialog.status.toPlainText():
        raise RuntimeError("The spectral depth UI did not report a depth estimate")
    grid_dialog.run_euler_sweep()
    if "Euler windows" not in grid_dialog.status.toPlainText():
        raise RuntimeError(f"The Euler window UI did not report its filters: {grid_dialog.status.toPlainText()}")
    grid_dialog.close()

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

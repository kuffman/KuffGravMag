"""Create and render the desktop window without requiring a physical display."""

import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QLibraryInfo, QTimer
from PyQt6.QtWidgets import QApplication, QMessageBox

from app import MainWindow
from engine.project import load_project
from ui.academic_lab import AcademicLabDialog
from engine.project_bundle import load_project_bundle
from engine.spatial import load_xyz_layer
from ui.grid_interpretation import GridInterpretationDialog


def main():
    app = QApplication.instance() or QApplication([])
    app.setOrganizationName("KuffGravMag")
    app.setApplicationName("KuffGravMag Studio")
    # Keep unexpected warning dialogs from blocking the headless smoke run.
    QMessageBox.warning = staticmethod(lambda *args, **kwargs: None)
    QMessageBox.critical = staticmethod(lambda *args, **kwargs: None)
    window = MainWindow(enable_recovery=False)
    window.show()
    app.processEvents()

    if not window.isVisible():
        raise RuntimeError("The desktop main window did not become visible")
    if not hasattr(window, "canvas"):
        raise RuntimeError("The desktop plot canvas was not initialized")
    if not hasattr(window, "studio_workspace") or window.studio_workspace.tree.topLevelItemCount() != 1:
        raise RuntimeError("The Studio project explorer was not initialized")
    if window.analysis_tabs.count() != 2 or window.prism_view.finite_body_count != 2:
        raise RuntimeError("The 3D finite-prism model view was not initialized")

    window.obs_gz = np.linspace(-0.2, 0.3, window.x_obs.size)
    window.refresh_project_explorer()
    if not window.studio_workspace.buttons["invert_gravity"].isEnabled():
        raise RuntimeError("The gravity inversion workflow did not enable for observed data")
    with tempfile.TemporaryDirectory(prefix="kgm-studio-smoke-") as temp_dir:
        project_path = Path(temp_dir) / "workflow.kgm.json"
        window._save_project_to_path(project_path)
        saved = load_project(project_path)
        if len(saved["bodies"]) != len(window.bodies):
            raise RuntimeError("The Studio project save did not preserve the model")
        np.testing.assert_array_equal(saved["observations"]["gravity"], window.obs_gz)

        raw_layer = b"x,y,value\n500000,663204.567,1\n500100,663204.567,2\n500000,663304.567,3\n"
        layer = load_xyz_layer(
            raw_layer, source_crs="EPSG:32630", target_crs="EPSG:4326", units="mGal",
            name="synthetic-ghana-grid.csv",
        )
        attachment_name = window._store_project_attachment(layer.name, raw_layer)
        window.spatial_layer = layer
        window.spatial_layer_attachment_name = attachment_name
        window.processing_steps.append({
            "name": "local_spatial_layer",
            "parameters": {
                "source_name": layer.name,
                "attachment_name": attachment_name,
                "source_sha256": layer.source_sha256,
                "source_crs": layer.source_crs,
                "target_crs": layer.target_crs,
                "units": layer.units,
                "point_count": int(layer.x.size),
                "local_only": True,
            },
        })
        bundle_path = Path(temp_dir) / "workflow.kgmproj"
        window._save_project_to_path(bundle_path)
        bundle = load_project_bundle(bundle_path)
        if bundle["attachments"] != {attachment_name: raw_layer}:
            raise RuntimeError("The portable project did not preserve its embedded XYZ layer")
        window._apply_project(
            bundle["project"], project_path=bundle_path, attachments=bundle["attachments"]
        )
        if window.spatial_layer is None or window.spatial_layer.source_sha256 != layer.source_sha256:
            raise RuntimeError("The portable project did not restore its spatial layer")

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

    lab_dialog = AcademicLabDialog(window)
    experiment = lab_dialog.run_experiment()
    if experiment.model_kind != "sphere" or experiment.seed != 20261124:
        raise RuntimeError("The Academic Lab did not produce its reproducible default experiment")
    lab_dialog.model.setCurrentIndex(1)
    lab_dialog.field.setCurrentIndex(1)
    magnetic_experiment = lab_dialog.run_experiment()
    if magnetic_experiment.response_units != "nT" or not np.all(
        np.isfinite(magnetic_experiment.observed_response)
    ):
        raise RuntimeError("The Academic Lab magnetic cylinder response is invalid")
    lab_dialog.close()

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

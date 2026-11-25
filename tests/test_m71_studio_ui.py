"""Offscreen UI regressions for linked Studio model views."""

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from app import MainWindow
from engine.visualization import finite_prism_faces
from ui.studio_workspace import StudioWorkspace


@pytest.fixture(scope="module")
def qt_app():
    probe_env = os.environ.copy()
    probe_env["QT_QPA_PLATFORM"] = "offscreen"
    probe = subprocess.run(
        [sys.executable, "-c", "from PyQt6.QtWidgets import QApplication; app=QApplication([])"],
        env=probe_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        reason = "The local Qt offscreen platform plugin could not start"
        if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
            pytest.fail(f"{reason}: {probe.stderr[-1500:]}")
        pytest.skip(reason)
    return QApplication.instance() or QApplication([])


def test_project_explorer_action_and_local_layer_tree_item(qt_app):
    workspace = StudioWorkspace()
    requested = []
    workspace.action_requested.connect(requested.append)
    workspace.update_project(spatial_layer_name="synthetic.xyz")

    workspace.buttons["load_spatial_layer"].click()
    tree_labels = [workspace.tree.topLevelItem(0).child(i).text(0)
                   for i in range(workspace.tree.topLevelItem(0).childCount())]

    assert requested == ["load_spatial_layer"]
    assert any("Local spatial layer · synthetic.xyz" in label for label in tree_labels)


def test_project_explorer_selection_links_2d_and_3d_body_editing(qt_app, tmp_path):
    window = MainWindow(recovery_path=tmp_path / "recovery.json", enable_recovery=False)
    try:
        window._activate_project_item(("body", 1))

        assert window.active_body_idx == 1
        assert window.body_list.currentRow() == 1
        assert window.prism_view.active_index == 1
        assert window.analysis_tabs.currentIndex() == 0
        assert window.ax_body.lines[1].get_linewidth() > window.ax_body.lines[0].get_linewidth()

        initial = np.asarray(finite_prism_faces(window.bodies[1]))
        window.spin_ymin.setValue(window.bodies[1].y_min + 75.0)
        changed = np.asarray(finite_prism_faces(window.bodies[1]))
        assert window.bodies[1].y_min == pytest.approx(-925.0)
        assert not np.array_equal(initial[:, :, 1].min(), changed[:, :, 1].min())
        assert window.prism_view.active_index == 1
    finally:
        window.close()

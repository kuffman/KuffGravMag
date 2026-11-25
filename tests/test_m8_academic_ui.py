"""Offscreen wiring checks for the interactive academic lab."""

import os
import subprocess
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from ui.academic_lab import AcademicLabDialog
from ui.studio_workspace import StudioWorkspace


@pytest.fixture(scope="module")
def qt_app():
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    probe = subprocess.run(
        [sys.executable, "-c", "from PyQt6.QtWidgets import QApplication; app=QApplication([])"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        reason = f"The local Qt offscreen platform plugin could not start: {probe.stderr[-1000:]}"
        if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
            pytest.fail(reason)
        pytest.skip(reason)
    return QApplication.instance() or QApplication([])


def test_studio_exposes_academic_lab_action(qt_app):
    workspace = StudioWorkspace()
    requested = []
    workspace.action_requested.connect(requested.append)
    workspace.buttons["academic_lab"].click()
    assert requested == ["academic_lab"]


def test_academic_lab_runs_seeded_sphere_and_magnetic_cylinder(qt_app):
    dialog = AcademicLabDialog()
    try:
        first = dialog.run_experiment()
        assert first.model_kind == "sphere"
        assert first.field_type == "gravity"
        assert first.seed == 20261124
        assert first.observed_response.size == dialog.station_count.value()
        dialog.seed.setValue(41)
        second = dialog.run_experiment()
        assert second.seed == 41
        assert not (first.observed_response == second.observed_response).all()

        dialog.model.setCurrentIndex(1)
        dialog.field.setCurrentIndex(1)
        magnetic = dialog.run_experiment()
        assert magnetic.model_kind == "cylinder"
        assert magnetic.field_type == "magnetics"
        assert magnetic.response_units == "nT"
        assert dialog.export_button.isEnabled()
    finally:
        dialog.close()

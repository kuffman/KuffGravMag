"""Workflow-state and 3D model-geometry regressions for KGM-M7."""

import numpy as np
import pytest

from engine.forward import Body2D
from engine.project import project_sha256
from engine.survey import SurveyProfile
from engine.visualization import finite_prism_faces
from ui.project_recovery import load_recovery_snapshot, save_recovery_snapshot
from ui.studio_workflow import prepare_model_observations


def _body(*, strike_mode="finite", y_min=-150.0, y_max=650.0):
    return Body2D(
        "Test prism",
        [-100.0, 100.0, 100.0, -100.0, -100.0],
        [200.0, 200.0, 450.0, 450.0, 200.0],
        density=420.0,
        k_succ=0.035,
        y_min=y_min,
        y_max=y_max,
        strike_mode=strike_mode,
    )


def test_finite_prism_mesh_uses_cross_section_and_declared_strike_bounds():
    body = _body()

    faces = finite_prism_faces(body)
    vertices = np.vstack(faces)

    assert len(faces) == 6  # Two end caps and four side faces.
    assert np.min(vertices[:, 0]) == pytest.approx(-100.0)
    assert np.max(vertices[:, 0]) == pytest.approx(100.0)
    assert np.min(vertices[:, 1]) == pytest.approx(-150.0)
    assert np.max(vertices[:, 1]) == pytest.approx(650.0)
    assert np.min(vertices[:, 2]) == pytest.approx(200.0)
    assert np.max(vertices[:, 2]) == pytest.approx(450.0)
    assert all(face.shape[1] == 3 for face in faces)


def test_infinite_strike_cannot_be_misrepresented_as_a_finite_prism():
    with pytest.raises(ValueError, match="Only finite-strike"):
        finite_prism_faces(_body(strike_mode="infinite", y_min=0.0, y_max=0.0))


def test_project_recovery_round_trip_keeps_workflow_inputs_and_processing_history(tmp_path):
    body = _body()
    observations = {
        "x": np.array([-50.0, 0.0, 50.0]),
        "z": np.array([0.0, 10.0, 0.0]),
        "gravity": np.array([0.2, 0.4, 0.2]),
        "magnetic": np.array([-3.0, 2.0, -3.0]),
    }
    steps = [{
        "name": "regular_grid_interpretation_source",
        "parameters": {"source_name": "grid.csv", "sha256": "a" * 64},
    }]
    recovery = tmp_path / "recovery.kgm.json"

    save_recovery_snapshot(
        recovery,
        [body],
        {"F": 48_500.0, "Inc": 5.0, "Dec": 45.0, "Alpha": 20.0},
        observations,
        processing_steps=steps,
    )
    loaded = load_recovery_snapshot(recovery)

    assert loaded["bodies"][0].name == "Test prism"
    assert loaded["field_params"]["F"] == pytest.approx(48_500.0)
    for key, values in observations.items():
        np.testing.assert_array_equal(loaded["observations"][key], values)
    assert loaded["processing_steps"] == steps


def test_canonical_project_hash_is_repeatable_and_sensitive_to_model_edits():
    body = _body()
    field = {"F": 48_000.0, "Inc": 55.0, "Dec": 0.0, "Alpha": 0.0}
    observations = {"x": np.array([0.0, 10.0]), "gravity": np.array([1.0, 2.0])}

    first = project_sha256([body], field, observations)
    repeat = project_sha256([body], dict(reversed(list(field.items()))), observations)
    body.density += 1.0
    changed = project_sha256([body], field, observations)

    assert first == repeat
    assert len(first) == 64
    assert changed != first


def test_regional_residual_handoff_filters_missing_stations_and_maps_elevation_sign():
    profile = SurveyProfile.from_csv(
        b"distance_m,elevation_m,gravity_mgal\n0,100,1\n10,110,\n20,120,4\n",
        metadata={
            "coordinate_reference": "local profile coordinates",
            "profile_azimuth_deg": 0.0,
            "elevation_datum": "local benchmark",
        },
        source_name="survey.csv",
    )

    observations, record = prepare_model_observations(
        profile,
        "gravity",
        np.array([0.1, np.nan, -0.1]),
        source_column="residual_mgal",
        units="mGal",
    )

    np.testing.assert_array_equal(observations["x"], [0.0, 20.0])
    np.testing.assert_array_equal(observations["z"], [-100.0, -120.0])
    np.testing.assert_array_equal(observations["gravity"], [0.1, -0.1])
    assert observations["magnetic"] is None
    assert record["parameters"]["retained_station_count"] == 2
    assert record["parameters"]["omitted_missing_station_count"] == 1

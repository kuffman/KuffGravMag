"""Project compatibility and report metadata regressions for KGM-M3."""

import json
import math

import numpy as np

from engine.forward import Body2D
from engine.model_status import model_status_lines
from engine.project import load_project, save_project
from engine.survey import SurveyProfile


def _body(name, *, mode, y_min, y_max, color="tab:blue"):
    return Body2D(
        name, [-100.0, 100.0, 100.0, -100.0, -100.0],
        [200.0, 200.0, 400.0, 400.0, 200.0],
        density=420.0, k_succ=0.035, J_r=1.25, I_r=32.0, D_r=71.0,
        y_min=y_min, y_max=y_max, color=color, strike_mode=mode,
    )


def test_project_round_trip_preserves_field_observations_and_physical_parameters(tmp_path):
    bodies = [
        _body("Finite", mode="finite", y_min=-150.0, y_max=650.0),
        _body("Infinite", mode="infinite", y_min=0.0, y_max=0.0, color="tab:orange"),
    ]
    field_params = {"F": 48_500.0, "Inc": 5.0, "Dec": 45.0, "Alpha": 20.0}
    observations = {
        "x": np.array([-50.0, 0.0, 50.0]),
        "z": np.array([0.0, 5.0, 0.0]),
        "gravity": np.array([0.2, 0.4, 0.2]),
        "magnetic": np.array([-3.0, 2.0, -3.0]),
    }
    path = tmp_path / "finite-project.kgm.json"
    save_project(path, bodies, field_params, observations)
    loaded = load_project(path)

    assert loaded["field_params"] == field_params
    for key, values in observations.items():
        np.testing.assert_array_equal(loaded["observations"][key], values)
    assert [(body.name, body.strike_mode, body.y_min, body.y_max) for body in loaded["bodies"]] == [
        ("Finite", "finite", -150.0, 650.0),
        ("Infinite", "infinite", 0.0, 0.0),
    ]
    magnetic = loaded["bodies"][0]
    assert (magnetic.density, magnetic.k_succ, magnetic.J_r,
            magnetic.I_r, magnetic.D_r, magnetic.color) == (
                420.0, 0.035, 1.25, 32.0, 71.0, "tab:blue"
            )
    assert json.loads(path.read_text())["schema_version"] == 2


def test_project_round_trip_preserves_survey_metadata_provenance_and_processing_steps(tmp_path):
    profile = SurveyProfile.from_csv(
        b"distance_m,elevation_m,gravity_mgal,magnetic_anomaly_nt\n0,10,1,2\n100,12,,3\n",
        metadata={
            "coordinate_reference": "EPSG:32630",
            "profile_azimuth_deg": 52.0,
            "elevation_datum": "local benchmark 100 m",
        },
        source_name="survey.csv",
    )
    steps = [{"name": "gravity_reduction", "datum_elevation_m": 100.0}]
    path = tmp_path / "survey-project.json"
    save_project(path, [_body("Finite", mode="finite", y_min=-10, y_max=10)], {},
                 survey_profile=profile, processing_steps=steps)
    loaded = load_project(path)

    assert loaded["survey_profile"].metadata == profile.metadata
    assert loaded["survey_profile"].provenance == profile.provenance
    np.testing.assert_array_equal(loaded["survey_profile"].distance_m, profile.distance_m)
    assert np.isnan(loaded["survey_profile"].gravity_mgal[1])
    assert loaded["processing_steps"] == steps
    assert json.loads(path.read_text())["schema_version"] == 2


def test_legacy_project_zero_bounds_remain_infinite_but_explicit_zero_width_is_finite():
    legacy = {
        "bodies": [{
            "name": "legacy", "x": [-1, 1, 1, -1], "z": [2, 2, 4, 4],
            "y_min": 0.0, "y_max": 0.0,
        }],
    }
    loaded = load_project(legacy)
    assert loaded["bodies"][0].strike_mode == "infinite"
    explicit = _body("physical zero width", mode="finite", y_min=0.0, y_max=0.0)
    loaded_explicit = load_project({"bodies": [
        {
            "name": explicit.name, "x": explicit.x.tolist(), "z": explicit.z.tolist(),
            "y_min": 0.0, "y_max": 0.0, "strike_mode": "finite",
        }
    ]})
    assert loaded_explicit["bodies"][0].strike_mode == "finite"


def test_project_preserves_unbounded_infinite_mode_without_nonstandard_json_numbers(tmp_path):
    body = _body("Infinite", mode="infinite", y_min=-math.inf, y_max=math.inf)
    path = tmp_path / "infinite-project.json"
    save_project(path, [body], {"F": 48_000.0, "Inc": 55.0, "Dec": 0.0, "Alpha": 0.0})
    raw = path.read_text()
    assert "-Infinity" not in raw and "Infinity" not in raw
    loaded = load_project(path)
    assert loaded["bodies"][0].strike_mode == "infinite"
    assert loaded["bodies"][0].y_min == -math.inf
    assert loaded["bodies"][0].y_max == math.inf


def test_status_text_distinguishes_validated_finite_and_infinite_models():
    finite = _body("Finite", mode="finite", y_min=-100.0, y_max=500.0)
    infinite = _body("Infinite", mode="infinite", y_min=0.0, y_max=0.0)
    lines = model_status_lines([finite, infinite])
    assert any("Finite Strike" in line and "closed-form" in line for line in lines)
    assert any("Infinite Strike" in line and "benchmarked" in line for line in lines)

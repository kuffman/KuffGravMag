"""Shared geological-model contract tests with independent forward references."""

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from engine.forward import PotentialField2D
from engine.geological_exchange import (
    GeologicalExchangeError,
    export_kgm_model,
    import_shared_model,
    validate_shared_model,
)
from engine.project import load_project, project_to_dict


FIXTURE = Path(__file__).parent / "fixtures" / "shared-geological-model-v1.json"
ELECTRICAL_FIXTURE = Path(__file__).parent / "fixtures" / "shared-geological-model-electrical-v0.1.0-draft.json"
MU0 = 4.0 * math.pi * 1e-7
G = 6.67430e-11


def _gravity_rectangle(a, top, bottom, density):
    def primitive(depth):
        return 2 * depth * math.atan(a / depth) + a * math.log((depth * depth + a * a) / (a * a))

    return 2 * G * density * 1e5 * (primitive(bottom) - primitive(top))


def test_shared_fixture_import_and_hash_are_stable():
    source = FIXTURE.read_bytes()
    model, digest = validate_shared_model(source)
    imported = import_shared_model(source)

    assert digest == hashlib.sha256(source).hexdigest()
    assert imported.source_sha256 == digest
    assert len(imported.bodies) == 3
    assert imported.bodies[0].z.tolist() == [0, 0, 2, 2, 0]
    assert imported.bodies[1].density == 500
    assert imported.bodies[1].k_succ == 0.025
    assert imported.bodies[1].J_r == 1.2
    assert imported.bodies[1].exchange_properties["seismic"]["pWaveVelocity"]["value"] == 1500
    assert imported.model_metadata["coordinateReference"] == model["coordinateReference"]
    assert imported.magnetic_field == {
        "intensity_nT": 48000.0, "inclination_deg": 60.0,
        "declination_deg": 0.0, "profileAzimuth_deg": 0.0,
    }


def test_gravity_and_magnetic_response_match_independent_layer_references():
    imported = import_shared_model(FIXTURE)
    engine = PotentialField2D(imported.bodies)
    x_obs = np.array([10.0])
    z_obs = np.array([-5.0])

    expected_gravity = sum(
        _gravity_rectangle(10.0, top + 5.0, bottom + 5.0, density)
        for top, bottom, density in ((0.0, 2.0, 250.0), (2.0, 7.0, 500.0), (7.0, 17.0, -100.0))
    )
    actual_gravity = engine.calculate_gravity(x_obs, z_obs)[0]
    assert actual_gravity == pytest.approx(expected_gravity, rel=2e-12, abs=2e-12)

    d_x, d_y, d_z, d_t = engine.calculate_magnetics_components(
        x_obs, z_obs, 48_000.0, 60.0, 0.0, 0.0
    )
    expected_x = expected_z = 0.0
    inc = math.radians(60.0)
    field_h = 48_000.0 * 1e-9 / MU0
    for top, bottom, susceptibility, remanence, rem_inc, rem_dec in (
        (0.0, 2.0, 0.012, 0.0, 0.0, 0.0),
        (2.0, 7.0, 0.025, 1.2, 35.0, 15.0),
        (7.0, 17.0, 0.006, 0.0, 0.0, 0.0),
    ):
        coefficient = MU0 / math.pi * (math.atan(10.0 / (top + 5.0)) - math.atan(10.0 / (bottom + 5.0))) * 1e9
        mx = susceptibility * field_h * math.cos(inc) + remanence * math.cos(math.radians(rem_inc)) * math.cos(math.radians(rem_dec))
        mz = susceptibility * field_h * math.sin(inc) + remanence * math.sin(math.radians(rem_inc))
        expected_x -= coefficient * mx
        expected_z += coefficient * mz
    expected_t = expected_x * math.cos(inc) + expected_z * math.sin(inc)
    assert d_x[0] == pytest.approx(expected_x, rel=2e-12, abs=2e-12)
    assert d_y[0] == pytest.approx(0.0, abs=1e-12)
    assert d_z[0] == pytest.approx(expected_z, rel=2e-12, abs=2e-12)
    assert d_t[0] == pytest.approx(expected_t, rel=2e-12, abs=2e-12)


def test_shared_model_round_trips_and_native_project_keeps_interop_metadata():
    imported = import_shared_model(FIXTURE)
    field = imported.magnetic_field
    exported, warnings = export_kgm_model(
        imported.bodies, field, model_metadata=imported.model_metadata,
    )
    assert not warnings
    assert [entity["id"] for entity in exported["geometry"]["entities"]] == [
        "layer:topsoil", "layer:weathered-rock", "layer:bedrock",
    ]
    assert exported["geometry"]["entities"][1]["geometry"]["ring"] == json.loads(FIXTURE.read_text())["geometry"]["entities"][1]["geometry"]["ring"]
    round_trip = import_shared_model(exported)
    assert [body.z.tolist() for body in round_trip.bodies] == [body.z.tolist() for body in imported.bodies]
    assert round_trip.bodies[1].exchange_properties["seismic"]["pWaveVelocity"]["value"] == 1500

    native = project_to_dict(imported.bodies, {"F": 48000, "Inc": 60, "Dec": 0, "Alpha": 0}, shared_model_metadata=imported.model_metadata)
    restored = load_project(native)
    restored_export, _ = export_kgm_model(restored["bodies"], field, model_metadata=restored["shared_model_metadata"])
    assert restored["shared_model_metadata"]["modelId"] == imported.model_metadata["modelId"]
    assert restored_export["geometry"]["entities"][1]["geometry"]["ring"] == exported["geometry"]["entities"][1]["geometry"]["ring"]


def test_draft_electrical_extension_states_survive_native_project_round_trip():
    source = json.loads(ELECTRICAL_FIXTURE.read_text())
    expected = source["geometry"]["entities"]
    imported = import_shared_model(ELECTRICAL_FIXTURE)
    native = project_to_dict(
        imported.bodies,
        {"F": 48000, "Inc": 60, "Dec": 0, "Alpha": 0},
        shared_model_metadata=imported.model_metadata,
    )
    restored = load_project(native)
    exported, warnings = export_kgm_model(
        restored["bodies"],
        imported.magnetic_field,
        model_metadata=restored["shared_model_metadata"],
    )
    validated, _ = validate_shared_model(exported)
    actual = validated["geometry"]["entities"]

    assert not warnings
    assert [entity["extensions"] for entity in actual] == [
        entity.get("extensions", {}) for entity in expected
    ]
    assert actual[0]["extensions"]["org.kuffman.kge:electrical"]["dc"]["state"] == "known"
    assert actual[1]["extensions"]["org.kuffman.kge:electrical"]["dc"] == {"state": "unknown"}
    assert "org.kuffman.kge:electrical" not in actual[2].get("extensions", {})
    for entity in actual:
        assert "electrical" not in entity["properties"]
    for before, after in zip(expected, actual):
        assert before["geometry"]["ring"] == after["geometry"]["ring"]


def test_layer_round_trip_preserves_imported_ring_across_float_datum_conversion():
    model = json.loads(FIXTURE.read_text())
    vertical = model["coordinateReference"]["vertical"]
    surface = 1.1
    vertical.update({"positiveDirection": "up", "groundSurfaceCoordinateM": surface})

    def transform(value):
        # Keep the valid source coordinate -0.9 exactly as written. The same
        # point converts to depth 2.0 and back to -0.8999999999999999.
        return -0.9 if value == -2.0 else surface + value

    for interface in model["geometry"]["interfaces"]:
        interface["coordinates"] = [[x, transform(z)] for x, z in interface["coordinates"]]
    for entity in model["geometry"]["entities"]:
        entity["geometry"]["ring"] = [
            [x, transform(z)] for x, z in entity["geometry"]["ring"]
        ]
    expected_entities = copy.deepcopy(model["geometry"]["entities"])

    imported = import_shared_model(model)
    exported, warnings = export_kgm_model(
        imported.bodies, imported.magnetic_field, model_metadata=imported.model_metadata,
    )

    assert not warnings
    for expected, actual in zip(expected_entities, exported["geometry"]["entities"]):
        assert actual["kind"] == "layer"
        assert actual["stratigraphicOrder"] == expected["stratigraphicOrder"]
        assert actual["topInterfaceId"] == expected["topInterfaceId"]
        assert actual["bottomInterfaceId"] == expected["bottomInterfaceId"]
        assert actual["geometry"]["ring"] == expected["geometry"]["ring"]

    native = project_to_dict(
        imported.bodies,
        {"F": 48000, "Inc": 60, "Dec": 0, "Alpha": 0},
        shared_model_metadata=imported.model_metadata,
    )
    restored = load_project(native)
    restored_export, restored_warnings = export_kgm_model(
        restored["bodies"], imported.magnetic_field,
        model_metadata=restored["shared_model_metadata"],
    )
    assert not restored_warnings
    assert [entity["geometry"]["ring"] for entity in restored_export["geometry"]["entities"]] == [
        entity["geometry"]["ring"] for entity in expected_entities
    ]

    changed = import_shared_model(model)
    changed.bodies[0].z[2:4] += 0.1
    changed_export, changed_warnings = export_kgm_model(
        changed.bodies, changed.magnetic_field, model_metadata=changed.model_metadata,
    )
    assert changed_export["geometry"]["entities"][0]["kind"] == "body"
    assert any("polygon geometry changed after import" in warning for warning in changed_warnings)


def test_explicit_density_reference_and_positive_down_datum_are_mapped_losslessly():
    model = json.loads(FIXTURE.read_text())
    first = model["geometry"]["entities"][0]["properties"]["density"]
    first.update({"basis": "absolute", "value": 2670.0, "referenceDensity": {"state": "known", "value": 2420.0, "unit": "kg/m3"}})
    model["coordinateReference"]["vertical"].update({"positiveDirection": "down", "groundSurfaceCoordinateM": 100.0})
    for interface in model["geometry"]["interfaces"]:
        interface["coordinates"] = [[x, 100.0 - z] for x, z in interface["coordinates"]]
    for entity in model["geometry"]["entities"]:
        entity["geometry"]["ring"] = [[x, 100.0 - z] for x, z in entity["geometry"]["ring"]]

    imported = import_shared_model(model)
    assert imported.bodies[0].density == pytest.approx(250.0)
    assert imported.bodies[0].z.tolist() == [0, 0, 2, 2, 0]
    exported, _ = export_kgm_model(imported.bodies, imported.magnetic_field, model_metadata=imported.model_metadata)
    assert exported["geometry"]["entities"][0]["properties"]["density"]["basis"] == "absolute"
    assert exported["geometry"]["entities"][0]["properties"]["density"]["value"] == pytest.approx(2670)
    assert exported["geometry"]["entities"][0]["geometry"]["ring"] == model["geometry"]["entities"][0]["geometry"]["ring"]


def test_profile_azimuth_is_mapped_to_the_local_transect_axis():
    model = json.loads(FIXTURE.read_text())
    model["coordinateReference"]["horizontal"]["azimuthDegFromNorthClockwise"] = 10.0
    model["magneticField"]["profileAzimuth_deg"] = 20.0
    with pytest.raises(GeologicalExchangeError, match="must match the declared local-transect"):
        validate_shared_model(model)

    imported = import_shared_model(FIXTURE)
    field = dict(imported.magnetic_field, profileAzimuth_deg=90.0)
    exported, _ = export_kgm_model(imported.bodies, field, model_metadata=imported.model_metadata)
    assert exported["coordinateReference"]["horizontal"]["azimuthDegFromNorthClockwise"] == 90.0
    assert exported["magneticField"]["profileAzimuth_deg"] == 90.0

    missing = json.loads(FIXTURE.read_text())
    del missing["coordinateReference"]["horizontal"]["azimuthDegFromNorthClockwise"]
    with pytest.raises(GeologicalExchangeError, match="explicit null when unknown"):
        validate_shared_model(missing)


def test_unknown_is_not_zero_and_invalid_or_unsupported_input_is_rejected():
    model = json.loads(FIXTURE.read_text())
    properties = model["geometry"]["entities"][0]["properties"]
    del properties["density"]["value"]
    properties["density"]["state"] = "unknown"
    with pytest.raises(GeologicalExchangeError, match="unknown is not equivalent to zero"):
        import_shared_model(model)

    bad = copy.deepcopy(model)
    bad["geometry"]["entities"][0]["properties"]["density"] = {
        "state": "known", "basis": "absolute", "value": 0, "unit": "kg/m3",
        "referenceDensity": {"state": "known", "value": 2500, "unit": "kg/m3"},
    }
    with pytest.raises(GeologicalExchangeError, match="absolute density must be positive"):
        validate_shared_model(bad)

    bad = json.loads(FIXTURE.read_text())
    bad["geometry"]["entities"][0]["geometry"] = {
        "kind": "polyhedron3d",
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, -1], [0, 0, -1]],
        "faces": [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
    }
    with pytest.raises(GeologicalExchangeError, match="polyhedron3d is not converted"):
        import_shared_model(bad)

    with pytest.raises(GeologicalExchangeError, match="Duplicate JSON object key"):
        validate_shared_model('{"format":"bad","format":"KuffGeoSharedGeologicalModel"}')
    with pytest.raises(GeologicalExchangeError, match="5 MB import limit"):
        validate_shared_model(b" " * (5 * 1024 * 1024 + 1))
    with pytest.raises(GeologicalExchangeError, match="ISO date-time"):
        malformed_date = json.loads(FIXTURE.read_text())
        malformed_date["provenance"]["createdAt"] = "not-a-date"
        validate_shared_model(malformed_date)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("magneticField",), None),
        (("geometry", "entities", 0, "properties", "density"), None),
        (("geometry", "entities", 0, "properties", "seismic"), None),
        (("geometry", "entities", 0, "properties", "elastic"), None),
    ],
)
def test_explicit_null_is_not_treated_as_an_absent_property(path, value):
    model = json.loads(FIXTURE.read_text())
    target = model
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(GeologicalExchangeError):
        validate_shared_model(model)

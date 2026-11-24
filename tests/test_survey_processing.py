from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from engine.survey import SurveyProfile, BaseStationSeries, export_processing_bundle
from engine.processing import (
    apply_gravity_reductions,
    apply_magnetic_diurnal_correction,
    separate_polynomial_regional,
    interpolate_profile,
    grid_scattered_xyz,
    regular_grid_from_xyz,
)


METADATA = {
    "coordinate_reference": "local tangent plane; distance along profile",
    "profile_azimuth_deg": 24.0,
    "elevation_datum": "local survey benchmark, 100 m",
}


def make_profile() -> SurveyProfile:
    csv_bytes = (Path(__file__).parent / "fixtures" / "survey-profile-v1.csv").read_bytes()
    return SurveyProfile.from_csv(csv_bytes, metadata=METADATA, source_name="line.csv")


def test_profile_schema_keeps_alignment_missing_values_and_source_provenance():
    payload = (
        "distance_m,elevation_m,gravity_mgal,magnetic_anomaly_nt,station_time_utc\n"
        "0,100,10,,2026-01-01T00:00:00Z\n"
        "100,110,,102,2026-01-01T00:01:00+00:00\n"
    ).encode()
    profile = SurveyProfile.from_csv(payload, metadata=METADATA, source_name="sparse.csv")

    assert profile.schema_version == 1
    assert profile.distance_m.tolist() == [0.0, 100.0]
    assert np.isnan(profile.gravity_mgal[1])
    assert np.isnan(profile.magnetic_anomaly_nt[0])
    assert profile.station_time_utc[1] == "2026-01-01T00:01:00+00:00"
    assert profile.provenance["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert profile.provenance["source_name"] == "sparse.csv"
    assert profile.metadata["coordinate_reference"] == METADATA["coordinate_reference"]


@pytest.mark.parametrize(
    "payload, metadata, message",
    [
        (b"distance_m,gravity_mgal\n0,1\n100,2\n", METADATA, "elevation_m"),
        (b"distance_m,elevation_m,gravity_mgal\n0,100,1\n100,101,2\n", {}, "coordinate_reference"),
        (b"distance_m,elevation_m,gravity_mgal\n0,100,1\n0,101,2\n", METADATA, "unique"),
        (
            b"distance_m,elevation_m,gravity_mgal\n0,100,1\n100,101,2\n",
            {**METADATA, "profile_azimuth_deg": 181},
            "profile_azimuth_deg",
        ),
        (
            b"distance_m,elevation_m\n0,100\n100,101\n",
            METADATA,
            "at least one magnetic or gravity",
        ),
    ],
)
def test_profile_schema_rejects_ambiguous_or_incomplete_inputs(payload, metadata, message):
    with pytest.raises(ValueError, match=message):
        SurveyProfile.from_csv(payload, metadata=metadata, source_name="bad.csv")


def test_free_air_and_bouguer_slab_reductions_use_declared_sign_and_keep_raw():
    profile = make_profile()
    result = apply_gravity_reductions(
        profile,
        datum_elevation_m=100.0,
        bouguer_density_kg_m3=2670.0,
    )

    height = profile.elevation_m - 100.0
    free_air = profile.gravity_mgal + 0.3086 * height
    bouguer = free_air - (0.00004191 * 2670.0 * height)
    assert result.columns["gravity_observed_mgal"].tolist() == profile.gravity_mgal.tolist()
    assert result.columns["gravity_free_air_reduced_mgal"] == pytest.approx(free_air)
    assert result.columns["gravity_bouguer_slab_reduced_mgal"] == pytest.approx(bouguer)
    assert result.parameters["bouguer_density_kg_m3"] == 2670.0
    assert any("not a complete Bouguer anomaly" in item for item in result.assumptions)


def test_gravity_reduction_requires_explicit_bouguer_density_and_preserves_missing_values():
    profile = SurveyProfile.from_csv(
        b"distance_m,elevation_m,gravity_mgal\n0,50,1\n100,60,\n",
        metadata=METADATA,
        source_name="missing.csv",
    )
    with pytest.raises(ValueError, match="density_kg_m3"):
        apply_gravity_reductions(profile, datum_elevation_m=0.0, bouguer_density_kg_m3=0.0)
    output = apply_gravity_reductions(profile, datum_elevation_m=0.0)
    assert np.isnan(output.columns["gravity_free_air_reduced_mgal"][1])
    assert "gravity_bouguer_slab_reduced_mgal" not in output.columns


def test_diurnal_correction_interpolates_base_series_and_preserves_reference_level():
    profile = make_profile()
    base = BaseStationSeries.from_csv(
        (Path(__file__).parent / "fixtures" / "base-station-v1.csv").read_bytes(),
        source_name="base.csv",
    )
    output = apply_magnetic_diurnal_correction(
        profile,
        base,
        reference_time_utc="2026-01-01T00:00:00Z",
    )

    # The base series rises 2 nT per minute; remove only its change from t0.
    assert output.columns["magnetic_diurnal_corrected_nt"] == pytest.approx([100, 100, 100, 100])
    assert output.parameters["reference_base_field_nt"] == 50000.0
    assert any("No IGRF or main-field subtraction" in item for item in output.assumptions)
    bundle = export_processing_bundle(profile, [output])
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        base_filename = f"base_station_{base.provenance['source_sha256'][:12]}.csv"
        assert base_filename in archive.namelist()
        assert b"50000" in archive.read(base_filename)
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["processing_steps"][0]["supporting_files"][base_filename]


def test_diurnal_correction_rejects_extrapolation_and_missing_station_times():
    profile = make_profile()
    base = BaseStationSeries.from_csv(
        b"time_utc,total_field_nt\n2026-01-01T00:00:00Z,50000\n2026-01-01T00:03:00Z,50003\n",
        source_name="base.csv",
    )
    with pytest.raises(ValueError, match="outside the base-station time span"):
        apply_magnetic_diurnal_correction(
            profile, base, reference_time_utc="2025-12-31T23:59:00Z"
        )

    untimed = SurveyProfile.from_csv(
        b"distance_m,elevation_m,magnetic_anomaly_nt\n0,0,1\n1,0,2\n",
        metadata=METADATA,
        source_name="untimed.csv",
    )
    with pytest.raises(ValueError, match="station_time_utc"):
        apply_magnetic_diurnal_correction(
            untimed, base, reference_time_utc="2026-01-01T00:00:00Z"
        )


def test_polynomial_regional_separation_recovers_known_quadratic_baseline():
    x = np.linspace(-500.0, 500.0, 101)
    regional = 2.0 + 0.004 * x - 0.000002 * x**2
    scaled = (x - x.mean()) / np.max(np.abs(x - x.mean()))
    design = np.polynomial.polynomial.polyvander(scaled, 2)
    basis, _ = np.linalg.qr(design)
    seed = np.sin(2.0 * np.pi * x / 100.0)
    residual = 0.6 * (seed - basis @ (basis.T @ seed))
    output = separate_polynomial_regional(x, regional + residual, degree=2, units="mGal")

    assert output.columns["regional_mgal"] == pytest.approx(regional, abs=1e-10)
    assert output.columns["residual_mgal"] == pytest.approx(residual, abs=1e-10)
    assert output.parameters["degree"] == 2
    assert any("not a uniquely determined geological background" in item for item in output.assumptions)


def test_profile_interpolation_recovers_linear_field_and_never_extrapolates():
    x = np.array([0.0, 70.0, 190.0, 310.0])
    values = 3.0 * x - 8.0
    output = interpolate_profile(x, values, spacing_m=50.0, units="nT")

    assert output.distance_m == pytest.approx(np.arange(0.0, 301.0, 50.0))
    assert output.values == pytest.approx(3.0 * output.distance_m - 8.0)
    assert output.distance_m[-1] < x[-1]
    with pytest.raises(ValueError, match="uniformly spaced|unique"):
        interpolate_profile([0, 1, 1], [1, 2, 3], spacing_m=1, units="nT")


def test_scattered_gridding_recovers_plane_and_leaves_outside_hull_masked():
    x = np.array([0, 100, 0, 100, 50, 25, 75], dtype=float)
    y = np.array([0, 0, 100, 100, 50, 75, 25], dtype=float)
    values = 3.0 * x - 2.0 * y + 7.0
    grid = grid_scattered_xyz(x, y, values, spacing_x_m=25.0, spacing_y_m=25.0)
    xx, yy = np.meshgrid(grid.x_m, grid.y_m)
    finite = np.isfinite(grid.values)

    assert grid.values.shape == (grid.y_m.size, grid.x_m.size)
    assert grid.values[finite] == pytest.approx((3.0 * xx - 2.0 * yy + 7.0)[finite])
    assert not np.isnan(grid.values[0, 0])
    assert grid.coordinate_reference == "unspecified"


def test_regular_grid_xyz_requires_a_complete_rectangular_uniform_lattice():
    grid = regular_grid_from_xyz(
        [0, 100, 0, 100], [0, 0, 50, 50], [1, 2, 3, 4], units="mGal"
    )
    assert grid.x_m.tolist() == [0.0, 100.0]
    assert grid.y_m.tolist() == [0.0, 50.0]
    assert grid.values.tolist() == [[1.0, 2.0], [3.0, 4.0]]

    with pytest.raises(ValueError, match="complete rectangular grid"):
        regular_grid_from_xyz([0, 100, 0], [0, 0, 50], [1, 2, 3])
    with pytest.raises(ValueError, match="duplicate"):
        regular_grid_from_xyz([0, 100, 0, 0], [0, 0, 50, 0], [1, 2, 3, 4])


def test_export_bundle_contains_raw_data_processing_parameters_and_hashes():
    profile = make_profile()
    product = apply_gravity_reductions(profile, datum_elevation_m=100.0)
    regular = interpolate_profile(profile.distance_m, profile.gravity_mgal, spacing_m=50, units="mGal")
    bundle = export_processing_bundle(profile, [product], regular_profiles=[regular])

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert sorted(archive.namelist()) == ["manifest.json", "profile.csv", "regular_profile_1.csv"]
        manifest = json.loads(archive.read("manifest.json"))
        table = archive.read("profile.csv").decode()
        regular_table = archive.read("regular_profile_1.csv").decode()
    assert "gravity_observed_mgal" in table
    assert "gravity_free_air_reduced_mgal" in table
    assert manifest["profile"]["provenance"]["source_name"] == "line.csv"
    assert manifest["processing_steps"][0]["parameters"]["datum_elevation_m"] == 100.0
    assert manifest["processing_steps"][0]["output_sha256"]["gravity_free_air_reduced_mgal"]
    assert manifest["regular_profiles"][0]["spacing_m"] == 50
    assert "distance_m,value" in regular_table
    assert export_processing_bundle(profile, [product], regular_profiles=[regular]) == bundle

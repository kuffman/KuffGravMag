from __future__ import annotations

from datetime import date
import math

import numpy as np
import pytest

from engine.euler import euler_window_sweep
from engine.igrf import igrf14_main_field
from engine.processing import apply_dem_terrain_correction, apply_igrf_main_field_correction
from engine.spectral import estimate_spectral_depth, stabilized_magnetic_transform_grid
from engine.terrain import TerrainDEM, terrain_correction
from validation.analytical import rectangular_prism_gravity_gz_mgal


def _small_dem():
    axis = np.array([0.0, 10.0, 20.0])
    elevations = np.array([
        [120.0, 90.0, 30.0],
        [50.0, 0.0, 0.0],
        [0.0, -30.0, np.nan],
    ])
    dem = TerrainDEM(axis, axis, elevations, "UTM 30N / metres", "survey orthometric datum")
    return axis, elevations, dem


def test_dem_terrain_attraction_matches_independent_prism_closed_form():
    axis, elevations, dem = _small_dem()
    density = 2670.0
    station = (12.0, 9.0, 250.0)
    result = terrain_correction(
        dem, [station[0]], [station[1]], [station[2]],
        station_coordinate_reference="UTM 30N / metres",
        reference_elevation_m=0.0,
        density_kg_m3=density,
        elevation_datum="survey orthometric datum",
    )
    expected = 0.0
    for row, y in enumerate(axis):
        for column, x in enumerate(axis):
            height = elevations[row, column]
            if not math.isfinite(height) or height == 0.0:
                continue
            top, bottom = max(height, 0.0), min(height, 0.0)
            expected += rectangular_prism_gravity_gz_mgal(
                (x - 5.0, x + 5.0),
                (y - 5.0, y + 5.0),
                (-top, -bottom),
                math.copysign(density, height),
                (station[0], station[1], -station[2]),
            )

    assert result.gravity_down_mgal[0] == pytest.approx(expected, rel=2e-10, abs=1e-12)
    assert result.parameters["dem_cell_count"] == 8
    assert result.parameters["active_prism_count"] == 5
    assert result.parameters["coordinate_reference"] == "UTM 30N / metres"
    assert result.assumptions[2].startswith("Cells above the reference plane use +density")
    with pytest.raises(ValueError, match="same declared coordinate reference"):
        terrain_correction(
            dem, [station[0]], [station[1]], [station[2]],
            station_coordinate_reference="different projected CRS",
            reference_elevation_m=0.0, density_kg_m3=density,
            elevation_datum="survey orthometric datum",
        )

    product = apply_dem_terrain_correction(
        dem, [station[0]], [station[1]], [station[2]],
        station_coordinate_reference="UTM 30N / metres",
        reference_elevation_m=0.0, density_kg_m3=density,
        elevation_datum="survey orthometric datum",
    )
    assert product.columns["terrain_residual_down_mgal"][0] == pytest.approx(expected)
    assert product.to_manifest()["output_sha256"]["terrain_residual_down_mgal"]
    assert set(product.supporting_files) == {"terrain_dem_xyz.csv", "terrain_stations.csv"}
    assert product.parameters["station_x_m"] == [station[0]]


def test_dem_terrain_attraction_matches_optional_harmonica_prism_reference():
    harmonica = pytest.importorskip("harmonica")
    axis, elevations, dem = _small_dem()
    density = 2670.0
    station = (12.0, 9.0, 250.0)
    result = terrain_correction(
        dem, [station[0]], [station[1]], [station[2]],
        station_coordinate_reference="UTM 30N / metres",
        reference_elevation_m=0.0, density_kg_m3=density,
        elevation_datum="survey orthometric datum",
    )
    prisms, contrasts = [], []
    for row, y in enumerate(axis):
        for column, x in enumerate(axis):
            height = elevations[row, column]
            if not math.isfinite(height) or height == 0.0:
                continue
            prisms.append([x - 5.0, x + 5.0, y - 5.0, y + 5.0, min(height, 0.0), max(height, 0.0)])
            contrasts.append(math.copysign(density, height))
    expected = harmonica.prism_gravity(
        (np.array([station[0]]), np.array([station[1]]), np.array([station[2]])),
        np.asarray(prisms), np.asarray(contrasts), field="g_z", parallel=False,
    )

    np.testing.assert_allclose(result.gravity_down_mgal, expected, rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize(
    ("latitude", "longitude", "altitude_m", "expected"),
    [
        (0.0, 0.0, 0.0, (27457.0, -1926.0, -15997.0, 31835.0, -4.014, -30.166)),
        (45.0, 10.0, 0.0, (22843.0, 1451.0, 41825.0, 47678.0, 3.634, 61.310)),
        (51.5, 0.0, 400000.0, (16632.0, 69.0, 37519.0, 41040.0, 0.239, 66.093)),
        (-33.86, 151.21, 0.0, (24040.0, 5450.0, -51381.0, 56988.0, 12.773, -64.371)),
    ],
)
def test_igrf14_matches_official_bgs_service_vectors(latitude, longitude, altitude_m, expected):
    field = igrf14_main_field(latitude, longitude, altitude_m, 2025.0)
    actual = (
        field.north_nt, field.east_nt, field.down_nt, field.total_nt,
        field.declination_deg, field.inclination_deg,
    )
    np.testing.assert_allclose(actual[:4], expected[:4], rtol=0.0, atol=2.0)
    np.testing.assert_allclose(actual[4:], expected[4:], rtol=0.0, atol=0.02)
    assert field.model == "IGRF-14"
    assert field.component_units == "nT"
    assert field.decimal_year == 2025.0
    assert field.total_nt == pytest.approx(math.sqrt(field.north_nt**2 + field.east_nt**2 + field.down_nt**2))


def test_igrf14_uses_utc_date_and_rejects_out_of_range_or_invalid_location():
    assert igrf14_main_field(0, 0, 0, date(2025, 1, 1)).decimal_year == 2025.0
    with pytest.raises(ValueError, match="1900.0 through 2030.0"):
        igrf14_main_field(0, 0, 0, "2030-12-31")
    with pytest.raises(ValueError, match="latitude_deg"):
        igrf14_main_field(91, 0, 0, 2025.0)


def test_igrf_main_field_correction_subtracts_f_only_from_raw_total_field():
    reference = [
        (0.0, 0.0, 0.0),
        (45.0, 10.0, 0.0),
        (51.5, 0.0, 400000.0),
        (-33.86, 151.21, 0.0),
    ]
    fields = [igrf14_main_field(*location, 2025.0) for location in reference]
    anomaly = np.array([12.0, -8.0, 2.5, 15.0])
    product = apply_igrf_main_field_correction(
        np.array([field.total_nt for field in fields]) + anomaly,
        latitude_geodetic_deg=[item[0] for item in reference],
        longitude_east_deg=[item[1] for item in reference],
        altitude_ellipsoid_m=[item[2] for item in reference],
        dates=["2025-01-01"] * 4,
    )

    np.testing.assert_allclose(product.columns["magnetic_igrf_corrected_nt"], anomaly, atol=2e-11)
    np.testing.assert_allclose(product.columns["igrf_north_nt"], [field.north_nt for field in fields])
    assert product.parameters["model"] == "IGRF-14"
    assert len(product.parameters["coefficient_sha256"]) == 64
    assert product.parameters["valid_decimal_year_range"] == [1900.0, 2030.0]
    assert "raw total-field observations" in product.assumptions[0]


def test_stabilized_low_latitude_transform_reports_and_limits_noise_gain():
    rng = np.random.default_rng(24)
    values = rng.normal(size=(32, 40))
    regularized = stabilized_magnetic_transform_grid(
        values, spacing_x_m=20.0, spacing_y_m=30.0,
        field_inclination_deg=2.0, field_declination_deg=-8.0,
        magnetization_inclination_deg=2.0, magnetization_declination_deg=-8.0,
        regularization=0.1, padding="none", max_gain_limit=20.0,
    )
    assert np.isfinite(regularized.values).all()
    assert regularized.max_gain <= 20.0
    assert regularized.rms_gain > 0.0
    assert regularized.fraction_gain_above_one > 0.0
    assert regularized.regularization == 0.1
    with pytest.raises(ValueError, match="increase regularization"):
        stabilized_magnetic_transform_grid(
            values, spacing_x_m=20.0, spacing_y_m=30.0,
            field_inclination_deg=2.0, field_declination_deg=-8.0,
            magnetization_inclination_deg=2.0, magnetization_declination_deg=-8.0,
            regularization=0.001, padding="none", max_gain_limit=20.0,
        )


def test_stabilized_magnetic_transform_matches_harmonica_at_moderate_inclination():
    harmonica = pytest.importorskip("harmonica")
    xr = pytest.importorskip("xarray")
    nx, ny = 40, 32
    dx, dy = 20.0, 30.0
    x, y = np.arange(nx) * dx, np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y)
    values = np.cos(2.0 * np.pi * 3 * xx / (nx * dx) + 2.0 * np.pi * 2 * yy / (ny * dy))
    result = stabilized_magnetic_transform_grid(
        values, spacing_x_m=dx, spacing_y_m=dy,
        field_inclination_deg=50.0, field_declination_deg=10.0,
        magnetization_inclination_deg=55.0, magnetization_declination_deg=15.0,
        regularization=1e-6, padding="none", max_gain_limit=20.0,
    )
    grid = xr.DataArray(values, coords={"northing": y, "easting": x}, dims=("northing", "easting"))
    expected = harmonica.reduction_to_pole(grid, 50.0, 10.0, 55.0, 15.0).values
    np.testing.assert_allclose(result.values, expected, rtol=2e-9, atol=2e-10)


def test_spectral_depth_recovers_band_limited_synthetic_source_depth_with_uncertainty():
    n = 128
    spacing = 25.0
    expected_depth = 300.0
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=spacing)
    kyy, kxx = np.meshgrid(ky, kx, indexing="ij")
    kr = np.hypot(kxx, kyy)
    rng = np.random.default_rng(7)
    spectrum = np.exp(-kr * expected_depth) * np.exp(1j * rng.uniform(-np.pi, np.pi, (n, n)))
    for row in range(n):
        for column in range(n):
            spectrum[(-row) % n, (-column) % n] = np.conj(spectrum[row, column])
    values = np.fft.ifft2(spectrum).real
    result = estimate_spectral_depth(
        values, spacing_x_m=spacing, spacing_y_m=spacing,
        wavenumber_band_rad_per_m=(0.008, 0.04), radial_bin_count=20,
    )

    assert result.depth_m == pytest.approx(expected_depth, rel=0.08)
    assert result.standard_error_m > 0.0
    assert result.r_squared > 0.98
    assert result.bin_count >= 4


def test_euler_window_index_and_rms_filters_retain_rejected_estimates():
    x = np.linspace(-1200.0, 1200.0, 31)
    y = np.linspace(-900.0, 900.0, 25)
    xx, yy = np.meshgrid(x, y)
    x0, y0, z0 = 120.0, -80.0, -600.0
    amplitude, base = 8.0e7, 25.0
    rx, ry, rz = xx - x0, yy - y0, -z0
    radius_squared = rx**2 + ry**2 + rz**2
    values = amplitude / radius_squared + base
    derivative_x = -2.0 * amplitude * rx / radius_squared**2
    derivative_y = -2.0 * amplitude * ry / radius_squared**2
    derivative_z_up = -2.0 * amplitude * rz / radius_squared**2
    estimates = euler_window_sweep(
        x, y, values, derivative_x, derivative_y, derivative_z_up,
        structural_indices=(1.0, 2.0, 3.0),
        window_size_samples=15, stride_samples=10,
        condition_limit=1e8, max_rms_residual=1e-6,
    )

    assert estimates
    center = [estimate for estimate in estimates if estimate.row_start == 0 and estimate.column_start == 0]
    assert {estimate.structural_index for estimate in center} == {1.0, 2.0, 3.0}
    assert next(item for item in center if item.structural_index == 2.0).accepted
    assert not next(item for item in center if item.structural_index == 1.0).accepted
    assert next(item for item in center if item.structural_index == 1.0).filter_reason == "rms_filter"


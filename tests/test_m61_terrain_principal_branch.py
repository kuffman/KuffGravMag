from __future__ import annotations

import numpy as np
import pytest

from engine.forward import Body2D, PotentialField2D
from engine.terrain import TerrainDEM, terrain_correction
from validation.analytical import (
    rectangular_prism_gravity_gz_mgal,
    rectangular_prism_volume_integral,
)


_DENSITY = 2670.0
_CRS = "local east/north metres"
_DATUM = "local vertical datum"


def _one_cell_dem(*, elevation_m: float = 20.0, reference_m: float = 10.0) -> TerrainDEM:
    # Cell (0, 0) spans [-5, 5] m in both horizontal dimensions. The other
    # cells have zero density contrast and make this a valid regular DEM.
    elevations = np.full((2, 2), reference_m, dtype=np.float64)
    elevations[0, 0] = elevation_m
    return TerrainDEM(
        np.array([0.0, 10.0]),
        np.array([0.0, 10.0]),
        elevations,
        _CRS,
        _DATUM,
    )


def _terrain(dem: TerrainDEM, stations: np.ndarray, *, density: float = _DENSITY):
    return terrain_correction(
        dem,
        stations[:, 0],
        stations[:, 1],
        stations[:, 2],
        station_coordinate_reference=_CRS,
        reference_elevation_m=10.0 if dem.elevation_m[0, 0] > 10.0 else 20.0,
        density_kg_m3=density,
        elevation_datum=_DATUM,
    ).gravity_down_mgal


def _direct_volume_gravity(
    station_up: tuple[float, float, float],
    *,
    density: float = _DENSITY,
    order: int = 40,
) -> float:
    gravity, _ = rectangular_prism_volume_integral(
        (-5.0, 5.0),
        (-5.0, 5.0),
        (-20.0, -10.0),
        density,
        (0.0, 0.0, 0.0),
        (station_up[0], station_up[1], -station_up[2]),
        order=order,
    )
    return gravity


def test_terrain_prism_matches_newtonian_volume_integral_above_below_and_inside():
    stations = np.array([
        [0.0, 0.0, 30.0],  # above the prism
        [0.0, 0.0, 0.0],   # below the prism
        [0.0, 0.0, 15.0],  # symmetric point inside the prism
    ])
    actual = _terrain(_one_cell_dem(), stations)
    expected = np.array([_direct_volume_gravity(tuple(station)) for station in stations])

    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
    assert actual[0] > 0.0
    assert actual[1] < 0.0
    assert actual[2] == pytest.approx(0.0, abs=2e-12)


def test_terrain_prism_matches_harmonica_above_below_and_inside():
    harmonica = pytest.importorskip("harmonica")
    stations = np.array([
        [0.0, 0.0, 30.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 15.0],
    ])
    actual = _terrain(_one_cell_dem(), stations)
    prism = np.array([[-5.0, 5.0, -5.0, 5.0, 10.0, 20.0]])
    expected = harmonica.prism_gravity(
        (stations[:, 0], stations[:, 1], stations[:, 2]),
        prism,
        np.array([_DENSITY]),
        field="g_z",
        parallel=False,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


def test_nagy_reference_principal_branch_and_corner_limits_match_harmonica():
    harmonica = pytest.importorskip("harmonica")
    # Observations use the reference function's positive-down convention.
    observations_down = np.array([
        [0.0, 0.0, -30.0],  # above
        [0.0, 0.0, 0.0],    # below
        [0.0, 0.0, -15.0],  # inside on the symmetry axis
        [5.0, 0.0, -15.0],  # side face
        [5.0, 5.0, -20.0],  # vertex
    ])
    expected = harmonica.prism_gravity(
        (observations_down[:, 0], observations_down[:, 1], -observations_down[:, 2]),
        np.array([[-5.0, 5.0, -5.0, 5.0, 10.0, 20.0]]),
        np.array([_DENSITY]),
        field="g_z",
        parallel=False,
    )
    with np.errstate(divide="raise", invalid="raise"):
        actual = np.array([
            rectangular_prism_gravity_gz_mgal(
                (-5.0, 5.0), (-5.0, 5.0), (-20.0, -10.0), _DENSITY,
                tuple(observation),
            )
            for observation in observations_down
        ])

    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


def test_finite_strike_gravity_is_consistent_below_and_inside_the_prism():
    harmonica = pytest.importorskip("harmonica")
    section = np.array([[-5.0, 10.0], [5.0, 10.0], [5.0, 20.0], [-5.0, 20.0]])
    body = Body2D(
        "finite terrain-check prism",
        np.r_[section[:, 0], section[0, 0]],
        np.r_[section[:, 1], section[0, 1]],
        density=_DENSITY,
        y_min=-5.0,
        y_max=5.0,
        strike_mode="finite",
    )
    observations_down = np.array([[0.0, 0.0], [0.0, 30.0], [0.0, 15.0]])
    actual = PotentialField2D([body]).calculate_gravity(
        observations_down[:, 0], observations_down[:, 1]
    )
    expected_volume = np.array([
        rectangular_prism_volume_integral(
            (-5.0, 5.0), (-5.0, 5.0), (10.0, 20.0), _DENSITY,
            (0.0, 0.0, 0.0), (x, 0.0, z), order=40,
        )[0]
        for x, z in observations_down
    ])
    expected_harmonica = harmonica.prism_gravity(
        (observations_down[:, 0], np.zeros(3), -observations_down[:, 1]),
        np.array([[-5.0, 5.0, -5.0, 5.0, -20.0, -10.0]]),
        np.array([_DENSITY]),
        field="g_z",
        parallel=False,
    )

    np.testing.assert_allclose(actual, expected_volume, rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(actual, expected_harmonica, rtol=2e-10, atol=2e-12)


def test_terrain_prism_matches_harmonica_on_axes_and_at_or_near_boundaries():
    harmonica = pytest.importorskip("harmonica")
    stations = np.array([
        [0.0, 0.0, 20.0],       # top face, on the vertical axis
        [0.0, 0.0, 10.0],       # bottom face, on the vertical axis
        [5.0, 0.0, 15.0],       # east face
        [-5.0, 0.0, 15.0],      # west face
        [0.0, 5.0, 15.0],       # north face
        [0.0, 0.0, 20.000001],  # just above top face
        [5.000001, 0.0, 15.0],  # just outside east face
        [5.0, 5.0, 20.0],       # prism vertex; finite gravity limit
    ])
    prism = np.array([[-5.0, 5.0, -5.0, 5.0, 10.0, 20.0]])

    with np.errstate(divide="raise", invalid="raise"):
        actual = _terrain(_one_cell_dem(), stations)
    expected = harmonica.prism_gravity(
        (stations[:, 0], stations[:, 1], stations[:, 2]),
        prism,
        np.array([_DENSITY]),
        field="g_z",
        parallel=False,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)


def test_terrain_prism_respects_horizontal_symmetry_and_negative_density_contrast():
    stations = np.array([
        [-1.25, -0.75, 30.0],
        [-1.25, 0.75, 30.0],
        [1.25, -0.75, 30.0],
        [1.25, 0.75, 30.0],
        [-1.25, -0.75, 0.0],
        [-1.25, 0.75, 0.0],
        [1.25, -0.75, 0.0],
        [1.25, 0.75, 0.0],
    ])
    positive = _terrain(_one_cell_dem(), stations)
    negative_dem = _one_cell_dem(elevation_m=10.0, reference_m=20.0)
    negative = _terrain(negative_dem, stations)
    expected_positive = np.array([_direct_volume_gravity(tuple(s)) for s in stations])
    expected_negative = np.array([
        _direct_volume_gravity(tuple(s), density=-_DENSITY) for s in stations
    ])

    np.testing.assert_allclose(positive, expected_positive, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(negative, expected_negative, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(positive[:4], positive[0], rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(positive[4:], positive[4], rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(negative, -positive, rtol=2e-10, atol=2e-12)


def test_terrain_prism_skips_masked_dem_cells_without_changing_the_active_prism():
    stations = np.array([[0.0, 0.0, 30.0], [0.0, 0.0, 0.0]])
    full = _one_cell_dem()
    masked_elevations = np.array(full.elevation_m, copy=True)
    masked_elevations[1, 1] = np.nan
    masked = TerrainDEM(full.x_m, full.y_m, masked_elevations, _CRS, _DATUM)

    full_result = _terrain(full, stations)
    masked_result = terrain_correction(
        masked,
        stations[:, 0],
        stations[:, 1],
        stations[:, 2],
        station_coordinate_reference=_CRS,
        reference_elevation_m=10.0,
        density_kg_m3=_DENSITY,
        elevation_datum=_DATUM,
    )

    np.testing.assert_allclose(masked_result.gravity_down_mgal, full_result, rtol=0.0, atol=0.0)
    assert masked_result.parameters["active_prism_count"] == 1
    assert masked_result.parameters["dem_cell_count"] == 3

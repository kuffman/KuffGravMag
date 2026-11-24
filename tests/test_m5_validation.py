"""M5 scientific validation regressions for independent references and studies."""

import math

import numpy as np
import pytest

from engine.forward import Body2D, PotentialField2D
from validation.analytical import (
    centered_rectangle_infinite_gravity_gz_mgal,
    centered_rectangle_infinite_magnetic_components_nT,
    dipole_magnetic_field_nT,
    point_mass_gravity_gz_mgal,
    polygon_infinite_strike_gravity_reference,
    polygon_infinite_strike_magnetic_reference_nT,
    profile_magnetization_a_m,
    rectangular_prism_gravity_gz_mgal,
    rectangular_prism_volume_integral,
)
from validation.harmonica_reference import (
    harmonica_rectangular_prism_gravity_gz_mgal,
    harmonica_rectangular_prism_magnetic_components_nT,
)
from validation.inversion_studies import (
    density_depth_tradeoff,
    geometry_identifiability_study,
    run_noise_recovery_ensemble,
)


def _rectangular_body(
    *,
    x_bounds=(-100.0, 100.0),
    y_bounds=(-300.0, 500.0),
    z_bounds=(200.0, 400.0),
    density=500.0,
    susceptibility=0.0,
    remanence=0.0,
    remanent_inc=0.0,
    remanent_dec=0.0,
    strike_mode="finite",
):
    x1, x2 = x_bounds
    z1, z2 = z_bounds
    return Body2D(
        "M5 reference prism",
        [x1, x2, x2, x1, x1],
        [z1, z1, z2, z2, z1],
        density=density,
        k_succ=susceptibility,
        J_r=remanence,
        I_r=remanent_inc,
        D_r=remanent_dec,
        y_min=y_bounds[0],
        y_max=y_bounds[1],
        strike_mode=strike_mode,
    )


def _harmonica_or_skip():
    try:
        import harmonica
    except ModuleNotFoundError as exc:
        if exc.name != "harmonica":
            raise
        pytest.skip("install the pinned optional validation group to run Harmonica comparisons")
    return harmonica


def test_reusable_infinite_strike_rectangle_references_match_the_engine():
    body = _rectangular_body(strike_mode="infinite", density=500.0)
    gravity = PotentialField2D([body]).calculate_gravity(np.array([0.0]), np.array([0.0]))[0]
    expected_gravity = centered_rectangle_infinite_gravity_gz_mgal(
        half_width=100.0, top=200.0, bottom=400.0, density=500.0
    )
    assert gravity == pytest.approx(expected_gravity, rel=1e-12, abs=1e-12)

    magnetic_body = _rectangular_body(
        strike_mode="infinite", density=0.0, remanence=1.0, remanent_inc=0.0
    )
    dX, _, dZ, _ = PotentialField2D([magnetic_body]).calculate_magnetics_components(
        np.array([0.0]), np.array([0.0]), 0.0, 0.0, 0.0, 0.0
    )
    expected_dX, expected_dZ = centered_rectangle_infinite_magnetic_components_nT(
        half_width=100.0,
        top=200.0,
        bottom=400.0,
        magnetization_x=1.0,
        magnetization_z_down=0.0,
    )
    assert dX[0] == pytest.approx(expected_dX, rel=1e-12, abs=1e-12)
    assert dZ[0] == pytest.approx(expected_dZ, rel=1e-12, abs=1e-12)


def test_nagy_prism_reference_agrees_with_volume_quadrature_and_engine():
    x_bounds = (-100.0, 100.0)
    y_bounds = (-300.0, 500.0)
    z_bounds = (200.0, 400.0)
    observation = (25.0, 0.0, 0.0)
    density = 500.0
    expected = rectangular_prism_gravity_gz_mgal(
        x_bounds, y_bounds, z_bounds, density, observation
    )
    numeric_gravity, _ = rectangular_prism_volume_integral(
        x_bounds,
        y_bounds,
        z_bounds,
        density,
        (0.8, -0.35, 0.42),
        observation,
        order=48,
    )
    body = _rectangular_body(x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds)
    actual = PotentialField2D([body]).calculate_gravity(
        np.array([observation[0]]), np.array([observation[2]])
    )[0]
    assert expected == pytest.approx(numeric_gravity, rel=2e-11, abs=2e-11)
    assert actual == pytest.approx(expected, rel=2e-11, abs=2e-11)


def test_independent_polygon_area_references_cover_concave_infinite_strike_body():
    vertices = np.array(
        [(-140.0, 160.0), (50.0, 160.0), (50.0, 280.0), (120.0, 280.0),
         (120.0, 430.0), (-140.0, 430.0)]
    )
    observation = (260.0, 12.0)
    density = 520.0
    magnetization_xz = (math.cos(math.radians(30.0)), math.sin(math.radians(30.0)))
    body = Body2D(
        "concave reference polygon",
        vertices[:, 0],
        vertices[:, 1],
        density=density,
        J_r=1.0,
        I_r=30.0,
        D_r=0.0,
        y_min=0.0,
        y_max=0.0,
        strike_mode="infinite",
    )
    engine = PotentialField2D([body])
    actual_gravity = engine.calculate_gravity(
        np.array([observation[0]]), np.array([observation[1]])
    )[0]
    actual_dX, _, actual_dZ, _ = engine.calculate_magnetics_components(
        np.array([observation[0]]), np.array([observation[1]]), 0.0, 0.0, 0.0, 0.0
    )
    expected_gravity = polygon_infinite_strike_gravity_reference(
        vertices, density, observation, order=64
    )
    expected_dX, expected_dZ = polygon_infinite_strike_magnetic_reference_nT(
        vertices, magnetization_xz, observation, order=64
    )
    np.testing.assert_allclose(actual_gravity, expected_gravity, rtol=2e-10, atol=2e-11)
    np.testing.assert_allclose(actual_dX[0], expected_dX, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(actual_dZ[0], expected_dZ, rtol=2e-10, atol=2e-10)

    reversed_body = Body2D(
        "reversed concave reference polygon",
        vertices[::-1, 0],
        vertices[::-1, 1],
        density=density,
        J_r=1.0,
        I_r=30.0,
        D_r=0.0,
        y_min=0.0,
        y_max=0.0,
        strike_mode="infinite",
    )
    reversed_engine = PotentialField2D([reversed_body])
    reversed_gravity = reversed_engine.calculate_gravity(
        np.array([observation[0]]), np.array([observation[1]])
    )[0]
    reversed_magnetic = reversed_engine.calculate_magnetics_components(
        np.array([observation[0]]), np.array([observation[1]]), 0.0, 0.0, 0.0, 0.0
    )
    np.testing.assert_allclose(reversed_gravity, actual_gravity, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(reversed_magnetic[0], actual_dX, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(reversed_magnetic[2], actual_dZ, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3], ids=["millimetre-scale", "metre-scale", "kilometre-scale"])
def test_prism_gravity_is_stable_across_geometry_scales(scale):
    x_bounds = (-100.0 * scale, 100.0 * scale)
    y_bounds = (-300.0 * scale, 500.0 * scale)
    z_bounds = (200.0 * scale, 400.0 * scale)
    x_observation = 25.0 * scale
    body = _rectangular_body(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        strike_mode="finite",
    )
    actual = PotentialField2D([body]).calculate_gravity(
        np.array([x_observation]), np.array([0.0])
    )[0]
    expected = rectangular_prism_gravity_gz_mgal(
        x_bounds,
        y_bounds,
        z_bounds,
        500.0,
        (x_observation, 0.0, 0.0),
    )
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, rel=5e-10, abs=1e-13 * scale)


def test_near_surface_prism_fields_remain_finite_and_match_the_analytic_gravity():
    body = _rectangular_body(
        density=500.0,
        susceptibility=0.0,
        remanence=1.0,
        remanent_inc=15.0,
        remanent_dec=25.0,
    )
    engine = PotentialField2D([body])
    offset = 1e-6
    x_observation = np.array([0.0, 99.999999])
    z_observation = np.array([200.0 - offset, 200.0 - offset])
    gravity = engine.calculate_gravity(x_observation, z_observation)
    for index, x_value in enumerate(x_observation):
        expected = rectangular_prism_gravity_gz_mgal(
            (-100.0, 100.0),
            (-300.0, 500.0),
            (200.0, 400.0),
            500.0,
            (x_value, 0.0, z_observation[index]),
        )
        assert gravity[index] == pytest.approx(expected, rel=2e-9, abs=2e-10)
    magnetic = engine.calculate_magnetics_components(
        x_observation, z_observation, 0.0, 0.0, 0.0, 0.0
    )
    assert all(np.all(np.isfinite(component)) for component in magnetic)


def test_far_field_matches_point_mass_and_dipole_asymptotes():
    x_bounds = (-100.0, 100.0)
    y_bounds = (-300.0, 500.0)
    z_bounds = (200.0, 400.0)
    observation = (20_000.0, 0.0, 0.0)
    center = (0.0, 100.0, 300.0)
    volume = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0]) * (z_bounds[1] - z_bounds[0])
    density = 500.0
    body = _rectangular_body(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        density=density,
        remanence=1.0,
        remanent_inc=0.0,
        remanent_dec=0.0,
    )
    engine = PotentialField2D([body])
    gravity = engine.calculate_gravity(np.array([observation[0]]), np.array([observation[2]]))[0]
    analytic = rectangular_prism_gravity_gz_mgal(
        x_bounds, y_bounds, z_bounds, density, observation
    )
    numeric_gravity, numeric_magnetic = rectangular_prism_volume_integral(
        x_bounds,
        y_bounds,
        z_bounds,
        density,
        (1.0, 0.0, 0.0),
        observation,
        order=16,
    )
    assert gravity == pytest.approx(analytic, rel=2e-5, abs=1e-18)
    assert numeric_gravity == pytest.approx(analytic, rel=2e-4, abs=1e-18)
    assert np.all(np.isfinite(numeric_magnetic))
    magnetic = np.array(
        [
            component[0]
            for component in engine.calculate_magnetics_components(
                np.array([observation[0]]), np.array([observation[2]]), 0.0, 0.0, 0.0, 0.0
            )[:3]
        ]
    )
    np.testing.assert_allclose(magnetic, numeric_magnetic, rtol=3e-5, atol=1e-12)
    dipole = dipole_magnetic_field_nT(
        volume=volume,
        magnetization=(1.0, 0.0, 0.0),
        center=center,
        observation=observation,
    )
    assert np.linalg.norm(magnetic - dipole) / np.linalg.norm(dipole) < 0.005
    point_mass = point_mass_gravity_gz_mgal(
        volume=volume, density=density, center=center, observation=observation
    )
    assert abs(gravity - point_mass) / abs(point_mass) < 0.005


def test_deterministic_noise_ensemble_reports_tradeoffs_and_uncertainty():
    first = run_noise_recovery_ensemble(seed=20261124, trials=16)
    second = run_noise_recovery_ensemble(seed=20261124, trials=16)
    assert first == second
    levels = first["noise_levels_mgal"]
    assert levels == [0.01, 0.03, 0.1]
    records = first["records"]
    assert len(records) == len(levels)
    assert all(record["identifiable_fraction"] == 1.0 for record in records)
    assert records[0]["rmse_density_kg_m3"] < records[1]["rmse_density_kg_m3"] < records[2]["rmse_density_kg_m3"]
    assert records[0]["mean_standard_error_kg_m3"] < records[1]["mean_standard_error_kg_m3"] < records[2]["mean_standard_error_kg_m3"]

    tradeoff = density_depth_tradeoff()
    assert abs(tradeoff["single_station_relative_residual"]) < 1e-12
    assert abs(tradeoff["deep_density_kg_m3"] - tradeoff["shallow_density_kg_m3"]) > 100.0
    assert tradeoff["full_profile_relative_rms_difference"] > 0.05

    identifiability = geometry_identifiability_study()
    assert identifiability["success"]
    assert identifiability["numerical_rank"] < identifiability["parameter_count"]
    assert not identifiability["identifiable"]
    assert not identifiability["uncertainty_available"]


def test_harmonica_rectangular_prism_gravity_comparison_maps_vertical_signs():
    harmonica = _harmonica_or_skip()
    assert str(harmonica.__version__).lstrip("v") == "0.7.0"
    x_bounds = (-100.0, 100.0)
    y_bounds = (-300.0, 500.0)
    z_bounds = (200.0, 400.0)
    observations = [(25.0, 0.0, 0.0), (-60.0, 0.0, 15.0)]
    body = _rectangular_body(x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds)
    actual = PotentialField2D([body]).calculate_gravity(
        np.array([point[0] for point in observations]),
        np.array([point[2] for point in observations]),
    )
    expected = np.array(
        [
            harmonica_rectangular_prism_gravity_gz_mgal(
                x_bounds, y_bounds, z_bounds, 500.0, point, profile_azimuth=90.0
            )
            for point in observations
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)


def test_harmonica_magnetic_comparison_maps_local_axes_and_field_projection():
    harmonica = _harmonica_or_skip()
    assert str(harmonica.__version__).lstrip("v") == "0.7.0"
    x_bounds = (-100.0, 100.0)
    y_bounds = (-300.0, 500.0)
    z_bounds = (200.0, 400.0)
    field = (48_000.0, 35.0, 18.0, 90.0)
    body = _rectangular_body(
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        z_bounds=z_bounds,
        density=0.0,
        susceptibility=0.04,
        remanence=1.7,
        remanent_inc=30.0,
        remanent_dec=65.0,
    )
    x_observations = np.array([-40.0, 0.0, 80.0])
    z_observations = np.array([0.0, 5.0, -20.0])
    actual = PotentialField2D([body]).calculate_magnetics_components(
        x_observations, z_observations, *field
    )
    magnetization = profile_magnetization_a_m(
        susceptibility=body.k_succ,
        remanent_intensity=body.J_r,
        remanent_inclination=body.I_r,
        remanent_declination=body.D_r,
        field_intensity_nT=field[0],
        field_inclination=field[1],
        field_declination=field[2],
        profile_azimuth=field[3],
    )
    expected = np.array(
        [
            harmonica_rectangular_prism_magnetic_components_nT(
                x_bounds,
                y_bounds,
                z_bounds,
                magnetization,
                x_obs,
                z_obs,
                *field,
            )
            for x_obs, z_obs in zip(x_observations, z_observations)
        ]
    )
    for index, component in enumerate(actual):
        np.testing.assert_allclose(component, expected[:, index], rtol=3e-8, atol=3e-8)

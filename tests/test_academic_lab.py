"""Independent scientific regressions for synthetic academic lab models."""

import csv
import io
import json
import math

import numpy as np
import pytest

from engine.academic import (
    GRAVITATIONAL_CONSTANT,
    SI_TO_MGAL,
    VACUUM_PERMEABILITY,
    cylinder_gravity_mgal,
    cylinder_magnetic_components_nT,
    export_teaching_experiment_csv,
    forward_teaching_model,
    invert_sphere_gravity_density_depth,
    simulate_teaching_experiment,
    sphere_density_depth_tradeoff,
    sphere_gravity_mgal,
    sphere_magnetic_components_nT,
)


def _cylinder_gravity_area_quadrature(x_obs, *, depth, radius, density):
    radial_nodes, radial_weights = np.polynomial.legendre.leggauss(72)
    radial = 0.5 * radius * (radial_nodes + 1.0)
    weights_r = 0.5 * radius * radial_weights
    theta = 2.0 * np.pi * (np.arange(512) + 0.5) / 512
    source_x = radial[:, None] * np.cos(theta)[None, :]
    source_z = depth + radial[:, None] * np.sin(theta)[None, :]
    delta_z = source_z
    values = []
    for x in x_obs:
        distance_squared = (x - source_x) ** 2 + delta_z**2
        integrand = 2.0 * GRAVITATIONAL_CONSTANT * density * delta_z / distance_squared
        values.append(np.sum(integrand * (weights_r * radial)[:, None]) * (2.0 * np.pi / 512))
    return np.asarray(values) * SI_TO_MGAL


def _dyke_gravity_area_quadrature(x_obs, *, width, top, bottom, density):
    x_nodes, x_weights = np.polynomial.legendre.leggauss(100)
    z_nodes, z_weights = np.polynomial.legendre.leggauss(100)
    source_x = 0.5 * width * x_nodes
    source_z = top + 0.5 * (bottom - top) * (z_nodes + 1.0)
    weights = np.outer(0.5 * width * x_weights, 0.5 * (bottom - top) * z_weights)
    values = []
    for x in x_obs:
        xx, zz = np.meshgrid(source_x, source_z, indexing="ij")
        values.append(
            np.sum(
                weights
                * 2.0
                * GRAVITATIONAL_CONSTANT
                * density
                * zz
                / ((x - xx) ** 2 + zz**2)
            )
        )
    return np.asarray(values) * SI_TO_MGAL


def test_sphere_gravity_matches_independent_external_point_mass_reference():
    x = np.array([-250.0, 0.0, 175.0])
    z = np.array([0.0, 0.0, 30.0])
    radius, density, depth = 120.0, 640.0, 420.0
    actual = sphere_gravity_mgal(
        x, z, radius_m=radius, density_contrast_kg_m3=density, center_depth_m=depth
    )
    mass = density * (4.0 / 3.0) * math.pi * radius**3
    dz = depth - z
    expected = (
        GRAVITATIONAL_CONSTANT
        * mass
        * dz
        / ((x**2 + dz**2) ** 1.5)
        * SI_TO_MGAL
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.0)


def test_infinite_cylinder_gravity_matches_independent_cross_section_quadrature():
    x = np.array([-180.0, 0.0, 240.0])
    actual = cylinder_gravity_mgal(
        x, center_depth_m=430.0, radius_m=140.0, density_contrast_kg_m3=580.0
    )
    expected = _cylinder_gravity_area_quadrature(
        x, depth=430.0, radius=140.0, density=580.0
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)


def test_infinite_dyke_gravity_matches_independent_cross_section_quadrature():
    x = np.array([-180.0, 0.0, 240.0])
    actual = forward_teaching_model(
        "dyke",
        "gravity",
        x,
        dyke_width_m=120.0,
        dyke_top_m=150.0,
        dyke_bottom_m=500.0,
        density_contrast_kg_m3=620.0,
    )
    expected = _dyke_gravity_area_quadrature(
        x, width=120.0, top=150.0, bottom=500.0, density=620.0
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-12)


def test_infinite_dyke_magnetic_teaching_model_returns_finite_total_field_response():
    response = forward_teaching_model(
        "dyke",
        "magnetics",
        np.linspace(-700.0, 700.0, 81),
        dyke_width_m=130.0,
        dyke_top_m=160.0,
        dyke_bottom_m=650.0,
        field_strength_nT=49_000.0,
        field_inclination_deg=9.0,
        field_declination_deg=-3.0,
        profile_azimuth_deg=90.0,
        susceptibility_si=0.04,
        remanence_a_m=0.6,
        remanent_inclination_deg=42.0,
        remanent_declination_deg=38.0,
    )
    assert response.shape == (81,)
    assert np.all(np.isfinite(response))
    assert np.max(np.abs(response)) > 0.0


def test_sphere_magnetic_components_match_independent_dipole_equation():
    x = np.array([100.0, -250.0])
    z = np.array([0.0, 40.0])
    radius = 120.0
    field = 50_000.0
    inclination, declination, azimuth = 18.0, -4.0, 90.0
    susceptibility, remanence = 0.025, 0.7
    actual = sphere_magnetic_components_nT(
        x,
        z,
        radius_m=radius,
        center_depth_m=430.0,
        field_strength_nT=field,
        field_inclination_deg=inclination,
        field_declination_deg=declination,
        profile_azimuth_deg=azimuth,
        susceptibility_si=susceptibility,
        remanence_a_m=remanence,
        remanent_inclination_deg=33.0,
        remanent_declination_deg=72.0,
    )
    to_radians = np.pi / 180.0
    field_direction = np.array([
        np.cos(inclination * to_radians) * np.cos((declination - azimuth) * to_radians),
        np.cos(inclination * to_radians) * np.sin((declination - azimuth) * to_radians),
        np.sin(inclination * to_radians),
    ])
    rem_direction = np.array([
        np.cos(33.0 * to_radians) * np.cos((72.0 - azimuth) * to_radians),
        np.cos(33.0 * to_radians) * np.sin((72.0 - azimuth) * to_radians),
        np.sin(33.0 * to_radians),
    ])
    magnetization = susceptibility * field * 1e-9 / VACUUM_PERMEABILITY * field_direction + remanence * rem_direction
    moment = (4.0 / 3.0) * np.pi * radius**3 * magnetization
    expected_components = []
    for px, pz in zip(x, z):
        r = np.array([px, 0.0, pz - 430.0])
        expected_components.append(
            VACUUM_PERMEABILITY
            / (4.0 * np.pi)
            * (3.0 * r * np.dot(moment, r) / np.linalg.norm(r) ** 5 - moment / np.linalg.norm(r) ** 3)
            * 1e9
        )
    expected_components = np.asarray(expected_components)
    expected_total = expected_components @ field_direction
    for component, expected in zip(actual[:3], expected_components.T):
        np.testing.assert_allclose(component, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual[3], expected_total, rtol=1e-13, atol=1e-13)


def test_cylinder_magnetics_obey_profile_mirror_symmetry_and_inside_limit():
    x = np.array([-100.0, 0.0, 100.0])
    components = cylinder_magnetic_components_nT(
        x,
        center_depth_m=400.0,
        radius_m=120.0,
        field_strength_nT=50_000.0,
        field_inclination_deg=0.0,
        field_declination_deg=90.0,
        profile_azimuth_deg=90.0,
        susceptibility_si=0.05,
    )
    np.testing.assert_allclose(components[0][0], components[0][2], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(components[2][0], -components[2][2], rtol=0.0, atol=1e-12)
    inside = cylinder_magnetic_components_nT(
        [0.0, 40.0],
        [400.0, 390.0],
        center_depth_m=400.0,
        radius_m=120.0,
        susceptibility_si=0.05,
    )
    np.testing.assert_allclose(inside[0], inside[0][0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(inside[2], inside[2][0], rtol=0.0, atol=0.0)
    assert np.all(inside[1] == 0.0)


def test_cylinder_magnetic_field_matches_independent_boundary_charge_quadrature():
    x = np.array([-220.0, 430.0])
    z = np.array([0.0, 25.0])
    radius = 125.0
    depth = 410.0
    field = 49_500.0
    field_inc, field_dec, alpha = 24.0, 11.0, 75.0
    susceptibility, remanence = 0.035, 0.9
    rem_inc, rem_dec = 41.0, -22.0
    actual = cylinder_magnetic_components_nT(
        x,
        z,
        center_depth_m=depth,
        radius_m=radius,
        field_strength_nT=field,
        field_inclination_deg=field_inc,
        field_declination_deg=field_dec,
        profile_azimuth_deg=alpha,
        susceptibility_si=susceptibility,
        remanence_a_m=remanence,
        remanent_inclination_deg=rem_inc,
        remanent_declination_deg=rem_dec,
    )
    radians = np.pi / 180.0
    field_direction = np.array([
        np.cos(field_inc * radians) * np.cos((field_dec - alpha) * radians),
        np.cos(field_inc * radians) * np.sin((field_dec - alpha) * radians),
        np.sin(field_inc * radians),
    ])
    rem_direction = np.array([
        np.cos(rem_inc * radians) * np.cos((rem_dec - alpha) * radians),
        np.cos(rem_inc * radians) * np.sin((rem_dec - alpha) * radians),
        np.sin(rem_inc * radians),
    ])
    magnetization = (
        susceptibility * field * 1e-9 / VACUUM_PERMEABILITY * field_direction
        + remanence * rem_direction
    )
    theta = 2.0 * np.pi * np.arange(8192) / 8192
    normals = np.column_stack((np.cos(theta), np.sin(theta)))
    surface_charge = magnetization[0] * normals[:, 0] + magnetization[2] * normals[:, 1]
    expected = []
    for x_station, z_station in zip(x, z):
        observation = np.column_stack((
            np.full(theta.size, x_station),
            np.full(theta.size, z_station - depth),
        ))
        displacement = observation - radius * normals
        radius_squared = np.einsum("ij,ij->i", displacement, displacement)
        # H = (1/2π) ∮ σ_m (r-r')/|r-r'|² ds, evaluated by a periodic trapezoid sum.
        h_xz = (
            radius
            * np.mean(surface_charge[:, None] * displacement / radius_squared[:, None], axis=0)
        )
        expected.append(np.array([h_xz[0], 0.0, h_xz[1]]) * VACUUM_PERMEABILITY * 1e9)
    expected = np.asarray(expected)
    for component, reference in zip(actual[:3], expected.T):
        np.testing.assert_allclose(component, reference, rtol=2e-12, atol=2e-11)
    np.testing.assert_allclose(actual[3], expected @ field_direction, rtol=2e-12, atol=2e-11)


def test_seeded_experiment_and_export_are_byte_reproducible_and_mark_synthetic():
    x = np.linspace(-800.0, 800.0, 41)
    kwargs = dict(
        x_m=x,
        noise_sigma=0.04,
        seed=73,
        center_depth_m=360.0,
        radius_m=120.0,
        density_contrast_kg_m3=450.0,
    )
    first = simulate_teaching_experiment("sphere", "gravity", **kwargs)
    repeated = simulate_teaching_experiment("sphere", "gravity", **kwargs)
    other_seed = simulate_teaching_experiment("sphere", "gravity", **(kwargs | {"seed": 74}))
    assert kwargs["x_m"].flags.writeable
    np.testing.assert_array_equal(first.observed_response, repeated.observed_response)
    assert export_teaching_experiment_csv(first) == export_teaching_experiment_csv(repeated)
    assert not np.array_equal(first.observed_response, other_seed.observed_response)
    payload = export_teaching_experiment_csv(first)
    metadata_line = payload.splitlines()[0].decode("utf-8")[2:]
    metadata = json.loads(metadata_line)
    assert metadata["synthetic"] is True
    assert metadata["seed"] == 73
    assert metadata["coordinate_context"]["crs"] == "EPSG:32630"
    assert metadata["coordinate_context"]["measured_data"] is False
    assert "point-mass" in metadata["equation"]
    rows = list(csv.reader(io.StringIO(payload.decode("utf-8").split("\n", 1)[1])))
    assert rows[0][-1] == "observed_mGal"
    assert len(rows) == x.size + 1


def test_sphere_depth_density_tradeoff_matches_one_station_but_not_full_profile():
    x = np.linspace(-1_000.0, 1_000.0, 101)
    comparison = sphere_density_depth_tradeoff(
        x, radius_m=100.0, shallow_density_kg_m3=500.0,
        shallow_depth_m=350.0, deep_depth_m=550.0,
    )
    shallow = comparison["shallow_response_mgal"]
    deep = comparison["deep_response_mgal"]
    assert abs(comparison["single_station_difference_mgal"]) < 1e-14
    assert comparison["deep_density_kg_m3"] == pytest.approx(500.0 * (550.0 / 350.0) ** 2)
    assert comparison["profile_relative_rms_difference"] > 0.1
    assert not np.allclose(shallow, deep)
    with pytest.raises(ValueError, match="must exceed"):
        sphere_density_depth_tradeoff(
            x,
            radius_m=100.0,
            shallow_density_kg_m3=500.0,
            shallow_depth_m=350.0,
            deep_depth_m=300.0,
        )


def test_sphere_density_depth_fit_recovers_seeded_noisy_model_and_uncertainty():
    x = np.linspace(-1_000.0, 1_000.0, 101)
    experiment = simulate_teaching_experiment(
        "sphere",
        "gravity",
        x_m=x,
        noise_sigma=0.02,
        seed=20261125,
        center_depth_m=430.0,
        radius_m=120.0,
        density_contrast_kg_m3=650.0,
    )
    result = invert_sphere_gravity_density_depth(
        x,
        experiment.observed_response,
        radius_m=120.0,
        observation_sigma_mgal=0.02,
        initial_density_kg_m3=420.0,
        initial_depth_m=540.0,
    )
    assert result.success
    assert result.identifiable
    assert result.numerical_rank == 2
    assert result.density_kg_m3 == pytest.approx(650.0, abs=20.0)
    assert result.center_depth_m == pytest.approx(430.0, abs=20.0)
    assert result.standard_errors is not None
    assert set(result.standard_errors) == {"density_kg_m3", "center_depth_m"}
    higher_sigma = invert_sphere_gravity_density_depth(
        x,
        experiment.observed_response,
        radius_m=120.0,
        observation_sigma_mgal=0.1,
        initial_density_kg_m3=420.0,
        initial_depth_m=540.0,
    )
    assert higher_sigma.standard_errors is not None
    assert higher_sigma.standard_errors["density_kg_m3"] > result.standard_errors["density_kg_m3"]
    assert higher_sigma.standard_errors["center_depth_m"] > result.standard_errors["center_depth_m"]


def test_single_station_sphere_fit_reports_nonuniqueness_without_uncertainty():
    value = sphere_gravity_mgal(
        [0.0], center_depth_m=450.0, radius_m=120.0, density_contrast_kg_m3=500.0
    )
    result = invert_sphere_gravity_density_depth(
        [0.0],
        value,
        radius_m=120.0,
        observation_sigma_mgal=0.05,
        initial_density_kg_m3=350.0,
        initial_depth_m=520.0,
    )
    assert result.success
    assert result.numerical_rank == 1
    assert not result.identifiable
    assert result.standard_errors is None


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda: sphere_gravity_mgal([0.0], center_depth_m=100.0, radius_m=100.0), "below the surface"),
        (lambda: cylinder_gravity_mgal([0.0], radius_m=0.0), "positive radius"),
        (lambda: forward_teaching_model("sphere", "gravity", [0.0], radius_m=-1.0), "positive radius"),
        (lambda: forward_teaching_model("unknown", "gravity", [0.0]), "model_kind"),
        (lambda: forward_teaching_model("sphere", "unknown", [0.0]), "field_type"),
    ],
)
def test_teaching_models_reject_unsupported_or_unphysical_inputs(call, message):
    with pytest.raises(ValueError, match=message):
        call()

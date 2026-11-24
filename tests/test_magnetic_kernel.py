"""Magnetic regression tests with independent 2D references."""

import math

import numpy as np
import pytest

from engine.forward import Body2D, PotentialField2D


MU0 = 4.0 * math.pi * 1e-7


def _body(
    points,
    mx=0.0,
    mz=0.0,
    susceptibility=0.0,
    strike=(0.0, 0.0),
    remanent_declination=0.0,
):
    points = np.asarray(points, dtype=np.float64)
    remanence = math.hypot(mx, mz)
    inclination = math.degrees(math.atan2(mz, mx)) if remanence else 0.0
    return Body2D(
        "test",
        np.append(points[:, 0], points[0, 0]),
        np.append(points[:, 1], points[0, 1]),
        k_succ=susceptibility,
        J_r=remanence,
        I_r=inclination,
        D_r=remanent_declination,
        y_min=strike[0],
        y_max=strike[1],
    )


def _centered_rectangle(a=100.0, z1=200.0, z2=400.0):
    return np.array([[-a, z1], [a, z1], [a, z2], [-a, z2]], dtype=np.float64)


def _rectangle_coefficient_nT(a, z1, z2):
    """Coefficient mapping unit A/m to nT at the centered surface point."""
    return MU0 / math.pi * (math.atan(a / z1) - math.atan(a / z2)) * 1e9


def _direct_infinite_strike_area_field(points, observation, magnetization, order=72):
    """Independent area integral of the 3D dipole field, integrated over strike."""
    points = np.asarray(points, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    magnetization = np.asarray(magnetization, dtype=np.float64)
    gauss, weights = np.polynomial.legendre.leggauss(order)
    unit = 0.5 * (gauss + 1.0)
    weights = 0.5 * weights
    total = np.zeros(2, dtype=np.float64)

    # Fan triangulation is exact for the test's convex triangles and rectangles.
    for i in range(1, len(points) - 1):
        p0, p1, p2 = points[0], points[i], points[i + 1]
        edge_a, edge_b = p1 - p0, p2 - p0
        jacobian = abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
        for iu, u in enumerate(unit):
            for iv, v in enumerate(unit):
                source = p0 + u * (p1 - p0) + (1.0 - u) * v * (p2 - p0)
                rx, rz = observation - source
                r2 = rx * rx + rz * rz
                dot = magnetization[0] * rx + magnetization[1] * rz
                kernel = np.array(
                    [2.0 * dot * rx / r2**2 - magnetization[0] / r2,
                     2.0 * dot * rz / r2**2 - magnetization[1] / r2]
                )
                total += (
                    weights[iu]
                    * weights[iv]
                    * jacobian
                    * (1.0 - u)
                    * MU0
                    / (2.0 * math.pi)
                    * kernel
                )
    return total * 1e9


def _calculate(body, x_obs, z_obs, *, F_nT=0.0, I_f=0.0, D_f=0.0, alpha=0.0):
    return PotentialField2D([body]).calculate_magnetics(
        np.asarray(x_obs, dtype=np.float64),
        np.asarray(z_obs, dtype=np.float64),
        F_nT,
        I_f,
        D_f,
        alpha,
    )


def test_horizontal_rectangle_matches_analytical_reference_and_winding_is_independent():
    points = _centered_rectangle()
    coefficient = _rectangle_coefficient_nT(100.0, 200.0, 400.0)
    expected = np.array([-coefficient, 0.0])

    forward = _calculate(_body(points, mx=1.0), [0.0], [0.0])
    reverse = _calculate(_body(points[::-1], mx=1.0), [0.0], [0.0])

    np.testing.assert_allclose(forward[:2], expected[:, None], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(reverse[:2], expected[:, None], rtol=1e-10, atol=1e-10)


def test_vertical_rectangle_matches_analytical_reference():
    points = _centered_rectangle()
    coefficient = _rectangle_coefficient_nT(100.0, 200.0, 400.0)
    d_x, d_z, _ = _calculate(_body(points, mz=1.0), [0.0], [0.0])
    np.testing.assert_allclose(d_x, [0.0], atol=1e-10)
    np.testing.assert_allclose(d_z, [coefficient], rtol=1e-10, atol=1e-10)


def test_remanent_declination_is_rotated_into_profile_coordinates():
    points = _centered_rectangle()
    coefficient = _rectangle_coefficient_nT(100.0, 200.0, 400.0)
    body = _body(points, mx=1.0, remanent_declination=75.0)
    d_x, d_z, _ = _calculate(body, [0.0], [0.0], alpha=20.0)
    expected_mx = math.cos(math.radians(75.0 - 20.0))
    np.testing.assert_allclose(d_x, [-coefficient * expected_mx], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(d_z, [0.0], atol=1e-10)


def test_triangle_matches_independent_area_dipole_quadrature():
    points = np.array([[-70.0, 170.0], [80.0, 220.0], [15.0, 360.0]])
    magnetization = np.array([0.7, -0.4])
    expected = _direct_infinite_strike_area_field(
        points, [20.0, 0.0], magnetization, order=72
    )
    d_x, d_z, _ = _calculate(
        _body(points, mx=magnetization[0], mz=magnetization[1]), [20.0], [0.0]
    )
    np.testing.assert_allclose([d_x[0], d_z[0]], expected, rtol=2e-7, atol=2e-7)


@pytest.mark.parametrize(
    "inclination,declination,profile_azimuth",
    [
        (0.0, 0.0, 0.0),
        (0.0, 75.0, 20.0),
        (5.0, 45.0, 20.0),
        (35.0, 110.0, 25.0),
        (89.9, 45.0, -15.0),
    ],
)
def test_induced_components_and_total_field_projection_follow_declared_angles(
    inclination, declination, profile_azimuth
):
    points = _centered_rectangle()
    susceptibility = 0.04
    field_nT = 50_000.0
    horizontal_angle = math.radians(declination - profile_azimuth)
    inc = math.radians(inclination)
    field_strength_a_per_m = field_nT * 1e-9 / MU0
    mx = susceptibility * field_strength_a_per_m * math.cos(inc) * math.cos(horizontal_angle)
    mz = susceptibility * field_strength_a_per_m * math.sin(inc)
    coefficient = _rectangle_coefficient_nT(100.0, 200.0, 400.0)
    expected_x, expected_z = -coefficient * mx, coefficient * mz
    expected_total = (
        expected_x * math.cos(inc) * math.cos(horizontal_angle)
        + expected_z * math.sin(inc)
    )

    d_x, d_z, d_t = _calculate(
        _body(points, susceptibility=susceptibility),
        [0.0],
        [0.0],
        F_nT=field_nT,
        I_f=inclination,
        D_f=declination,
        alpha=profile_azimuth,
    )
    np.testing.assert_allclose(d_x, [expected_x], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(d_z, [expected_z], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(d_t, [expected_total], rtol=1e-10, atol=1e-10)


def test_oblique_total_field_anomaly_is_dot_product_of_background_unit_field_and_components():
    body = _body(_centered_rectangle(), mx=0.8, mz=-0.35)
    inclination, declination, profile_azimuth = 60.0, 60.0, 0.0
    d_x, d_z, d_t = _calculate(
        body,
        [15.0],
        [0.0],
        I_f=inclination,
        D_f=declination,
        alpha=profile_azimuth,
    )
    inc = math.radians(inclination)
    horizontal_angle = math.radians(declination - profile_azimuth)
    expected = d_x * math.cos(inc) * math.cos(horizontal_angle) + d_z * math.sin(inc)
    np.testing.assert_allclose(d_t, expected, rtol=1e-12, atol=1e-12)


def test_horizontal_symmetry_for_horizontal_magnetization():
    body = _body(_centered_rectangle(), mx=1.0)
    d_x, d_z, _ = _calculate(body, [-75.0, 75.0], [0.0, 0.0])
    np.testing.assert_allclose(d_x[0], d_x[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d_z[0], -d_z[1], rtol=1e-12, atol=1e-12)


def test_horizontal_symmetry_for_vertical_magnetization():
    body = _body(_centered_rectangle(), mz=1.0)
    d_x, d_z, _ = _calculate(body, [-75.0, 75.0], [0.0, 0.0])
    np.testing.assert_allclose(d_x[0], -d_x[1], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(d_z[0], d_z[1], rtol=1e-12, atol=1e-12)


def test_edge_interior_uses_finite_principal_value_and_is_winding_independent():
    points = _centered_rectangle()
    forward = _calculate(_body(points, mx=0.7, mz=-0.4), [0.0], [200.0])
    reverse = _calculate(_body(points[::-1], mx=0.7, mz=-0.4), [0.0], [200.0])
    for field in forward + reverse:
        assert np.all(np.isfinite(field))
    for first, second in zip(forward, reverse):
        np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)


def test_magnetic_fields_superpose_across_bodies():
    first = _body(_centered_rectangle(), mx=0.8, mz=-0.2)
    second = _body(_centered_rectangle(60.0, 500.0, 650.0), mx=-0.1, mz=0.6)
    observations = (np.array([-25.0, 0.0, 35.0]), np.zeros(3))
    combined = PotentialField2D([first, second]).calculate_magnetics(
        *observations, 0.0, 30.0, 40.0, 10.0
    )
    outputs = [
        PotentialField2D([body]).calculate_magnetics(*observations, 0.0, 30.0, 40.0, 10.0)
        for body in (first, second)
    ]
    for field_index in range(3):
        np.testing.assert_allclose(
            combined[field_index],
            outputs[0][field_index] + outputs[1][field_index],
            rtol=1e-12,
            atol=1e-12,
        )


def test_susceptibility_scaling_and_zero_magnetization():
    points = _centered_rectangle()
    base = _calculate(_body(points, susceptibility=0.02), [0.0], [0.0], F_nT=48_000, I_f=27, D_f=63, alpha=19)
    doubled = _calculate(_body(points, susceptibility=0.04), [0.0], [0.0], F_nT=48_000, I_f=27, D_f=63, alpha=19)
    zero = _calculate(_body(points), [0.0], [0.0], F_nT=48_000, I_f=27, D_f=63, alpha=19)
    zero_at_vertex = _calculate(_body(points), [-100.0], [200.0], F_nT=48_000, I_f=27, D_f=63, alpha=19)
    for first, twice in zip(base, doubled):
        np.testing.assert_allclose(twice, 2.0 * first, rtol=1e-12, atol=1e-12)
    for component in zero:
        np.testing.assert_array_equal(component, [0.0])
    for component in zero_at_vertex:
        np.testing.assert_array_equal(component, [0.0])


def test_invalid_magnetic_geometry_and_nonfinite_observations_are_rejected():
    crossed = np.array([[-50.0, 200.0], [50.0, 400.0], [-50.0, 400.0], [40.0, 200.0]])
    with pytest.raises(ValueError, match="Self-intersecting"):
        _calculate(_body(crossed, mx=1.0), [0.0], [0.0])
    with pytest.raises(ValueError, match="finite"):
        _calculate(_body(_centered_rectangle(), mx=1.0), [math.inf], [0.0])


def test_observation_at_polygon_vertex_reports_singularity():
    with pytest.raises(ValueError, match="singular at a polygon vertex"):
        _calculate(_body(_centered_rectangle(), mx=1.0), [-100.0], [200.0])

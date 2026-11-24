import math

import numpy as np
import pytest

from engine.forward import Body2D, PotentialField2D


G = 6.67430e-11
SI2MGAL = 1.0e5


def rectangle(xmin=-100.0, xmax=100.0, ztop=200.0, zbottom=400.0):
    return np.array([xmin, xmax, xmax, xmin, xmin]), np.array(
        [ztop, ztop, zbottom, zbottom, ztop]
    )


def analytical_rectangle_gz(a, ztop, zbottom, density, z_obs=0.0):
    """Positive-down 2D infinite-strike rectangle in mGal."""
    near = ztop - z_obs
    far = zbottom - z_obs
    if near < 0 or far < 0:
        raise ValueError("This closed form expects the observation above the body")

    def primitive(depth):
        log_term = math.log((depth * depth + a * a) / (a * a))
        if depth == 0.0:
            return a * log_term
        return 2.0 * depth * math.atan(a / depth) + a * log_term

    integral = primitive(far) - primitive(near)
    return 2.0 * G * density * SI2MGAL * integral


def triangle_area_integral_gz(vertices, density, x_obs, z_obs, order=64):
    """Independent Gaussian integration of the 2D area integral over a triangle."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    a, b, c = np.asarray(vertices, dtype=np.float64)
    ab, ac = b - a, c - a
    jacobian_scale = abs(ab[0] * ac[1] - ab[1] * ac[0])
    integral = 0.0

    for u, weight_u in zip(nodes, weights):
        for v, weight_v in zip(nodes, weights):
            point = a + u * (b - a) + (1.0 - u) * v * (c - a)
            x, z = point
            dz = z - z_obs
            radius_squared = (x - x_obs) ** 2 + dz**2
            integrand = 2.0 * G * density * dz / radius_squared
            integral += weight_u * weight_v * (1.0 - u) * integrand

    return integral * jacobian_scale * SI2MGAL


def gravity_for_polygon(x, z, density=500.0, x_obs=(0.0,), z_obs=(0.0,), y1=-math.inf, y2=math.inf):
    return PotentialField2D()._calc_single_gravity_25d(
        np.asarray(x_obs, dtype=np.float64),
        np.asarray(z_obs, dtype=np.float64),
        np.asarray(x, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
        density,
        y1,
        y2,
    )


def test_reported_rectangle_matches_independent_positive_down_solution():
    x, z = rectangle()
    # The legacy zero-strike-extent setting disables its finite-strike multiplier.
    actual = gravity_for_polygon(x, z, y1=0.0, y2=0.0)[0]
    expected = analytical_rectangle_gz(100.0, 200.0, 400.0, 500.0)

    assert expected == pytest.approx(0.887024, abs=5e-7)
    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-10)


def test_explicit_infinite_strike_matches_legacy_no_correction_sentinel():
    x, z = rectangle()
    explicit_infinite = gravity_for_polygon(x, z, y1=-math.inf, y2=math.inf)
    legacy_no_correction = gravity_for_polygon(x, z, y1=0.0, y2=0.0)

    np.testing.assert_allclose(explicit_infinite, legacy_no_correction, rtol=1e-12, atol=1e-12)


def test_nonrectangular_polygon_matches_independent_area_quadrature():
    vertices = np.array([[-80.0, 180.0], [110.0, 220.0], [0.0, 410.0]])
    x = np.append(vertices[:, 0], vertices[0, 0])
    z = np.append(vertices[:, 1], vertices[0, 1])
    actual = gravity_for_polygon(x, z, x_obs=(15.0,), z_obs=(-20.0,))[0]
    reference = triangle_area_integral_gz(vertices, 500.0, 15.0, -20.0)

    assert actual == pytest.approx(reference, rel=2e-6, abs=1e-10)


def test_polygon_winding_and_start_vertex_do_not_change_gravity():
    x, z = rectangle()
    forward = gravity_for_polygon(x, z)
    reversed_order = gravity_for_polygon(x[::-1], z[::-1])
    rotated_start = gravity_for_polygon(np.roll(x, 2), np.roll(z, 2))

    np.testing.assert_allclose(reversed_order, forward, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(rotated_start, forward, rtol=1e-12, atol=1e-12)


def test_zero_and_negative_density_contrasts():
    x, z = rectangle()
    zero = gravity_for_polygon(x, z, density=0.0)
    positive = gravity_for_polygon(x, z, density=500.0)
    negative = gravity_for_polygon(x, z, density=-500.0)

    np.testing.assert_array_equal(zero, [0.0])
    np.testing.assert_allclose(negative, -positive, rtol=1e-12, atol=1e-12)


def test_symmetric_body_has_symmetric_vertical_gravity():
    x, z = rectangle()
    result = gravity_for_polygon(x, z, x_obs=(-50.0, 50.0), z_obs=(0.0, 0.0))

    assert result[0] == pytest.approx(result[1], rel=1e-12, abs=1e-12)


def test_gravity_is_linear_and_superposes_across_bodies():
    x, z = rectangle()
    first = Body2D("first", x, z, density=300.0, y_min=-math.inf, y_max=math.inf)
    second = Body2D("second", x + 250.0, z, density=200.0, y_min=-math.inf, y_max=math.inf)
    engine = PotentialField2D([first, second])

    combined = engine.calculate_gravity(np.array([0.0, 100.0]), np.array([0.0, 0.0]))
    separate = PotentialField2D([first]).calculate_gravity(
        np.array([0.0, 100.0]), np.array([0.0, 0.0])
    ) + PotentialField2D([second]).calculate_gravity(
        np.array([0.0, 100.0]), np.array([0.0, 0.0])
    )

    np.testing.assert_allclose(combined, separate, rtol=1e-12, atol=1e-12)


def test_depth_and_observation_elevation_match_closed_form():
    x, z = rectangle()
    shallow = gravity_for_polygon(x, z)[0]
    deeper_x, deeper_z = rectangle(ztop=300.0, zbottom=500.0)
    deeper = gravity_for_polygon(deeper_x, deeper_z)[0]
    elevated = gravity_for_polygon(x, z, z_obs=(-50.0,))[0]

    assert deeper > 0.0
    assert deeper < shallow
    assert elevated > 0.0
    assert elevated < shallow
    assert elevated == pytest.approx(
        analytical_rectangle_gz(100.0, 200.0, 400.0, 500.0, z_obs=-50.0),
        rel=1e-6,
    )


def test_vertical_translation_of_body_and_observation_preserves_response():
    x, z = rectangle()
    shifted = gravity_for_polygon(x, z + 750.0, z_obs=(750.0,))

    assert shifted[0] == pytest.approx(gravity_for_polygon(x, z)[0], rel=1e-12)


def test_observation_on_polygon_edge_is_finite_and_uses_integrable_limit():
    x, z = rectangle()
    actual = gravity_for_polygon(x, z, z_obs=(200.0,))[0]

    assert math.isfinite(actual)
    assert actual == pytest.approx(
        analytical_rectangle_gz(100.0, 200.0, 400.0, 500.0, z_obs=200.0),
        rel=1e-6,
    )


def test_repeated_consecutive_vertices_are_ignored():
    x, z = rectangle()
    x_with_repeat = np.insert(x, 2, x[1])
    z_with_repeat = np.insert(z, 2, z[1])

    actual = gravity_for_polygon(x_with_repeat, z_with_repeat)[0]
    expected = analytical_rectangle_gz(100.0, 200.0, 400.0, 500.0)

    assert actual == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    ("x", "z"),
    [
        ([0.0, 1.0], [0.0]),
        ([0.0, 1.0, 0.0], [0.0, math.nan, 1.0]),
        ([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]),
        ([0.0, 3.0, 0.0, 2.0, 0.0], [0.0, 3.0, 2.0, 0.0, 0.0]),
    ],
    ids=["mismatched", "nonfinite", "zero-area", "self-intersection"],
)
def test_invalid_polygon_geometry_is_rejected(x, z):
    with pytest.raises(ValueError):
        gravity_for_polygon(x, z)


@pytest.mark.parametrize(
    ("x_obs", "z_obs"),
    [
        ((0.0, 1.0), (0.0,)),
        ((math.nan,), (0.0,)),
        ((0.0,), (math.inf,)),
    ],
    ids=["mismatched", "nonfinite-x", "nonfinite-z"],
)
def test_invalid_observations_are_rejected(x_obs, z_obs):
    x, z = rectangle()
    with pytest.raises(ValueError):
        gravity_for_polygon(x, z, x_obs=x_obs, z_obs=z_obs)

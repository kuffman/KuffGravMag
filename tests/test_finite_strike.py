"""Finite-strike prism regressions with independent analytical/numeric oracles."""

import math

import numpy as np
import pytest

from engine.forward import Body2D, PotentialField2D
from validation.analytical import rectangular_prism_gravity_gz_mgal


G = 6.67430e-11
MU0 = 4.0 * math.pi * 1e-7


def _rect_body(*, strike=(-300.0, 500.0), density=500.0, susceptibility=0.0,
               remanence=0.0, remanent_inc=0.0, remanent_dec=0.0,
               winding=1, strike_mode="finite"):
    points = np.array([[-100.0, 200.0], [100.0, 200.0],
                       [100.0, 400.0], [-100.0, 400.0]])
    if winding < 0:
        points = points[::-1]
    points = np.vstack((points, points[0]))
    return Body2D(
        "finite prism", points[:, 0], points[:, 1], density=density,
        k_succ=susceptibility, J_r=remanence, I_r=remanent_inc,
        D_r=remanent_dec, y_min=strike[0], y_max=strike[1],
        strike_mode=strike_mode,
    )


def _box_volume_integral(x_bounds, y_bounds, z_bounds, density, magnetization,
                         observation, order=32):
    """Independent tensor-Gauss volume integration of gravity and dipole B."""
    nodes, weights = np.polynomial.legendre.leggauss(order)

    def axis(bounds):
        low, high = bounds
        return (0.5 * (high - low) * nodes + 0.5 * (high + low),
                0.5 * (high - low) * weights)

    xs, wx = axis(x_bounds)
    ys, wy = axis(y_bounds)
    zs, wz = axis(z_bounds)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    volume_weights = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    xo, yo, zo = observation
    R = np.stack((xo - X, yo - Y, zo - Z), axis=-1)
    radius2 = np.sum(R * R, axis=-1)
    radius = np.sqrt(radius2)
    gz = G * density * np.sum(volume_weights * (Z - zo) / radius**3) * 1e5
    M = np.asarray(magnetization, dtype=np.float64)
    dot = np.einsum("...i,i->...", R, M)
    kernel = 3.0 * dot[..., None] * R / radius[..., None]**5 - M / radius[..., None]**3
    B = MU0 / (4.0 * math.pi) * np.sum(volume_weights[..., None] * kernel,
                                       axis=(0, 1, 2)) * 1e9
    return gz, B


def _magnetization_from_parameters(body, F_nT, I_f, D_f, alpha):
    rad = math.pi / 180.0
    inc, rel_dec = I_f * rad, (D_f - alpha) * rad
    rem_inc, rem_dec = body.I_r * rad, (body.D_r - alpha) * rad
    H0 = F_nT * 1e-9 / MU0
    return np.array([
        body.k_succ * H0 * math.cos(inc) * math.cos(rel_dec)
        + body.J_r * math.cos(rem_inc) * math.cos(rem_dec),
        body.k_succ * H0 * math.cos(inc) * math.sin(rel_dec)
        + body.J_r * math.cos(rem_inc) * math.sin(rem_dec),
        body.k_succ * H0 * math.sin(inc) + body.J_r * math.sin(rem_inc),
    ])


def test_finite_gravity_matches_prism_analytical_and_independent_volume_integral():
    body = _rect_body()
    observation = (25.0, 0.0, 0.0)
    expected_analytic = rectangular_prism_gravity_gz_mgal(
        (-100.0, 100.0), (-300.0, 500.0), (200.0, 400.0), 500.0, observation
    )
    expected_numeric, _ = _box_volume_integral(
        (-100.0, 100.0), (-300.0, 500.0), (200.0, 400.0), 500.0,
        (0.0, 0.0, 0.0), observation, order=32,
    )
    actual = PotentialField2D([body]).calculate_gravity(
        np.array([observation[0]]), np.array([observation[2]])
    )[0]

    assert expected_analytic == pytest.approx(expected_numeric, rel=2e-12, abs=2e-12)
    assert actual == pytest.approx(expected_analytic, rel=2e-11, abs=2e-11)


@pytest.mark.parametrize(
    ("inclination", "declination", "profile_azimuth"),
    [(0.0, 75.0, 20.0), (5.0, 45.0, 20.0), (35.0, 110.0, 25.0)],
    ids=["equatorial", "low-inclination", "oblique-profile"],
)
def test_finite_magnetic_all_components_and_total_field_match_volume_integral(
    inclination, declination, profile_azimuth
):
    body = _rect_body(susceptibility=0.04, remanence=1.7,
                      remanent_inc=30.0, remanent_dec=65.0)
    observation_x = np.array([25.0])
    observation_z = np.array([0.0])
    field = PotentialField2D([body]).calculate_magnetics_components(
        observation_x, observation_z, 48_000.0, inclination,
        declination, profile_azimuth,
    )
    magnetization = _magnetization_from_parameters(
        body, 48_000.0, inclination, declination, profile_azimuth
    )
    _, expected = _box_volume_integral(
        (-100.0, 100.0), (-300.0, 500.0), (200.0, 400.0), 0.0,
        magnetization, (25.0, 0.0, 0.0), order=32,
    )
    inc = math.radians(inclination)
    rel_dec = math.radians(declination - profile_azimuth)
    direction = np.array([math.cos(inc) * math.cos(rel_dec),
                          math.cos(inc) * math.sin(rel_dec), math.sin(inc)])
    expected_total = float(expected @ direction)

    for component, expected_component in zip(field[:3], expected):
        assert component[0] == pytest.approx(expected_component, rel=3e-8, abs=3e-8)
    assert field[3][0] == pytest.approx(expected_total, rel=3e-8, abs=3e-8)


def test_asymmetric_off_profile_bounds_create_expected_strike_component():
    body = _rect_body(strike=(100.0, 500.0), remanence=1.0,
                      remanent_inc=0.0, remanent_dec=60.0)
    actual = PotentialField2D([body]).calculate_magnetics_components(
        np.array([20.0]), np.array([0.0]), 0.0, 0.0, 0.0, 0.0
    )
    _, expected = _box_volume_integral(
        (-100.0, 100.0), (100.0, 500.0), (200.0, 400.0), 0.0,
        (0.5, math.sin(math.radians(60.0)), 0.0), (20.0, 0.0, 0.0), order=32,
    )
    assert abs(expected[1]) > 1.0e-3
    np.testing.assert_allclose([part[0] for part in actual[:3]], expected,
                               rtol=3e-8, atol=3e-8)


def test_finite_gravity_symmetric_response_zero_contrast_and_density_scaling():
    body = _rect_body(strike=(-500.0, 500.0), density=450.0)
    x_obs, z_obs = np.array([-65.0, 65.0]), np.zeros(2)
    response = PotentialField2D([body]).calculate_gravity(x_obs, z_obs)
    assert response[0] == pytest.approx(response[1], rel=1e-12, abs=1e-12)

    zero = _rect_body(strike=(-500.0, 500.0), density=0.0)
    np.testing.assert_array_equal(PotentialField2D([zero]).calculate_gravity(x_obs, z_obs),
                                  [0.0, 0.0])
    doubled = _rect_body(strike=(-500.0, 500.0), density=900.0)
    np.testing.assert_allclose(PotentialField2D([doubled]).calculate_gravity(x_obs, z_obs),
                               2.0 * response, rtol=1e-12, atol=1e-12)


def test_finite_prism_winding_independence_and_symmetric_profile_response():
    forward = _rect_body(strike=(-500.0, 500.0), remanence=1.0,
                         remanent_inc=0.0, remanent_dec=0.0, winding=1)
    reversed_body = _rect_body(strike=(-500.0, 500.0), remanence=1.0,
                               remanent_inc=0.0, remanent_dec=0.0, winding=-1)
    xs = np.array([-50.0, 50.0])
    zs = np.zeros(2)
    first = PotentialField2D([forward]).calculate_magnetics_components(xs, zs, 0, 0, 0, 0)
    second = PotentialField2D([reversed_body]).calculate_magnetics_components(xs, zs, 0, 0, 0, 0)
    for a, b in zip(first, second):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    assert first[0][0] == pytest.approx(first[0][1], rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(first[1], [0.0, 0.0], atol=1e-12)
    assert first[2][0] == pytest.approx(-first[2][1], rel=1e-12, abs=1e-12)


def test_finite_gravity_and_magnetic_superposition_and_scaling():
    first = _rect_body(strike=(-300.0, 500.0), density=300.0,
                       susceptibility=0.02, remanence=0.3, remanent_inc=20,
                       remanent_dec=50)
    second = _rect_body(strike=(100.0, 800.0), density=-125.0,
                        susceptibility=0.01, remanence=0.0)
    x_obs, z_obs = np.array([-35.0, 0.0, 45.0]), np.zeros(3)
    both = PotentialField2D([first, second])
    separate = [PotentialField2D([body]) for body in (first, second)]
    np.testing.assert_allclose(
        both.calculate_gravity(x_obs, z_obs),
        separate[0].calculate_gravity(x_obs, z_obs)
        + separate[1].calculate_gravity(x_obs, z_obs), rtol=1e-12, atol=1e-12,
    )
    fields = [engine.calculate_magnetics_components(x_obs, z_obs, 48_000, 15, 70, 25)
              for engine in (both, *separate)]
    for i in range(4):
        np.testing.assert_allclose(fields[0][i], fields[1][i] + fields[2][i],
                                   rtol=1e-12, atol=1e-12)

    doubled = _rect_body(strike=(-300.0, 500.0), susceptibility=0.08)
    single = _rect_body(strike=(-300.0, 500.0), susceptibility=0.04)
    zero = _rect_body(strike=(-300.0, 500.0))
    args = (x_obs[:1], z_obs[:1], 48_000.0, 0.0, 60.0, 20.0)
    one_field = PotentialField2D([single]).calculate_magnetics_components(*args)
    two_field = PotentialField2D([doubled]).calculate_magnetics_components(*args)
    zero_field = PotentialField2D([zero]).calculate_magnetics_components(*args)
    for a, b, c in zip(one_field, two_field, zero_field):
        np.testing.assert_allclose(b, 2.0 * a, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(c, [0.0])


def test_explicit_finite_zero_width_is_physical_zero_but_legacy_zero_bounds_are_infinite():
    x_obs, z_obs = np.array([0.0]), np.array([0.0])
    zero_width = _rect_body(strike=(0.0, 0.0), strike_mode="finite",
                            density=500.0, remanence=1.0)
    infinite = _rect_body(strike=(0.0, 0.0), strike_mode="infinite",
                          density=500.0, remanence=1.0)
    legacy = _rect_body(strike=(0.0, 0.0), strike_mode=None,
                        density=500.0, remanence=1.0)
    zero_engine = PotentialField2D([zero_width])
    infinite_engine = PotentialField2D([infinite])
    legacy_engine = PotentialField2D([legacy])

    np.testing.assert_array_equal(zero_engine.calculate_gravity(x_obs, z_obs), [0.0])
    for component in zero_engine.calculate_magnetics_components(x_obs, z_obs, 0, 0, 0, 0):
        np.testing.assert_array_equal(component, [0.0])
    assert infinite_engine.calculate_gravity(x_obs, z_obs)[0] > 0.0
    np.testing.assert_allclose(legacy_engine.calculate_gravity(x_obs, z_obs),
                               infinite_engine.calculate_gravity(x_obs, z_obs),
                               rtol=1e-12, atol=1e-12)


def test_finite_strike_converges_to_infinite_strike_kernels():
    finite = _rect_body(strike=(-1.0e6, 1.0e6), density=500.0,
                        susceptibility=0.03, remanence=0.7,
                        remanent_inc=25.0, remanent_dec=60.0)
    infinite = _rect_body(strike=(-math.inf, math.inf), density=500.0,
                          susceptibility=0.03, remanence=0.7,
                          remanent_inc=25.0, remanent_dec=60.0,
                          strike_mode="infinite")
    x_obs, z_obs = np.array([-25.0, 0.0, 25.0]), np.zeros(3)
    finite_engine, infinite_engine = PotentialField2D([finite]), PotentialField2D([infinite])
    np.testing.assert_allclose(finite_engine.calculate_gravity(x_obs, z_obs),
                               infinite_engine.calculate_gravity(x_obs, z_obs),
                               rtol=1e-7, atol=1e-9)
    finite_mag = finite_engine.calculate_magnetics_components(x_obs, z_obs, 48_000, 25, 60, 15)
    infinite_mag = infinite_engine.calculate_magnetics_components(x_obs, z_obs, 48_000, 25, 60, 15)
    for actual, expected in zip(finite_mag, infinite_mag):
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=2e-5)

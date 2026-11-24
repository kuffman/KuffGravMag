"""Synthetic tests for bounded property inversion and diagnostics."""

import numpy as np
import pytest

from engine.forward import Body2D, PotentialField2D
from engine.inversion import InversionResult, invert_body_parameters


def _body(*, density=250.0, susceptibility=0.02, x_shift=0.0, top=200.0, bottom=400.0):
    return Body2D(
        "synthetic body",
        np.array([-100.0, 100.0, 100.0, -100.0, -100.0]) + x_shift,
        [top, top, bottom, bottom, top],
        density=density,
        k_succ=susceptibility,
        J_r=0.0,
        I_r=0.0,
        D_r=0.0,
        y_min=0.0,
        y_max=0.0,
        strike_mode="infinite",
    )


def _gravity(body, x_obs):
    return PotentialField2D([body]).calculate_gravity(x_obs, np.zeros_like(x_obs))


def _magnetic(body, x_obs, field_params):
    return PotentialField2D([body]).calculate_magnetics_components(
        x_obs,
        np.zeros_like(x_obs),
        field_params["F"],
        field_params["Inc"],
        field_params["Dec"],
        field_params["alpha"],
    )[3]


def test_synthetic_gravity_recovers_density_and_reports_uncertainty():
    x_obs = np.linspace(-500.0, 500.0, 61)
    truth = _body(density=640.0)
    target = _body(density=180.0)
    observed = _gravity(truth, x_obs)

    result = invert_body_parameters(
        target,
        x_obs,
        np.zeros_like(x_obs),
        observed,
        field_type="gravity",
        observation_sigma=0.01,
    )

    assert isinstance(result, InversionResult)
    assert result.success
    assert result.identifiable
    assert result.parameters["density"] == pytest.approx(640.0, abs=1e-5)
    assert result.final_rms < 1e-8
    assert result.initial_rms > result.final_rms
    assert result.standard_errors is not None
    assert result.parameter_sensitivity["density"] > 0.0


def test_synthetic_magnetics_recovers_susceptibility_with_orientation_inputs():
    x_obs = np.linspace(-500.0, 500.0, 61)
    field_params = {"F": 48_500.0, "Inc": 35.0, "Dec": 18.0, "alpha": 7.0}
    truth = _body(density=0.0, susceptibility=0.036)
    target = _body(density=0.0, susceptibility=0.012)
    observed = _magnetic(truth, x_obs, field_params)

    result = invert_body_parameters(
        target,
        x_obs,
        np.zeros_like(x_obs),
        observed,
        field_type="magnetics",
        field_params=field_params,
        observation_sigma=0.01,
    )

    assert result.success
    assert result.parameters["k_succ"] == pytest.approx(0.036, abs=1e-7)
    assert target.k_succ == pytest.approx(0.036, abs=1e-7)


def test_all_bodies_contribute_while_only_the_target_body_is_updated():
    x_obs = np.linspace(-600.0, 600.0, 71)
    truth = _body(density=520.0)
    fixed = _body(density=-180.0, x_shift=250.0)
    target = _body(density=120.0)
    observed = PotentialField2D([truth, fixed]).calculate_gravity(x_obs, np.zeros_like(x_obs))

    result = invert_body_parameters(
        target,
        x_obs,
        np.zeros_like(x_obs),
        observed,
        all_bodies=[target, fixed],
        field_type="gravity",
    )

    assert result.success
    assert target.density == pytest.approx(520.0, abs=1e-5)
    assert fixed.density == -180.0


def test_observation_weights_reduce_the_effect_of_a_noisy_outlier():
    x_obs = np.linspace(-500.0, 500.0, 51)
    truth = _body(density=700.0)
    exact = _gravity(truth, x_obs)
    noisy = exact.copy()
    noisy[0] += 12.0
    sigma = np.full(x_obs.size, 0.05)
    sigma[0] = 100.0

    unweighted = invert_body_parameters(
        _body(density=300.0), x_obs, np.zeros_like(x_obs), noisy, field_type="gravity"
    )
    weighted = invert_body_parameters(
        _body(density=300.0),
        x_obs,
        np.zeros_like(x_obs),
        noisy,
        field_type="gravity",
        observation_sigma=sigma,
    )

    assert unweighted.success and weighted.success
    assert abs(weighted.parameters["density"] - 700.0) < abs(unweighted.parameters["density"] - 700.0)


def test_regularization_pulls_toward_reference_without_hiding_weighted_misfit():
    x_obs = np.linspace(-500.0, 500.0, 51)
    truth = _body(density=700.0)
    observed = _gravity(truth, x_obs)

    unregularized = invert_body_parameters(
        _body(density=100.0),
        x_obs,
        np.zeros_like(x_obs),
        observed,
        field_type="gravity",
        observation_sigma=20.0,
    )
    regularized = invert_body_parameters(
        _body(density=100.0),
        x_obs,
        np.zeros_like(x_obs),
        observed,
        field_type="gravity",
        observation_sigma=20.0,
        regularization=2.0,
        reference_model={"density": 100.0},
    )

    assert unregularized.success and regularized.success
    assert regularized.parameters["density"] < unregularized.parameters["density"]
    assert regularized.initial_cost > regularized.final_cost


def test_geometry_fit_respects_depth_bounds_and_preserves_polygon_closure():
    x_obs = np.linspace(-500.0, 500.0, 61)
    truth = _body(density=500.0, top=250.0, bottom=500.0)
    target = _body(density=430.0)
    observed = _gravity(truth, x_obs)
    x_before = target.x.copy()

    result = invert_body_parameters(
        target,
        x_obs,
        np.zeros_like(x_obs),
        observed,
        field_type="gravity",
        fit_geometry=True,
        depth_bounds=(0.0, 800.0),
        parameter_bounds={
            "density": (-1000.0, 1000.0),
            "depth_offset_m": (-100.0, 200.0),
            "depth_scale": (0.5, 2.0),
        },
    )

    assert result.success
    assert result.parameters["depth_offset_m"] >= -200.0
    assert result.parameters["depth_scale"] > 0.0
    assert target.z[0] == target.z[-1]
    assert target.z.min() >= 0.0
    assert target.z.max() <= 800.0
    np.testing.assert_array_equal(target.x, x_before)


def test_underdetermined_geometry_fit_is_marked_nonidentifiable():
    x_obs = np.array([-250.0, 250.0])
    truth = _body(density=500.0, top=240.0, bottom=460.0)
    target = _body(density=300.0)
    observed = _gravity(truth, x_obs)

    result = invert_body_parameters(
        target,
        x_obs,
        np.zeros_like(x_obs),
        observed,
        field_type="gravity",
        fit_geometry=True,
        regularization=0.1,
    )

    assert result.success
    assert not result.identifiable
    assert result.numerical_rank < result.parameter_count
    assert result.standard_errors is None
    assert "non-unique" in result.message


def test_noisy_synthetic_density_recovery_is_stable_with_reported_noise():
    rng = np.random.default_rng(20261124)
    x_obs = np.linspace(-500.0, 500.0, 81)
    truth = _body(density=620.0)
    sigma = 0.02
    observed = _gravity(truth, x_obs) + rng.normal(0.0, sigma, x_obs.size)

    result = invert_body_parameters(
        _body(density=100.0),
        x_obs,
        np.zeros_like(x_obs),
        observed,
        field_type="gravity",
        observation_sigma=sigma,
    )

    assert result.success
    assert result.parameters["density"] == pytest.approx(620.0, abs=5.0)
    assert result.weighted_rms < 1.5


def test_invalid_input_reports_failure_without_mutating_model():
    x_obs = np.linspace(-100.0, 100.0, 5)
    target = _body(density=321.0)
    before_x, before_z = target.x.copy(), target.z.copy()

    result = invert_body_parameters(
        target,
        x_obs,
        np.zeros_like(x_obs),
        np.zeros_like(x_obs),
        field_type="gravity",
        observation_sigma=0.0,
    )

    assert not result.success
    assert result.status == "invalid_input"
    assert not bool(result)
    assert target.density == 321.0
    np.testing.assert_array_equal(target.x, before_x)
    np.testing.assert_array_equal(target.z, before_z)


def test_invalid_geometry_is_rejected_and_restored():
    x_obs = np.linspace(-100.0, 100.0, 5)
    crossed = Body2D(
        "crossed",
        [0.0, 100.0, 0.0, 100.0, 0.0],
        [100.0, 200.0, 200.0, 100.0, 100.0],
        density=321.0,
        strike_mode="infinite",
        y_min=0.0,
        y_max=0.0,
    )
    before_x, before_z = crossed.x.copy(), crossed.z.copy()

    result = invert_body_parameters(
        crossed,
        x_obs,
        np.zeros_like(x_obs),
        np.zeros_like(x_obs),
        field_type="gravity",
    )

    assert not result.success
    assert result.status == "invalid_model"
    np.testing.assert_array_equal(crossed.x, before_x)
    np.testing.assert_array_equal(crossed.z, before_z)


def test_maximum_evaluations_is_not_reported_as_convergence_and_restores_model():
    x_obs = np.linspace(-500.0, 500.0, 61)
    observed = _gravity(_body(density=900.0), x_obs)
    target = _body(density=100.0)
    before = target.density

    result = invert_body_parameters(
        target,
        x_obs,
        np.zeros_like(x_obs),
        observed,
        field_type="gravity",
        max_nfev=1,
    )

    assert not result.success
    assert result.status == "no_convergence"
    assert target.density == before


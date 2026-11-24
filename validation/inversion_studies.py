"""Deterministic synthetic studies of noise, uncertainty, and non-uniqueness."""

from __future__ import annotations

import math

import numpy as np

from engine.forward import Body2D, PotentialField2D
from engine.inversion import invert_body_parameters


NOISE_LEVELS_MGAL = (0.01, 0.03, 0.1)
DEFAULT_SEED = 20261124


def _rectangle(density: float, top: float = 200.0, bottom: float = 400.0) -> Body2D:
    return Body2D(
        "synthetic inversion prism",
        [-100.0, 100.0, 100.0, -100.0, -100.0],
        [top, top, bottom, bottom, top],
        density=density,
        y_min=0.0,
        y_max=0.0,
        strike_mode="infinite",
    )


def run_noise_recovery_ensemble(*, seed: int = DEFAULT_SEED, trials: int = 32) -> dict:
    """Run a fixed-seed density-recovery ensemble at three reported noise levels.

    Each noise level uses the same standardized draws scaled by its stated
    standard deviation, making the error and uncertainty trend comparisons
    deterministic and paired. Coverage is the empirical fraction of fitted
    local linear 95% intervals that include the known synthetic density.
    """
    trials = int(trials)
    if trials < 2:
        raise ValueError("At least two trials are required to summarize noise sensitivity")
    seed = int(seed)
    x_obs = np.linspace(-500.0, 500.0, 61)
    z_obs = np.zeros_like(x_obs)
    true_density = 620.0
    truth = _rectangle(true_density)
    clean = PotentialField2D([truth]).calculate_gravity(x_obs, z_obs)
    standardized_noise = np.random.default_rng(seed).standard_normal((trials, x_obs.size))
    records = []

    for noise_index, sigma in enumerate(NOISE_LEVELS_MGAL):
        errors = []
        standard_errors = []
        coverages = []
        identifiable = []
        for trial in range(trials):
            observations = clean + sigma * standardized_noise[trial]
            target = _rectangle(250.0)
            result = invert_body_parameters(
                target,
                x_obs,
                z_obs,
                observations,
                field_type="gravity",
                observation_sigma=sigma,
                parameter_bounds={"density": (0.0, 1500.0)},
                parameter_scales={"density": 500.0},
                max_nfev=100,
            )
            if not result.success:
                raise RuntimeError(f"Noise ensemble trial did not converge: {result.message}")
            errors.append(result.parameters["density"] - true_density)
            identifiable.append(result.identifiable)
            standard_error = (
                None if result.standard_errors is None else result.standard_errors.get("density")
            )
            if standard_error is not None and math.isfinite(standard_error):
                standard_errors.append(standard_error)
                coverages.append(abs(errors[-1]) <= 1.96 * standard_error)

        errors_array = np.asarray(errors, dtype=np.float64)
        records.append(
            {
                "sigma_mgal": float(sigma),
                "mean_bias_density_kg_m3": float(np.mean(errors_array)),
                "rmse_density_kg_m3": float(np.sqrt(np.mean(errors_array**2))),
                "mean_standard_error_kg_m3": float(np.mean(standard_errors))
                if standard_errors
                else None,
                "coverage_95": float(np.mean(coverages)) if coverages else None,
                "identifiable_fraction": float(np.mean(identifiable)),
                "uncertainty_available_fraction": float(len(standard_errors) / trials),
                "successful_trials": len(errors),
            }
        )
    return {
        "seed": seed,
        "trials": trials,
        "true_density_kg_m3": true_density,
        "noise_levels_mgal": list(NOISE_LEVELS_MGAL),
        "records": records,
    }


def density_depth_tradeoff() -> dict:
    """Construct two different density/depth models that match one station.

    Density is linear in the forward response, so a deeper geometry can be
    assigned a different density to match a single measurement exactly. Their
    residual profile shows why broader spatial sampling is needed to reduce the
    tradeoff; it does not imply that the model space is otherwise unique.
    """
    x_obs = np.linspace(-500.0, 500.0, 101)
    z_obs = np.zeros_like(x_obs)
    shallow = _rectangle(500.0, top=200.0, bottom=400.0)
    shallow_curve = PotentialField2D([shallow]).calculate_gravity(x_obs, z_obs)
    station_index = int(np.argmin(np.abs(x_obs)))

    unit_deep = _rectangle(1.0, top=350.0, bottom=550.0)
    unit_curve = PotentialField2D([unit_deep]).calculate_gravity(x_obs, z_obs)
    deep_density = float(shallow_curve[station_index] / unit_curve[station_index])
    deep = _rectangle(deep_density, top=350.0, bottom=550.0)
    deep_curve = PotentialField2D([deep]).calculate_gravity(x_obs, z_obs)
    one_station_relative = float(
        (deep_curve[station_index] - shallow_curve[station_index])
        / shallow_curve[station_index]
    )
    profile_relative = float(
        np.sqrt(np.mean((deep_curve - shallow_curve) ** 2))
        / np.sqrt(np.mean(shallow_curve**2))
    )
    return {
        "shallow_density_kg_m3": 500.0,
        "shallow_top_m": 200.0,
        "deep_density_kg_m3": deep_density,
        "deep_top_m": 350.0,
        "single_station_relative_residual": one_station_relative,
        "full_profile_relative_rms_difference": profile_relative,
        "profile_observation_count": int(x_obs.size),
    }


def geometry_identifiability_study() -> dict:
    """Fit three geometry/property parameters from two gravity observations.

    This deliberately underdetermined synthetic problem records optimizer
    convergence separately from local identifiability and uncertainty. The
    returned diagnostics describe the data Jacobian, not global uniqueness.
    """
    x_obs = np.array([-250.0, 250.0], dtype=np.float64)
    z_obs = np.zeros_like(x_obs)
    truth = _rectangle(500.0, top=240.0, bottom=460.0)
    target = _rectangle(300.0)
    observed = PotentialField2D([truth]).calculate_gravity(x_obs, z_obs)
    result = invert_body_parameters(
        target,
        x_obs,
        z_obs,
        observed,
        field_type="gravity",
        fit_geometry=True,
        regularization=0.1,
    )
    if not result.success:
        raise RuntimeError(f"Underdetermined geometry study did not converge: {result.message}")
    return {
        "observation_count": int(x_obs.size),
        "parameter_count": int(result.parameter_count),
        "numerical_rank": int(result.numerical_rank),
        "identifiable": bool(result.identifiable),
        "uncertainty_available": result.standard_errors is not None,
        "success": bool(result.success),
    }

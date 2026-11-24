"""Bounded, weighted inversions for the validated KuffGravMag forward models."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np
from scipy.optimize import least_squares

from engine.forward import PotentialField2D


@dataclass(frozen=True)
class InversionResult:
    """Optimizer outcome and local identifiability diagnostics.

    ``success`` describes numerical optimizer convergence. It does not imply that
    the data uniquely identify every fitted parameter; see ``identifiable``.
    """

    success: bool
    status: str
    message: str
    field_type: str
    parameters: Mapping[str, float] = field(default_factory=dict)
    initial_rms: float | None = None
    final_rms: float | None = None
    weighted_rms: float | None = None
    initial_cost: float | None = None
    final_cost: float | None = None
    function_evaluations: int = 0
    jacobian_evaluations: int | None = None
    optimality: float | None = None
    identifiable: bool = False
    numerical_rank: int = 0
    parameter_count: int = 0
    condition_number: float | None = None
    parameter_sensitivity: Mapping[str, float] = field(default_factory=dict)
    standard_errors: Mapping[str, float] | None = None
    covariance: tuple[tuple[float, ...], ...] | None = None

    def __bool__(self) -> bool:
        """Preserve useful truth-value behavior for older callers."""
        return self.success


def _failure(field_type: str, status: str, message: str) -> InversionResult:
    return InversionResult(
        success=False,
        status=status,
        message=message,
        field_type=field_type,
    )


def _finite_vector(values, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return result


def _positive_vector(values, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim == 0:
        result = np.full(size, float(result), dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain one finite positive value per observation")
    return result


def _ordered_bounds(value, name: str) -> tuple[float, float]:
    try:
        lower, upper = map(float, value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bounds for {name} must be a pair of finite numbers") from exc
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError(f"Bounds for {name} must be finite and strictly increasing")
    return lower, upper


def _field_prediction(engine, field_type, x_obs, z_obs, field_params):
    if field_type == "gravity":
        return engine.calculate_gravity(x_obs, z_obs)
    return engine.calculate_magnetics_components(
        x_obs,
        z_obs,
        field_params["F"],
        field_params["Inc"],
        field_params["Dec"],
        field_params["alpha"],
    )[3]


def invert_body_parameters(
    target_body,
    x_obs,
    z_obs,
    obs_data,
    *,
    all_bodies=None,
    field_type="gravity",
    field_params=None,
    observation_sigma=1.0,
    parameter_bounds=None,
    parameter_scales=None,
    reference_model=None,
    regularization=0.0,
    fit_geometry=False,
    depth_bounds=(0.0, 20_000.0),
    loss="linear",
    f_scale=1.0,
    max_nfev=2000,
) -> InversionResult:
    """Fit density or susceptibility, optionally with constrained depth shape.

    The fitted property is selected by ``field_type``. Geometry is fixed by
    default; when requested, two shape-preserving parameters translate the
    section in depth and scale its vertical relief. The latter keeps polygon
    topology intact while enforcing positive, bounded depths.

    ``observation_sigma`` is a scalar or per-observation standard deviation in
    mGal (gravity) or nT (magnetics). Residuals are divided by these values.
    ``regularization`` is lambda in the dimensionless objective
    ``||W_d r||² + lambda² ||q-q_ref||²``. Parameters are scaled into the
    dimensionless vector ``q`` before the trust-region solve.

    A converged result may still be non-identifiable. The data-only Jacobian
    rank and condition number report local identifiability; uncertainty is
    returned only for a full-rank linear-loss fit with positive degrees of
    freedom.
    """
    kind = str(field_type).strip().lower()
    if kind in {"magnetic", "magnetics"}:
        kind = "magnetics"
        property_name = "k_succ"
        default_property_bounds = (0.0, 1.0)
        default_property_scale = 0.01
    elif kind == "gravity":
        property_name = "density"
        default_property_bounds = (-5000.0, 5000.0)
        default_property_scale = 500.0
    else:
        return _failure(str(field_type), "invalid_input", "field_type must be 'gravity' or 'magnetics'")

    try:
        x_values = _finite_vector(x_obs, "x_obs")
        z_values = _finite_vector(z_obs, "z_obs")
        observed = _finite_vector(obs_data, "obs_data")
        if x_values.size == 0 or z_values.shape != x_values.shape or observed.shape != x_values.shape:
            raise ValueError("x_obs, z_obs, and obs_data must have the same nonzero length")
        sigma = _positive_vector(observation_sigma, x_values.size, "observation_sigma")

        regularization = float(regularization)
        f_scale = float(f_scale)
        max_nfev = int(max_nfev)
        if not math.isfinite(regularization) or regularization < 0.0:
            raise ValueError("regularization must be finite and nonnegative")
        if not math.isfinite(f_scale) or f_scale <= 0.0:
            raise ValueError("f_scale must be finite and positive")
        if max_nfev <= 0:
            raise ValueError("max_nfev must be positive")
        if loss not in {"linear", "soft_l1", "huber"}:
            raise ValueError("loss must be 'linear', 'soft_l1', or 'huber'")

        field_values = dict(field_params or {})
        magnetic_params = {
            "F": float(field_values.get("F", 50_000.0)),
            "Inc": float(field_values.get("Inc", 60.0)),
            "Dec": float(field_values.get("Dec", field_values.get("D", 0.0))),
            "alpha": float(field_values.get("alpha", field_values.get("Alpha", 0.0))),
        }
        if not all(math.isfinite(value) for value in magnetic_params.values()):
            raise ValueError("Magnetic field parameters must be finite")
        if magnetic_params["F"] <= 0.0:
            raise ValueError("Magnetic field strength F must be positive")

        x_original = np.asarray(target_body.x, dtype=np.float64).copy()
        z_original = np.asarray(target_body.z, dtype=np.float64).copy()
        if x_original.ndim != 1 or z_original.ndim != 1 or x_original.shape != z_original.shape:
            raise ValueError("The target body must have matching one-dimensional x and z coordinates")
        if x_original.size < 4 or not np.all(np.isfinite(x_original)) or not np.all(np.isfinite(z_original)):
            raise ValueError("The target body must be a finite, closed polygon with at least three vertices")
        if not (
            np.isclose(x_original[0], x_original[-1], rtol=0.0, atol=1e-12)
            and np.isclose(z_original[0], z_original[-1], rtol=0.0, atol=1e-12)
        ):
            raise ValueError("The target body polygon must be closed")

        initial_property = float(getattr(target_body, property_name))
        if not math.isfinite(initial_property):
            raise ValueError(f"The target body's {property_name} must be finite")

        if isinstance(all_bodies, (str, bytes)):
            raise ValueError("all_bodies must be a sequence of body objects")
        bodies = list(all_bodies) if all_bodies is not None else []
        if any(body is None for body in bodies):
            raise ValueError("all_bodies cannot contain None")
        if any(not hasattr(body, "x") or not hasattr(body, "z") for body in bodies):
            raise ValueError("all_bodies must contain body objects with x and z coordinates")
        if not any(body is target_body for body in bodies):
            bodies.append(target_body)
        if len({id(body) for body in bodies}) != len(bodies):
            raise ValueError("all_bodies contains the same body more than once")

        bounds = dict(parameter_bounds or {})
        scales = dict(parameter_scales or {})
        property_bounds = _ordered_bounds(
            bounds.get(property_name, default_property_bounds), property_name
        )
        if not property_bounds[0] <= initial_property <= property_bounds[1]:
            raise ValueError(f"Initial {property_name} lies outside its bounds")

        parameter_names = [property_name]
        initial_values = [initial_property]
        lower_values = [property_bounds[0]]
        upper_values = [property_bounds[1]]
        scale_values = [float(scales.get(property_name, default_property_scale))]

        original_unique_z = z_original[:-1]
        depth_min = depth_max = None
        if fit_geometry:
            try:
                depth_min, depth_max = map(float, depth_bounds)
            except (TypeError, ValueError) as exc:
                raise ValueError("depth_bounds must be a finite increasing pair") from exc
            if not (math.isfinite(depth_min) and math.isfinite(depth_max) and depth_min < depth_max):
                raise ValueError("depth_bounds must be a finite increasing pair")

            z_min = float(np.min(original_unique_z))
            z_max = float(np.max(original_unique_z))
            vertical_range = z_max - z_min
            if vertical_range <= 0.0 or z_min < depth_min or z_max > depth_max:
                raise ValueError("Initial body depths must lie inside depth_bounds and have positive relief")

            shift_bounds = _ordered_bounds(bounds.get("depth_offset_m", (-2000.0, 2000.0)), "depth_offset_m")
            scale_bounds = _ordered_bounds(bounds.get("depth_scale", (0.25, 4.0)), "depth_scale")
            if scale_bounds[0] <= 0.0:
                raise ValueError("depth_scale bounds must remain positive to preserve polygon topology")
            if not shift_bounds[0] <= 0.0 <= shift_bounds[1]:
                raise ValueError("depth_offset_m bounds must include the unchanged model")
            if not scale_bounds[0] <= 1.0 <= scale_bounds[1]:
                raise ValueError("depth_scale bounds must include the unchanged model")

            # Fixed bounds guarantee all combinations in the trust region stay
            # inside the requested depth interval, with scale=1 and offset=0 valid.
            shift_lower = max(shift_bounds[0], depth_min - z_min)
            shift_upper = min(
                shift_bounds[1],
                depth_max - z_min - vertical_range,
                depth_max - z_min - scale_bounds[0] * vertical_range,
            )
            if shift_lower > 0.0 or shift_upper < 0.0 or shift_lower >= shift_upper:
                raise ValueError("depth_offset_m bounds have no feasible range inside depth_bounds")
            safe_scale_upper = min(
                scale_bounds[1],
                (depth_max - z_min - shift_upper) / vertical_range,
            )
            if safe_scale_upper < 1.0 or safe_scale_upper <= scale_bounds[0]:
                raise ValueError("depth_scale bounds have no feasible range inside depth_bounds")

            parameter_names.extend(("depth_offset_m", "depth_scale"))
            initial_values.extend((0.0, 1.0))
            lower_values.extend((shift_lower, scale_bounds[0]))
            upper_values.extend((shift_upper, safe_scale_upper))
            scale_values.extend((float(scales.get("depth_offset_m", 200.0)),
                                 float(scales.get("depth_scale", 1.0))))

        initial_values = np.asarray(initial_values, dtype=np.float64)
        scale_values = np.asarray(scale_values, dtype=np.float64)
        lower_values = np.asarray(lower_values, dtype=np.float64)
        upper_values = np.asarray(upper_values, dtype=np.float64)
        if not np.all(np.isfinite(scale_values)) or np.any(scale_values <= 0.0):
            raise ValueError("parameter_scales must be finite and positive")
        if np.any(initial_values < lower_values) or np.any(initial_values > upper_values):
            raise ValueError("The starting model lies outside the requested parameter bounds")

        reference_values = dict(reference_model or {})
        reference = initial_values.copy()
        for index, name in enumerate(parameter_names):
            if name in reference_values:
                reference[index] = float(reference_values[name])
        if not np.all(np.isfinite(reference)):
            raise ValueError("reference_model values must be finite")

        engine = PotentialField2D(bodies)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return _failure(kind, "invalid_input", str(exc))

    original_property = initial_property
    initial_q = np.zeros(len(parameter_names), dtype=np.float64)
    lower_q = (lower_values - initial_values) / scale_values
    upper_q = (upper_values - initial_values) / scale_values
    reference_q = (reference - initial_values) / scale_values

    def set_model(q):
        physical = initial_values + scale_values * np.asarray(q, dtype=np.float64)
        setattr(target_body, property_name, float(physical[0]))
        if fit_geometry:
            offset, vertical_scale = float(physical[1]), float(physical[2])
            z_min = float(np.min(original_unique_z))
            new_unique_z = z_min + offset + vertical_scale * (original_unique_z - z_min)
            target_body.update_polygon(x_original, np.append(new_unique_z, new_unique_z[0]))
        return physical

    def predict(q):
        set_model(q)
        values = np.asarray(
            _field_prediction(engine, kind, x_values, z_values, magnetic_params),
            dtype=np.float64,
        )
        if values.shape != observed.shape or not np.all(np.isfinite(values)):
            raise ValueError("Forward model returned nonfinite or mismatched values")
        return values

    try:
        initial_prediction = predict(initial_q)
        initial_residual = initial_prediction - observed
        initial_weighted = initial_residual / sigma
    except (ArithmeticError, ValueError, FloatingPointError) as exc:
        setattr(target_body, property_name, original_property)
        target_body.update_polygon(x_original, z_original)
        return _failure(kind, "invalid_model", f"Initial model is invalid: {exc}")

    invalid_residual = np.full(observed.size + len(parameter_names), 1.0e12, dtype=np.float64)

    def objective(q):
        try:
            predicted = predict(q)
            weighted = (predicted - observed) / sigma
            regularized = regularization * (np.asarray(q, dtype=np.float64) - reference_q)
            return np.concatenate((weighted, regularized))
        except (ArithmeticError, ValueError, FloatingPointError):
            return invalid_residual.copy()

    try:
        result = least_squares(
            objective,
            initial_q,
            method="trf",
            bounds=(lower_q, upper_q),
            x_scale=1.0,
            loss=loss,
            f_scale=f_scale,
            max_nfev=max_nfev,
        )
        best_prediction = predict(result.x)
    except (ArithmeticError, ValueError, FloatingPointError, RuntimeError) as exc:
        setattr(target_body, property_name, original_property)
        target_body.update_polygon(x_original, z_original)
        return _failure(kind, "optimization_error", f"Optimizer could not evaluate a valid model: {exc}")

    final_raw_residual = best_prediction - observed
    final_weighted_residual = final_raw_residual / sigma
    final_rms = float(np.sqrt(np.mean(final_raw_residual**2)))
    weighted_rms = float(np.sqrt(np.mean(final_weighted_residual**2)))
    initial_rms = float(np.sqrt(np.mean(initial_residual**2)))
    initial_cost = float(
        0.5 * np.dot(initial_weighted, initial_weighted)
        + 0.5 * regularization**2 * np.dot(initial_q - reference_q, initial_q - reference_q)
    )

    data_jacobian = np.asarray(result.jac[:observed.size, :], dtype=np.float64)
    if data_jacobian.ndim != 2 or not np.all(np.isfinite(data_jacobian)):
        singular_values = np.asarray([], dtype=np.float64)
        rank = 0
        condition_number = None
    else:
        singular_values = np.linalg.svd(data_jacobian, compute_uv=False)
        tolerance = (
            singular_values[0] * max(data_jacobian.shape) * np.finfo(np.float64).eps
            if singular_values.size and singular_values[0] > 0.0
            else 0.0
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        condition_number = (
            float(singular_values[0] / singular_values[-1])
            if rank == len(parameter_names) and singular_values[-1] > 0.0
            else None
        )
    identifiable = rank == len(parameter_names) and condition_number is not None and condition_number < 1.0e12
    sensitivity = {
        name: float(np.linalg.norm(data_jacobian[:, index]))
        for index, name in enumerate(parameter_names)
    }

    standard_errors = None
    covariance = None
    degrees_of_freedom = observed.size - rank
    if identifiable and loss == "linear" and degrees_of_freedom > 0:
        try:
            variance = float(np.dot(final_weighted_residual, final_weighted_residual) / degrees_of_freedom)
            hessian = data_jacobian.T @ data_jacobian
            hessian += (regularization**2) * np.eye(len(parameter_names))
            covariance_q = variance * np.linalg.inv(hessian)
            covariance_physical = covariance_q * np.outer(scale_values, scale_values)
            diagonal = np.maximum(np.diag(covariance_physical), 0.0)
            standard_errors = {
                name: float(math.sqrt(diagonal[index]))
                for index, name in enumerate(parameter_names)
            }
            covariance = tuple(tuple(float(value) for value in row) for row in covariance_physical)
        except np.linalg.LinAlgError:
            standard_errors = None
            covariance = None

    physical_solution = initial_values + scale_values * result.x
    parameters = {name: float(value) for name, value in zip(parameter_names, physical_solution)}
    success = bool(result.success and np.all(np.isfinite(physical_solution)))
    if success:
        set_model(result.x)
        status = "converged"
        message = str(result.message)
        if not identifiable:
            message += " Data do not locally identify all fitted parameters; interpret this model as non-unique."
        if standard_errors is None:
            message += " Local uncertainty was not reported because the data Jacobian is rank-deficient, has no degrees of freedom, or uses a robust loss."
    else:
        setattr(target_body, property_name, original_property)
        target_body.update_polygon(x_original, z_original)
        status = "no_convergence"
        message = str(result.message)

    return InversionResult(
        success=success,
        status=status,
        message=message,
        field_type=kind,
        parameters=parameters,
        initial_rms=initial_rms,
        final_rms=final_rms,
        weighted_rms=weighted_rms,
        initial_cost=initial_cost,
        final_cost=float(result.cost),
        function_evaluations=int(result.nfev),
        jacobian_evaluations=None if result.njev is None else int(result.njev),
        optimality=float(result.optimality),
        identifiable=bool(identifiable),
        numerical_rank=rank,
        parameter_count=len(parameter_names),
        condition_number=condition_number,
        parameter_sensitivity=sensitivity,
        standard_errors=standard_errors,
        covariance=covariance,
    )

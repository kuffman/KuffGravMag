"""Conditioned Euler deconvolution for gridded potential-field profiles."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EulerResult:
    success: bool
    status: str
    message: str
    x0_m: float | None = None
    y0_m: float | None = None
    z0_up_m: float | None = None
    depth_below_observation_m: float | None = None
    base_level: float | None = None
    structural_index: float | None = None
    rank: int = 0
    condition_number: float | None = None
    rms_residual: float | None = None
    sample_count: int = 0


def _failure(status: str, message: str, *, rank: int = 0, condition: float | None = None, count: int = 0) -> EulerResult:
    return EulerResult(False, status, message, rank=rank, condition_number=condition, sample_count=count)


def euler_deconvolution_grid(
    x_m,
    y_m,
    values,
    derivative_x,
    derivative_y,
    derivative_z_up,
    *,
    structural_index: float,
    observation_z_up_m: float = 0.0,
    mask=None,
    condition_limit: float = 1.0e10,
) -> EulerResult:
    """Solve the fixed-index Euler equation over one selected regular-grid window.

    Coordinates use x/y in metres and z positive upward. ``derivative_z_up``
    must therefore be a derivative with respect to positive-up elevation. The
    linear solve estimates source (x0,y0,z0) and base level jointly. A result
    is rejected when the system is underdetermined, rank deficient, ill
    conditioned, or places the source above the observation plane.

    This solver reports a local Euler estimate for the selected structural
    index and window; it does not infer the correct index or establish unique
    geology. Window choice, derivative noise, remanence, and edge effects remain
    interpretation inputs.
    """
    index = float(structural_index)
    z_obs = float(observation_z_up_m)
    max_condition = float(condition_limit)
    if not math.isfinite(index) or not 0.0 < index <= 5.0:
        raise ValueError("structural_index must be finite and in the interval (0, 5]")
    if not math.isfinite(z_obs):
        raise ValueError("observation_z_up_m must be finite")
    if not math.isfinite(max_condition) or max_condition <= 1.0:
        raise ValueError("condition_limit must be finite and greater than one")

    x_axis = np.asarray(x_m, dtype=np.float64)
    y_axis = np.asarray(y_m, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    dx = np.asarray(derivative_x, dtype=np.float64)
    dy = np.asarray(derivative_y, dtype=np.float64)
    dz = np.asarray(derivative_z_up, dtype=np.float64)
    if x_axis.ndim != 1 or y_axis.ndim != 1 or x_axis.size < 2 or y_axis.size < 2:
        raise ValueError("x_m and y_m must be regular one-dimensional coordinate axes")
    if not np.all(np.isfinite(x_axis)) or not np.all(np.isfinite(y_axis)):
        raise ValueError("Euler coordinate axes must be finite")
    if np.any(np.diff(x_axis) <= 0.0) or np.any(np.diff(y_axis) <= 0.0):
        raise ValueError("Euler coordinate axes must be strictly increasing")
    if not np.allclose(np.diff(x_axis), np.diff(x_axis)[0]) or not np.allclose(np.diff(y_axis), np.diff(y_axis)[0]):
        raise ValueError("Euler deconvolution requires regular grid spacing")
    expected_shape = (y_axis.size, x_axis.size)
    if any(array.shape != expected_shape for array in (data, dx, dy, dz)):
        raise ValueError("values and derivative grids must have shape (len(y_m), len(x_m))")
    if mask is None:
        selected = np.ones(expected_shape, dtype=bool)
    else:
        selected = np.array(mask, dtype=bool, copy=True)
        if selected.shape != expected_shape:
            raise ValueError("mask must match the grid shape")
    xx, yy = np.meshgrid(x_axis, y_axis)
    selected &= np.isfinite(data) & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dz)
    count = int(np.count_nonzero(selected))
    if count < 8:
        return _failure("underdetermined", "Euler window needs at least eight finite stations", count=count)

    x_ref = float(np.mean(xx[selected]))
    y_ref = float(np.mean(yy[selected]))
    x_relative = xx[selected] - x_ref
    y_relative = yy[selected] - y_ref
    z_relative = np.full(count, z_obs - z_obs, dtype=np.float64)
    gx, gy, gz = dx[selected], dy[selected], dz[selected]
    field = data[selected]

    # x Tx + y Ty + z Tz + SI*T = x0 Tx + y0 Ty + z0 Tz + SI*B.
    design = np.column_stack((gx, gy, gz, np.full(count, index)))
    target = x_relative * gx + y_relative * gy + z_relative * gz + index * field
    column_norms = np.linalg.norm(design, axis=0)
    if np.any(column_norms <= np.finfo(np.float64).tiny):
        return _failure("singular", "Euler system has a zero-information parameter column", count=count)
    scaled_design = design / column_norms
    singular_values = np.linalg.svd(scaled_design, compute_uv=False)
    tolerance = singular_values[0] * max(scaled_design.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if rank == 4 and singular_values[-1] > 0.0 else None
    )
    if rank < 4:
        return _failure("singular", "Euler system is rank deficient for this window and structural index", rank=rank, condition=condition, count=count)
    if condition is None or condition > max_condition:
        return _failure("ill_conditioned", "Euler system exceeds the declared condition limit", rank=rank, condition=condition, count=count)

    scaled_solution, _, solved_rank, _ = np.linalg.lstsq(scaled_design, target, rcond=None)
    solution = scaled_solution / column_norms
    x0 = x_ref + float(solution[0])
    y0 = y_ref + float(solution[1])
    z0 = z_obs + float(solution[2])
    base = float(solution[3])
    residual = design @ solution - target
    rms = float(np.sqrt(np.mean(residual**2)))
    depth = z_obs - z0
    if solved_rank < 4:
        return _failure("singular", "Least-squares solve found a rank-deficient Euler system", rank=int(solved_rank), condition=condition, count=count)
    if not all(math.isfinite(value) for value in (x0, y0, z0, base, rms)):
        return _failure("invalid_solution", "Euler solution contains nonfinite values", rank=rank, condition=condition, count=count)
    if depth <= 0.0:
        return _failure("nonpositive_depth", "Euler source estimate is not below the observation plane", rank=rank, condition=condition, count=count)
    return EulerResult(
        True,
        "converged",
        "Euler source estimate converged for the supplied structural index and window.",
        x0_m=x0,
        y0_m=y0,
        z0_up_m=z0,
        depth_below_observation_m=depth,
        base_level=base,
        structural_index=index,
        rank=rank,
        condition_number=condition,
        rms_residual=rms,
        sample_count=count,
    )

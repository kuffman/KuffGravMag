"""Pinned Harmonica prism comparisons with explicit local/global axis mapping."""

from __future__ import annotations

import math

import numpy as np


PINNED_HARMONICA_VERSION = "0.7.0"


def _harmonica():
    try:
        import harmonica
    except ImportError as exc:  # pragma: no cover - exercised without the optional group
        raise RuntimeError(
            "Harmonica comparisons require `uv sync --locked --group validation`."
        ) from exc
    version = str(getattr(harmonica, "__version__", "")).lstrip("v")
    if version != PINNED_HARMONICA_VERSION:
        raise RuntimeError(
            f"M5 comparisons require Harmonica {PINNED_HARMONICA_VERSION}; found {version!r}."
        )
    return harmonica


def _axis_aligned_prism(x_bounds, y_bounds, z_bounds, profile_azimuth):
    x1, x2 = map(float, x_bounds)
    y1, y2 = map(float, y_bounds)
    z1, z2 = map(float, z_bounds)
    if not (x1 < x2 and y1 < y2 and z1 < z2):
        raise ValueError("Prism bounds must be strictly increasing")
    angle = float(profile_azimuth) % 360.0
    if not math.isclose(angle % 90.0, 0.0, abs_tol=1e-12):
        raise ValueError("Axis-aligned Harmonica comparisons support 90-degree profile increments")
    angle = int(round(angle)) % 360
    if angle == 0:
        east, north = (y1, y2), (x1, x2)
    elif angle == 90:
        east, north = (x1, x2), (-y2, -y1)
    elif angle == 180:
        east, north = (-y2, -y1), (-x2, -x1)
    else:
        east, north = (-x2, -x1), (y1, y2)
    upward = (-z2, -z1)
    return [east[0], east[1], north[0], north[1], upward[0], upward[1]], angle


def _global_observation(x_obs, y_obs, z_down, angle):
    alpha = math.radians(angle)
    x_obs, y_obs, z_down = map(float, (x_obs, y_obs, z_down))
    east = x_obs * math.sin(alpha) + y_obs * math.cos(alpha)
    north = x_obs * math.cos(alpha) - y_obs * math.sin(alpha)
    return east, north, -z_down


def _global_vector(x, y, z_down, angle):
    alpha = math.radians(angle)
    east = x * math.sin(alpha) + y * math.cos(alpha)
    north = x * math.cos(alpha) - y * math.sin(alpha)
    return east, north, -z_down


def _local_vector(east, north, up, angle):
    alpha = math.radians(angle)
    x = east * math.sin(alpha) + north * math.cos(alpha)
    y = east * math.cos(alpha) - north * math.sin(alpha)
    return float(x), float(y), float(-up)


def harmonica_rectangular_prism_gravity_gz_mgal(
    x_bounds,
    y_bounds,
    z_bounds,
    density,
    observation,
    *,
    profile_azimuth=90.0,
):
    """Return Harmonica's downward prism gravity component in mGal."""
    harmonica = _harmonica()
    prism, angle = _axis_aligned_prism(x_bounds, y_bounds, z_bounds, profile_azimuth)
    east, north, upward = _global_observation(*observation, angle)
    result = harmonica.prism_gravity(
        coordinates=(np.array([east]), np.array([north]), np.array([upward])),
        prisms=prism,
        density=float(density),
        field="g_z",
        parallel=False,
    )
    return float(np.asarray(result, dtype=np.float64)[0])


def harmonica_rectangular_prism_magnetic_components_nT(
    x_bounds,
    y_bounds,
    z_bounds,
    magnetization_local,
    x_obs,
    z_obs,
    F_nT,
    I_f,
    D_f,
    profile_azimuth,
    *,
    y_obs=0.0,
):
    """Return Harmonica prism dX/dY/dZ in KuffGravMag's local frame.

    Harmonica uses easting/northing/upward and returns its vector in nT. This
    adapter rotates the prism, magnetization, observation, and output using the
    documented clockwise-from-north profile azimuth, then flips the upward
    component into KuffGravMag's positive-down vertical convention.
    """
    harmonica = _harmonica()
    prism, angle = _axis_aligned_prism(x_bounds, y_bounds, z_bounds, profile_azimuth)
    mx, my, mz_down = map(float, magnetization_local)
    me, mn, mu = _global_vector(mx, my, mz_down, angle)
    east, north, upward = _global_observation(x_obs, y_obs, z_obs, angle)
    components = harmonica.prism_magnetic(
        coordinates=(np.array([east]), np.array([north]), np.array([upward])),
        prisms=prism,
        magnetization=(np.array([me]), np.array([mn]), np.array([mu])),
        field="b",
        parallel=False,
    )
    be, bn, bu = (float(np.asarray(component, dtype=np.float64)[0]) for component in components)
    dx, dy, dz_down = _local_vector(be, bn, bu, angle)

    inc, dec_rel = math.radians(float(I_f)), math.radians(float(D_f) - angle)
    direction = (
        math.cos(inc) * math.cos(dec_rel),
        math.cos(inc) * math.sin(dec_rel),
        math.sin(inc),
    )
    total = dx * direction[0] + dy * direction[1] + dz_down * direction[2]
    return dx, dy, dz_down, total

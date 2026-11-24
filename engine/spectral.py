"""Fourier potential-field transforms for evenly sampled profiles and grids."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


def _valid_spacing(value: float, name: str) -> float:
    spacing = float(value)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return spacing


def _valid_height(value: float) -> float:
    height = float(value)
    if not math.isfinite(height) or height < 0.0:
        raise ValueError("height_m must be finite and nonnegative for upward continuation")
    return height


def _valid_padding(padding: str) -> str:
    if padding not in {"none", "reflect"}:
        raise ValueError("padding must be 'none' (periodic) or 'reflect'")
    return padding


def _real_crop(values: np.ndarray, begin: int, end: int | None = None) -> np.ndarray:
    output = np.real_if_close(values, tol=1000).real
    return np.asarray(output[begin:end], dtype=np.float64)


@dataclass(frozen=True)
class SpectralProfileResult:
    upward_continued: np.ndarray
    horizontal_derivative: np.ndarray
    vertical_derivative_down: np.ndarray
    analytic_signal: np.ndarray
    tilt_angle_deg: np.ndarray
    edge_mask: np.ndarray
    edge_effect_samples: int
    units: Mapping[str, str]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class SpectralGridResult:
    upward_continued: np.ndarray
    derivative_x: np.ndarray
    derivative_y: np.ndarray
    vertical_derivative_down: np.ndarray
    analytic_signal: np.ndarray
    tilt_angle_deg: np.ndarray
    edge_mask: np.ndarray
    edge_effect_samples: tuple[int, int]
    units: Mapping[str, str]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class MagneticGridTransformResult:
    """Direction-change transform and explicit spectral noise-gain diagnostics."""

    values: np.ndarray
    filter_gain: np.ndarray
    max_gain: float
    rms_gain: float
    fraction_gain_above_one: float
    regularization: float
    edge_mask: np.ndarray
    edge_effect_samples: tuple[int, int]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class SpectralDepthResult:
    depth_m: float
    standard_error_m: float
    slope_m: float
    r_squared: float
    wavenumber_band_rad_per_m: tuple[float, float]
    bin_count: int
    assumptions: tuple[str, ...]


def transform_profile(
    values,
    *,
    spacing_m: float,
    height_m: float = 0.0,
    padding: str = "reflect",
    units: str = "input_unit",
) -> SpectralProfileResult:
    """Continue and differentiate a uniformly sampled 1D profile.

    This uses the 2D profile approximation (invariant along strike), so the
    Fourier wavenumber is along-profile only. ``padding='none'`` assumes the
    supplied line repeats periodically. Reflect padding is the default and
    marks a conservative edge band that must be reviewed before interpretation.
    The vertical derivative is positive downward; the tilt is
    ``atan2(VDR, abs(HDR))``, the 1D equivalent of using horizontal-gradient
    magnitude in the 2D grid calculation.
    """
    data = np.asarray(values, dtype=np.float64)
    spacing = _valid_spacing(spacing_m, "spacing_m")
    height = _valid_height(height_m)
    padding = _valid_padding(padding)
    if data.ndim != 1 or data.size < 8 or not np.all(np.isfinite(data)):
        raise ValueError("Profile transforms require at least eight finite, uniformly sampled values")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("units must be declared")
    pad = data.size // 2 if padding == "reflect" else 0
    padded = np.pad(data, (pad, pad), mode="reflect") if pad else data.copy()
    k = 2.0 * np.pi * np.fft.fftfreq(padded.size, d=spacing)
    spectrum = np.fft.fft(padded)
    attenuation = np.exp(-np.abs(k) * height)
    continued = np.fft.ifft(spectrum * attenuation)
    derivative_x = np.fft.ifft(spectrum * (1j * k) * attenuation)
    derivative_down = np.fft.ifft(spectrum * np.abs(k) * attenuation)
    continued = _real_crop(continued, pad, -pad if pad else None)
    derivative_x = _real_crop(derivative_x, pad, -pad if pad else None)
    derivative_down = _real_crop(derivative_down, pad, -pad if pad else None)
    analytic = np.hypot(derivative_x, derivative_down)
    tilt = np.degrees(np.arctan2(derivative_down, np.abs(derivative_x)))
    edge_mask = np.zeros(data.size, dtype=bool)
    if pad:
        edge_mask[:pad] = True
        edge_mask[-pad:] = True
    edge_mask.setflags(write=False)
    for array in (continued, derivative_x, derivative_down, analytic, tilt):
        array.setflags(write=False)
    return SpectralProfileResult(
        continued, derivative_x, derivative_down, analytic, tilt, edge_mask, pad,
        {
            "upward_continued": units,
            "horizontal_derivative": f"{units}/m",
            "vertical_derivative_down": f"{units}/m",
            "analytic_signal": f"{units}/m",
            "tilt_angle_deg": "degree",
        },
        (
            "1D Fourier transform assumes a 2D profile field invariant along strike.",
            "Samples are uniformly spaced; no missing or irregular stations are allowed.",
            "Reflect padding mirrors the profile; edge_mask marks the padded-width edge band for review.",
            "Upward continuation uses exp(-|k|h); vertical derivative is positive downward.",
        ) if padding == "reflect" else (
            "1D Fourier transform assumes a 2D profile field invariant along strike.",
            "Samples are uniformly spaced; no missing or irregular stations are allowed.",
            "No padding assumes periodic repetition at both profile ends; edge effects may wrap around.",
            "Upward continuation uses exp(-|k|h); vertical derivative is positive downward.",
        ),
    )


def transform_grid(
    values,
    *,
    spacing_x_m: float,
    spacing_y_m: float,
    height_m: float = 0.0,
    padding: str = "reflect",
    units: str = "input_unit",
) -> SpectralGridResult:
    """Apply 2D Fourier continuation and derivatives on a regular grid.

    Grid values are ordered ``(ny, nx)`` for ascending y and x coordinates.
    The 2D wavenumber uses both horizontal axes. ``padding='none'`` is periodic;
    reflect padding returns an edge-review mask. Vertical derivative is positive
    downward and tilt uses atan2(VDR, horizontal-gradient magnitude).
    """
    data = np.asarray(values, dtype=np.float64)
    dx, dy = _valid_spacing(spacing_x_m, "spacing_x_m"), _valid_spacing(spacing_y_m, "spacing_y_m")
    height = _valid_height(height_m)
    padding = _valid_padding(padding)
    if data.ndim != 2 or min(data.shape) < 8 or not np.all(np.isfinite(data)):
        raise ValueError("Grid transforms require at least an 8 by 8 grid of finite, uniformly sampled values")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("units must be declared")
    ny, nx = data.shape
    pad_y, pad_x = (ny // 2, nx // 2) if padding == "reflect" else (0, 0)
    padded = np.pad(data, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect") if pad_y else data.copy()
    ky = 2.0 * np.pi * np.fft.fftfreq(padded.shape[0], d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(padded.shape[1], d=dx)
    kyy, kxx = np.meshgrid(ky, kx, indexing="ij")
    kr = np.hypot(kxx, kyy)
    spectrum = np.fft.fft2(padded)
    attenuation = np.exp(-kr * height)
    continued = np.fft.ifft2(spectrum * attenuation)
    derivative_x = np.fft.ifft2(spectrum * (1j * kxx) * attenuation)
    derivative_y = np.fft.ifft2(spectrum * (1j * kyy) * attenuation)
    derivative_down = np.fft.ifft2(spectrum * kr * attenuation)
    region = (slice(pad_y, -pad_y if pad_y else None), slice(pad_x, -pad_x if pad_x else None))
    continued = _real_crop(continued[region], 0)
    derivative_x = _real_crop(derivative_x[region], 0)
    derivative_y = _real_crop(derivative_y[region], 0)
    derivative_down = _real_crop(derivative_down[region], 0)
    horizontal = np.hypot(derivative_x, derivative_y)
    analytic = np.hypot(horizontal, derivative_down)
    tilt = np.degrees(np.arctan2(derivative_down, horizontal))
    edge_mask = np.zeros(data.shape, dtype=bool)
    if pad_y:
        edge_mask[:pad_y, :] = True
        edge_mask[-pad_y:, :] = True
    if pad_x:
        edge_mask[:, :pad_x] = True
        edge_mask[:, -pad_x:] = True
    edge_mask.setflags(write=False)
    for array in (continued, derivative_x, derivative_y, derivative_down, analytic, tilt):
        array.setflags(write=False)
    return SpectralGridResult(
        continued, derivative_x, derivative_y, derivative_down, analytic, tilt,
        edge_mask, (pad_y, pad_x),
        {
            "upward_continued": units,
            "derivative_x": f"{units}/m",
            "derivative_y": f"{units}/m",
            "vertical_derivative_down": f"{units}/m",
            "analytic_signal": f"{units}/m",
            "tilt_angle_deg": "degree",
        },
        (
            "2D Fourier transform assumes a complete, regular grid and a harmonic potential field above sources.",
            "Grid values are ordered (ny, nx); x and y spacings are uniform and positive.",
            "Reflect padding mirrors the grid; edge_mask marks the padded-width boundary band for review.",
            "Upward continuation uses exp(-|k|h); vertical derivative is positive downward.",
        ) if padding == "reflect" else (
            "2D Fourier transform assumes a complete, regular grid and a harmonic potential field above sources.",
            "Grid values are ordered (ny, nx); x and y spacings are uniform and positive.",
            "No padding assumes periodic repetition at each grid edge; edge effects may wrap around.",
            "Upward continuation uses exp(-|k|h); vertical derivative is positive downward.",
        ),
    )


def _direction_factor(inclination_deg: float, declination_deg: float, kx: np.ndarray,
                      ky: np.ndarray, kr: np.ndarray) -> np.ndarray:
    inclination = math.radians(float(inclination_deg))
    declination = math.radians(float(declination_deg))
    if not all(math.isfinite(value) for value in (inclination, declination)):
        raise ValueError("magnetic inclinations and declinations must be finite")
    if abs(float(inclination_deg)) > 90.0:
        raise ValueError("magnetic inclinations must be within [-90, 90] degrees")
    safe = np.where(kr > 0.0, kr, 1.0)
    east = math.sin(declination) * math.cos(inclination)
    north = math.cos(declination) * math.cos(inclination)
    down = math.sin(inclination)
    return down + 1j * (east * kx + north * ky) / safe


def stabilized_magnetic_transform_grid(
    values,
    *,
    spacing_x_m: float,
    spacing_y_m: float,
    field_inclination_deg: float,
    field_declination_deg: float,
    magnetization_inclination_deg: float,
    magnetization_declination_deg: float,
    target_inclination_deg: float = 90.0,
    target_declination_deg: float = 0.0,
    regularization: float,
    padding: str = "reflect",
    max_gain_limit: float = 20.0,
) -> MagneticGridTransformResult:
    """Stabilized FFT direction transform (RTP by default) for a regular grid.

    The unregularized operator is the ratio of the target and observed magnetic
    direction factors. A Tikhonov inverse is used at every nonzero wavenumber:

        H = K_target * conj(K_observed) / (|K_observed|² + lambda²)

    ``regularization`` is a required, dimensionless positive value. The output
    reports the maximum and RMS Fourier gain and is rejected when the requested
    filter exceeds ``max_gain_limit``. Declinations are east of north; positive
    inclination points down. Magnetization direction must be supplied
    explicitly, so remanence is not silently assumed absent.
    """
    data = np.asarray(values, dtype=np.float64)
    dx = _valid_spacing(spacing_x_m, "spacing_x_m")
    dy = _valid_spacing(spacing_y_m, "spacing_y_m")
    padding = _valid_padding(padding)
    lam = float(regularization)
    gain_limit = float(max_gain_limit)
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("regularization must be finite and positive")
    if not math.isfinite(gain_limit) or gain_limit <= 0.0:
        raise ValueError("max_gain_limit must be finite and positive")
    if data.ndim != 2 or min(data.shape) < 8 or not np.all(np.isfinite(data)):
        raise ValueError("Magnetic grid transforms require a complete finite grid of at least 8 by 8 cells")
    directional_values = (
        field_inclination_deg, field_declination_deg, magnetization_inclination_deg,
        magnetization_declination_deg, target_inclination_deg, target_declination_deg,
    )
    if not all(math.isfinite(float(value)) for value in directional_values):
        raise ValueError("all field, magnetization, and target angles must be finite")
    for value in (field_inclination_deg, magnetization_inclination_deg, target_inclination_deg):
        if abs(float(value)) > 90.0:
            raise ValueError("magnetic inclinations must be within [-90, 90] degrees")

    ny, nx = data.shape
    pad_y, pad_x = (ny // 2, nx // 2) if padding == "reflect" else (0, 0)
    padded = np.pad(data, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect") if pad_y else data.copy()
    ky = 2.0 * np.pi * np.fft.fftfreq(padded.shape[0], d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(padded.shape[1], d=dx)
    kyy, kxx = np.meshgrid(ky, kx, indexing="ij")
    kr = np.hypot(kxx, kyy)
    observed = (
        _direction_factor(field_inclination_deg, field_declination_deg, kxx, kyy, kr)
        * _direction_factor(magnetization_inclination_deg, magnetization_declination_deg, kxx, kyy, kr)
    )
    target = (
        _direction_factor(target_inclination_deg, target_declination_deg, kxx, kyy, kr)
        * _direction_factor(target_inclination_deg, target_declination_deg, kxx, kyy, kr)
    )
    transfer = target * np.conj(observed) / (np.abs(observed) ** 2 + lam**2)
    transfer[kr == 0.0] = 1.0
    gain = np.abs(transfer)
    max_gain = float(np.max(gain))
    rms_gain = float(np.sqrt(np.mean(gain**2)))
    fraction_above_one = float(np.mean(gain > 1.0))
    if max_gain > gain_limit:
        raise ValueError(
            f"regularized transform reaches gain {max_gain:.3g}, above the declared limit {gain_limit:.3g}; increase regularization"
        )
    transformed = np.fft.ifft2(np.fft.fft2(padded) * transfer)
    region = (slice(pad_y, -pad_y if pad_y else None), slice(pad_x, -pad_x if pad_x else None))
    transformed = _real_crop(transformed[region], 0)
    edge_mask = np.zeros(data.shape, dtype=bool)
    if pad_y:
        edge_mask[:pad_y, :] = True
        edge_mask[-pad_y:, :] = True
    if pad_x:
        edge_mask[:, :pad_x] = True
        edge_mask[:, -pad_x:] = True
    gain = np.asarray(gain, dtype=np.float64)
    for array in (transformed, gain, edge_mask):
        array.setflags(write=False)
    return MagneticGridTransformResult(
        transformed, gain, max_gain, rms_gain, fraction_above_one, lam, edge_mask,
        (pad_y, pad_x),
        (
            "The input is a complete, regular total-field magnetic grid; no missing cells are filled.",
            "The Fourier direction kernel follows the planar potential-field RTP relation; positive z is down and declination is east of north.",
            "The supplied source magnetization direction is treated as spatially uniform; remanence is only represented through that declared direction.",
            "Tikhonov regularization damps poorly conditioned wavenumbers; reported gain diagnostics quantify potential white-noise amplification.",
            "The result is an interpretive transform, not a unique recovered magnetization or a validated low-latitude inversion.",
            "Reflect padding marks a boundary band for review." if padding == "reflect" else "No padding assumes periodic repetition at the grid boundaries.",
        ),
    )


def estimate_spectral_depth(
    values,
    *,
    spacing_x_m: float,
    spacing_y_m: float,
    wavenumber_band_rad_per_m: tuple[float, float],
    radial_bin_count: int = 24,
) -> SpectralDepthResult:
    """Estimate an ensemble/source depth from log radial power versus wavenumber.

    The fitted model is ``log(power) = intercept - 2 * depth * k``. This is a
    band-limited summary for a source population, not a depth pick for an
    individual body. It returns the ordinary least-squares slope error and
    goodness of fit so users can reject unstable or poorly linear bands.
    """
    data = np.asarray(values, dtype=np.float64)
    dx, dy = _valid_spacing(spacing_x_m, "spacing_x_m"), _valid_spacing(spacing_y_m, "spacing_y_m")
    if data.ndim != 2 or min(data.shape) < 8 or not np.all(np.isfinite(data)):
        raise ValueError("Spectral depth requires a complete finite grid of at least 8 by 8 cells")
    try:
        k_min, k_max = map(float, wavenumber_band_rad_per_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("wavenumber_band_rad_per_m must contain two numbers") from exc
    if not all(math.isfinite(value) for value in (k_min, k_max)) or k_min <= 0.0 or k_max <= k_min:
        raise ValueError("wavenumber band must be finite, positive, and increasing")
    bins = int(radial_bin_count)
    if bins < 4:
        raise ValueError("radial_bin_count must be at least four")

    centered = data - float(np.mean(data))
    window = np.outer(np.hanning(data.shape[0]), np.hanning(data.shape[1]))
    spectrum = np.fft.fft2(centered * window)
    ky = 2.0 * np.pi * np.fft.fftfreq(data.shape[0], d=dy)
    kx = 2.0 * np.pi * np.fft.fftfreq(data.shape[1], d=dx)
    kyy, kxx = np.meshgrid(ky, kx, indexing="ij")
    kr = np.hypot(kxx, kyy)
    power = np.abs(spectrum) ** 2
    edges = np.linspace(k_min, k_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radial_power = np.full(bins, np.nan, dtype=np.float64)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (kr >= low) & (kr < high) & (power > 0.0)
        if np.count_nonzero(selected) >= 2:
            radial_power[index] = float(np.mean(power[selected]))
    valid = np.isfinite(radial_power) & (radial_power > 0.0)
    if np.count_nonzero(valid) < 4:
        raise ValueError("Selected spectral band has fewer than four populated radial bins")
    x = centers[valid]
    y = np.log(radial_power[valid])
    design = np.column_stack((x, np.ones_like(x)))
    solution, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise ValueError("Selected spectral band is rank deficient")
    slope, intercept = map(float, solution)
    residual = y - (slope * x + intercept)
    dof = x.size - 2
    if dof <= 0:
        raise ValueError("Selected spectral band has no residual degrees of freedom")
    variance = float(np.sum(residual**2) / dof)
    covariance = variance * np.linalg.inv(design.T @ design)
    slope_error = math.sqrt(max(0.0, float(covariance[0, 0])))
    total = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = 1.0 if total == 0.0 else 1.0 - float(np.sum(residual**2)) / total
    depth = -0.5 * slope
    standard_error = 0.5 * slope_error
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("Selected spectral band does not yield a positive depth estimate")
    return SpectralDepthResult(
        depth, standard_error, slope, r_squared, (k_min, k_max), int(x.size),
        (
            "The estimate fits log radial power as log(P)=intercept-2*depth*k over the explicitly selected band.",
            "The grid is detrended only by mean subtraction and tapered by a 2D Hann window; finite-window leakage may remain.",
            "The estimate summarizes a source ensemble; it does not resolve individual-body depth or source geometry.",
            "The reported standard error is the linear-fit slope uncertainty and excludes model, sampling, and geological uncertainty.",
        ),
    )

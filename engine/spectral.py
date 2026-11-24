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
    The vertical derivative is positive downward; the tilt is ``atan2(VDR,HDR)``.
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
    tilt = np.degrees(np.arctan2(derivative_down, derivative_x))
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

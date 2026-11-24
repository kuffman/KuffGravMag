from __future__ import annotations

import numpy as np
import pytest

from engine.spectral import transform_profile, transform_grid
from engine.euler import euler_deconvolution_grid


def test_profile_spectral_filters_match_a_periodic_fourier_mode():
    n = 256
    spacing = 10.0
    wavelength = n * spacing / 4.0
    x = np.arange(n) * spacing
    k = 2.0 * np.pi / wavelength
    values = np.cos(k * x)
    height = 75.0
    result = transform_profile(values, spacing_m=spacing, height_m=height, padding="none")
    attenuation = np.exp(-k * height)

    assert result.upward_continued == pytest.approx(attenuation * values, abs=1e-12)
    assert result.horizontal_derivative == pytest.approx(-k * attenuation * np.sin(k * x), abs=1e-12)
    assert result.vertical_derivative_down == pytest.approx(k * attenuation * values, abs=1e-12)
    assert result.analytic_signal == pytest.approx(np.full(n, k * attenuation), abs=1e-12)
    assert result.units["horizontal_derivative"] == "input_unit/m"
    assert result.edge_effect_samples == 0


def test_profile_reflect_padding_marks_edge_affected_samples():
    values = np.exp(-((np.arange(64) - 32.0) / 8.0) ** 2)
    result = transform_profile(values, spacing_m=25.0, height_m=100.0, padding="reflect")

    assert result.upward_continued.shape == values.shape
    assert np.isfinite(result.upward_continued).all()
    assert result.edge_effect_samples > 0
    assert result.edge_mask[: result.edge_effect_samples].all()
    assert result.edge_mask[-result.edge_effect_samples :].all()
    assert not result.edge_mask[result.edge_effect_samples : -result.edge_effect_samples].any()


def test_grid_spectral_filters_match_a_two_dimensional_plane_wave():
    nx, ny = 96, 80
    dx, dy = 20.0, 30.0
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    xx, yy = np.meshgrid(x, y)
    kx = 2.0 * np.pi * 3 / (nx * dx)
    ky = 2.0 * np.pi * 2 / (ny * dy)
    kr = np.hypot(kx, ky)
    phase = kx * xx + ky * yy
    field = np.cos(phase)
    height = 40.0
    attenuation = np.exp(-kr * height)
    result = transform_grid(field, spacing_x_m=dx, spacing_y_m=dy,
                            height_m=height, padding="none")

    assert result.upward_continued == pytest.approx(attenuation * field, abs=1e-12)
    assert result.derivative_x == pytest.approx(-kx * attenuation * np.sin(phase), abs=1e-12)
    assert result.derivative_y == pytest.approx(-ky * attenuation * np.sin(phase), abs=1e-12)
    assert result.vertical_derivative_down == pytest.approx(kr * attenuation * field, abs=1e-12)
    assert result.analytic_signal == pytest.approx(np.full_like(field, kr * attenuation), abs=1e-12)
    assert result.edge_effect_samples == (0, 0)


def test_euler_deconvolution_recovers_analytic_point_source_and_reports_conditioning():
    x_m = np.linspace(-1200.0, 1200.0, 31)
    y_m = np.linspace(-900.0, 900.0, 25)
    xx, yy = np.meshgrid(x_m, y_m)
    x0, y0, z0 = 120.0, -80.0, -600.0
    z_obs = 0.0
    amplitude, base = 8.0e7, 25.0
    rx, ry, rz = xx - x0, yy - y0, z_obs - z0
    r2 = rx**2 + ry**2 + rz**2
    field = amplitude / r2 + base
    d_dx = -2.0 * amplitude * rx / r2**2
    d_dy = -2.0 * amplitude * ry / r2**2
    d_dz_up = -2.0 * amplitude * rz / r2**2

    result = euler_deconvolution_grid(
        x_m, y_m, field, d_dx, d_dy, d_dz_up,
        structural_index=2.0,
        observation_z_up_m=z_obs,
        condition_limit=1e8,
    )

    assert result.success
    assert result.x0_m == pytest.approx(x0, abs=1e-7)
    assert result.y0_m == pytest.approx(y0, abs=1e-7)
    assert result.z0_up_m == pytest.approx(z0, abs=1e-7)
    assert result.depth_below_observation_m == pytest.approx(600.0, abs=1e-7)
    assert result.base_level == pytest.approx(base, abs=1e-9)
    assert result.rank == 4
    assert result.condition_number < 1e8
    assert result.rms_residual < 1e-9


def test_euler_deconvolution_rejects_underdetermined_or_singular_windows():
    x = np.arange(3.0)
    y = np.arange(3.0)
    xx, yy = np.meshgrid(x, y)
    values = xx + yy
    zeros = np.zeros_like(values)
    result = euler_deconvolution_grid(
        x, y, values, zeros, zeros, zeros, structural_index=1.0,
    )

    assert not result.success
    assert result.rank < 4
    assert result.status in {"underdetermined", "singular"}


def test_euler_deconvolution_rejects_invalid_structural_index():
    x = np.arange(4.0)
    xx, yy = np.meshgrid(x, x)
    values = xx + yy + 1.0
    with pytest.raises(ValueError, match="structural_index"):
        euler_deconvolution_grid(
            x, x, values, values, values, values, structural_index=-1.0,
        )

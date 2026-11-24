"""Offline IGRF-14 main-field evaluation using the pinned official coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
import hashlib
from pathlib import Path
import math

import numpy as np

MODEL_NAME = "IGRF-14"
MODEL_VALID_FROM = 1900.0
MODEL_VALID_TO = 2030.0
COEFFICIENT_SHA256 = "717f6dce821a8f2bfcc6a77f79cc227ba91f61aeb458d5433e8c72450d48f8e0"
REFERENCE_RADIUS_KM = 6371.2
WGS84_A_KM = 6378.137
WGS84_B_KM = 6356.752314245


@dataclass(frozen=True)
class IGRFVector:
    north_nt: float
    east_nt: float
    down_nt: float
    horizontal_nt: float
    total_nt: float
    declination_deg: float
    inclination_deg: float
    latitude_geodetic_deg: float
    longitude_east_deg: float
    altitude_ellipsoid_m: float
    decimal_year: float
    model: str = MODEL_NAME
    component_units: str = "nT"


@lru_cache(maxsize=1)
def _read_coefficients() -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray]]:
    path = Path(__file__).with_name("data") / "IGRF14.shc"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != COEFFICIENT_SHA256:
        raise RuntimeError("Bundled IGRF-14 coefficient file hash does not match the pinned release")
    rows = [line for line in payload.decode("ascii").splitlines() if line and not line.startswith("#")]
    if len(rows) < 3:
        raise RuntimeError("Pinned IGRF-14 coefficient file is incomplete")
    _n_min, n_max, n_epochs, _order, _step = map(int, rows[0].split()[:5])
    epochs = np.asarray([float(value) for value in rows[1].split()], dtype=np.float64)
    if epochs.size != n_epochs or epochs[0] != MODEL_VALID_FROM or epochs[-1] != MODEL_VALID_TO:
        raise RuntimeError("Pinned IGRF-14 coefficient epochs do not match the supported validity range")
    coefficients: dict[tuple[int, int], np.ndarray] = {}
    for line in rows[2:]:
        parts = line.split()
        n, signed_m = int(parts[0]), int(parts[1])
        values = np.asarray(parts[2:], dtype=np.float64)
        if values.size != n_epochs or n > n_max:
            raise RuntimeError("Malformed pinned IGRF-14 coefficient row")
        coefficients[(n, signed_m)] = values
    return epochs, coefficients


def _decimal_year(value: date | datetime | float | int | str) -> float:
    if isinstance(value, (int, float)):
        year = float(value)
    else:
        parsed: date | datetime
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("date must be ISO YYYY-MM-DD, a date/datetime, or a decimal year") from exc
        elif isinstance(value, (date, datetime)):
            parsed = value
        else:
            raise ValueError("date must be ISO YYYY-MM-DD, a date/datetime, or a decimal year")
        if isinstance(parsed, datetime):
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            instant = parsed.astimezone(timezone.utc)
            start = datetime(instant.year, 1, 1, tzinfo=timezone.utc)
            end = datetime(instant.year + 1, 1, 1, tzinfo=timezone.utc)
            year = instant.year + (instant - start).total_seconds() / (end - start).total_seconds()
        else:
            start = date(parsed.year, 1, 1)
            end = date(parsed.year + 1, 1, 1)
            year = parsed.year + (parsed - start).days / (end - start).days
    if not math.isfinite(year) or year < MODEL_VALID_FROM or year > MODEL_VALID_TO:
        raise ValueError("IGRF-14 supports decimal years from 1900.0 through 2030.0 only")
    return year


def _schmidt_legendre(theta_rad: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    theta = float(theta_rad)
    sin_theta, cos_theta = math.sin(theta), math.cos(theta)
    p = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    dp = np.zeros_like(p)
    schmidt = np.zeros_like(p)
    p[0, 0] = schmidt[0, 0] = 1.0
    for n in range(1, n_max + 1):
        for m in range(n + 1):
            if n == m:
                p[n, n] = sin_theta * p[n - 1, m - 1]
                dp[n, n] = sin_theta * dp[n - 1, m - 1] + cos_theta * p[n - 1, n - 1]
            elif n == 1:
                p[n, m] = cos_theta * p[n - 1, m]
                dp[n, m] = cos_theta * dp[n - 1, m] - sin_theta * p[n - 1, m]
            else:
                factor = ((n - 1) ** 2 - m ** 2) / ((2 * n - 1) * (2 * n - 3))
                p[n, m] = cos_theta * p[n - 1, m] - factor * p[n - 2, m]
                dp[n, m] = (
                    cos_theta * dp[n - 1, m] - sin_theta * p[n - 1, m]
                    - factor * dp[n - 2, m]
                )
            if m == 0:
                schmidt[n, 0] = schmidt[n - 1, 0] * (2 * n - 1) / n
            else:
                schmidt[n, m] = schmidt[n, m - 1] * math.sqrt(
                    (n - m + 1) * (2.0 if m == 1 else 1.0) / (n + m)
                )
    for n in range(1, n_max + 1):
        for m in range(n + 1):
            p[n, m] *= schmidt[n, m]
            dp[n, m] *= schmidt[n, m]
    return p, dp


def igrf14_main_field(
    latitude_deg: float,
    longitude_east_deg: float,
    altitude_ellipsoid_m: float,
    when: date | datetime | float | int | str,
) -> IGRFVector:
    """Return IGRF-14 north/east/down main-field components at one location.

    Latitude is geodetic degrees north, longitude is degrees east, and altitude
    is metres above the WGS-84 ellipsoid. Dates are UTC calendar dates or a
    decimal year. The bundled coefficients cover 1900.0–2030.0; this function
    performs no online request or extrapolation beyond that range.
    """
    latitude = float(latitude_deg)
    longitude = float(longitude_east_deg)
    altitude = float(altitude_ellipsoid_m)
    year = _decimal_year(when)
    if not math.isfinite(latitude) or not -90.0 <= latitude <= 90.0:
        raise ValueError("latitude_deg must be finite and within [-90, 90]")
    if not math.isfinite(longitude) or not -180.0 <= longitude <= 360.0:
        raise ValueError("longitude_east_deg must be finite and within [-180, 360]")
    if not math.isfinite(altitude) or altitude <= -2_300_000.0:
        raise ValueError("altitude_ellipsoid_m must be finite and above -2300 km")

    epochs, coeff = _read_coefficients()
    right = int(np.searchsorted(epochs, year, side="right"))
    if right == 0:
        left, right = 0, 1
    elif right >= len(epochs):
        left, right = len(epochs) - 2, len(epochs) - 1
    else:
        left = right - 1
    weight = (year - epochs[left]) / (epochs[right] - epochs[left])

    lat = math.radians(latitude)
    lon = math.radians(longitude)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    a2, b2 = WGS84_A_KM**2, WGS84_B_KM**2
    one, two = a2 * cos_lat**2, b2 * sin_lat**2
    three = one + two
    rho = math.sqrt(three)
    height_km = altitude / 1000.0
    radius = math.sqrt(height_km * (height_km + 2.0 * rho) + (a2 * one + b2 * two) / three)
    cd = (height_km + rho) / radius
    sd = (a2 - b2) / rho * sin_lat * cos_lat / radius
    sin_lat_gc = sin_lat * cd - cos_lat * sd
    cos_lat_gc = cos_lat * cd + sin_lat * sd
    theta = math.atan2(cos_lat_gc, sin_lat_gc)
    # At an exact geographic pole, use the declared longitude to define a local
    # meridian and evaluate the finite limiting field just off the singular axis.
    theta_eval = min(max(theta, 1.0e-10), math.pi - 1.0e-10)
    p, dp = _schmidt_legendre(theta_eval, 13)
    sin_theta = math.sin(theta_eval)
    br = bt = bp = 0.0
    for (n, signed_m), epochs_values in coeff.items():
        if signed_m < 0:
            continue
        m = abs(signed_m)
        value = float(epochs_values[left] + weight * (epochs_values[right] - epochs_values[left]))
        g_value = value
        h_epochs = coeff.get((n, -m))
        h_value = 0.0 if h_epochs is None else float(h_epochs[left] + weight * (h_epochs[right] - h_epochs[left]))
        gh = g_value * math.cos(m * lon) + h_value * math.sin(m * lon)
        dgh = -g_value * math.sin(m * lon) + h_value * math.cos(m * lon)
        radial_scale = (REFERENCE_RADIUS_KM / radius) ** (n + 2)
        br += radial_scale * (n + 1) * gh * p[n, m]
        bt -= radial_scale * gh * dp[n, m]
        bp -= radial_scale * m * dgh * p[n, m] / sin_theta

    # Geocentric north/east/down followed by the geodetic normal rotation.
    north_gc, east, down_gc = -bt, bp, -br
    north = north_gc * cd + down_gc * sd
    down = down_gc * cd - north_gc * sd
    horizontal = math.hypot(north, east)
    total = math.hypot(horizontal, down)
    return IGRFVector(
        north, east, down, horizontal, total,
        math.degrees(math.atan2(east, north)),
        math.degrees(math.atan2(down, horizontal)),
        latitude, longitude, altitude, year,
    )

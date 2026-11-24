"""Reversible, assumption-recording survey corrections and resampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import csv
import io
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.interpolate import griddata

from engine.survey import BaseStationSeries, SurveyProfile, _utc_datetime


def _array_digest(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").copy()
    canonical[np.isnan(canonical)] = np.nan
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _profile_digest(profile: SurveyProfile) -> str:
    return str(profile.provenance["source_sha256"])


@dataclass(frozen=True)
class ProcessingProduct:
    """Named outputs plus their units, assumptions, parameters, and hashes."""

    name: str
    columns: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    parameters: Mapping[str, Any]
    assumptions: tuple[str, ...]
    input_sha256: str
    supporting_files: Mapping[str, bytes] = field(default_factory=dict)

    def __post_init__(self) -> None:
        clean_columns: dict[str, np.ndarray] = {}
        for name, values in self.columns.items():
            array = np.asarray(values, dtype=np.float64).copy()
            if array.ndim != 1 or np.any(np.isinf(array)):
                raise ValueError(f"Processing output {name} must be a one-dimensional finite-or-missing vector")
            array.setflags(write=False)
            clean_columns[str(name)] = array
        if set(clean_columns) != set(self.units):
            raise ValueError("Every processing output column must have exactly one unit label")
        if any(not isinstance(unit, str) or not unit.strip() for unit in self.units.values()):
            raise ValueError("Processing output units must be non-empty strings")
        clean_parameters = dict(self.parameters)
        json.dumps(clean_parameters, allow_nan=False)
        clean_supporting_files = {}
        for filename, content in self.supporting_files.items():
            if not isinstance(filename, str) or not filename or filename != filename.split("/")[-1]:
                raise ValueError("Supporting filenames must be safe bundle-local names")
            clean_supporting_files[filename] = bytes(content)
        object.__setattr__(self, "columns", clean_columns)
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "parameters", clean_parameters)
        object.__setattr__(self, "assumptions", tuple(str(item) for item in self.assumptions))
        object.__setattr__(self, "supporting_files", clean_supporting_files)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "assumptions": list(self.assumptions),
            "input_sha256": self.input_sha256,
            "output_sha256": {name: _array_digest(values) for name, values in self.columns.items()},
            "columns": {name: {"units": self.units[name]} for name in self.columns},
            "supporting_files": {
                name: hashlib.sha256(content).hexdigest() for name, content in self.supporting_files.items()
            },
        }


@dataclass(frozen=True)
class RegularProfile:
    distance_m: np.ndarray
    values: np.ndarray
    units: str
    name: str = "regular_profile"
    parameters: Mapping[str, Any] | None = None
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        distance = np.asarray(self.distance_m, dtype=np.float64).copy()
        values = np.asarray(self.values, dtype=np.float64).copy()
        if distance.ndim != 1 or values.shape != distance.shape or distance.size < 2:
            raise ValueError("Regular profile distance and values must be aligned vectors with at least two samples")
        if not np.all(np.isfinite(distance)) or not np.all(np.isfinite(values)):
            raise ValueError("Regular profile output must be finite")
        if not np.allclose(np.diff(distance), np.diff(distance)[0], rtol=1e-10, atol=1e-10):
            raise ValueError("Regular profile distance must be uniformly spaced")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("Regular profile units must be declared")
        distance.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "values", values)
        parameters = dict(self.parameters or {})
        json.dumps(parameters, allow_nan=False)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "assumptions", tuple(str(item) for item in self.assumptions))

    @property
    def spacing_m(self) -> float:
        return float(self.distance_m[1] - self.distance_m[0])


@dataclass(frozen=True)
class RegularGrid:
    x_m: np.ndarray
    y_m: np.ndarray
    values: np.ndarray
    units: str
    coordinate_reference: str = "unspecified"

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=np.float64).copy()
        y = np.asarray(self.y_m, dtype=np.float64).copy()
        values = np.asarray(self.values, dtype=np.float64).copy()
        if x.ndim != 1 or y.ndim != 1 or x.size < 2 or y.size < 2:
            raise ValueError("Grid axes must be one-dimensional with at least two coordinates each")
        if values.shape != (y.size, x.size) or np.any(np.isinf(values)):
            raise ValueError("Grid values must have shape (len(y_m), len(x_m)) and contain no infinities")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Grid coordinates must be finite")
        if not np.allclose(np.diff(x), np.diff(x)[0]) or not np.allclose(np.diff(y), np.diff(y)[0]):
            raise ValueError("Grid axes must be uniformly spaced")
        if not np.isfinite(values).any():
            raise ValueError("Grid must include at least one interpolated value")
        for array in (x, y, values):
            array.setflags(write=False)
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "y_m", y)
        object.__setattr__(self, "values", values)


def apply_gravity_reductions(
    profile: SurveyProfile,
    *,
    datum_elevation_m: float,
    bouguer_density_kg_m3: float | None = None,
) -> ProcessingProduct:
    """Apply free-air and optional infinite Bouguer-slab reductions to observed gravity.

    The result retains the measured channel. It does not subtract normal gravity,
    include terrain, curvature, tides, drift, or claim to be a complete Bouguer anomaly.
    """
    if profile.gravity_mgal is None:
        raise ValueError("Gravity reductions require the gravity_mgal input channel")
    datum = float(datum_elevation_m)
    if not math.isfinite(datum):
        raise ValueError("datum_elevation_m must be finite")
    height = profile.elevation_m - datum
    raw = np.asarray(profile.gravity_mgal)
    free_air = raw + 0.3086 * height
    columns: dict[str, np.ndarray] = {
        "gravity_observed_mgal": raw,
        "gravity_free_air_reduced_mgal": free_air,
    }
    units = {name: "mGal" for name in columns}
    parameters: dict[str, Any] = {
        "datum_elevation_m": datum,
        "free_air_gradient_mgal_per_m": 0.3086,
    }
    assumptions = [
        "Positive elevation above the declared reduction datum receives +0.3086 mGal per metre.",
        "Free-air reduction only; no normal-gravity, tide, drift, latitude, or curvature correction is applied.",
        "The result is reduced gravity and is not a complete free-air anomaly without a reference-gravity correction.",
        "No terrain correction is included.",
    ]
    if bouguer_density_kg_m3 is not None:
        density = float(bouguer_density_kg_m3)
        if not math.isfinite(density) or density <= 0.0:
            raise ValueError("bouguer_density_kg_m3 must be finite and positive")
        slab = 0.00004191 * density * height
        columns["gravity_bouguer_slab_reduced_mgal"] = free_air - slab
        units["gravity_bouguer_slab_reduced_mgal"] = "mGal"
        parameters.update({
            "bouguer_density_kg_m3": density,
            "bouguer_slab_coefficient_mgal_per_kg_m2": 0.00004191,
        })
        assumptions.append(
            "Infinite horizontal Bouguer slab is subtracted using the declared density; this is not a complete Bouguer anomaly."
        )
    else:
        parameters["bouguer_density_kg_m3"] = None
    return ProcessingProduct(
        "gravity_reduction", columns, units, parameters, tuple(assumptions), _profile_digest(profile)
    )


def _base_station_csv(base_station: BaseStationSeries) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["time_utc", "total_field_nt"])
    writer.writerows(zip(base_station.time_utc, (format(float(value), ".17g") for value in base_station.total_field_nt)))
    return output.getvalue().encode("utf-8")


def apply_magnetic_diurnal_correction(
    profile: SurveyProfile,
    base_station: BaseStationSeries,
    *,
    reference_time_utc: str,
) -> ProcessingProduct:
    """Remove base-station temporal variation without subtracting an IGRF/main field."""
    if profile.magnetic_anomaly_nt is None:
        raise ValueError("Diurnal correction requires the magnetic_anomaly_nt input channel")
    if profile.station_time_utc is None or any(value is None for value in profile.station_time_utc):
        raise ValueError("Diurnal correction requires station_time_utc for every profile station")
    reference_seconds = _utc_datetime(reference_time_utc, "reference_time_utc").timestamp()
    base_seconds = base_station.time_seconds
    if reference_seconds < base_seconds[0] or reference_seconds > base_seconds[-1]:
        raise ValueError("reference_time_utc is outside the base-station time span")
    station_seconds = np.asarray([
        _utc_datetime(value, "station_time_utc").timestamp() for value in profile.station_time_utc
    ])
    if np.any(station_seconds < base_seconds[0]) or np.any(station_seconds > base_seconds[-1]):
        raise ValueError("Profile station time is outside the base-station time span; extrapolation is disabled")
    reference_field = float(np.interp(reference_seconds, base_seconds, base_station.total_field_nt))
    base_at_station = np.interp(station_seconds, base_seconds, base_station.total_field_nt)
    raw = np.asarray(profile.magnetic_anomaly_nt)
    corrected = raw - (base_at_station - reference_field)
    output = "magnetic_diurnal_corrected_nt"
    return ProcessingProduct(
        "magnetic_diurnal_correction",
        {"magnetic_observed_nt": raw, output: corrected},
        {"magnetic_observed_nt": "nT total-field anomaly", output: "nT total-field anomaly"},
        {
            "reference_time_utc": _utc_datetime(reference_time_utc, "reference_time_utc").isoformat(),
            "reference_base_field_nt": reference_field,
            "base_station_source_sha256": base_station.provenance["source_sha256"],
            "interpolation": "linear; no time extrapolation",
        },
        (
            "Subtracts only base-station temporal change relative to the declared reference time.",
            "No IGRF or main-field subtraction is applied.",
            "Station and base timestamps must be UTC-offset ISO-8601; base data are linearly interpolated.",
        ),
        _profile_digest(profile),
        supporting_files={
            f"base_station_{base_station.provenance['source_sha256'][:12]}.csv": _base_station_csv(base_station)
        },
    )


def separate_polynomial_regional(
    distance_m: Sequence[float],
    values: Sequence[float],
    *,
    degree: int = 1,
    units: str,
) -> ProcessingProduct:
    """Fit a declared polynomial regional on profile distance and return residuals."""
    x = np.asarray(distance_m, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or data.shape != x.shape or x.size < 2:
        raise ValueError("distance_m and values must be aligned one-dimensional vectors")
    if not np.all(np.isfinite(x)):
        raise ValueError("distance_m must be finite")
    if not isinstance(degree, (int, np.integer)) or degree < 0 or degree > 5:
        raise ValueError("degree must be an integer from 0 through 5")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("units must be declared")
    valid = np.isfinite(data)
    if valid.sum() <= degree + 1:
        raise ValueError("Polynomial regional fit needs more finite observations than coefficients")
    if np.unique(x[valid]).size < degree + 1:
        raise ValueError("Polynomial regional fit requires enough distinct station distances")
    polynomial = np.polynomial.Polynomial.fit(x[valid], data[valid], deg=int(degree))
    regional = np.full(data.shape, np.nan, dtype=np.float64)
    regional[valid] = polynomial(x[valid])
    residual = data - regional
    suffix = "".join(char.lower() if char.isalnum() else "_" for char in units).strip("_")
    regional_name = f"regional_{suffix}"
    residual_name = f"residual_{suffix}"
    return ProcessingProduct(
        "polynomial_regional_residual",
        {regional_name: regional, residual_name: residual},
        {regional_name: units, residual_name: units},
        {"degree": int(degree), "distance_origin_m": float(np.mean(x[valid])), "fit": "least-squares polynomial"},
        (
            "Polynomial regional is fit across the supplied profile and is not a uniquely determined geological background.",
            "Missing channel samples are excluded from fitting and remain missing in both outputs.",
            "The input channel is preserved by the export bundle.",
        ),
        hashlib.sha256(np.asarray(x, dtype="<f8").tobytes() + np.asarray(data, dtype="<f8").tobytes()).hexdigest(),
    )


def interpolate_profile(
    distance_m: Sequence[float],
    values: Sequence[float],
    *,
    spacing_m: float,
    units: str,
) -> RegularProfile:
    """Linearly sample a profile on a regular grid, without extrapolating."""
    x = np.asarray(distance_m, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    spacing = float(spacing_m)
    if x.ndim != 1 or data.shape != x.shape or x.size < 2:
        raise ValueError("distance_m and values must be aligned one-dimensional vectors")
    if not np.all(np.isfinite(x)):
        raise ValueError("distance_m must be finite")
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    valid = np.isfinite(data)
    x_valid, data_valid = x[valid], data[valid]
    order = np.argsort(x_valid)
    x_valid, data_valid = x_valid[order], data_valid[order]
    if x_valid.size < 2:
        raise ValueError("Interpolation requires at least two finite observations")
    if np.any(np.diff(x_valid) <= 0.0):
        raise ValueError("Interpolation requires unique station distances for finite observations")
    count = int(math.floor((x_valid[-1] - x_valid[0]) / spacing + 1e-12)) + 1
    if count < 2:
        raise ValueError("spacing_m is too large to create two profile samples")
    grid = x_valid[0] + spacing * np.arange(count, dtype=np.float64)
    output = np.interp(grid, x_valid, data_valid)
    return RegularProfile(
        grid,
        output,
        units,
        name="linear_interpolation",
        parameters={"spacing_m": spacing, "source_distance_min_m": float(x_valid[0]), "source_distance_max_m": float(x_valid[-1])},
        assumptions=("Linear interpolation within the finite input interval; no extrapolation.", "Missing channel values are omitted before interpolation."),
    )


def grid_scattered_xyz(
    x_m: Sequence[float],
    y_m: Sequence[float],
    values: Sequence[float],
    *,
    spacing_x_m: float,
    spacing_y_m: float,
    units: str = "unspecified",
    coordinate_reference: str = "unspecified",
) -> RegularGrid:
    """Linearly grid XYZ samples; cells outside their convex hull remain NaN."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or data.shape != x.shape:
        raise ValueError("x_m, y_m, and values must be aligned one-dimensional vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Scattered coordinates must be finite")
    spacing_x, spacing_y = float(spacing_x_m), float(spacing_y_m)
    if not math.isfinite(spacing_x) or spacing_x <= 0.0:
        raise ValueError("spacing_x_m must be finite and positive")
    if not math.isfinite(spacing_y) or spacing_y <= 0.0:
        raise ValueError("spacing_y_m must be finite and positive")
    valid = np.isfinite(data)
    points = np.column_stack((x[valid], y[valid]))
    samples = data[valid]
    if np.unique(points, axis=0).shape[0] < 3:
        raise ValueError("Scattered gridding requires at least three unique finite points")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("Scattered gridding requires non-collinear points")
    nx = int(math.floor((x.max() - x.min()) / spacing_x + 1e-12)) + 1
    ny = int(math.floor((y.max() - y.min()) / spacing_y + 1e-12)) + 1
    if nx < 2 or ny < 2:
        raise ValueError("Grid spacing must produce at least two cells along each axis")
    gx = x.min() + spacing_x * np.arange(nx, dtype=np.float64)
    gy = y.min() + spacing_y * np.arange(ny, dtype=np.float64)
    xx, yy = np.meshgrid(gx, gy)
    grid_values = griddata(points, samples, (xx, yy), method="linear")
    return RegularGrid(gx, gy, grid_values, units, coordinate_reference)


def regular_grid_from_xyz(
    x_m: Sequence[float],
    y_m: Sequence[float],
    values: Sequence[float],
    *,
    units: str = "unspecified",
    coordinate_reference: str = "unspecified",
) -> RegularGrid:
    """Rebuild a complete regular grid from XYZ rows without interpolation."""
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or data.shape != x.shape:
        raise ValueError("x_m, y_m, and values must be aligned one-dimensional vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(data)):
        raise ValueError("Regular XYZ grids require finite coordinates and values")
    points = list(zip(x.tolist(), y.tolist()))
    if len(set(points)) != len(points):
        raise ValueError("Regular XYZ grid contains a duplicate coordinate pair")
    x_axis, y_axis = np.unique(x), np.unique(y)
    if x_axis.size < 2 or y_axis.size < 2 or x_axis.size * y_axis.size != x.size:
        raise ValueError("XYZ rows must form a complete rectangular grid")
    if not np.allclose(np.diff(x_axis), np.diff(x_axis)[0], rtol=1e-10, atol=1e-10):
        raise ValueError("Regular XYZ grid x coordinates must have uniform spacing")
    if not np.allclose(np.diff(y_axis), np.diff(y_axis)[0], rtol=1e-10, atol=1e-10):
        raise ValueError("Regular XYZ grid y coordinates must have uniform spacing")
    if not isinstance(units, str) or not units.strip():
        raise ValueError("Grid value units must be declared")
    if not isinstance(coordinate_reference, str) or not coordinate_reference.strip():
        raise ValueError("Grid coordinate_reference must be declared")
    x_lookup = {value: index for index, value in enumerate(x_axis.tolist())}
    y_lookup = {value: index for index, value in enumerate(y_axis.tolist())}
    grid = np.empty((y_axis.size, x_axis.size), dtype=np.float64)
    for x_value, y_value, sample in zip(x, y, data):
        grid[y_lookup[float(y_value)], x_lookup[float(x_value)]] = float(sample)
    return RegularGrid(x_axis, y_axis, grid, units, coordinate_reference)

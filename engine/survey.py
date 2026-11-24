"""Versioned, unit-explicit profile and base-station input records."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


SURVEY_SCHEMA_VERSION = 1
BASE_STATION_SCHEMA_VERSION = 1


def _read_csv_source(source: bytes | bytearray | str | Path) -> tuple[bytes, str | None]:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source), None
    if isinstance(source, Path):
        return source.read_bytes(), source.name
    if isinstance(source, str):
        candidate = Path(source)
        try:
            if candidate.is_file():
                return candidate.read_bytes(), candidate.name
        except OSError:
            pass
        return source.encode("utf-8"), None
    raise TypeError("CSV source must be bytes, text, or a path")


def _read_rows(payload: bytes) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Survey CSV must be UTF-8 encoded") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if not reader.fieldnames:
        raise ValueError("Survey CSV requires a header row")
    names = [str(name or "").strip().lower() for name in reader.fieldnames]
    if len(names) != len(set(names)):
        raise ValueError("Survey CSV has duplicate column names")
    rows = []
    for row in reader:
        if None in row:
            raise ValueError("Survey CSV rows must match the declared header columns")
        rows.append({names[index]: str(value or "").strip() for index, value in enumerate(row.values())})
    if not rows:
        raise ValueError("Survey CSV must contain at least two station rows")
    return names, rows


def _number(value: str, name: str, *, allow_missing: bool = False) -> float:
    if not value and allow_missing:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric" if value else f"{name} is required") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _readonly_vector(values: Sequence[float], name: str, *, allow_nan: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).copy()
    if array.ndim != 1 or (not allow_nan and not np.all(np.isfinite(array))):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    if allow_nan and np.any(np.isinf(array)):
        raise ValueError(f"{name} cannot contain infinite values")
    array.setflags(write=False)
    return array


def _utc_datetime(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp with UTC offset") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _iso_utc(value: str, name: str) -> str:
    return _utc_datetime(value, name).isoformat(timespec="auto")


def _validate_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("Survey metadata must include coordinate_reference, profile_azimuth_deg, and elevation_datum")
    result = dict(metadata)
    for key in ("coordinate_reference", "elevation_datum"):
        value = result.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Survey metadata requires a non-empty {key}")
        result[key] = value.strip()
    azimuth = _number(str(result.get("profile_azimuth_deg", "")), "profile_azimuth_deg")
    if not -180.0 <= azimuth <= 180.0:
        raise ValueError("profile_azimuth_deg must be between -180 and 180 degrees")
    result["profile_azimuth_deg"] = azimuth
    return result


@dataclass(frozen=True)
class SurveyProfile:
    """Raw profile observations with an explicit local coordinate and unit schema.

    Empty gravity or magnetic cells are retained as NaN and remain aligned to
    their original station rows. ``magnetic_anomaly_nt`` is the measured total-field
    anomaly in nT; it is not an IGRF-corrected value unless the source says so.
    """

    distance_m: np.ndarray
    elevation_m: np.ndarray
    gravity_mgal: np.ndarray | None
    magnetic_anomaly_nt: np.ndarray | None
    station_time_utc: tuple[str | None, ...] | None
    station_id: tuple[str, ...] | None
    metadata: Mapping[str, Any]
    provenance: Mapping[str, Any]
    schema_version: int = SURVEY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        distance = _readonly_vector(self.distance_m, "distance_m")
        elevation = _readonly_vector(self.elevation_m, "elevation_m")
        if distance.size < 2 or elevation.shape != distance.shape:
            raise ValueError("A survey profile needs at least two aligned distance/elevation stations")
        if np.unique(distance).size != distance.size:
            raise ValueError("distance_m stations must be unique")
        gravity = None if self.gravity_mgal is None else _readonly_vector(
            self.gravity_mgal, "gravity_mgal", allow_nan=True
        )
        magnetic = None if self.magnetic_anomaly_nt is None else _readonly_vector(
            self.magnetic_anomaly_nt, "magnetic_anomaly_nt", allow_nan=True
        )
        if gravity is not None and gravity.shape != distance.shape:
            raise ValueError("gravity_mgal must align with distance_m")
        if magnetic is not None and magnetic.shape != distance.shape:
            raise ValueError("magnetic_anomaly_nt must align with distance_m")
        if gravity is None and magnetic is None:
            raise ValueError("Survey profile requires at least one magnetic or gravity channel")
        if gravity is not None and not np.isfinite(gravity).any():
            raise ValueError("gravity_mgal must contain at least one finite observation")
        if magnetic is not None and not np.isfinite(magnetic).any():
            raise ValueError("magnetic_anomaly_nt must contain at least one finite observation")
        if self.station_time_utc is not None and len(self.station_time_utc) != distance.size:
            raise ValueError("station_time_utc must align with distance_m")
        if self.station_id is not None and len(self.station_id) != distance.size:
            raise ValueError("station_id must align with distance_m")
        object.__setattr__(self, "distance_m", distance)
        object.__setattr__(self, "elevation_m", elevation)
        object.__setattr__(self, "gravity_mgal", gravity)
        object.__setattr__(self, "magnetic_anomaly_nt", magnetic)
        metadata = _validate_metadata(self.metadata)
        provenance = dict(self.provenance)
        digest = provenance.get("source_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError("Survey provenance requires the source_sha256 digest")
        if not isinstance(provenance.get("source_name"), str) or not provenance["source_name"].strip():
            raise ValueError("Survey provenance requires source_name")
        if isinstance(self.schema_version, bool) or self.schema_version != SURVEY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported survey profile schema version: {self.schema_version!r}")
        json.dumps(metadata, allow_nan=False)
        json.dumps(provenance, allow_nan=False)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_csv(
        cls,
        source: bytes | bytearray | str | Path,
        *,
        metadata: Mapping[str, Any],
        source_name: str | None = None,
    ) -> "SurveyProfile":
        payload, path_name = _read_csv_source(source)
        names, rows = _read_rows(payload)
        required = {"distance_m", "elevation_m"}
        allowed = required | {"gravity_mgal", "magnetic_anomaly_nt", "station_time_utc", "station_id"}
        missing = required - set(names)
        unknown = set(names) - allowed
        if missing:
            raise ValueError(f"Survey CSV missing required columns: {', '.join(sorted(missing))}")
        if unknown:
            raise ValueError(f"Unsupported survey CSV columns: {', '.join(sorted(unknown))}")
        distance = [_number(row["distance_m"], "distance_m") for row in rows]
        elevation = [_number(row["elevation_m"], "elevation_m") for row in rows]
        if len(set(distance)) != len(distance):
            raise ValueError("distance_m stations must be unique")

        gravity = (
            [_number(row["gravity_mgal"], "gravity_mgal", allow_missing=True) for row in rows]
            if "gravity_mgal" in names else None
        )
        magnetic = (
            [_number(row["magnetic_anomaly_nt"], "magnetic_anomaly_nt", allow_missing=True) for row in rows]
            if "magnetic_anomaly_nt" in names else None
        )
        times = None
        if "station_time_utc" in names:
            times = tuple(
                _iso_utc(row["station_time_utc"], "station_time_utc") if row["station_time_utc"] else None
                for row in rows
            )
        station_ids = tuple(row["station_id"] for row in rows) if "station_id" in names else None
        actual_name = source_name or path_name or "uploaded-survey.csv"
        provenance = {
            "source_name": Path(actual_name).name,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "schema": "kuffgravmag-survey-profile",
            "schema_version": SURVEY_SCHEMA_VERSION,
            "parser": "KuffGravMag SurveyProfile.from_csv",
        }
        return cls(
            np.asarray(distance), np.asarray(elevation),
            None if gravity is None else np.asarray(gravity),
            None if magnetic is None else np.asarray(magnetic),
            times, station_ids, metadata, provenance,
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": "kuffgravmag-survey-profile",
            "schema_version": self.schema_version,
            "coordinate_reference": self.metadata["coordinate_reference"],
            "profile_azimuth_deg": self.metadata["profile_azimuth_deg"],
            "elevation_datum": self.metadata["elevation_datum"],
            "units": {
                "distance_m": "m",
                "elevation_m": "m",
                "gravity_mgal": "mGal",
                "magnetic_nt": "nT total-field anomaly",
                "station_time_utc": "ISO-8601 UTC",
            },
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
            "station_count": int(self.distance_m.size),
        }

    def to_dict(self) -> dict[str, Any]:
        def encode_channel(values):
            if values is None:
                return None
            return [None if math.isnan(float(value)) else float(value) for value in values]

        return {
            "schema_version": self.schema_version,
            "distance_m": self.distance_m.tolist(),
            "elevation_m": self.elevation_m.tolist(),
            "gravity_mgal": encode_channel(self.gravity_mgal),
            "magnetic_anomaly_nt": encode_channel(self.magnetic_anomaly_nt),
            "station_time_utc": None if self.station_time_utc is None else list(self.station_time_utc),
            "station_id": None if self.station_id is None else list(self.station_id),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SurveyProfile":
        if not isinstance(data, Mapping):
            raise ValueError("Survey profile record must be an object")
        version = data.get("schema_version")
        if isinstance(version, bool) or version != SURVEY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported survey profile schema version: {version!r}")

        def decode_channel(name):
            values = data.get(name)
            if values is None:
                return None
            return np.asarray([math.nan if value is None else float(value) for value in values], dtype=np.float64)

        raw_times = data.get("station_time_utc")
        raw_ids = data.get("station_id")
        return cls(
            distance_m=np.asarray(data.get("distance_m"), dtype=np.float64),
            elevation_m=np.asarray(data.get("elevation_m"), dtype=np.float64),
            gravity_mgal=decode_channel("gravity_mgal"),
            magnetic_anomaly_nt=decode_channel("magnetic_anomaly_nt"),
            station_time_utc=None if raw_times is None else tuple(
                None if value is None else _iso_utc(str(value), "station_time_utc") for value in raw_times
            ),
            station_id=None if raw_ids is None else tuple(str(value) for value in raw_ids),
            metadata=data.get("metadata", {}),
            provenance=data.get("provenance", {}),
            schema_version=version,
        )


@dataclass(frozen=True)
class BaseStationSeries:
    """Time-tagged total-field observations from a declared magnetic base station."""

    time_utc: tuple[str, ...]
    total_field_nt: np.ndarray
    provenance: Mapping[str, Any]
    schema_version: int = BASE_STATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        times = tuple(_iso_utc(value, "time_utc") for value in self.time_utc)
        values = _readonly_vector(self.total_field_nt, "total_field_nt")
        if len(times) < 2 or values.size != len(times):
            raise ValueError("Base-station series requires at least two aligned readings")
        seconds = np.asarray([_utc_datetime(value, "time_utc").timestamp() for value in times])
        if np.any(np.diff(seconds) <= 0.0):
            raise ValueError("Base-station time_utc values must be strictly increasing")
        provenance = dict(self.provenance)
        digest = provenance.get("source_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError("Base-station provenance requires the source_sha256 digest")
        json.dumps(provenance, allow_nan=False)
        object.__setattr__(self, "time_utc", times)
        object.__setattr__(self, "total_field_nt", values)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_csv(
        cls,
        source: bytes | bytearray | str | Path,
        *,
        source_name: str | None = None,
    ) -> "BaseStationSeries":
        payload, path_name = _read_csv_source(source)
        names, rows = _read_rows(payload)
        required = {"time_utc", "total_field_nt"}
        if required - set(names):
            raise ValueError("Base-station CSV requires time_utc and total_field_nt columns")
        unknown = set(names) - required
        if unknown:
            raise ValueError(f"Unsupported base-station columns: {', '.join(sorted(unknown))}")
        times = tuple(_iso_utc(row["time_utc"], "time_utc") for row in rows)
        values = [_number(row["total_field_nt"], "total_field_nt") for row in rows]
        actual_name = source_name or path_name or "uploaded-base-station.csv"
        provenance = {
            "source_name": Path(actual_name).name,
            "source_sha256": hashlib.sha256(payload).hexdigest(),
            "schema": "kuffgravmag-base-station",
            "schema_version": BASE_STATION_SCHEMA_VERSION,
        }
        return cls(times, np.asarray(values), provenance)

    @property
    def time_seconds(self) -> np.ndarray:
        return np.asarray([_utc_datetime(value, "time_utc").timestamp() for value in self.time_utc])


def _csv_number(value: float) -> str:
    number = float(value)
    return "" if math.isnan(number) else format(number, ".17g")


def export_processing_bundle(
    profile: SurveyProfile,
    products: Sequence[Any] = (),
    *,
    regular_profiles: Sequence[Any] = (),
) -> bytes:
    """Build a deterministic ZIP with station data and a JSON provenance manifest."""
    products = tuple(products)
    regular_profiles = tuple(regular_profiles)
    column_arrays: dict[str, np.ndarray] = {}
    column_units: dict[str, str] = {}
    supporting_files: dict[str, bytes] = {}
    for product in products:
        if not hasattr(product, "columns") or not hasattr(product, "to_manifest"):
            raise TypeError("Processing bundle products must be ProcessingProduct values")
        for name, values in product.columns.items():
            if name in column_arrays:
                if not np.array_equal(column_arrays[name], values, equal_nan=True):
                    raise ValueError(f"Conflicting processing product column: {name}")
                continue
            column_arrays[name] = np.asarray(values, dtype=np.float64)
            column_units[name] = str(product.units[name])
        for filename, content in getattr(product, "supporting_files", {}).items():
            content = bytes(content)
            if filename in {"profile.csv", "manifest.json"} or filename.startswith("regular_profile_"):
                raise ValueError(f"Processing support file uses a reserved bundle name: {filename}")
            if filename in supporting_files and supporting_files[filename] != content:
                raise ValueError(f"Conflicting processing support file: {filename}")
            supporting_files[filename] = content

    base_columns: dict[str, Sequence[Any]] = {
        "distance_m": profile.distance_m,
        "elevation_m": profile.elevation_m,
        "gravity_mgal": profile.gravity_mgal if profile.gravity_mgal is not None else [None] * profile.distance_m.size,
        "magnetic_anomaly_nt": profile.magnetic_anomaly_nt if profile.magnetic_anomaly_nt is not None else [None] * profile.distance_m.size,
        "station_time_utc": profile.station_time_utc if profile.station_time_utc is not None else [None] * profile.distance_m.size,
        "station_id": profile.station_id if profile.station_id is not None else [None] * profile.distance_m.size,
    }
    ordered_names = list(base_columns) + list(column_arrays)
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(ordered_names)
    for index in range(profile.distance_m.size):
        row = []
        for name in base_columns:
            value = base_columns[name][index]
            row.append(_csv_number(value) if isinstance(value, (float, np.floating, int, np.integer)) else (value or ""))
        for values in column_arrays.values():
            if values.shape != profile.distance_m.shape:
                raise ValueError("Every exported processing product must align with profile stations")
            row.append(_csv_number(values[index]))
        writer.writerow(row)

    regular_entries: list[tuple[str, bytes]] = []
    regular_manifest = []
    for index, regular in enumerate(regular_profiles, start=1):
        distance = np.asarray(regular.distance_m, dtype=np.float64)
        values = np.asarray(regular.values, dtype=np.float64)
        if distance.ndim != 1 or values.shape != distance.shape or not np.all(np.isfinite(distance)) or not np.all(np.isfinite(values)):
            raise ValueError("Regular-profile exports must contain aligned finite coordinates and values")
        filename = f"regular_profile_{index}.csv"
        regular_csv = io.StringIO(newline="")
        regular_writer = csv.writer(regular_csv, lineterminator="\n")
        regular_writer.writerow(["distance_m", "value"])
        regular_writer.writerows((_csv_number(x), _csv_number(value)) for x, value in zip(distance, values))
        regular_bytes = regular_csv.getvalue().encode("utf-8")
        regular_entries.append((filename, regular_bytes))
        regular_manifest.append({
            "file": filename,
            "name": str(getattr(regular, "name", f"regular_profile_{index}")),
            "units": str(regular.units),
            "spacing_m": float(regular.spacing_m),
            "sample_count": int(distance.size),
            "data_sha256": hashlib.sha256(
                np.asarray(distance, dtype="<f8").tobytes() + np.asarray(values, dtype="<f8").tobytes()
            ).hexdigest(),
            "parameters": dict(getattr(regular, "parameters", {}) or {}),
            "assumptions": list(getattr(regular, "assumptions", ())),
        })
    manifest = {
        "bundle_schema": "kuffgravmag-processing-bundle",
        "bundle_schema_version": 1,
        "profile": profile.to_manifest(),
        "processing_steps": [product.to_manifest() for product in products],
        "regular_profiles": regular_manifest,
        "exported_columns": {
            name: {"units": column_units[name]} for name in column_arrays
        },
    }
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n"
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        entries = [("profile.csv", output.getvalue().encode("utf-8"))]
        entries.extend(regular_entries)
        entries.extend(sorted(supporting_files.items()))
        entries.append(("manifest.json", manifest_text.encode("utf-8")))
        for filename, content in entries:
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zipped.writestr(info, content)
    return archive.getvalue()

"""Validated local XYZ point layers and explicit CRS coordinate conversion."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from pyproj import CRS, Transformer


MAX_SPATIAL_LAYER_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class SpatialPointLayer:
    """Read-only scattered XYZ values with explicit coordinate provenance."""

    name: str
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    units: str
    source_crs: str
    target_crs: str
    source_sha256: str
    x_unit: str
    y_unit: str


def transform_coordinates(
    x: Sequence[float],
    y: Sequence[float],
    source_crs: str,
    target_crs: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform X/Y arrays using explicit CRS definitions and always-XY order."""
    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.ndim != 1 or y_array.ndim != 1 or x_array.shape != y_array.shape:
        raise ValueError("CRS conversion requires aligned one-dimensional X and Y arrays")
    if x_array.size == 0 or not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("CRS conversion requires non-empty finite coordinates")
    if not isinstance(source_crs, str) or not source_crs.strip():
        raise ValueError("A declared source CRS is required for coordinate conversion")
    if not isinstance(target_crs, str) or not target_crs.strip():
        raise ValueError("A declared target CRS is required for coordinate conversion")
    try:
        source = CRS.from_user_input(source_crs.strip())
        target = CRS.from_user_input(target_crs.strip())
        transformer = Transformer.from_crs(source, target, always_xy=True)
        tx, ty = transformer.transform(x_array, y_array, errcheck=True)
    except Exception as exc:
        raise ValueError(f"Could not transform coordinates from {source_crs!r} to {target_crs!r}: {exc}") from exc
    tx = np.asarray(tx, dtype=np.float64)
    ty = np.asarray(ty, dtype=np.float64)
    if tx.shape != x_array.shape or ty.shape != y_array.shape or not np.all(np.isfinite(tx)) or not np.all(np.isfinite(ty)):
        raise ValueError("CRS conversion produced invalid coordinates")
    return tx, ty


def _read_source(source) -> tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
        name = "local-spatial-layer.xyz"
    elif isinstance(source, (str, Path)):
        path = Path(source)
        try:
            size = path.stat().st_size
            if size > MAX_SPATIAL_LAYER_BYTES:
                raise ValueError("Spatial layer is larger than the supported 32 MiB limit")
            with path.open("rb") as stream:
                payload = stream.read(MAX_SPATIAL_LAYER_BYTES + 1)
        except OSError as exc:
            raise ValueError(f"Could not read spatial layer: {exc}") from exc
        name = path.name
    else:
        raise TypeError("Spatial layer source must be bytes or a filesystem path")
    if len(payload) > MAX_SPATIAL_LAYER_BYTES:
        raise ValueError("Spatial layer is larger than the supported 32 MiB limit")
    return payload, name


def load_xyz_layer(
    source,
    *,
    source_crs: str,
    target_crs: str | None = None,
    units: str = "input units",
    name: str | None = None,
) -> SpatialPointLayer:
    """Load local XYZ text without network access and optionally convert its CRS.

    One initial non-numeric header row and comment lines beginning with ``#`` are
    accepted. Any malformed row after data starts is rejected to avoid silently
    dropping survey points.
    """
    if not isinstance(source_crs, str) or not source_crs.strip():
        raise ValueError("A declared source CRS is required for a local spatial layer")
    if target_crs is not None and (not isinstance(target_crs, str) or not target_crs.strip()):
        raise ValueError("Set a non-empty target CRS or leave it unspecified to keep source coordinates")
    payload, source_name = _read_source(source)
    if not payload:
        raise ValueError("Spatial layer is empty")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Spatial layer must be UTF-8 text") from exc
    x, y, values = [], [], []
    saw_data = False
    saw_header = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.replace(",", " ").split()
        if len(fields) < 3:
            if not saw_data and not saw_header:
                saw_header = True
                continue
            raise ValueError(f"Spatial layer row {line_number} must contain X, Y, and value columns")
        try:
            row_x, row_y, row_value = map(float, fields[:3])
        except ValueError as exc:
            if not saw_data and not saw_header:
                saw_header = True
                continue
            raise ValueError(f"Spatial layer row {line_number} contains non-numeric values") from exc
        if not np.all(np.isfinite([row_x, row_y, row_value])):
            raise ValueError(f"Spatial layer row {line_number} contains non-finite values")
        saw_data = True
        x.append(row_x)
        y.append(row_y)
        values.append(row_value)
    if len(x) < 3:
        raise ValueError("A local XYZ spatial layer requires at least three valid points")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target_label = source_crs.strip() if target_crs is None else target_crs.strip()
    if target_crs is not None:
        x, y = transform_coordinates(x, y, source_crs, target_label)
    else:
        # Validate the source definition even when preserving source coordinates.
        try:
            CRS.from_user_input(source_crs)
        except Exception as exc:
            raise ValueError(f"Could not parse source CRS {source_crs!r}: {exc}") from exc
    target = CRS.from_user_input(target_label)
    axes = list(target.axis_info)
    x_unit = axes[0].unit_name if axes else "CRS units"
    y_unit = axes[1].unit_name if len(axes) > 1 else x_unit
    for array in (x, y, values):
        array.setflags(write=False)
    return SpatialPointLayer(
        name=name or source_name,
        x=x,
        y=y,
        values=values,
        units=str(units).strip() or "input units",
        source_crs=CRS.from_user_input(source_crs).to_string(),
        target_crs=target.to_string(),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        x_unit=x_unit,
        y_unit=y_unit,
    )


def spatial_layer_csv(layer: SpatialPointLayer) -> bytes:
    """Return a stable UTF-8 CSV with explicit coordinate and value metadata."""
    output = io.StringIO(newline="")
    output.write(f"# source_crs: {layer.source_crs}\n")
    output.write(f"# target_crs: {layer.target_crs}\n")
    output.write(f"# units: {layer.units}\n")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow((f"x_{layer.x_unit}", f"y_{layer.y_unit}", f"value_{layer.units}"))
    writer.writerows(zip(layer.x.tolist(), layer.y.tolist(), layer.values.tolist()))
    return output.getvalue().encode("utf-8")

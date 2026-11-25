"""Bounded, loss-aware adapters for the KuffGeo shared model contract v1."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from engine.forward import Body2D


FORMAT = "KuffGeoSharedGeologicalModel"
SCHEMA_VERSION = 1
SCHEMA_REVISION = "1.0.0"
MAX_IMPORT_BYTES = 5 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_ENTITIES = 256
MAX_INTERFACES = 512
MAX_VERTICES_PER_GEOMETRY = 2048
MAX_TOTAL_VERTICES = 100_000
MAX_GEOMETRY_VALIDATION_OPERATIONS = 5_000_000
MAX_ABSOLUTE_COORDINATE_M = 1_000_000_000.0


class GeologicalExchangeError(ValueError):
    """A bounded input or an unsupported conversion with an actionable path."""


@dataclass(frozen=True)
class KgmImportResult:
    bodies: tuple[Body2D, ...]
    magnetic_field: dict[str, float] | None
    model_metadata: dict[str, Any]
    warnings: tuple[str, ...]
    source_sha256: str | None


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GeologicalExchangeError(f"Duplicate JSON object key: {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise GeologicalExchangeError(f"Non-finite JSON number {value!r} is not allowed.")


def _safe_tree(value, *, max_depth=MAX_JSON_DEPTH):
    stack = [(value, 0, "$" )]
    while stack:
        current, depth, path = stack.pop()
        if depth > max_depth:
            raise GeologicalExchangeError(f"{path}: maximum JSON nesting depth is {max_depth}.")
        if isinstance(current, dict):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise GeologicalExchangeError(f"{path}: object keys must be strings.")
                stack.append((item, depth + 1, f"{path}.{key}"))
        elif isinstance(current, list):
            for index, item in enumerate(current):
                stack.append((item, depth + 1, f"{path}[{index}]"))
        elif isinstance(current, float) and not math.isfinite(current):
            raise GeologicalExchangeError(f"{path}: number must be finite.")
        elif not (current is None or isinstance(current, (str, int, float, bool))):
            raise GeologicalExchangeError(f"{path}: value is not JSON-compatible.")


def _read_payload(source) -> tuple[dict[str, Any], bytes | None]:
    raw: bytes | None = None
    if isinstance(source, Path):
        try:
            if source.stat().st_size > MAX_IMPORT_BYTES:
                raise GeologicalExchangeError(f"Model file exceeds the {MAX_IMPORT_BYTES // (1024 * 1024)} MB import limit.")
            raw = source.read_bytes()
        except OSError as exc:
            raise GeologicalExchangeError(f"Could not read model file: {exc}") from exc
    elif isinstance(source, bytes):
        raw = source
    elif isinstance(source, str):
        raw = source.encode("utf-8")
    elif isinstance(source, Mapping):
        try:
            data = copy.deepcopy(dict(source))
            encoded = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise GeologicalExchangeError(f"Model object is not finite JSON: {exc}") from exc
        if len(encoded) > MAX_IMPORT_BYTES:
            raise GeologicalExchangeError(f"Model object exceeds the {MAX_IMPORT_BYTES // (1024 * 1024)} MB import limit.")
        _safe_tree(data)
        return data, None
    else:
        raise TypeError("Model input must be UTF-8 JSON bytes/text, a Path, or a mapping")

    if len(raw) > MAX_IMPORT_BYTES:
        raise GeologicalExchangeError(f"Model file exceeds the {MAX_IMPORT_BYTES // (1024 * 1024)} MB import limit.")
    try:
        data = json.loads(
            raw.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except GeologicalExchangeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise GeologicalExchangeError(f"Model file is not valid bounded UTF-8 JSON: {exc}") from exc
    _safe_tree(data)
    return data, raw


def _record(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GeologicalExchangeError(f"{path}: expected an object.")
    return value


def _finite(value: Any, path: str, *, limit=MAX_ABSOLUTE_COORDINATE_M) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeologicalExchangeError(f"{path}: expected a finite number.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise GeologicalExchangeError(f"{path}: expected a finite number within the supported bound.") from exc
    if not math.isfinite(result):
        raise GeologicalExchangeError(f"{path}: expected a finite number.")
    if abs(result) > limit:
        raise GeologicalExchangeError(f"{path}: magnitude exceeds the supported numeric bound {limit:g}.")
    return result


def _text(value: Any, path: str, *, maximum=128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise GeologicalExchangeError(f"{path}: expected non-empty text of at most {maximum} characters.")
    return value


def _identifier(value: Any, path: str) -> str:
    text = _text(value, path, maximum=128)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", text):
        raise GeologicalExchangeError(f"{path}: identifier contains unsupported characters.")
    return text


def _only_keys(value: Mapping[str, Any], path: str, allowed) -> None:
    unknown = set(value) - set(allowed)
    if unknown:
        raise GeologicalExchangeError(f"{path}: unsupported field {sorted(unknown)[0]!r}.")


def _validate_extensions(value, path):
    extensions = _record(value, path)
    if len(extensions) > 64:
        raise GeologicalExchangeError(f"{path}: expected at most 64 namespaced extension fields.")
    for key in extensions:
        if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,127}:[A-Za-z0-9._-]{1,128}", key):
            raise GeologicalExchangeError(f"{path}: extension keys must use a namespace:name form.")
    return extensions


def _point(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise GeologicalExchangeError(f"{path}: section coordinates must be [x, z] pairs in metres.")
    return _finite(value[0], f"{path}[0]"), _finite(value[1], f"{path}[1]")


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p, tol):
    return min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol


def _validate_ring(value: Any, path: str) -> list[tuple[float, float]]:
    if not isinstance(value, list) or not 4 <= len(value) <= MAX_VERTICES_PER_GEOMETRY + 1:
        raise GeologicalExchangeError(f"{path}: polygon ring must have 4 to {MAX_VERTICES_PER_GEOMETRY + 1} points including closure.")
    points = [_point(point, f"{path}[{index}]") for index, point in enumerate(value)]
    if points[0] != points[-1]:
        raise GeologicalExchangeError(f"{path}: polygon ring must be explicitly closed (last point equals first point).")
    if len(points) - 1 > MAX_VERTICES_PER_GEOMETRY:
        raise GeologicalExchangeError(f"{path}: polygon has too many vertices.")
    ring = points[:-1]
    if any(ring[index] == ring[(index + 1) % len(ring)] for index in range(len(ring))):
        raise GeologicalExchangeError(f"{path}: adjacent polygon vertices must be distinct.")
    span = max(max(p[0] for p in ring) - min(p[0] for p in ring), max(p[1] for p in ring) - min(p[1] for p in ring))
    if span <= 0:
        raise GeologicalExchangeError(f"{path}: polygon must have non-zero extent.")
    twice_area = sum(ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1] for i in range(len(ring)))
    if abs(twice_area) <= 64 * math.ulp(1.0) * span * span:
        raise GeologicalExchangeError(f"{path}: polygon must enclose non-zero area.")

    tol = 32 * math.ulp(1.0) * max(1.0, *(abs(c) for p in ring for c in p))
    area_tol = 64 * math.ulp(1.0) * max(1.0, span * span)
    count = len(ring)
    for i in range(count):
        a, b = ring[i], ring[(i + 1) % count]
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            c, d = ring[j], ring[(j + 1) % count]
            o1, o2, o3, o4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
            crossing = ((o1 > area_tol and o2 < -area_tol) or (o1 < -area_tol and o2 > area_tol)) and ((o3 > area_tol and o4 < -area_tol) or (o3 < -area_tol and o4 > area_tol))
            touching = (abs(o1) <= area_tol and _on_segment(a, b, c, tol)) or (abs(o2) <= area_tol and _on_segment(a, b, d, tol)) or (abs(o3) <= area_tol and _on_segment(c, d, a, tol)) or (abs(o4) <= area_tol and _on_segment(c, d, b, tol))
            if crossing or touching:
                raise GeologicalExchangeError(f"{path}: self-intersecting polygons are not supported.")
    return points


def _validate_polyhedron(vertices, faces, path):
    points = [tuple(_finite(value, f"{path}.vertices[{i}][{axis}]") for axis, value in enumerate(vertex)) for i, vertex in enumerate(vertices)]
    scale = max(1.0, *(max(axis) - min(axis) for axis in zip(*points)))
    plane_tolerance = 1e-9 * scale
    area_tolerance = 128 * math.ulp(1.0) * scale * scale
    used_vertices = set()
    edges = {}
    adjacency = {index: set() for index in range(len(faces))}
    face_set = set()
    volume6 = 0.0
    validation_operations = 0
    center = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))

    for face_index, face in enumerate(faces):
        if not isinstance(face, list) or len(face) < 3 or len(face) > min(len(points), 2048) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= len(points) for item in face) or len(set(face)) != len(face):
            raise GeologicalExchangeError(f"{path}.faces[{face_index}]: face needs 3 to 2048 distinct, in-range vertex indices.")
        validation_operations += len(face) ** 2
        if validation_operations > MAX_GEOMETRY_VALIDATION_OPERATIONS:
            raise GeologicalExchangeError(f"{path}: polyhedron topology exceeds the bounded geometry-validation budget.")
        canonical_face = tuple(sorted(face))
        if canonical_face in face_set:
            raise GeologicalExchangeError(f"{path}.faces[{face_index}]: duplicate face.")
        face_set.add(canonical_face)
        used_vertices.update(face)
        origin = points[face[0]]
        normal = None
        for offset in range(1, len(face) - 1):
            a = tuple(points[face[offset]][axis] - origin[axis] for axis in range(3))
            b = tuple(points[face[offset + 1]][axis] - origin[axis] for axis in range(3))
            candidate = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
            if math.sqrt(sum(value * value for value in candidate)) > area_tolerance:
                normal = candidate
                break
        if normal is None:
            raise GeologicalExchangeError(f"{path}.faces[{face_index}]: face must have non-zero area.")
        normal_length = math.sqrt(sum(value * value for value in normal))
        if any(abs(sum(normal[axis] * (points[index][axis] - origin[axis]) for axis in range(3))) / normal_length > plane_tolerance for index in face):
            raise GeologicalExchangeError(f"{path}.faces[{face_index}]: face vertices must be coplanar.")
        drop_axis = max(range(3), key=lambda axis: abs(normal[axis]))
        projected_axes = [axis for axis in range(3) if axis != drop_axis]
        projected = [[points[index][projected_axes[0]], points[index][projected_axes[1]]] for index in face]
        _validate_ring(projected + [projected[0]], f"{path}.faces[{face_index}]")
        for edge_index, start in enumerate(face):
            end = face[(edge_index + 1) % len(face)]
            key = (min(start, end), max(start, end))
            edges.setdefault(key, []).append((face_index, start, end))
        p0 = tuple(points[face[0]][axis] - center[axis] for axis in range(3))
        for offset in range(1, len(face) - 1):
            p1 = tuple(points[face[offset]][axis] - center[axis] for axis in range(3))
            p2 = tuple(points[face[offset + 1]][axis] - center[axis] for axis in range(3))
            cross = (p1[1] * p2[2] - p1[2] * p2[1], p1[2] * p2[0] - p1[0] * p2[2], p1[0] * p2[1] - p1[1] * p2[0])
            volume6 += sum(p0[axis] * cross[axis] for axis in range(3))

    if used_vertices != set(range(len(points))):
        raise GeologicalExchangeError(f"{path}: every polyhedron vertex must belong to a face.")
    for edge, uses in edges.items():
        if len(uses) != 2 or uses[0][1:] != uses[1][1:][::-1]:
            raise GeologicalExchangeError(f"{path}: polyhedron must be a closed, consistently oriented two-manifold (edge {edge} is invalid).")
        adjacency[uses[0][0]].add(uses[1][0])
        adjacency[uses[1][0]].add(uses[0][0])
    reached = set()
    pending = [0]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(adjacency[current] - reached)
    if len(reached) != len(faces):
        raise GeologicalExchangeError(f"{path}: polyhedron must be one connected solid.")
    volume_tolerance = 128 * math.ulp(1.0) * scale**3
    if abs(volume6) <= volume_tolerance:
        raise GeologicalExchangeError(f"{path}: polyhedron must enclose non-zero volume.")
    return validation_operations


def validate_shared_model(source) -> tuple[dict[str, Any], str | None]:
    """Parse and validate a v1 exchange document; return a defensive copy and exact source hash."""
    data, raw = _read_payload(source)
    if not isinstance(data, dict):
        raise GeologicalExchangeError("$: model root must be an object.")
    _only_keys(data, "$", ("format", "schemaVersion", "schemaRevision", "modelId", "name", "coordinateReference", "magneticField", "geometry", "provenance", "extensions"))
    if data.get("format") != FORMAT:
        raise GeologicalExchangeError("format: unrecognized shared geological model.")
    if isinstance(data.get("schemaVersion"), bool) or data.get("schemaVersion") != SCHEMA_VERSION or data.get("schemaRevision") != SCHEMA_REVISION:
        raise GeologicalExchangeError(f"schemaVersion/schemaRevision: unsupported exchange version {data.get('schemaVersion')!r}/{data.get('schemaRevision')!r}; this adapter reads {SCHEMA_VERSION}/{SCHEMA_REVISION}.")
    _identifier(data.get("modelId"), "modelId")
    _text(data.get("name"), "name", maximum=160)

    coordinate = _record(data.get("coordinateReference"), "coordinateReference")
    _only_keys(coordinate, "coordinateReference", ("axisOrder", "horizontal", "vertical"))
    if coordinate.get("axisOrder") != ["x", "y", "z"]:
        raise GeologicalExchangeError("coordinateReference.axisOrder: expected [x, y, z].")
    horizontal = _record(coordinate.get("horizontal"), "coordinateReference.horizontal")
    _only_keys(horizontal, "coordinateReference.horizontal", ("kind", "id", "unit", "axisOrder", "xAxis", "yAxis", "azimuthDegFromNorthClockwise"))
    if horizontal.get("kind") != "local_transect":
        raise GeologicalExchangeError("coordinateReference.horizontal: projected CRS transformation is not supported by this adapter; convert explicitly to a local transect in metres.")
    if horizontal.get("unit") != "m" or horizontal.get("axisOrder") != ["x", "y"] or horizontal.get("xAxis") != "along_profile" or horizontal.get("yAxis") != "left_of_profile":
        raise GeologicalExchangeError("coordinateReference.horizontal: expected metre axes [x, y] with x along profile and y left of profile.")
    _identifier(horizontal.get("id"), "coordinateReference.horizontal.id")
    if "azimuthDegFromNorthClockwise" not in horizontal:
        raise GeologicalExchangeError("coordinateReference.horizontal.azimuthDegFromNorthClockwise: specify a bearing or explicit null when unknown.")
    azimuth = horizontal.get("azimuthDegFromNorthClockwise")
    if azimuth is not None and not 0 <= _finite(azimuth, "coordinateReference.horizontal.azimuthDegFromNorthClockwise", limit=360) < 360:
        raise GeologicalExchangeError("coordinateReference.horizontal.azimuthDegFromNorthClockwise: expected a value from 0 inclusive to 360 exclusive, or null when unknown.")
    vertical = _record(coordinate.get("vertical"), "coordinateReference.vertical")
    _only_keys(vertical, "coordinateReference.vertical", ("datumId", "unit", "positiveDirection", "groundSurfaceCoordinateM"))
    _identifier(vertical.get("datumId"), "coordinateReference.vertical.datumId")
    if vertical.get("unit") != "m" or vertical.get("positiveDirection") not in ("up", "down"):
        raise GeologicalExchangeError("coordinateReference.vertical: expected metres and an explicit positiveDirection of 'up' or 'down'.")
    _finite(vertical.get("groundSurfaceCoordinateM"), "coordinateReference.vertical.groundSurfaceCoordinateM")

    field = data.get("magneticField")
    if "magneticField" in data:
        field = _record(field, "magneticField")
        _only_keys(field, "magneticField", ("intensity_nT", "inclination_deg", "declination_deg", "profileAzimuth_deg"))
        for key in ("intensity_nT", "inclination_deg", "declination_deg", "profileAzimuth_deg"):
            _finite(field.get(key), f"magneticField.{key}", limit=1_000_000)
        if field["intensity_nT"] <= 0 or not -90 <= field["inclination_deg"] <= 90 or not -180 <= field["declination_deg"] <= 180 or not 0 <= field["profileAzimuth_deg"] < 360:
            raise GeologicalExchangeError("magneticField: field intensity or orientation is outside its supported range.")
        if azimuth is not None and not math.isclose(
            float(azimuth), float(field["profileAzimuth_deg"]), rel_tol=0.0, abs_tol=1e-9
        ):
            raise GeologicalExchangeError(
                "magneticField.profileAzimuth_deg: must match the declared local-transect x-axis azimuth."
            )

    geometry = _record(data.get("geometry"), "geometry")
    _only_keys(geometry, "geometry", ("interfaces", "entities"))
    interfaces = geometry.get("interfaces")
    entities = geometry.get("entities")
    if not isinstance(interfaces, list) or len(interfaces) > MAX_INTERFACES:
        raise GeologicalExchangeError(f"geometry.interfaces: expected an array with at most {MAX_INTERFACES} entries.")
    if not isinstance(entities, list) or len(entities) > MAX_ENTITIES:
        raise GeologicalExchangeError(f"geometry.entities: expected an array with at most {MAX_ENTITIES} entries.")
    ids: set[str] = set()
    interface_ids: set[str] = set()
    total_vertices = 0
    for index, interface in enumerate(interfaces):
        path = f"geometry.interfaces[{index}]"
        interface = _record(interface, path)
        _only_keys(interface, path, ("id", "name", "coordinates", "extensions"))
        identifier = _identifier(interface.get("id"), f"{path}.id")
        _text(interface.get("name"), f"{path}.name")
        if identifier in ids:
            raise GeologicalExchangeError(f"{path}.id: duplicate stable identifier {identifier!r}.")
        ids.add(identifier)
        interface_ids.add(identifier)
        points = interface.get("coordinates")
        if not isinstance(points, list) or not 2 <= len(points) <= MAX_VERTICES_PER_GEOMETRY:
            raise GeologicalExchangeError(f"{path}.coordinates: expected 2 to {MAX_VERTICES_PER_GEOMETRY} points.")
        for point_index, point in enumerate(points):
            _point(point, f"{path}.coordinates[{point_index}]")
        if "extensions" in interface:
            _validate_extensions(interface["extensions"], f"{path}.extensions")
        total_vertices += len(points)

    layer_orders: set[int] = set()
    geometry_validation_operations = 0
    for index, entity in enumerate(entities):
        path = f"geometry.entities[{index}]"
        entity = _record(entity, path)
        _only_keys(entity, path, ("id", "name", "kind", "stratigraphicOrder", "topInterfaceId", "bottomInterfaceId", "geometry", "properties", "extensions"))
        identifier = _identifier(entity.get("id"), f"{path}.id")
        _text(entity.get("name"), f"{path}.name")
        if identifier in ids:
            raise GeologicalExchangeError(f"{path}.id: duplicate stable identifier {identifier!r}.")
        ids.add(identifier)
        kind = entity.get("kind")
        if kind not in ("layer", "body"):
            raise GeologicalExchangeError(f"{path}.kind: expected 'layer' or 'body'.")
        if kind == "layer":
            order = entity.get("stratigraphicOrder")
            if isinstance(order, bool) or not isinstance(order, int) or not 0 <= order < 256 or order in layer_orders:
                raise GeologicalExchangeError(f"{path}.stratigraphicOrder: expected a unique integer from 0 to 255.")
            layer_orders.add(order)
            for ref_name in ("topInterfaceId", "bottomInterfaceId"):
                ref = _identifier(entity.get(ref_name), f"{path}.{ref_name}")
                if ref not in interface_ids:
                    raise GeologicalExchangeError(f"{path}.{ref_name}: interface {ref!r} does not exist.")
        shape = _record(entity.get("geometry"), f"{path}.geometry")
        if shape.get("kind") == "section_polygon":
            _only_keys(shape, f"{path}.geometry", ("kind", "ring", "strikeExtent"))
            ring = _validate_ring(shape.get("ring"), f"{path}.geometry.ring")
            geometry_validation_operations += len(ring) ** 2
            if geometry_validation_operations > MAX_GEOMETRY_VALIDATION_OPERATIONS:
                raise GeologicalExchangeError("geometry: polygon topology exceeds the bounded geometry-validation budget.")
            total_vertices += len(ring)
            strike = _record(shape.get("strikeExtent"), f"{path}.geometry.strikeExtent")
            if strike.get("kind") == "infinite":
                _only_keys(strike, f"{path}.geometry.strikeExtent", ("kind",))
            else:
                _only_keys(strike, f"{path}.geometry.strikeExtent", ("kind", "yMinM", "yMaxM"))
            if strike.get("kind") == "finite":
                lower = _finite(strike.get("yMinM"), f"{path}.geometry.strikeExtent.yMinM")
                upper = _finite(strike.get("yMaxM"), f"{path}.geometry.strikeExtent.yMaxM")
                if lower >= upper:
                    raise GeologicalExchangeError(f"{path}.geometry.strikeExtent: finite strike requires yMinM < yMaxM.")
            elif strike.get("kind") != "infinite":
                raise GeologicalExchangeError(f"{path}.geometry.strikeExtent.kind: expected 'infinite' or 'finite'.")
        elif shape.get("kind") == "polyhedron3d":
            _only_keys(shape, f"{path}.geometry", ("kind", "vertices", "faces"))
            vertices, faces = shape.get("vertices"), shape.get("faces")
            if not isinstance(vertices, list) or not 4 <= len(vertices) <= 20_000 or not isinstance(faces, list) or not 4 <= len(faces) <= 20_000:
                raise GeologicalExchangeError(f"{path}.geometry: 3D geometry needs bounded vertices and faces.")
            if any(not isinstance(vertex, list) or len(vertex) != 3 for vertex in vertices):
                raise GeologicalExchangeError(f"{path}.geometry.vertices: expected [x, y, z] triples in metres.")
            geometry_validation_operations += _validate_polyhedron(vertices, faces, f"{path}.geometry")
            if geometry_validation_operations > MAX_GEOMETRY_VALIDATION_OPERATIONS:
                raise GeologicalExchangeError("geometry: 3D topology exceeds the bounded geometry-validation budget.")
            total_vertices += len(vertices)
        else:
            raise GeologicalExchangeError(f"{path}.geometry.kind: unsupported geometry representation.")
        properties = _record(entity.get("properties"), f"{path}.properties")
        _validate_properties(properties, f"{path}.properties")
        if "extensions" in entity:
            _validate_extensions(entity["extensions"], f"{path}.extensions")
    if total_vertices > MAX_TOTAL_VERTICES:
        raise GeologicalExchangeError(f"geometry: total vertex count exceeds {MAX_TOTAL_VERTICES}.")

    _validate_provenance(data.get("provenance"))
    if "extensions" in data:
        _validate_extensions(data["extensions"], "extensions")
    return copy.deepcopy(data), hashlib.sha256(raw).hexdigest() if raw is not None else None


def _validate_provenance(value):
    provenance = _record(value, "provenance")
    _only_keys(provenance, "provenance", ("createdAt", "sourceApplication", "sourceVersion", "sourceModelId", "sourceHashes", "transformations"))
    _text(provenance.get("createdAt"), "provenance.createdAt", maximum=64)
    _text(provenance.get("sourceApplication"), "provenance.sourceApplication", maximum=128)
    _text(provenance.get("sourceVersion"), "provenance.sourceVersion", maximum=64)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", provenance["createdAt"]):
        raise GeologicalExchangeError("provenance.createdAt: expected an ISO date-time with a UTC offset.")
    try:
        datetime.fromisoformat(provenance["createdAt"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise GeologicalExchangeError("provenance.createdAt: expected an ISO date-time.") from exc
    if "T" not in provenance["createdAt"]:
        raise GeologicalExchangeError("provenance.createdAt: expected an ISO date-time.")
    if "sourceModelId" in provenance:
        _identifier(provenance["sourceModelId"], "provenance.sourceModelId")
    hashes = provenance.get("sourceHashes", [])
    if not isinstance(hashes, list) or len(hashes) > 64:
        raise GeologicalExchangeError("provenance.sourceHashes: expected at most 64 source hashes.")
    for index, item in enumerate(hashes):
        item = _record(item, f"provenance.sourceHashes[{index}]")
        _only_keys(item, f"provenance.sourceHashes[{index}]", ("name", "sha256"))
        _text(item.get("name"), f"provenance.sourceHashes[{index}].name", maximum=256)
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise GeologicalExchangeError(f"provenance.sourceHashes[{index}].sha256: expected a lowercase SHA-256 hex digest.")
    transformations = provenance.get("transformations", [])
    if not isinstance(transformations, list) or len(transformations) > 128 or any(not isinstance(item, str) or not item.strip() or len(item) > 512 for item in transformations):
        raise GeologicalExchangeError("provenance.transformations: expected at most 128 bounded descriptions.")


def _validate_properties(properties, path):
    _only_keys(properties, path, ("density", "magneticSusceptibility", "remanence", "seismic", "elastic"))
    density = properties.get("density")
    if "density" in properties:
        density = _record(density, f"{path}.density")
        _only_keys(density, f"{path}.density", ("state", "basis", "value", "unit", "referenceDensity"))
        state, basis = density.get("state"), density.get("basis")
        if state not in ("known", "unknown") or basis not in ("contrast", "absolute") or density.get("unit") != "kg/m3":
            raise GeologicalExchangeError(f"{path}.density: expected a known/unknown absolute or contrast value in kg/m3.")
        if state == "known":
            density_value = _finite(density.get("value"), f"{path}.density.value")
            if basis == "absolute" and density_value <= 0:
                raise GeologicalExchangeError(f"{path}.density.value: absolute density must be positive.")
        elif "value" in density:
            raise GeologicalExchangeError(f"{path}.density.value: unknown density must not carry a numeric value; zero is a known value.")
        if basis == "absolute":
            reference = _record(density.get("referenceDensity"), f"{path}.density.referenceDensity")
            _only_keys(reference, f"{path}.density.referenceDensity", ("state", "value", "unit"))
            if reference.get("state") not in ("known", "unknown") or reference.get("unit") != "kg/m3":
                raise GeologicalExchangeError(f"{path}.density.referenceDensity: expected a known/unknown value in kg/m3.")
            if reference.get("state") == "known":
                value = _finite(reference.get("value"), f"{path}.density.referenceDensity.value")
                if value <= 0:
                    raise GeologicalExchangeError(f"{path}.density.referenceDensity.value: absolute reference density must be positive.")
            elif "value" in reference:
                raise GeologicalExchangeError(f"{path}.density.referenceDensity.value: unknown reference density must not carry a number.")
        elif "referenceDensity" in density:
            raise GeologicalExchangeError(f"{path}.density.referenceDensity: reference density is only valid for absolute density.")
    susceptibility = properties.get("magneticSusceptibility")
    if susceptibility is not None:
        susceptibility = _record(susceptibility, f"{path}.magneticSusceptibility")
        _only_keys(susceptibility, f"{path}.magneticSusceptibility", ("state", "value", "unit"))
        if susceptibility.get("state") not in ("known", "unknown") or susceptibility.get("unit") != "SI":
            raise GeologicalExchangeError(f"{path}.magneticSusceptibility: expected known/unknown SI susceptibility.")
        if susceptibility.get("state") == "known":
            _finite(susceptibility.get("value"), f"{path}.magneticSusceptibility.value")
        elif "value" in susceptibility:
            raise GeologicalExchangeError(f"{path}.magneticSusceptibility.value: unknown susceptibility must not carry a numeric value; zero is known.")
    remanence = properties.get("remanence")
    if remanence is not None:
        remanence = _record(remanence, f"{path}.remanence")
        _only_keys(remanence, f"{path}.remanence", ("state", "magnitude_A_per_m", "inclination_deg", "declination_deg", "orientationReference"))
        if remanence.get("state") == "known":
            magnitude = _finite(remanence.get("magnitude_A_per_m"), f"{path}.remanence.magnitude_A_per_m")
            inc = _finite(remanence.get("inclination_deg"), f"{path}.remanence.inclination_deg", limit=180)
            dec = _finite(remanence.get("declination_deg"), f"{path}.remanence.declination_deg", limit=360)
            if magnitude < 0 or not -90 <= inc <= 90 or not -180 <= dec <= 180 or remanence.get("orientationReference") != "geographic_north_inclination_positive_down":
                raise GeologicalExchangeError(f"{path}.remanence: invalid magnitude or geographic orientation convention.")
        elif remanence.get("state") != "unknown" or set(remanence) != {"state"}:
            raise GeologicalExchangeError(f"{path}.remanence: expected explicit known or unknown state; zero magnitude is known.")
    seismic = properties.get("seismic", {})
    if "seismic" in properties:
        seismic = _record(seismic, f"{path}.seismic")
        _only_keys(seismic, f"{path}.seismic", ("pWaveVelocity", "sWaveVelocity"))
        for key in ("pWaveVelocity", "sWaveVelocity"):
            item = seismic.get(key)
            if item is None:
                continue
            item = _record(item, f"{path}.seismic.{key}")
            _only_keys(item, f"{path}.seismic.{key}", ("state", "value", "unit"))
            expected_unit = "m/s"
            if item.get("unit") != expected_unit or item.get("state") not in ("known", "unknown"):
                raise GeologicalExchangeError(f"{path}.seismic.{key}: expected known/unknown velocity in m/s.")
            if item.get("state") == "known":
                if _finite(item.get("value"), f"{path}.seismic.{key}.value") <= 0:
                    raise GeologicalExchangeError(f"{path}.seismic.{key}.value: seismic velocity must be positive.")
            elif "value" in item:
                raise GeologicalExchangeError(f"{path}.seismic.{key}.value: unknown velocity must not carry a number.")
    elastic = properties.get("elastic", {})
    if "elastic" in properties:
        elastic = _record(elastic, f"{path}.elastic")
        _only_keys(elastic, f"{path}.elastic", ("bulkModulus", "shearModulus", "youngModulus", "poissonRatio"))
        for key in ("bulkModulus", "shearModulus", "youngModulus", "poissonRatio"):
            item = elastic.get(key)
            if item is None:
                continue
            item = _record(item, f"{path}.elastic.{key}")
            _only_keys(item, f"{path}.elastic.{key}", ("state", "value", "unit"))
            expected_unit = "1" if key == "poissonRatio" else "Pa"
            if item.get("unit") != expected_unit or item.get("state") not in ("known", "unknown"):
                raise GeologicalExchangeError(f"{path}.elastic.{key}: expected known/unknown value in {expected_unit}.")
            if item.get("state") == "known":
                value = _finite(item.get("value"), f"{path}.elastic.{key}.value")
                if key != "poissonRatio" and value <= 0:
                    raise GeologicalExchangeError(f"{path}.elastic.{key}.value: modulus must be positive.")
                if key == "poissonRatio" and not -1 < value < 0.5:
                    raise GeologicalExchangeError(f"{path}.elastic.{key}.value: Poisson ratio must be between -1 and 0.5.")
            elif "value" in item:
                raise GeologicalExchangeError(f"{path}.elastic.{key}.value: unknown value must not carry a number.")


def _canonical_z_to_kgm_depth(z: float, vertical: Mapping[str, Any]) -> float:
    surface = float(vertical["groundSurfaceCoordinateM"])
    height_up = z - surface if vertical["positiveDirection"] == "up" else surface - z
    return -height_up


def _density_contrast(properties: Mapping[str, Any], path: str) -> float:
    value = properties.get("density")
    if not isinstance(value, dict) or value.get("state") != "known":
        raise GeologicalExchangeError(f"{path}.density: KuffGravMag import requires a known density contrast; unknown is not equivalent to zero.")
    raw = float(value["value"])
    if value.get("basis") == "contrast":
        return raw
    reference = value.get("referenceDensity")
    if not isinstance(reference, dict) or reference.get("state") != "known":
        raise GeologicalExchangeError(f"{path}.density.referenceDensity: absolute density needs a known reference density before converting to contrast.")
    return raw - float(reference["value"])


def _known_value(properties, key, path, *, unit=None):
    value = properties.get(key)
    if not isinstance(value, dict) or value.get("state") != "known":
        raise GeologicalExchangeError(f"{path}.{key}: KuffGravMag import requires an explicit known value; unknown is not equivalent to zero.")
    if unit is not None and value.get("unit") != unit:
        raise GeologicalExchangeError(f"{path}.{key}.unit: expected {unit}.")
    return float(value["value"])


def import_shared_model(source) -> KgmImportResult:
    """Convert supported section polygons to Body2D instances without defaulting unknown properties."""
    data, source_sha256 = validate_shared_model(source)
    vertical = data["coordinateReference"]["vertical"]
    bodies: list[Body2D] = []
    warnings: list[str] = []
    for index, entity in enumerate(data["geometry"]["entities"]):
        path = f"geometry.entities[{index}]"
        shape = entity["geometry"]
        if shape["kind"] != "section_polygon":
            raise GeologicalExchangeError(f"{path}.geometry: KuffGravMag 2.5D supports section polygons with an explicit strike extent; polyhedron3d is not converted.")
        properties = entity["properties"]
        density = _density_contrast(properties, f"{path}.properties")
        susceptibility = _known_value(properties, "magneticSusceptibility", f"{path}.properties", unit="SI")
        remanence = properties.get("remanence")
        if not isinstance(remanence, dict) or remanence.get("state") != "known":
            raise GeologicalExchangeError(f"{path}.properties.remanence: KuffGravMag import needs a known remanence state; use a known 0 A/m value to mean no remanence.")
        strike = shape["strikeExtent"]
        if strike["kind"] == "infinite":
            y_min, y_max, strike_mode = -math.inf, math.inf, "infinite"
        else:
            y_min, y_max, strike_mode = float(strike["yMinM"]), float(strike["yMaxM"]), "finite"
        ring = shape["ring"]
        x = [point[0] for point in ring]
        z = [_canonical_z_to_kgm_depth(point[1], vertical) for point in ring]
        body = Body2D(
            entity["name"], x, z, density=density, k_succ=susceptibility,
            J_r=float(remanence["magnitude_A_per_m"]),
            I_r=float(remanence["inclination_deg"]),
            D_r=float(remanence["declination_deg"]),
            y_min=y_min, y_max=y_max, strike_mode=strike_mode,
        )
        body.exchange_id = entity["id"]
        body.exchange_kind = entity["kind"]
        body.exchange_stratigraphic_order = entity.get("stratigraphicOrder")
        body.exchange_top_interface_id = entity.get("topInterfaceId")
        body.exchange_bottom_interface_id = entity.get("bottomInterfaceId")
        body.exchange_properties = copy.deepcopy(properties)
        body.exchange_extensions = copy.deepcopy(entity.get("extensions", {}))
        body.exchange_original_ring = copy.deepcopy(ring)
        bodies.append(body)
        if properties.get("seismic") or properties.get("elastic"):
            warnings.append(f"{entity['id']}: seismic and elastic properties are retained as exchange metadata; KuffGravMag does not use them.")
    model_metadata = {
        "modelId": data["modelId"],
        "name": data["name"],
        "coordinateReference": copy.deepcopy(data["coordinateReference"]),
        "magneticField": copy.deepcopy(data.get("magneticField")),
        "geometryInterfaces": copy.deepcopy(data["geometry"]["interfaces"]),
        "provenance": copy.deepcopy(data["provenance"]),
        "extensions": copy.deepcopy(data.get("extensions", {})),
        "sourceSha256": source_sha256,
    }
    if source_sha256:
        hashes = model_metadata["provenance"].setdefault("sourceHashes", [])
        source_name = "shared-geological-model.json"
        if not any(item.get("sha256") == source_sha256 for item in hashes):
            hashes.append({"name": source_name, "sha256": source_sha256})
    field = data.get("magneticField")
    magnetic_field = None if field is None else {
        "intensity_nT": float(field["intensity_nT"]),
        "inclination_deg": float(field["inclination_deg"]),
        "declination_deg": float(field["declination_deg"]),
        "profileAzimuth_deg": float(field["profileAzimuth_deg"]),
    }
    if magnetic_field is None:
        warnings.append("The shared model has no inducing-field declaration. KuffGravMag keeps its current field setting active for calculations; a shared-model export will declare that active field as a new setting.")
    return KgmImportResult(tuple(bodies), magnetic_field, model_metadata, tuple(dict.fromkeys(warnings)), source_sha256)


def _body_ring(body: Body2D, vertical: Mapping[str, Any]) -> list[list[float]]:
    surface = float(vertical["groundSurfaceCoordinateM"])
    result = []
    for x, depth in zip(body.x, body.z):
        height_up = -float(depth)
        coordinate = surface + height_up if vertical["positiveDirection"] == "up" else surface - height_up
        result.append([float(x), coordinate])
    return result


def export_kgm_model(
    bodies,
    magnetic_field: Mapping[str, float] | None,
    *,
    model_id: str = "kgm:exported-model",
    name: str = "KuffGravMag model",
    coordinate_reference: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Export KuffGravMag bodies; imported layer metadata is preserved until its geometry changes."""
    metadata = dict(model_metadata or {})
    coordinate = copy.deepcopy(coordinate_reference or metadata.get("coordinateReference") or {
        "axisOrder": ["x", "y", "z"],
        "horizontal": {
            "kind": "local_transect", "id": "local:kgm-profile", "unit": "m",
            "axisOrder": ["x", "y"], "xAxis": "along_profile", "yAxis": "left_of_profile",
            "azimuthDegFromNorthClockwise": (magnetic_field or {}).get("profileAzimuth_deg"),
        },
        "vertical": {"datumId": "local:ground-surface", "unit": "m", "positiveDirection": "up", "groundSurfaceCoordinateM": 0},
    })
    vertical = coordinate["vertical"]
    horizontal = coordinate.get("horizontal")
    if magnetic_field is not None and isinstance(horizontal, dict) and horizontal.get("kind") == "local_transect":
        horizontal["azimuthDegFromNorthClockwise"] = float(magnetic_field["profileAzimuth_deg"]) % 360.0
    entities = []
    warnings = []
    for index, body in enumerate(bodies):
        ring = _body_ring(body, vertical)
        original_ring = getattr(body, "exchange_original_ring", None)
        geometry_unchanged = original_ring is not None and ring == original_ring
        original_kind = getattr(body, "exchange_kind", None)
        kind = original_kind if geometry_unchanged and original_kind in ("layer", "body") else "body"
        entity = {
            "id": getattr(body, "exchange_id", f"body:{index + 1}"),
            "name": str(body.name)[:128] or f"Body {index + 1}",
            "kind": kind,
            "geometry": {
                "kind": "section_polygon",
                "ring": ring,
                "strikeExtent": {"kind": "infinite"} if body.strike_mode == "infinite" else {"kind": "finite", "yMinM": float(body.y_min), "yMaxM": float(body.y_max)},
            },
            "properties": copy.deepcopy(getattr(body, "exchange_properties", {})),
            "extensions": copy.deepcopy(getattr(body, "exchange_extensions", {})),
        }
        if kind == "layer":
            entity["stratigraphicOrder"] = int(getattr(body, "exchange_stratigraphic_order"))
            entity["topInterfaceId"] = getattr(body, "exchange_top_interface_id")
            entity["bottomInterfaceId"] = getattr(body, "exchange_bottom_interface_id")
        elif original_kind == "layer":
            warnings.append(f"{entity['id']}: polygon geometry changed after import; exported as a body because the original layer interfaces may no longer match.")
        original_density = entity["properties"].get("density", {})
        reference = original_density.get("referenceDensity") if isinstance(original_density, dict) else None
        if isinstance(original_density, dict) and original_density.get("basis") == "absolute" and isinstance(reference, dict) and reference.get("state") == "known":
            entity["properties"]["density"] = {
                "state": "known", "basis": "absolute",
                "value": float(body.density) + float(reference["value"]),
                "unit": "kg/m3", "referenceDensity": copy.deepcopy(reference),
            }
        else:
            entity["properties"]["density"] = {"state": "known", "basis": "contrast", "value": float(body.density), "unit": "kg/m3"}
        entity["properties"]["magneticSusceptibility"] = {"state": "known", "value": float(body.k_succ), "unit": "SI"}
        entity["properties"]["remanence"] = {
            "state": "known", "magnitude_A_per_m": float(body.J_r),
            "inclination_deg": float(body.I_r), "declination_deg": float(body.D_r),
            "orientationReference": "geographic_north_inclination_positive_down",
        }
        entities.append(entity)

    interfaces = copy.deepcopy(metadata.get("geometryInterfaces", [])) if metadata else []
    payload = {
        "format": FORMAT,
        "schemaVersion": SCHEMA_VERSION,
        "schemaRevision": SCHEMA_REVISION,
        "modelId": str(metadata.get("modelId", model_id)),
        "name": str(metadata.get("name", name))[:160],
        "coordinateReference": coordinate,
        "geometry": {"interfaces": interfaces, "entities": entities},
        "provenance": copy.deepcopy(provenance or metadata.get("provenance") or {
            "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "sourceApplication": "KuffGravMag",
            "sourceVersion": "0.1.0",
        }),
        "extensions": copy.deepcopy(metadata.get("extensions", {})),
    }
    if magnetic_field is not None:
        payload["magneticField"] = {
            "intensity_nT": float(magnetic_field["intensity_nT"]),
            "inclination_deg": float(magnetic_field["inclination_deg"]),
            "declination_deg": float(magnetic_field["declination_deg"]),
            "profileAzimuth_deg": float(magnetic_field["profileAzimuth_deg"]),
        }
    validate_shared_model(payload)
    return payload, tuple(warnings)

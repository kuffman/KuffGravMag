"""Versioned JSON project persistence for KuffGravMag model inputs."""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from engine.forward import Body2D
from engine.survey import SurveyProfile


SCHEMA_VERSION = 2
MAX_SHARED_METADATA_BYTES = 4 * 1024 * 1024
MAX_BODY_EXCHANGE_BYTES = 256 * 1024


def _bounded_json_copy(value, label, max_bytes):
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} must contain finite JSON data") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes // 1024} KB limit")
    try:
        decoded = json.loads(encoded)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{label} is not bounded JSON data") from exc
    stack = [(decoded, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > 24:
            raise ValueError(f"{label} exceeds the maximum JSON nesting depth of 24")
        if isinstance(current, dict):
            if len(current) > 2048:
                raise ValueError(f"{label} contains too many object fields")
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            if len(current) > 20_000:
                raise ValueError(f"{label} contains too many array items")
            stack.extend((child, depth + 1) for child in current)
    return decoded


def _bound_to_json(value):
    value = float(value)
    if math.isinf(value):
        return "-inf" if value < 0.0 else "+inf"
    if not math.isfinite(value):
        raise ValueError("Strike bounds cannot be NaN")
    return value


def _bound_from_json(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("-inf", "-infinity"):
            return -math.inf
        if lowered in ("+inf", "inf", "+infinity", "infinity"):
            return math.inf
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Strike bounds must be numeric") from exc
    if math.isnan(result):
        raise ValueError("Strike bounds cannot be NaN")
    return result


def _body_to_dict(body):
    result = {
        "name": str(body.name),
        "x": np.asarray(body.x, dtype=np.float64).tolist(),
        "z": np.asarray(body.z, dtype=np.float64).tolist(),
        "density": float(body.density),
        "susceptibility": float(body.k_succ),
        "remanence_A_per_m": float(body.J_r),
        "remanent_inclination_deg": float(body.I_r),
        "remanent_declination_deg": float(body.D_r),
        "strike_mode": str(body.strike_mode),
        "strike_y_min_m": _bound_to_json(body.y_min),
        "strike_y_max_m": _bound_to_json(body.y_max),
        "color": str(body.color),
    }
    exchange = {}
    for key, attribute in (
        ("id", "exchange_id"), ("kind", "exchange_kind"),
        ("stratigraphicOrder", "exchange_stratigraphic_order"),
        ("topInterfaceId", "exchange_top_interface_id"),
        ("bottomInterfaceId", "exchange_bottom_interface_id"),
        ("properties", "exchange_properties"), ("extensions", "exchange_extensions"),
        ("originalRing", "exchange_original_ring"),
    ):
        if hasattr(body, attribute):
            exchange[key] = getattr(body, attribute)
    if exchange:
        result["exchange"] = _bounded_json_copy(exchange, "Body shared-model metadata", MAX_BODY_EXCHANGE_BYTES)
    return result


def project_to_dict(
    bodies, field_params, observations=None, *, survey_profile=None, processing_steps=None,
    shared_model_metadata=None,
):
    """Return a JSON-serializable project snapshot of all model settings."""
    observations = observations or {}
    field_params = dict(field_params or {})
    clean_observations = {}
    for key in ("x", "z", "gravity", "magnetic"):
        value = observations.get(key)
        clean_observations[key] = (
            None if value is None else np.asarray(value, dtype=np.float64).tolist()
        )
    survey_record = None
    if survey_profile is not None:
        survey_record = survey_profile.to_dict() if isinstance(survey_profile, SurveyProfile) else survey_profile
        if not isinstance(survey_record, dict):
            raise ValueError("survey_profile must be a SurveyProfile or mapping")
        json.dumps(survey_record, allow_nan=False)
    steps = list(processing_steps or ())
    if any(not isinstance(step, dict) for step in steps):
        raise ValueError("processing_steps must be a sequence of objects")
    json.dumps(steps, allow_nan=False)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "field_params": {str(key): float(value) for key, value in field_params.items()},
        "observations": clean_observations,
        "bodies": [_body_to_dict(body) for body in bodies],
        "survey_profile": survey_record,
        "processing_steps": steps,
    }
    if shared_model_metadata is not None:
        payload["shared_model_metadata"] = _bounded_json_copy(
            shared_model_metadata, "Shared geological model metadata", MAX_SHARED_METADATA_BYTES
        )
    return payload


def save_project(filename, bodies, field_params, observations=None, *, survey_profile=None, processing_steps=None,
                 shared_model_metadata=None):
    """Atomically write a project JSON file with no non-standard NaN/Infinity tokens."""
    payload = project_to_dict(
        bodies,
        field_params,
        observations,
        survey_profile=survey_profile,
        processing_steps=processing_steps,
        shared_model_metadata=shared_model_metadata,
    )
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(contents)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def project_sha256(
    bodies, field_params, observations=None, *, survey_profile=None, processing_steps=None,
    shared_model_metadata=None,
):
    """Return a stable SHA-256 fingerprint of the canonical project inputs."""
    payload = project_to_dict(
        bodies,
        field_params,
        observations,
        survey_profile=survey_profile,
        processing_steps=processing_steps,
        shared_model_metadata=shared_model_metadata,
    )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _read_source(source):
    if isinstance(source, dict):
        return source
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    if isinstance(source, Path):
        return json.loads(source.read_text(encoding="utf-8"))
    if isinstance(source, str):
        candidate = Path(source)
        try:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        except OSError:
            pass
        return json.loads(source)
    raise TypeError("Project input must be a path, JSON text, bytes, or mapping")


def _body_from_dict(data):
    if not isinstance(data, dict):
        raise ValueError("Each project body must be an object")
    x = data.get("x", data.get("x_verts"))
    z = data.get("z", data.get("z_verts"))
    if x is None or z is None:
        raise ValueError("Project bodies require x and z vertices")
    y_min = _bound_from_json(data.get("strike_y_min_m", data.get("y_min")), 0.0)
    y_max = _bound_from_json(data.get("strike_y_max_m", data.get("y_max")), 0.0)
    # Missing strike_mode is an old-project record. Body2D resolves only its
    # historical (0, 0) sentinel to infinite strike; explicit mode is honored.
    mode = data.get("strike_mode")
    body = Body2D(
        data.get("name", "Body"), x, z,
        density=data.get("density", 300.0),
        k_succ=data.get("susceptibility", data.get("k_succ", 0.02)),
        J_r=data.get("remanence_A_per_m", data.get("J_r", 0.0)),
        I_r=data.get("remanent_inclination_deg", data.get("I_r", 0.0)),
        D_r=data.get("remanent_declination_deg", data.get("D_r", 0.0)),
        y_min=y_min, y_max=y_max, color=data.get("color", "tab:blue"),
        strike_mode=mode,
    )
    exchange = data.get("exchange")
    if exchange is not None:
        exchange = _bounded_json_copy(exchange, "Body shared-model metadata", MAX_BODY_EXCHANGE_BYTES)
        allowed = {"id", "kind", "stratigraphicOrder", "topInterfaceId", "bottomInterfaceId", "properties", "extensions", "originalRing"}
        if not isinstance(exchange, dict) or set(exchange) - allowed:
            raise ValueError("Body shared-model metadata has unsupported fields")
        if not isinstance(exchange.get("id"), str) or not exchange["id"] or len(exchange["id"]) > 128:
            raise ValueError("Body shared-model metadata requires a stable identifier")
        if exchange.get("kind") not in ("layer", "body"):
            raise ValueError("Body shared-model metadata has an invalid geometry kind")
        if not isinstance(exchange.get("properties", {}), dict) or not isinstance(exchange.get("extensions", {}), dict):
            raise ValueError("Body shared-model properties and extensions must be objects")
        if exchange.get("originalRing") is not None and not isinstance(exchange["originalRing"], list):
            raise ValueError("Body shared-model original ring must be an array")
        body.exchange_id = exchange["id"]
        body.exchange_kind = exchange["kind"]
        if "stratigraphicOrder" in exchange:
            body.exchange_stratigraphic_order = exchange["stratigraphicOrder"]
        if "topInterfaceId" in exchange:
            body.exchange_top_interface_id = exchange["topInterfaceId"]
        if "bottomInterfaceId" in exchange:
            body.exchange_bottom_interface_id = exchange["bottomInterfaceId"]
        body.exchange_properties = exchange.get("properties", {})
        body.exchange_extensions = exchange.get("extensions", {})
        body.exchange_original_ring = exchange.get("originalRing")
    return body


def load_project(source):
    """Load a versioned or legacy JSON project into application-ready values."""
    data = _read_source(source)
    if not isinstance(data, dict):
        raise ValueError("Project root must be an object")
    version = data.get("schema_version", 0)
    if not isinstance(version, int) or version < 0 or version > SCHEMA_VERSION:
        raise ValueError(f"Unsupported project schema version: {version!r}")
    body_data = data.get("bodies", [])
    if not isinstance(body_data, list):
        raise ValueError("Project bodies must be a list")
    field_params = data.get("field_params", {})
    if not isinstance(field_params, dict):
        raise ValueError("Project field_params must be an object")
    field_params = {str(key): float(value) for key, value in field_params.items()}
    if not all(math.isfinite(value) for value in field_params.values()):
        raise ValueError("Project field parameters must be finite")
    raw_observations = data.get("observations", {})
    if not isinstance(raw_observations, dict):
        raise ValueError("Project observations must be an object")
    observations = {}
    for key in ("x", "z", "gravity", "magnetic"):
        value = raw_observations.get(key)
        observations[key] = None if value is None else np.asarray(value, dtype=np.float64)
    for coord in ("x", "z"):
        if observations[coord] is not None and not np.all(np.isfinite(observations[coord])):
            raise ValueError(f"Project observation {coord} values must be finite")
    survey_data = data.get("survey_profile")
    survey_profile = None if survey_data is None else SurveyProfile.from_dict(survey_data)
    processing_steps = data.get("processing_steps", [])
    if not isinstance(processing_steps, list) or any(not isinstance(step, dict) for step in processing_steps):
        raise ValueError("Project processing_steps must be a list of objects")
    json.dumps(processing_steps, allow_nan=False)
    shared_model_metadata = data.get("shared_model_metadata")
    if shared_model_metadata is not None:
        if not isinstance(shared_model_metadata, dict):
            raise ValueError("Shared geological model metadata must be an object")
        shared_model_metadata = _bounded_json_copy(
            shared_model_metadata, "Shared geological model metadata", MAX_SHARED_METADATA_BYTES
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "bodies": [_body_from_dict(body) for body in body_data],
        "field_params": field_params,
        "observations": observations,
        "survey_profile": survey_profile,
        "processing_steps": processing_steps,
    }
    if shared_model_metadata is not None:
        result["shared_model_metadata"] = shared_model_metadata
    return result

"""Portable project bundles with bounded, non-extracting ZIP parsing."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from engine.project import load_project


BUNDLE_FORMAT = "kuffgravmag-project-bundle"
BUNDLE_VERSION = 1
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_PROJECT_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 32 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 64
MAX_COMPRESSION_RATIO = 1000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _safe_attachment_name(name: str) -> str:
    if not isinstance(name, str) or not name or len(name) > 160:
        raise ValueError("Attachment names must be short, non-empty strings")
    if name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError("Attachment names must be plain filenames without path components")
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise ValueError("Attachment names cannot contain control characters")
    return name


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Project bundle JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Project bundle JSON contains invalid numeric value {value}")


def _json_load(payload: bytes, label: str):
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Project bundle {label} is not valid UTF-8 JSON") from exc


def build_project_bundle(project_data, attachments=None) -> bytes:
    """Build a reproducible ``.kgmproj`` ZIP from validated project JSON and files.

    Attachments are stored as opaque bytes and are never extracted while loading.
    Their manifest paths are generated from content hashes rather than filenames.
    """
    if not isinstance(project_data, dict):
        raise ValueError("Project data must be a JSON object")
    load_project(project_data)
    project_payload = _json_bytes(project_data)
    if len(project_payload) > MAX_PROJECT_BYTES:
        raise ValueError("Project JSON is larger than the supported bundle limit")

    source_files = attachments or {}
    if not isinstance(source_files, dict):
        raise ValueError("Project attachments must be a filename-to-bytes mapping")
    if len(source_files) > MAX_ARCHIVE_ENTRIES - 2:
        raise ValueError("Project bundle contains too many attachments")

    normalized = []
    seen_names = set()
    total_size = len(project_payload)
    for raw_name, raw_content in sorted(source_files.items()):
        name = _safe_attachment_name(raw_name)
        if name in seen_names:
            raise ValueError(f"Duplicate attachment name {name!r}")
        seen_names.add(name)
        if not isinstance(raw_content, (bytes, bytearray, memoryview)):
            raise ValueError(f"Attachment {name!r} must contain bytes")
        content = bytes(raw_content)
        if len(content) > MAX_ATTACHMENT_BYTES:
            raise ValueError(f"Attachment {name!r} is larger than the supported limit")
        total_size += len(content)
        if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("Project bundle attachments exceed the total size limit")
        digest = hashlib.sha256(content).hexdigest()
        archive_path = f"data/{digest}-{len(normalized):04d}.bin"
        normalized.append((name, archive_path, digest, content))

    manifest = {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "project": {"path": "project.json", "sha256": hashlib.sha256(project_payload).hexdigest(), "size": len(project_payload)},
        "attachments": [
            {"name": name, "path": archive_path, "sha256": digest, "size": len(content)}
            for name, archive_path, digest, content in sorted(normalized, key=lambda item: item[0])
        ],
    }
    manifest_payload = _json_bytes(manifest)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        entries = [("manifest.json", manifest_payload), ("project.json", project_payload)]
        entries.extend((path, content) for _name, path, _digest, content in normalized)
        for filename, content in sorted(entries):
            info = zipfile.ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            zipped.writestr(info, content)
    result = archive.getvalue()
    if len(result) > MAX_ARCHIVE_BYTES:
        raise ValueError("Project bundle exceeds the supported compressed size limit")
    return result


def save_project_bundle(filename, project_data, attachments=None):
    """Atomically write a portable project bundle."""
    path = Path(filename)
    payload = build_project_bundle(project_data, attachments)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return path


def _read_archive_source(source) -> bytes:
    if isinstance(source, (bytes, bytearray, memoryview)):
        payload = bytes(source)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"Could not read project bundle: {exc}") from exc
        if size > MAX_ARCHIVE_BYTES:
            raise ValueError("Project bundle is larger than the supported compressed size limit")
        with path.open("rb") as stream:
            payload = stream.read(MAX_ARCHIVE_BYTES + 1)
    else:
        raise TypeError("Project bundle input must be bytes or a filesystem path")
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError("Project bundle is larger than the supported compressed size limit")
    return payload


def _validate_archive_info(info: zipfile.ZipInfo):
    name = info.filename
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or name.endswith("/")
    ):
        raise ValueError("Project bundle contains an unsafe archive path")
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type not in (0, stat.S_IFREG):
        raise ValueError("Project bundle may contain regular files only")
    if info.flag_bits & 0x1:
        raise ValueError("Encrypted project bundle entries are unsupported")
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise ValueError("Project bundle uses an unsupported compression method")
    if info.file_size < 0 or info.compress_size < 0:
        raise ValueError("Project bundle contains invalid entry sizes")
    if info.file_size and not info.compress_size:
        raise ValueError("Project bundle contains an invalid compressed entry")
    if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        raise ValueError("Project bundle contains a suspiciously compressed entry")


def load_project_bundle(source):
    """Read and validate a bundle without writing or extracting any archive member."""
    payload = _read_archive_source(source)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload), "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("Project bundle is not a valid ZIP archive") from exc

    with archive:
        infos = archive.infolist()
        if len(infos) < 2 or len(infos) > MAX_ARCHIVE_ENTRIES:
            raise ValueError("Project bundle contains an invalid number of archive entries")
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise ValueError("Project bundle contains duplicate archive paths")
        for info in infos:
            _validate_archive_info(info)
        total_size = sum(info.file_size for info in infos)
        if total_size > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise ValueError("Project bundle exceeds the total uncompressed size limit")
        info_by_name = {info.filename: info for info in infos}
        if not {"manifest.json", "project.json"}.issubset(info_by_name):
            raise ValueError("Project bundle is missing its manifest or project data")
        if info_by_name["manifest.json"].file_size > 256 * 1024:
            raise ValueError("Project bundle manifest is too large")
        if info_by_name["project.json"].file_size > MAX_PROJECT_BYTES:
            raise ValueError("Project bundle project JSON is too large")
        try:
            manifest_payload = archive.read(info_by_name["manifest.json"])
            project_payload = archive.read(info_by_name["project.json"])
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise ValueError("Project bundle contains unreadable data") from exc

        manifest = _json_load(manifest_payload, "manifest")
        if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
            raise ValueError("Unsupported project bundle format")
        if type(manifest.get("version")) is not int or manifest.get("version") != BUNDLE_VERSION:
            raise ValueError(f"Unsupported project bundle version: {manifest.get('version')!r}")
        project_record = manifest.get("project")
        if not isinstance(project_record, dict) or project_record.get("path") != "project.json":
            raise ValueError("Project bundle manifest has an invalid project record")
        if project_record.get("size") != len(project_payload) or project_record.get("sha256") != hashlib.sha256(project_payload).hexdigest():
            raise ValueError("Project bundle project checksum does not match")
        project_data = _json_load(project_payload, "project")
        project = load_project(project_data)

        attachment_records = manifest.get("attachments")
        if not isinstance(attachment_records, list) or len(attachment_records) > MAX_ARCHIVE_ENTRIES - 2:
            raise ValueError("Project bundle manifest has an invalid attachment list")
        attachments = {}
        expected_paths = {"manifest.json", "project.json"}
        for record in attachment_records:
            if not isinstance(record, dict):
                raise ValueError("Project bundle attachment records must be objects")
            name = _safe_attachment_name(record.get("name"))
            path = record.get("path")
            digest = record.get("sha256")
            size = record.get("size")
            if not isinstance(path, str) or not re.fullmatch(r"data/[0-9a-f]{64}-[0-9]{4}\.bin", path):
                raise ValueError("Project bundle manifest contains an unsafe attachment path")
            if path in expected_paths or name in attachments:
                raise ValueError("Project bundle contains duplicate attachment metadata")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise ValueError("Project bundle attachment checksum is invalid")
            if type(size) is not int or size < 0 or size > MAX_ATTACHMENT_BYTES:
                raise ValueError("Project bundle attachment size is invalid")
            if path not in info_by_name or info_by_name[path].file_size != size:
                raise ValueError(f"Project bundle attachment {name!r} is missing or has an invalid size")
            expected_paths.add(path)
            try:
                content = archive.read(info_by_name[path])
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise ValueError(f"Project bundle attachment {name!r} cannot be read") from exc
            if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Project bundle attachment {name!r} checksum does not match")
            attachments[name] = content

        if set(info_by_name) != expected_paths:
            raise ValueError("Project bundle contains unlisted or unsupported files")

    return {
        "project": project,
        "project_data": project_data,
        "attachments": attachments,
        "manifest": manifest,
    }

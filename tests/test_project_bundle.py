"""Security and round-trip checks for portable project bundles."""

import io
import hashlib
import json
import zipfile

import numpy as np
import pytest

from engine.forward import Body2D
from engine.project import project_to_dict
from engine.project_bundle import build_project_bundle, load_project_bundle


def _project():
    body = Body2D(
        "Bundle body",
        [-100.0, 100.0, 100.0, -100.0, -100.0],
        [200.0, 200.0, 400.0, 400.0, 200.0],
        density=320.0,
        y_min=-250.0,
        y_max=350.0,
    )
    return project_to_dict(
        [body],
        {"F": 48_500.0, "Inc": 7.0, "Dec": 23.0, "Alpha": 45.0},
        {"x": [0.0, 50.0], "z": [0.0, 4.0], "gravity": [0.3, 0.2]},
    )


def _archive(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, payload in entries:
            zipped.writestr(name, payload)
    return stream.getvalue()


def test_project_bundle_round_trip_preserves_project_and_embedded_bytes():
    raw_grid = b"x_m,y_m,value\n0,0,1.5\n100,0,2.0\n"
    bundle = build_project_bundle(_project(), {"synthetic-grid.csv": raw_grid})
    loaded = load_project_bundle(bundle)

    assert loaded["project"]["bodies"][0].name == "Bundle body"
    assert loaded["project"]["field_params"]["Inc"] == pytest.approx(7.0)
    np.testing.assert_array_equal(loaded["project"]["observations"]["x"], [0.0, 50.0])
    assert loaded["attachments"] == {"synthetic-grid.csv": raw_grid}
    assert loaded["manifest"]["format"] == "kuffgravmag-project-bundle"


def test_project_bundle_is_byte_reproducible():
    files = {"layer.xyz": b"0 0 1\n1 0 2\n0 1 3\n"}
    assert build_project_bundle(_project(), files) == build_project_bundle(_project(), files)


def test_project_bundle_supports_distinct_names_with_identical_content():
    content = b"same embedded source bytes"
    bundle = build_project_bundle(_project(), {"first.csv": content, "second.csv": content})
    loaded = load_project_bundle(bundle)

    assert loaded["attachments"] == {"first.csv": content, "second.csv": content}


@pytest.mark.parametrize("name", ["../outside.csv", "/absolute.csv", "folder\\file.csv", "", ".."])
def test_project_bundle_writer_rejects_attachment_path_components(name):
    with pytest.raises(ValueError, match="filename|Attachment"):
        build_project_bundle(_project(), {name: b"x"})


def test_project_bundle_reader_rejects_path_traversal_and_unlisted_members():
    archive = _archive([
        ("manifest.json", b"{}"),
        ("project.json", json.dumps(_project()).encode()),
        ("../outside.csv", b"should never be extracted"),
    ])
    with pytest.raises(ValueError, match="unsafe archive path"):
        load_project_bundle(archive)


def test_project_bundle_reader_rejects_duplicate_archive_paths():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr("manifest.json", b"{}")
        zipped.writestr("project.json", json.dumps(_project()).encode())
        with pytest.warns(UserWarning, match="Duplicate name"):
            zipped.writestr("project.json", json.dumps(_project()).encode())
    with pytest.raises(ValueError, match="duplicate archive paths"):
        load_project_bundle(stream.getvalue())


def test_project_bundle_reader_rejects_unlisted_and_corrupt_attachments():
    valid = build_project_bundle(_project(), {"layer.csv": b"x,y,v\n0,0,1\n"})
    with zipfile.ZipFile(io.BytesIO(valid), "r") as source:
        entries = [(name, source.read(name)) for name in source.namelist()]
    corrupt = [
        (name, b"z" * len(content) if name.startswith("data/") else content)
        for name, content in entries
    ]
    with pytest.raises(ValueError, match="checksum does not match"):
        load_project_bundle(_archive(corrupt))

    unlisted = _archive(entries + [("notes.txt", b"not in manifest")])
    with pytest.raises(ValueError, match="unlisted or unsupported"):
        load_project_bundle(unlisted)


def test_project_bundle_reader_rejects_high_ratio_compression():
    archive = _archive([
        ("manifest.json", b"{}"),
        ("project.json", json.dumps(_project()).encode()),
        ("data/" + "a" * 64 + ".bin", b"0" * 1_100_000),
    ])
    with pytest.raises(ValueError, match="suspiciously compressed"):
        load_project_bundle(archive)


def test_project_bundle_reader_rejects_duplicate_json_keys():
    project = b'{"schema_version":2,"schema_version":2,"bodies":[]}'
    manifest = json.dumps({
        "format": "kuffgravmag-project-bundle",
        "version": 1,
        "project": {
            "path": "project.json",
            "sha256": hashlib.sha256(project).hexdigest(),
            "size": len(project),
        },
        "attachments": [],
    }).encode()
    archive = _archive([("manifest.json", manifest), ("project.json", project)])

    with pytest.raises(ValueError, match="duplicate key"):
        load_project_bundle(archive)

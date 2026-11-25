"""Atomic local recovery snapshots using the normal project schema."""

from __future__ import annotations

from pathlib import Path

from engine.project import load_project, save_project


def save_recovery_snapshot(
    filename,
    bodies,
    field_params,
    observations=None,
    *,
    survey_profile=None,
    processing_steps=None,
):
    """Persist the current project inputs to a recoverable schema-compatible file."""
    return save_project(
        filename,
        bodies,
        field_params,
        observations,
        survey_profile=survey_profile,
        processing_steps=processing_steps,
    )


def load_recovery_snapshot(filename):
    return load_project(Path(filename))


def clear_recovery_snapshot(filename):
    Path(filename).unlink(missing_ok=True)


def recovery_is_newer(recovery_path, project_path=None):
    """Return whether a recovery snapshot exists and is newer than the saved project."""
    recovery = Path(recovery_path)
    if not recovery.is_file():
        return False
    if project_path is None:
        return True
    project = Path(project_path)
    if not project.is_file():
        return True
    return recovery.stat().st_mtime_ns > project.stat().st_mtime_ns

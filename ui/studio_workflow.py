"""Pure helpers for handing processed survey residuals to model workflows."""

from __future__ import annotations

import numpy as np


def prepare_model_observations(profile, channel, values, *, source_column, units):
    """Build finite model inputs from a selected profile residual.

    Survey elevation is positive upward; the model cross section uses depth
    positive downward. Coordinates are therefore negated relative to the
    declared zero of the survey elevation datum.
    """
    if channel not in {"gravity", "magnetic"}:
        raise ValueError("Model handoff channel must be gravity or magnetic")
    expected_units = "mGal" if channel == "gravity" else "nT total-field anomaly"
    if units != expected_units:
        raise ValueError(f"{channel} model inputs must use {expected_units}")
    if not str(source_column).startswith("residual_"):
        raise ValueError("Only an explicitly separated regional residual can be handed to inversion")

    distance = np.asarray(profile.distance_m, dtype=np.float64)
    elevation = np.asarray(profile.elevation_m, dtype=np.float64)
    observations = np.asarray(values, dtype=np.float64)
    if observations.shape != distance.shape:
        raise ValueError("Residual values must align with survey stations")
    valid = np.isfinite(distance) & np.isfinite(elevation) & np.isfinite(observations)
    if int(valid.sum()) < 2:
        raise ValueError("At least two finite residual stations are required for model plotting")

    retained = int(valid.sum())
    omitted = int(valid.size - retained)
    selected = observations[valid].copy()
    model_observations = {
        "x": distance[valid].copy(),
        "z": -elevation[valid].copy(),
        "gravity": selected if channel == "gravity" else None,
        "magnetic": selected if channel == "magnetic" else None,
    }
    record = {
        "name": "model_observation_handoff",
        "parameters": {
            "channel": channel,
            "source_column": str(source_column),
            "units": expected_units,
            "retained_station_count": retained,
            "omitted_missing_station_count": omitted,
            "model_vertical_coordinate": "z_m = -elevation_m; depth is positive down",
            "elevation_datum": str(profile.metadata.get("elevation_datum", "unspecified")),
            "survey_source_sha256": str(profile.provenance["source_sha256"]),
        },
        "assumptions": [
            "The selected polynomial residual is treated as the observed model channel.",
            "Survey elevation is negated into the model's positive-down vertical coordinate.",
            "The residual is not proof that the regional separation is geologically unique.",
        ],
    }
    return model_observations, record

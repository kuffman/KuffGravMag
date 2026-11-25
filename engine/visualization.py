"""Geometry helpers for rendering finite-strike potential-field models."""

from __future__ import annotations

import math

import numpy as np


def finite_prism_faces(body):
    """Return the exact extrusion faces for a closed finite-strike cross section.

    Coordinates are (profile distance X, strike Y, depth Z), with depth positive
    downward to match the desktop cross-section plot. Infinite-strike bodies are
    intentionally excluded because a finite prism would misstate their geometry.
    """
    if getattr(body, "strike_mode", None) != "finite":
        raise ValueError("Only finite-strike bodies have a finite 3D prism view")
    x = np.asarray(body.x, dtype=np.float64)
    z = np.asarray(body.z, dtype=np.float64)
    y_min, y_max = float(body.y_min), float(body.y_max)
    if x.ndim != 1 or z.ndim != 1 or x.size != z.size or x.size < 4:
        raise ValueError("A finite prism needs at least three closed cross-section vertices")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
        raise ValueError("Prism vertices must be finite")
    if not (math.isfinite(y_min) and math.isfinite(y_max) and y_max > y_min):
        raise ValueError("A finite prism needs ordered strike bounds with positive width")
    if not (x[0] == x[-1] and z[0] == z[-1]):
        raise ValueError("The model cross section must be closed")

    ring_x, ring_z = x[:-1], z[:-1]
    at_min = np.column_stack((ring_x, np.full(ring_x.size, y_min), ring_z))
    at_max = np.column_stack((ring_x, np.full(ring_x.size, y_max), ring_z))
    faces = [at_min, at_max[::-1]]
    for index in range(ring_x.size):
        following = (index + 1) % ring_x.size
        faces.append(np.array([
            at_min[index], at_max[index], at_max[following], at_min[following],
        ]))
    return tuple(faces)

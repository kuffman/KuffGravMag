"""Terrain residual corrections from a regular, datum-controlled DEM.

The DEM is represented as square/rectangular cells whose elevations are
positive upward. Each cell contributes the density contrast between the DEM
surface and a user-declared horizontal reference plane. The returned quantity
is the positive-down vertical attraction of those residual prisms in mGal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

_G = 6.67430e-11
_SI_TO_MGAL = 1.0e5


@dataclass(frozen=True)
class TerrainDEM:
    """A complete or explicitly masked regular DEM on a local metric grid.

    ``x_m`` and ``y_m`` are cell-centre coordinates in a projected Cartesian
    CRS (or a declared local east/north frame), in metres. Elevations are
    positive upward and share the declared vertical datum.
    """

    x_m: np.ndarray
    y_m: np.ndarray
    elevation_m: np.ndarray
    coordinate_reference: str
    elevation_datum: str

    def __post_init__(self) -> None:
        x = np.asarray(self.x_m, dtype=np.float64).copy()
        y = np.asarray(self.y_m, dtype=np.float64).copy()
        elevation = np.asarray(self.elevation_m, dtype=np.float64).copy()
        if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
            raise ValueError("DEM axes must be one-dimensional and contain at least two cell centres")
        if elevation.shape != (y.size, x.size) or np.isinf(elevation).any():
            raise ValueError("DEM elevations must have shape (len(y_m), len(x_m)) and no infinities")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("DEM coordinates must be finite")
        if np.any(np.diff(x) <= 0.0) or np.any(np.diff(y) <= 0.0):
            raise ValueError("DEM axes must be strictly increasing")
        if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-10, atol=1e-8):
            raise ValueError("DEM x coordinates must be uniformly spaced")
        if not np.allclose(np.diff(y), np.diff(y)[0], rtol=1e-10, atol=1e-8):
            raise ValueError("DEM y coordinates must be uniformly spaced")
        if not np.isfinite(elevation).any():
            raise ValueError("DEM must contain at least one finite elevation")
        if not isinstance(self.coordinate_reference, str) or not self.coordinate_reference.strip():
            raise ValueError("coordinate_reference must identify a projected metre-based CRS or local frame")
        if not isinstance(self.elevation_datum, str) or not self.elevation_datum.strip():
            raise ValueError("elevation_datum must be declared")
        for array in (x, y, elevation):
            array.setflags(write=False)
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "y_m", y)
        object.__setattr__(self, "elevation_m", elevation)

    @property
    def cell_size_m(self) -> tuple[float, float]:
        return float(self.x_m[1] - self.x_m[0]), float(self.y_m[1] - self.y_m[0])


@dataclass(frozen=True)
class TerrainCorrectionResult:
    gravity_down_mgal: np.ndarray
    parameters: dict[str, float | str | int]
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.gravity_down_mgal, dtype=np.float64).copy()
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("Terrain correction must be a finite vector")
        values.setflags(write=False)
        object.__setattr__(self, "gravity_down_mgal", values)
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))


def _corner_primitive(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    radius = np.sqrt(x * x + y * y + z * z)

    def log_sum(coordinate: np.ndarray, other_a: np.ndarray, other_b: np.ndarray) -> np.ndarray:
        positive = coordinate >= 0.0
        numerator = other_a * other_a + other_b * other_b
        result = np.zeros_like(coordinate)
        # At an exact corner, coordinate + radius can be zero. The associated
        # leading x or y multiplier is also zero, so its limiting product is 0.
        positive_argument = coordinate + radius
        positive_valid = positive & (positive_argument > 0.0)
        result[positive_valid] = np.log(positive_argument[positive_valid])
        negative = ~positive
        negative_valid = negative & (numerator > 0.0)
        # Rationalize r + coordinate to avoid cancellation close to a face or
        # edge. When numerator is zero, the leading x or y multiplier is zero
        # and the product has the finite zero limit used above.
        result[negative_valid] = (
            np.log(numerator[negative_valid])
            - np.log(radius[negative_valid] - coordinate[negative_valid])
        )
        return result

    x_term = np.where(x == 0.0, 0.0, x * log_sum(y, x, z))
    y_term = np.where(y == 0.0, 0.0, y * log_sum(x, y, z))
    # Nagy's vertical-field primitive uses the principal atan(x*y/(z*r))
    # branch. atan2 adds or subtracts pi when z*r < 0, creating a spurious
    # finite term that is especially visible below and inside the prism.
    atan_numerator = x * y
    atan_denominator = z * radius
    atan = np.arctan2(atan_numerator, atan_denominator)
    negative_denominator = atan_denominator < 0.0
    atan = np.where(
        negative_denominator,
        atan - np.copysign(np.pi, atan_numerator),
        atan,
    )
    return x_term + y_term - z * atan


def _rectangular_prism_down_mgal(
    x_obs: float,
    y_obs: float,
    z_obs_up: float,
    west: np.ndarray,
    east: np.ndarray,
    south: np.ndarray,
    north: np.ndarray,
    bottom_up: np.ndarray,
    top_up: np.ndarray,
    signed_density: np.ndarray,
) -> float:
    """Vectorized Nagy corner sum over prisms, with upward-positive input z."""
    corner_sum = np.zeros(west.size, dtype=np.float64)
    x_corners = (west - x_obs, east - x_obs)
    y_corners = (south - y_obs, north - y_obs)
    # Convert upward-positive elevations to the positive-down convention of the
    # corner primitive; the top of each prism is the smaller down coordinate.
    z_corners = (-top_up + z_obs_up, -bottom_up + z_obs_up)
    for ix, x in enumerate(x_corners):
        for iy, y in enumerate(y_corners):
            for iz, z in enumerate(z_corners):
                sign = -1.0 if (ix + iy + iz) % 2 == 0 else 1.0
                corner_sum += sign * _corner_primitive(x, y, z)
    return float(-_G * _SI_TO_MGAL * np.sum(signed_density * corner_sum))


def terrain_correction(
    dem: TerrainDEM,
    station_x_m,
    station_y_m,
    station_elevation_m,
    *,
    station_coordinate_reference: str,
    reference_elevation_m: float,
    density_kg_m3: float,
    elevation_datum: str,
) -> TerrainCorrectionResult:
    """Calculate the DEM residual-terrain attraction at survey stations.

    The DEM cells are rectangular prisms between ``reference_elevation_m`` and
    their cell elevation. Cells above the plane use positive density contrast;
    cells below it use negative contrast. The output is the additive,
    positive-down residual attraction in mGal. It is a finite DEM-domain term,
    not an infinite Bouguer slab and not an extrapolation outside the DEM.
    """
    if station_coordinate_reference != dem.coordinate_reference:
        raise ValueError("station and DEM coordinates must use the same declared coordinate reference")
    if elevation_datum != dem.elevation_datum:
        raise ValueError("station and DEM elevation datums must match exactly")
    reference = float(reference_elevation_m)
    density = float(density_kg_m3)
    if not math.isfinite(reference):
        raise ValueError("reference_elevation_m must be finite")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("density_kg_m3 must be finite and positive")
    xs, ys, zs = (np.asarray(v, dtype=np.float64) for v in (station_x_m, station_y_m, station_elevation_m))
    if xs.ndim != 1 or ys.shape != xs.shape or zs.shape != xs.shape or xs.size == 0:
        raise ValueError("station coordinates and elevations must be aligned non-empty vectors")
    if not all(np.all(np.isfinite(v)) for v in (xs, ys, zs)):
        raise ValueError("station coordinates and elevations must be finite")

    dx, dy = dem.cell_size_m
    xx, yy = np.meshgrid(dem.x_m, dem.y_m)
    finite = np.isfinite(dem.elevation_m)
    elev = dem.elevation_m[finite]
    delta = elev - reference
    nonzero = delta != 0.0
    if not np.any(nonzero):
        return TerrainCorrectionResult(
            np.zeros(xs.size),
            {
                "density_kg_m3": density,
                "reference_elevation_m": reference,
                "coordinate_reference": dem.coordinate_reference,
                "elevation_datum": dem.elevation_datum,
                "dem_cell_count": int(elev.size),
                "active_prism_count": 0,
                "cell_size_x_m": dx,
                "cell_size_y_m": dy,
            },
            ("DEM cells are cell-centred rectangular prisms; no edge or exterior terrain is inferred.",
             "The DEM-minus-plane density contrast is zero in every finite cell."),
        )
    x_flat, y_flat = xx[finite][nonzero], yy[finite][nonzero]
    delta = delta[nonzero]
    top = np.maximum(elev[nonzero], reference)
    bottom = np.minimum(elev[nonzero], reference)
    signed_density = np.sign(delta) * density
    west, east = x_flat - dx / 2.0, x_flat + dx / 2.0
    south, north = y_flat - dy / 2.0, y_flat + dy / 2.0
    result = np.array([
        _rectangular_prism_down_mgal(
            x, y, z, west, east, south, north, bottom, top, signed_density
        )
        for x, y, z in zip(xs, ys, zs)
    ], dtype=np.float64)
    return TerrainCorrectionResult(
        result,
        {
            "density_kg_m3": density,
            "reference_elevation_m": reference,
            "coordinate_reference": dem.coordinate_reference,
            "elevation_datum": dem.elevation_datum,
            "dem_cell_count": int(elev.size),
            "active_prism_count": int(np.count_nonzero(nonzero)),
            "cell_size_x_m": dx,
            "cell_size_y_m": dy,
        },
        (
            "DEM cell elevations and station elevations share the declared vertical datum and positive-up convention.",
            "Each finite DEM cell is modelled as a rectangular prism between its elevation and the explicit reference plane.",
            "Cells above the reference plane use +density; cells below use -density as terrain mass deficit.",
            "The output is the positive-down vertical attraction of residual topography in the DEM footprint; no terrain outside the grid is included.",
            "Horizontal coordinates are planar metres; no CRS conversion, Earth curvature, or latitude correction is applied.",
        ),
    )

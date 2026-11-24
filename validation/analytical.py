"""Independent analytical and quadrature references for supported body shapes.

Coordinates in this module use KuffGravMag's local profile frame: x increases
along profile, y is positive strike, and z is positive down. These oracles do
not call the production forward-model implementation.
"""

from __future__ import annotations

import math

import numpy as np


G = 6.67430e-11
MU0 = 4.0 * math.pi * 1e-7
SI_TO_MGAL = 1e5
TESLA_TO_NT = 1e9


def _ordered_bounds(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    try:
        low, high = map(float, bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two numeric bounds") from exc
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return low, high


def centered_rectangle_infinite_gravity_gz_mgal(
    *, half_width: float, top: float, bottom: float, density: float
) -> float:
    """Closed positive-down attraction for a centered infinite-strike rectangle."""
    half_width = float(half_width)
    top, bottom = float(top), float(bottom)
    density = float(density)
    if not all(math.isfinite(value) for value in (half_width, top, bottom, density)):
        raise ValueError("Rectangle parameters must be finite")
    if half_width <= 0.0 or top < 0.0 or bottom <= top:
        raise ValueError("Expected positive width and 0 <= top < bottom")

    def primitive(depth: float) -> float:
        log_ratio = math.log1p((depth / half_width) ** 2)
        if depth == 0.0:
            return half_width * log_ratio
        return 2.0 * depth * math.atan(half_width / depth) + half_width * log_ratio

    area_integral = primitive(bottom) - primitive(top)
    return 2.0 * G * density * SI_TO_MGAL * area_integral


def centered_rectangle_infinite_magnetic_components_nT(
    *,
    half_width: float,
    top: float,
    bottom: float,
    magnetization_x: float,
    magnetization_z_down: float,
) -> tuple[float, float]:
    """Closed dX/dZ response for a centered infinite-strike rectangular section."""
    values = tuple(
        float(value)
        for value in (half_width, top, bottom, magnetization_x, magnetization_z_down)
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Rectangle and magnetization parameters must be finite")
    half_width, top, bottom, magnetization_x, magnetization_z_down = values
    if half_width <= 0.0 or top <= 0.0 or bottom <= top:
        raise ValueError("Expected positive width and 0 < top < bottom")
    coefficient = (
        MU0
        / math.pi
        * (math.atan(half_width / top) - math.atan(half_width / bottom))
        * TESLA_TO_NT
    )
    return -coefficient * magnetization_x, coefficient * magnetization_z_down


def profile_magnetization_a_m(
    *,
    susceptibility: float,
    remanent_intensity: float,
    remanent_inclination: float,
    remanent_declination: float,
    field_intensity_nT: float,
    field_inclination: float,
    field_declination: float,
    profile_azimuth: float,
) -> np.ndarray:
    """Independent induced-plus-remanent magnetization in local (x,y,z-down)."""
    values = tuple(
        float(value)
        for value in (
            susceptibility,
            remanent_intensity,
            remanent_inclination,
            remanent_declination,
            field_intensity_nT,
            field_inclination,
            field_declination,
            profile_azimuth,
        )
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Magnetization and field parameters must be finite")
    susceptibility, remanence, Ir, Dr, field, If, Df, alpha = values
    mu0 = MU0
    H0 = field * 1e-9 / mu0
    If, Ir = math.radians(If), math.radians(Ir)
    field_delta, remanence_delta = math.radians(Df - alpha), math.radians(Dr - alpha)
    return np.array(
        [
            susceptibility * H0 * math.cos(If) * math.cos(field_delta)
            + remanence * math.cos(Ir) * math.cos(remanence_delta),
            susceptibility * H0 * math.cos(If) * math.sin(field_delta)
            + remanence * math.cos(Ir) * math.sin(remanence_delta),
            susceptibility * H0 * math.sin(If) + remanence * math.sin(Ir),
        ],
        dtype=np.float64,
    )


def rectangular_prism_gravity_gz_mgal(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    density: float,
    observation: tuple[float, float, float],
) -> float:
    """Nagy corner sum for a right rectangular prism, with z positive down.

    The corner primitive is evaluated directly from the closed form. Bounds are
    ordered source coordinates in metres, density is kg/m³, and the returned
    downward acceleration is mGal. Gravity observations on or inside a prism
    are supported, with finite limiting values at faces, edges, and vertices.
    """
    xb = _ordered_bounds(x_bounds, "x_bounds")
    yb = _ordered_bounds(y_bounds, "y_bounds")
    zb = _ordered_bounds(z_bounds, "z_bounds")
    density = float(density)
    try:
        xo, yo, zo = map(float, observation)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation must contain three numeric coordinates") from exc
    if not all(math.isfinite(value) for value in (density, xo, yo, zo)):
        raise ValueError("density and observation must be finite")

    def corner_primitive(x: float, y: float, z: float) -> float:
        radius = math.sqrt(x * x + y * y + z * z)
        if radius == 0.0:
            # Each coordinate-log product tends to zero as the corner is
            # approached; the gravity component itself remains finite.
            return 0.0

        def log_coordinate_sum(coordinate: float, other_a: float, other_b: float) -> float:
            if coordinate < 0.0:
                # Rationalize r + coordinate to avoid catastrophic cancellation
                # for observations close to a prism face or edge.
                numerator = other_a * other_a + other_b * other_b
                if numerator == 0.0:
                    # The leading x or y multiplier is zero in this limit.
                    return 0.0
                return math.log(numerator) - math.log(radius - coordinate)
            positive_sum = coordinate + radius
            return math.log(positive_sum) if positive_sum > 0.0 else 0.0

        x_log = 0.0 if x == 0.0 else x * log_coordinate_sum(y, x, z)
        y_log = 0.0 if y == 0.0 else y * log_coordinate_sum(x, y, z)
        atan_numerator = x * y
        atan_denominator = z * radius
        angle = math.atan2(atan_numerator, atan_denominator)
        if atan_denominator < 0.0:
            # The Nagy formula uses the principal atan ratio branch. Convert
            # atan2's quadrant II/III result back to that (-pi/2, pi/2) branch.
            angle -= math.copysign(math.pi, atan_numerator)
        return x_log + y_log - z * angle

    corner_sum = 0.0
    for ix, x_source in enumerate(xb):
        for iy, y_source in enumerate(yb):
            for iz, z_source in enumerate(zb):
                dx, dy, dz = x_source - xo, y_source - yo, z_source - zo
                sign = -1.0 if (ix + iy + iz) % 2 == 0 else 1.0
                corner_sum += sign * corner_primitive(dx, dy, dz)
    return -G * density * corner_sum * SI_TO_MGAL


def rectangular_prism_volume_integral(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    density: float,
    magnetization: tuple[float, float, float],
    observation: tuple[float, float, float],
    *,
    order: int = 32,
) -> tuple[float, np.ndarray]:
    """Independent tensor-Gauss volume reference for gravity and dipole field."""
    xb = _ordered_bounds(x_bounds, "x_bounds")
    yb = _ordered_bounds(y_bounds, "y_bounds")
    zb = _ordered_bounds(z_bounds, "z_bounds")
    density = float(density)
    magnetization = np.asarray(magnetization, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if magnetization.shape != (3,) or observation.shape != (3,):
        raise ValueError("magnetization and observation must be three-component vectors")
    if not np.all(np.isfinite(magnetization)) or not np.all(np.isfinite(observation)):
        raise ValueError("magnetization and observation must be finite")
    if not math.isfinite(density) or order < 2:
        raise ValueError("density must be finite and quadrature order must be at least two")

    nodes, weights = np.polynomial.legendre.leggauss(int(order))

    def quadrature_axis(bounds):
        low, high = bounds
        return (
            0.5 * (high - low) * nodes + 0.5 * (high + low),
            0.5 * (high - low) * weights,
        )

    xs, wx = quadrature_axis(xb)
    ys, wy = quadrature_axis(yb)
    zs, wz = quadrature_axis(zb)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    volume_weights = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    dx = observation[0] - X
    dy = observation[1] - Y
    dz = observation[2] - Z
    radius_squared = dx * dx + dy * dy + dz * dz
    if np.any(radius_squared == 0.0):
        raise ValueError("Volume quadrature nodes cannot coincide with the observation")
    radius = np.sqrt(radius_squared)
    gravity = (
        G
        * density
        * np.sum(volume_weights * (Z - observation[2]) / radius**3)
        * SI_TO_MGAL
    )
    displacement = np.stack((dx, dy, dz), axis=-1)
    dot = np.einsum("...i,i->...", displacement, magnetization)
    kernel = (
        3.0 * dot[..., None] * displacement / radius[..., None] ** 5
        - magnetization / radius[..., None] ** 3
    )
    magnetic = (
        MU0
        / (4.0 * math.pi)
        * np.sum(volume_weights[..., None] * kernel, axis=(0, 1, 2))
        * TESLA_TO_NT
    )
    return float(gravity), np.asarray(magnetic, dtype=np.float64)


def _triangulate_polygon(vertices: np.ndarray) -> list[np.ndarray]:
    """Ear-clip a simple polygon for independent area quadrature."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("polygon vertices must be a finite (n, 2) array")
    if points.shape[0] > 3 and np.allclose(points[0], points[-1], rtol=0.0, atol=0.0):
        points = points[:-1]
    if points.shape[0] < 3:
        raise ValueError("a polygon must contain at least three vertices")
    cross = lambda a, b, c: (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    twice_area = float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1)
               - np.roll(points[:, 0], -1) * points[:, 1])
    )
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    tolerance = 64.0 * np.finfo(np.float64).eps * span * span
    if abs(twice_area) <= tolerance:
        raise ValueError("polygon area must be nonzero")
    orientation = 1.0 if twice_area > 0.0 else -1.0
    remaining = list(range(points.shape[0]))
    triangles: list[np.ndarray] = []

    def inside_triangle(point, a, b, c):
        return (
            orientation * cross(a, b, point) >= -tolerance
            and orientation * cross(b, c, point) >= -tolerance
            and orientation * cross(c, a, point) >= -tolerance
        )

    while len(remaining) > 3:
        clipped = False
        for position, middle_index in enumerate(remaining):
            previous_index = remaining[position - 1]
            next_index = remaining[(position + 1) % len(remaining)]
            a, b, c = points[previous_index], points[middle_index], points[next_index]
            if orientation * cross(a, b, c) <= tolerance:
                continue
            if any(
                inside_triangle(points[index], a, b, c)
                for index in remaining
                if index not in {previous_index, middle_index, next_index}
            ):
                continue
            triangles.append(np.array([a, b, c], dtype=np.float64))
            remaining.pop(position)
            clipped = True
            break
        if not clipped:
            raise ValueError("polygon must be simple and free of degenerate edges")
    triangles.append(points[remaining])
    return triangles


def _triangle_quadrature(vertices: np.ndarray, order: int):
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    unit_nodes = 0.5 * (nodes + 1.0)
    unit_weights = 0.5 * weights
    a, b, c = vertices
    edge_a, edge_b = b - a, c - a
    jacobian = abs(float(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]))
    for iu, u in enumerate(unit_nodes):
        for iv, v in enumerate(unit_nodes):
            point = a + u * (b - a) + (1.0 - u) * v * (c - a)
            weight = unit_weights[iu] * unit_weights[iv] * (1.0 - u) * jacobian
            yield point, float(weight)


def polygon_infinite_strike_gravity_reference(
    vertices,
    density: float,
    observation: tuple[float, float],
    *,
    order: int = 64,
) -> float:
    """Independent area-quadrature gravity reference for a simple polygon.

    The strike direction is integrated analytically before tensor Gaussian
    quadrature over an ear-clipped polygon. This works for convex and concave
    simple polygons whose source area does not contain the observation.
    """
    density = float(density)
    observation = np.asarray(observation, dtype=np.float64)
    if observation.shape != (2,) or not np.all(np.isfinite(observation)):
        raise ValueError("observation must be a finite (x, z) pair")
    if not math.isfinite(density) or order < 2:
        raise ValueError("density must be finite and quadrature order at least two")
    total = 0.0
    for triangle in _triangulate_polygon(vertices):
        for source, weight in _triangle_quadrature(triangle, order):
            dx = float(source[0] - observation[0])
            dz = float(source[1] - observation[1])
            radius_squared = dx * dx + dz * dz
            if radius_squared == 0.0:
                raise ValueError("observation intersects a quadrature source point")
            total += weight * 2.0 * G * density * dz / radius_squared
    return total * SI_TO_MGAL


def polygon_infinite_strike_magnetic_reference_nT(
    vertices,
    magnetization_xz: tuple[float, float],
    observation: tuple[float, float],
    *,
    order: int = 64,
) -> tuple[float, float]:
    """Independent area-quadrature dX/dZ reference for a simple 2D polygon."""
    magnetization = np.asarray(magnetization_xz, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if magnetization.shape != (2,) or observation.shape != (2,):
        raise ValueError("magnetization and observation must be two-component vectors")
    if not np.all(np.isfinite(magnetization)) or not np.all(np.isfinite(observation)):
        raise ValueError("magnetization and observation must be finite")
    if order < 2:
        raise ValueError("quadrature order must be at least two")
    total = np.zeros(2, dtype=np.float64)
    for triangle in _triangulate_polygon(vertices):
        for source, weight in _triangle_quadrature(triangle, order):
            rx, rz = observation - source
            radius_squared = float(rx * rx + rz * rz)
            if radius_squared == 0.0:
                raise ValueError("observation intersects a quadrature source point")
            dot = float(np.dot(magnetization, (rx, rz)))
            kernel = 2.0 * dot * np.array([rx, rz]) / radius_squared**2 - magnetization / radius_squared
            total += weight * kernel
    total *= MU0 / (2.0 * math.pi) * TESLA_TO_NT
    return float(total[0]), float(total[1])


def point_mass_gravity_gz_mgal(
    *, volume: float, density: float, center: tuple[float, float, float], observation: tuple[float, float, float]
) -> float:
    """Positive-down point-mass asymptote for a compact body."""
    center = np.asarray(center, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    if center.shape != (3,) or observation.shape != (3,):
        raise ValueError("center and observation must be three-component vectors")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(observation)):
        raise ValueError("center and observation must be finite")
    displacement = observation - center
    radius = float(np.linalg.norm(displacement))
    if radius == 0.0:
        raise ValueError("Point-mass asymptote is singular at the source center")
    return -G * float(density) * float(volume) * displacement[2] / radius**3 * SI_TO_MGAL


def dipole_magnetic_field_nT(
    *,
    volume: float,
    magnetization: tuple[float, float, float],
    center: tuple[float, float, float],
    observation: tuple[float, float, float],
) -> np.ndarray:
    """Uniform-body magnetic dipole asymptote in the local positive-down frame."""
    center = np.asarray(center, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    magnetization = np.asarray(magnetization, dtype=np.float64)
    if center.shape != (3,) or observation.shape != (3,) or magnetization.shape != (3,):
        raise ValueError("center, observation, and magnetization must be 3-vectors")
    if not all(np.all(np.isfinite(vector)) for vector in (center, observation, magnetization)):
        raise ValueError("center, observation, and magnetization must be finite")
    displacement = observation - center
    radius = float(np.linalg.norm(displacement))
    if radius == 0.0:
        raise ValueError("Dipole asymptote is singular at the source center")
    field = (
        3.0 * float(np.dot(magnetization, displacement)) * displacement / radius**5
        - magnetization / radius**3
    )
    return MU0 / (4.0 * math.pi) * float(volume) * field * TESLA_TO_NT

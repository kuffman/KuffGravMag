import math

import numpy as np


def is_infinite_strike_reference(y_min, y_max):
    """Return whether legacy bounds select the historical infinite-strike model."""
    try:
        y_min, y_max = float(y_min), float(y_max)
    except (TypeError, ValueError):
        return False
    return (y_min == 0.0 and y_max == 0.0) or (y_min == -math.inf and y_max == math.inf)


def resolve_strike_mode(y_min, y_max, strike_mode=None):
    """Resolve explicit modes while preserving old project zero-bound semantics."""
    if strike_mode is None:
        return "infinite" if is_infinite_strike_reference(y_min, y_max) else "finite"
    mode = str(strike_mode).strip().lower().replace(" ", "_")
    aliases = {"infinite": "infinite", "infinite_strike": "infinite",
               "finite": "finite", "finite_strike": "finite"}
    if mode not in aliases:
        raise ValueError("strike_mode must be 'infinite' or 'finite'")
    return aliases[mode]


def _validate_strike_bounds(y_min, y_max, strike_mode):
    try:
        y_min, y_max = float(y_min), float(y_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("Strike bounds must be numeric") from exc
    if math.isnan(y_min) or math.isnan(y_max) or y_min > y_max:
        raise ValueError("Strike bounds must be ordered and cannot be NaN")
    if strike_mode == "finite" and (not math.isfinite(y_min) or not math.isfinite(y_max)):
        raise ValueError("Finite strike bounds must be finite")
    if strike_mode == "infinite" and not (
        (math.isinf(y_min) and y_min < 0 and math.isinf(y_max) and y_max > 0)
        or (math.isfinite(y_min) and math.isfinite(y_max))
    ):
        raise ValueError("Infinite strike mode requires finite bounds or (-inf, +inf)")
    return y_min, y_max


class Body2D:
    """
    Represents a 2.5D polygonal geological body with finite strike length (Y_min, Y_max).
    """

    def __init__(self, name, x_verts, z_verts, density=300.0, k_succ=0.02,
                 J_r=0.0, I_r=0.0, D_r=0.0, y_min=-500.0, y_max=500.0,
                 color='tab:blue', strike_mode=None):
        self.name = name
        self.x = np.asarray(x_verts, dtype=np.float64)
        self.z = np.asarray(z_verts, dtype=np.float64)

        # Ensure polygon closure
        if self.x[0] != self.x[-1] or self.z[0] != self.z[-1]:
            self.x = np.append(self.x, self.x[0])
            self.z = np.append(self.z, self.z[0])

        self.density = density  # kg/m^3
        self.k_succ = k_succ  # SI units
        self.J_r = J_r  # A/m
        self.I_r = I_r  # degrees
        self.D_r = D_r  # degrees
        self.y_min = y_min  # Left strike extent (m)
        self.y_max = y_max  # Right strike extent (m)
        # Missing mode is the legacy-project path: (0, 0) remains the old
        # infinite-strike sentinel, while all other finite bounds are physical.
        self.strike_mode = resolve_strike_mode(y_min, y_max, strike_mode)
        self.color = color  # Canvas color

    def update_polygon(self, x_verts, z_verts):
        self.x = np.asarray(x_verts, dtype=np.float64)
        self.z = np.asarray(z_verts, dtype=np.float64)
        if self.x[0] != self.x[-1] or self.z[0] != self.z[-1]:
            self.x = np.append(self.x, self.x[0])
            self.z = np.append(self.z, self.z[0])


def _extruded_prism_faces(x, z, y_min, y_max, orientation):
    """Build outward-oriented polygonal faces for a cross-section extrusion."""
    section = np.column_stack((x[:-1], z[:-1]))
    if orientation < 0.0:
        section = section[::-1]
    at_min = np.column_stack((section[:, 0], np.full(len(section), y_min), section[:, 1]))
    at_max = np.column_stack((section[:, 0], np.full(len(section), y_max), section[:, 1]))
    faces = [at_min, at_max[::-1]]
    for index in range(len(section)):
        following = (index + 1) % len(section)
        # Walking along +y first gives the outward side normal for a CCW x-z ring.
        faces.append(np.array([at_min[index], at_max[index], at_max[following], at_min[following]]))
    return faces


def _face_normal(vertices):
    origin = vertices[0]
    area_vector = np.zeros(3, dtype=np.float64)
    for index in range(1, len(vertices) - 1):
        area_vector += np.cross(vertices[index] - origin, vertices[index + 1] - origin)
    magnitude = float(np.linalg.norm(area_vector))
    if not math.isfinite(magnitude) or magnitude == 0.0:
        return None
    return area_vector / magnitude


def _gravity_face_integral(vertices, observation):
    """Exact planar-polygon integral of 1/R using a boundary primitive."""
    normal = _face_normal(vertices)
    if normal is None:
        return 0.0
    first_edge = None
    for index in range(len(vertices)):
        candidate = vertices[(index + 1) % len(vertices)] - vertices[index]
        length = float(np.linalg.norm(candidate))
        if length > 0.0:
            first_edge = candidate / length
            break
    if first_edge is None:
        return 0.0
    e1 = first_edge
    e2 = np.cross(normal, e1)
    origin = vertices[0]
    local = np.column_stack(((vertices - origin) @ e1, (vertices - origin) @ e2))
    height = float(np.dot(observation - origin, normal))
    projected_observation = observation - origin - height * normal
    obs2 = np.array([np.dot(projected_observation, e1), np.dot(projected_observation, e2)])
    scale = max(1.0, float(np.max(np.abs(vertices))), float(np.max(np.abs(observation))))
    tolerance = 32.0 * np.finfo(np.float64).eps * scale
    if abs(height) <= tolerance:
        height = 0.0
    b = abs(height)
    total = 0.0
    for index in range(len(local)):
        a = local[index]
        endpoint = local[(index + 1) % len(local)]
        edge = endpoint - a
        length = float(np.linalg.norm(edge))
        if length <= tolerance:
            continue
        tangent = edge / length
        outward = np.array([tangent[1], -tangent[0]])
        q = float(np.dot(a - obs2, outward))
        if abs(q) <= tolerance:
            q = 0.0
        s1 = float(np.dot(a - obs2, tangent))
        s2 = s1 + length
        if q == 0.0:
            continue
        a_radius = math.hypot(q, height)
        ratio_scale = abs(q) / (a_radius + b)

        def primitive(s):
            radius = math.hypot(s, a_radius)
            angle = math.atan(ratio_scale * s / (radius + a_radius))
            return q * math.asinh(s / a_radius) - 2.0 * math.copysign(b, q) * angle

        total += primitive(s2) - primitive(s1)
    return total


def _solid_angle(vertices, observation, normal):
    """Signed solid angle of an oriented planar polygon at an observation."""
    origin = vertices[0] - observation
    angle = 0.0
    for index in range(1, len(vertices) - 1):
        a = origin
        b = vertices[index] - observation
        c = vertices[index + 1] - observation
        la, lb, lc = float(np.linalg.norm(a)), float(np.linalg.norm(b)), float(np.linalg.norm(c))
        if min(la, lb, lc) == 0.0:
            raise ValueError("Magnetic field is singular at a prism vertex")
        numerator = float(np.dot(a, np.cross(b, c)))
        denominator = (la * lb * lc + float(np.dot(a, b)) * lc
                      + float(np.dot(b, c)) * la + float(np.dot(c, a)) * lb)
        angle += 2.0 * math.atan2(numerator, denominator)
    return angle


def _magnetic_face_vector_integral(vertices, observation):
    """Exact integral of R/R^3 over one planar polygonal face."""
    normal = _face_normal(vertices)
    if normal is None:
        return np.zeros(3, dtype=np.float64)
    first_edge = None
    for index in range(len(vertices)):
        candidate = vertices[(index + 1) % len(vertices)] - vertices[index]
        length = float(np.linalg.norm(candidate))
        if length > 0.0:
            first_edge = candidate / length
            break
    if first_edge is None:
        return np.zeros(3, dtype=np.float64)
    e1 = first_edge
    e2 = np.cross(normal, e1)
    origin = vertices[0]
    local = np.column_stack(((vertices - origin) @ e1, (vertices - origin) @ e2))
    height = float(np.dot(observation - origin, normal))
    projected_observation = observation - origin - height * normal
    obs2 = np.array([np.dot(projected_observation, e1), np.dot(projected_observation, e2)])
    scale = max(1.0, float(np.max(np.abs(vertices))), float(np.max(np.abs(observation))))
    tolerance = 32.0 * np.finfo(np.float64).eps * scale
    if abs(height) <= tolerance:
        height = 0.0
    tangent_local = np.zeros(2, dtype=np.float64)
    for index in range(len(local)):
        a = local[index]
        endpoint = local[(index + 1) % len(local)]
        edge = endpoint - a
        length = float(np.linalg.norm(edge))
        if length <= tolerance:
            continue
        tangent = edge / length
        outward = np.array([tangent[1], -tangent[0]])
        q = float(np.dot(a - obs2, outward))
        if abs(q) <= tolerance:
            q = 0.0
        s1 = float(np.dot(a - obs2, tangent))
        s2 = s1 + length
        perpendicular_distance = math.hypot(q, height)
        if perpendicular_distance <= tolerance:
            raise ValueError("Magnetic field is singular on a finite-prism edge")
        edge_integral = (math.asinh(s2 / perpendicular_distance)
                         - math.asinh(s1 / perpendicular_distance))
        tangent_local += outward * edge_integral
    tangent_global = tangent_local[0] * e1 + tangent_local[1] * e2
    solid_angle = 0.0 if height == 0.0 else _solid_angle(vertices, observation, normal)
    return tangent_global - solid_angle * normal


class PotentialField2D:
    """
    2.5D Forward Engine with finite strike bounds and variable topography support.
    """
    G = 6.67430e-11  # m^3 kg^-1 s^-2
    SI2MGAL = 1e5  # m/s^2 to mGal

    def __init__(self, bodies=None):
        self.bodies = bodies if bodies is not None else []

    @staticmethod
    def _validate_observations(x_obs, z_obs):
        try:
            x_obs = np.asarray(x_obs, dtype=np.float64)
            z_obs = np.asarray(z_obs, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Observation coordinates must be numeric vectors") from exc

        if x_obs.ndim != 1 or z_obs.ndim != 1 or x_obs.shape != z_obs.shape:
            raise ValueError("x_obs and z_obs must be one-dimensional vectors of equal length")
        if not np.all(np.isfinite(x_obs)) or not np.all(np.isfinite(z_obs)):
            raise ValueError("Observation coordinates must be finite")
        return x_obs, z_obs

    @staticmethod
    def _prepare_polygon(x, z):
        try:
            x = np.asarray(x, dtype=np.float64)
            z = np.asarray(z, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Polygon coordinates must be numeric vectors") from exc

        if x.ndim != 1 or z.ndim != 1 or x.shape != z.shape:
            raise ValueError("Polygon x and z coordinates must be one-dimensional vectors of equal length")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(z)):
            raise ValueError("Polygon coordinates must be finite")
        if x.size < 3:
            raise ValueError("A polygon must contain at least three distinct vertices")

        points = np.column_stack((x, z))
        coordinate_scale = max(1.0, float(np.max(np.abs(points))))
        coordinate_tolerance = 16.0 * np.finfo(np.float64).eps * coordinate_scale

        # Drop repeated adjacent vertices, including a repeated closing vertex.
        cleaned = []
        for point in points:
            if not cleaned or np.hypot(*(point - cleaned[-1])) > coordinate_tolerance:
                cleaned.append(point)
        if len(cleaned) > 1 and np.hypot(*(cleaned[0] - cleaned[-1])) <= coordinate_tolerance:
            cleaned.pop()
        if len(cleaned) < 3:
            raise ValueError("A polygon must contain at least three distinct vertices")

        points = np.asarray(cleaned, dtype=np.float64)
        span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
        if not math.isfinite(span) or span <= coordinate_tolerance:
            raise ValueError("Polygon has zero or numerically negligible extent")

        relative = points - points[0]
        next_relative = np.roll(relative, -1, axis=0)
        twice_area = float(
            np.sum(
                relative[:, 0] * next_relative[:, 1]
                - next_relative[:, 0] * relative[:, 1]
            )
        )
        area_tolerance = 64.0 * np.finfo(np.float64).eps * span * span
        if abs(twice_area) <= area_tolerance:
            raise ValueError("Polygon must enclose a nonzero area")

        def cross(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        def on_segment(a, b, point):
            return (
                min(a[0], b[0]) - coordinate_tolerance
                <= point[0]
                <= max(a[0], b[0]) + coordinate_tolerance
                and min(a[1], b[1]) - coordinate_tolerance
                <= point[1]
                <= max(a[1], b[1]) + coordinate_tolerance
            )

        vertex_count = len(points)
        for i in range(vertex_count):
            a, b = points[i], points[(i + 1) % vertex_count]
            for j in range(i + 1, vertex_count):
                if j == i + 1 or (i == 0 and j == vertex_count - 1):
                    continue
                c, d = points[j], points[(j + 1) % vertex_count]
                o1, o2 = cross(a, b, c), cross(a, b, d)
                o3, o4 = cross(c, d, a), cross(c, d, b)
                proper_crossing = (
                    (o1 > area_tolerance and o2 < -area_tolerance)
                    or (o1 < -area_tolerance and o2 > area_tolerance)
                ) and (
                    (o3 > area_tolerance and o4 < -area_tolerance)
                    or (o3 < -area_tolerance and o4 > area_tolerance)
                )
                touching = (
                    (abs(o1) <= area_tolerance and on_segment(a, b, c))
                    or (abs(o2) <= area_tolerance and on_segment(a, b, d))
                    or (abs(o3) <= area_tolerance and on_segment(c, d, a))
                    or (abs(o4) <= area_tolerance and on_segment(c, d, b))
                )
                if proper_crossing or touching:
                    raise ValueError("Self-intersecting polygons are not supported")

        closed = np.vstack((points, points[0]))
        orientation = 1.0 if twice_area > 0.0 else -1.0
        reference_length = max(span, 1.0)
        return closed[:, 0], closed[:, 1], orientation, reference_length, coordinate_tolerance

    @staticmethod
    def _log_edge_primitive(s, q, reference_length):
        radius = math.hypot(s, q)
        if radius == 0.0:
            log_term = 0.0
        else:
            log_term = 2.0 * s * (math.log(radius) - math.log(reference_length))

        if q == 0.0:
            angle_term = 0.0
        else:
            # This is atan(s / q) on a continuous branch, without forming the ratio.
            angle_argument = s if q > 0.0 else -s
            angle = math.atan2(angle_argument, abs(q))
            angle_term = 2.0 * q * angle
        return log_term - 2.0 * s + angle_term

    def calculate_gravity(self, x_obs, z_obs):
        x_obs, z_obs = self._validate_observations(x_obs, z_obs)
        gz_total = np.zeros(x_obs.shape, dtype=np.float64)
        for body in self.bodies:
            gz_total += self._calc_single_gravity_25d(
                x_obs, z_obs, body.x, body.z, body.density, body.y_min, body.y_max,
                strike_mode=body.strike_mode,
            )
        return gz_total

    def _calc_single_gravity_25d(self, x_obs, z_obs, x, z, density, y1, y2,
                                 strike_mode=None):
        x_obs, z_obs = self._validate_observations(x_obs, z_obs)
        x, z, orientation, reference_length, coordinate_tolerance = self._prepare_polygon(x, z)

        try:
            density = float(density)
        except (TypeError, ValueError) as exc:
            raise ValueError("Density must be numeric") from exc
        if not math.isfinite(density):
            raise ValueError("Density contrast must be finite")
        mode = resolve_strike_mode(y1, y2, strike_mode)
        y1, y2 = _validate_strike_bounds(y1, y2, mode)
        if mode == "finite":
            if y1 == y2 or density == 0.0:
                return np.zeros_like(x_obs, dtype=np.float64)
            faces = _extruded_prism_faces(x, z, y1, y2, orientation)
            result = np.zeros_like(x_obs, dtype=np.float64)
            for index, (xo, zo) in enumerate(zip(x_obs, z_obs)):
                observation = np.array([xo, 0.0, zo], dtype=np.float64)
                face_sum = sum(
                    _face_normal(face)[2] * _gravity_face_integral(face, observation)
                    for face in faces
                    if _face_normal(face) is not None
                )
                result[index] = -self.G * density * face_sum * self.SI2MGAL
            return result

        # Infinite-strike gravity retains the validated 2D boundary integral.
        gz = np.zeros(x_obs.shape, dtype=np.float64)

        for j, (xo, zo) in enumerate(zip(x_obs, z_obs)):
            edge_sum = 0.0
            for i in range(len(x) - 1):
                x1, z1 = x[i], z[i]
                x2, z2 = x[i + 1], z[i + 1]
                dx, dz = x2 - x1, z2 - z1
                edge_length = math.hypot(dx, dz)
                if edge_length <= coordinate_tolerance:
                    continue

                ux, uz = dx / edge_length, dz / edge_length
                rx1, rz1 = x1 - xo, z1 - zo
                rx2, rz2 = x2 - xo, z2 - zo
                s1 = rx1 * ux + rz1 * uz
                s2 = rx2 * ux + rz2 * uz
                q = -rx1 * uz + rz1 * ux

                primitive_delta = self._log_edge_primitive(
                    s2, q, reference_length
                ) - self._log_edge_primitive(s1, q, reference_length)
                edge_integral = ux * primitive_delta

                # Green's theorem gives g_z = -G Δρ ∮ log(r²) dx for a
                # positively wound polygon in the (x, positive-down z) plane.
                edge_sum -= orientation * edge_integral

            gz[j] = self.G * density * self.SI2MGAL * edge_sum

        return gz

    def calculate_magnetics(self, x_obs, z_obs, F_nT, I_f, D_f, alpha):
        """Return the legacy profile-horizontal, vertical, and total-field outputs."""
        dX, _, dZ, dT = self.calculate_magnetics_components(
            x_obs, z_obs, F_nT, I_f, D_f, alpha
        )
        return dX, dZ, dT

    def calculate_magnetics_components(self, x_obs, z_obs, F_nT, I_f, D_f, alpha):
        """Return (dX, dY, dZ, dT) in nT, with z positive down."""
        x_obs, z_obs = self._validate_observations(x_obs, z_obs)
        dX_total = np.zeros_like(x_obs, dtype=np.float64)
        dY_total = np.zeros_like(x_obs, dtype=np.float64)
        dZ_total = np.zeros_like(x_obs, dtype=np.float64)
        dT_total = np.zeros_like(x_obs, dtype=np.float64)

        for body in self.bodies:
            dX, dY, dZ, dT = self._calc_single_magnetic_components(
                x_obs, z_obs, body.x, body.z, body.k_succ, F_nT, I_f, D_f, alpha,
                body.J_r, body.I_r, body.D_r, body.y_min, body.y_max,
                strike_mode=body.strike_mode,
            )
            dX_total += dX
            dY_total += dY
            dZ_total += dZ
            dT_total += dT

        return dX_total, dY_total, dZ_total, dT_total

    def _calc_single_magnetics_25d(self, x_obs, z_obs, x, z, k_succ, F_nT, I_f,
                                  D_f, alpha, J_r, I_r, D_r, y1, y2,
                                  strike_mode=None):
        """Compatibility wrapper retained for callers of the old private helper."""
        dX, _, dZ, dT = self._calc_single_magnetic_components(
            x_obs, z_obs, x, z, k_succ, F_nT, I_f, D_f, alpha,
            J_r, I_r, D_r, y1, y2, strike_mode=strike_mode,
        )
        return dX, dZ, dT

    def _calc_single_magnetic_components(self, x_obs, z_obs, x, z, k_succ,
                                         F_nT, I_f, D_f, alpha, J_r, I_r,
                                         D_r, y1, y2, strike_mode=None):
        x_obs, z_obs = self._validate_observations(x_obs, z_obs)
        x, z, orientation, _, coordinate_tolerance = self._prepare_polygon(x, z)
        try:
            k_succ, F_nT, I_f, D_f, alpha, J_r, I_r, D_r = map(
                float, (k_succ, F_nT, I_f, D_f, alpha, J_r, I_r, D_r)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("Magnetic parameters must be numeric") from exc
        parameters = (k_succ, F_nT, I_f, D_f, alpha, J_r, I_r, D_r)
        if not all(math.isfinite(value) for value in parameters):
            raise ValueError("Magnetic parameters must be finite")
        mode = resolve_strike_mode(y1, y2, strike_mode)
        y1, y2 = _validate_strike_bounds(y1, y2, mode)

        rad = math.pi / 180.0
        inclination = I_f * rad
        horizontal_angle = (D_f - alpha) * rad
        remanent_inclination = I_r * rad
        remanent_declination = (D_r - alpha) * rad
        H0 = F_nT * 1e-9 / (4.0 * math.pi * 1e-7)
        Mx = (
            k_succ * H0 * math.cos(inclination) * math.cos(horizontal_angle)
            + J_r * math.cos(remanent_inclination) * math.cos(remanent_declination)
        )
        My = (
            k_succ * H0 * math.cos(inclination) * math.sin(horizontal_angle)
            + J_r * math.cos(remanent_inclination) * math.sin(remanent_declination)
        )
        Mz = k_succ * H0 * math.sin(inclination) + J_r * math.sin(remanent_inclination)
        magnetization = np.array([Mx, My, Mz], dtype=np.float64)

        if not np.any(magnetization):
            zeros = np.zeros_like(x_obs, dtype=np.float64)
            return zeros, zeros.copy(), zeros.copy(), zeros.copy()

        if mode == "finite" and y1 == y2:
            zeros = np.zeros_like(x_obs, dtype=np.float64)
            return zeros, zeros.copy(), zeros.copy(), zeros.copy()

        if mode == "infinite":
            dX_nT, dZ_nT = self._infinite_strike_magnetic_components(
                x_obs, z_obs, x, z, orientation, coordinate_tolerance, Mx, Mz
            )
            dY_nT = np.zeros_like(x_obs, dtype=np.float64)
        else:
            faces = _extruded_prism_faces(x, z, y1, y2, orientation)
            vertices = np.concatenate(faces, axis=0)
            scale = max(1.0, float(np.max(np.abs(vertices))))
            tolerance = 32.0 * np.finfo(np.float64).eps * scale
            if np.any(np.linalg.norm(vertices[None, :, :] -
                                     np.column_stack((x_obs, np.zeros_like(x_obs), z_obs))[:, None, :],
                                     axis=2) <= tolerance):
                raise ValueError("Magnetic field is singular at a prism vertex")
            dX_nT = np.zeros_like(x_obs, dtype=np.float64)
            dY_nT = np.zeros_like(x_obs, dtype=np.float64)
            dZ_nT = np.zeros_like(x_obs, dtype=np.float64)
            scale_nT = (4.0 * math.pi * 1e-7) / (4.0 * math.pi) * 1e9
            normals = [_face_normal(face) for face in faces]
            for j, (xo, zo) in enumerate(zip(x_obs, z_obs)):
                observation = np.array([xo, 0.0, zo], dtype=np.float64)
                field_sum = np.zeros(3, dtype=np.float64)
                for face, normal in zip(faces, normals):
                    if normal is None:
                        continue
                    surface_charge = float(np.dot(magnetization, normal))
                    if surface_charge == 0.0:
                        continue
                    field_sum += surface_charge * _magnetic_face_vector_integral(
                        face, observation
                    )
                dX_nT[j], dY_nT[j], dZ_nT[j] = scale_nT * field_sum

        unit_field = np.array([
            math.cos(inclination) * math.cos(horizontal_angle),
            math.cos(inclination) * math.sin(horizontal_angle),
            math.sin(inclination),
        ])
        dT_nT = dX_nT * unit_field[0] + dY_nT * unit_field[1] + dZ_nT * unit_field[2]
        return dX_nT, dY_nT, dZ_nT, dT_nT

    @staticmethod
    def _infinite_strike_magnetic_components(x_obs, z_obs, x, z,
                                             orientation, coordinate_tolerance,
                                             Mx, Mz):
        """Validated infinite-strike 2D boundary-charge field (dX, dZ)."""
        dX_nT = np.zeros_like(x_obs, dtype=np.float64)
        dZ_nT = np.zeros_like(x_obs, dtype=np.float64)
        for j, (xo, zo) in enumerate(zip(x_obs, z_obs)):
            bx_sum = 0.0
            bz_sum = 0.0
            for i in range(len(x) - 1):
                x1, z1 = x[i], z[i]
                dx, dz = x[i + 1] - x1, z[i + 1] - z1
                edge_length = math.hypot(dx, dz)
                if edge_length <= coordinate_tolerance:
                    continue
                ux, uz = dx / edge_length, dz / edge_length

                # The left normal points inward for a positively wound ring.
                nx, nz = -uz, ux
                rx, rz = xo - x1, zo - z1
                u1 = rx * ux + rz * uz
                q = rx * nx + rz * nz
                u2 = u1 - edge_length
                radius1, radius2 = math.hypot(u1, q), math.hypot(u2, q)
                if radius1 == 0.0 or radius2 == 0.0:
                    raise ValueError("Magnetic field is singular at a polygon vertex")

                tangent_integral = math.log(radius1) - math.log(radius2)
                if q == 0.0:
                    # Principal-value convention at an edge interior: mean of
                    # the two one-sided normal fields.
                    normal_integral = 0.0
                else:
                    q_sign = 1.0 if q > 0.0 else -1.0
                    normal_integral = math.atan2(q_sign * u1, abs(q)) - math.atan2(
                        q_sign * u2, abs(q)
                    )

                integral_x = ux * tangent_integral + nx * normal_integral
                integral_z = uz * tangent_integral + nz * normal_integral
                surface_charge_factor = orientation * (Mx * uz - Mz * ux)
                bx_sum += surface_charge_factor * integral_x
                bz_sum += surface_charge_factor * integral_z

            scale_nT = (4.0 * math.pi * 1e-7) / (2.0 * math.pi) * 1e9
            dX_nT[j] = scale_nT * bx_sum
            dZ_nT[j] = scale_nT * bz_sum
        return dX_nT, dZ_nT

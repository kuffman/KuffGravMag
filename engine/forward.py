import math

import numpy as np


class Body2D:
    """
    Represents a 2.5D polygonal geological body with finite strike length (Y_min, Y_max).
    """

    def __init__(self, name, x_verts, z_verts, density=300.0, k_succ=0.02,
                 J_r=0.0, I_r=0.0, D_r=0.0, y_min=-500.0, y_max=500.0, color='tab:blue'):
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
        self.color = color  # Canvas color

    def update_polygon(self, x_verts, z_verts):
        self.x = np.asarray(x_verts, dtype=np.float64)
        self.z = np.asarray(z_verts, dtype=np.float64)
        if self.x[0] != self.x[-1] or self.z[0] != self.z[-1]:
            self.x = np.append(self.x, self.x[0])
            self.z = np.append(self.z, self.z[0])


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
    def _prepare_gravity_polygon(x, z):
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
                x_obs, z_obs, body.x, body.z, body.density, body.y_min, body.y_max
            )
        return gz_total

    def _calc_single_gravity_25d(self, x_obs, z_obs, x, z, density, y1, y2):
        x_obs, z_obs = self._validate_observations(x_obs, z_obs)
        x, z, orientation, reference_length, coordinate_tolerance = self._prepare_gravity_polygon(x, z)

        try:
            density = float(density)
            y1, y2 = float(y1), float(y2)
        except (TypeError, ValueError) as exc:
            raise ValueError("Density and strike bounds must be numeric") from exc
        if not math.isfinite(density):
            raise ValueError("Density contrast must be finite")
        if math.isnan(y1) or math.isnan(y2) or y1 > y2:
            raise ValueError("Strike bounds must be ordered and cannot be NaN")
        infinite_strike = y1 == -math.inf and y2 == math.inf
        if not infinite_strike and (not math.isfinite(y1) or not math.isfinite(y2)):
            raise ValueError("Only the symmetric infinite-strike bounds (-inf, inf) are supported")

        # Keep the existing finite-strike midpoint correction unchanged.
        y_avg = 0.5 * abs(y1) + 0.5 * abs(y2) if not infinite_strike else math.inf
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

                if infinite_strike or y_avg == 0.0:
                    strike_factor = 1.0
                else:
                    r1 = math.hypot(rx1, rz1)
                    r2 = math.hypot(rx2, rz2)
                    r_mid = 0.5 * (r1 + r2)
                    strike_factor = y_avg / math.hypot(r_mid, y_avg)

                # Green's theorem gives g_z = -G Δρ ∮ log(r²) dx for a
                # positively wound polygon in the (x, positive-down z) plane.
                edge_sum -= orientation * edge_integral * strike_factor

            gz[j] = self.G * density * self.SI2MGAL * edge_sum

        return gz

    def calculate_magnetics(self, x_obs, z_obs, F_nT, I_f, D_f, alpha):
        dX_total = np.zeros_like(x_obs, dtype=np.float64)
        dZ_total = np.zeros_like(x_obs, dtype=np.float64)
        dT_total = np.zeros_like(x_obs, dtype=np.float64)

        for body in self.bodies:
            dX, dZ, dT = self._calc_single_magnetics_25d(
                x_obs, z_obs, body.x, body.z, body.k_succ, F_nT, I_f, D_f, alpha,
                body.J_r, body.I_r, body.D_r, body.y_min, body.y_max
            )
            dX_total += dX
            dZ_total += dZ
            dT_total += dT

        return dX_total, dZ_total, dT_total

    def _calc_single_magnetics_25d(self, x_obs, z_obs, x, z, k_succ, F_nT, I_f, D_f, alpha, J_r, I_r, D_r, y1, y2):
        rad = np.pi / 180.0
        I_f_r, D_f_r, alpha_r = I_f * rad, D_f * rad, alpha * rad
        I_r_r, D_r_r = I_r * rad, D_r * rad

        mu0 = 4 * np.pi * 1e-7
        H_i = (F_nT * 1e-9) / mu0
        J_i = k_succ * H_i

        Jx = J_i * np.cos(I_f_r) * np.cos(D_f_r - alpha_r) + J_r * np.cos(I_r_r) * np.cos(D_r_r - alpha_r)
        Jz = J_i * np.sin(I_f_r) + J_r * np.sin(I_r_r)

        J_prime = np.hypot(Jx, Jz)
        I_prime = np.arctan2(Jz, Jx)

        dX = np.zeros_like(x_obs, dtype=np.float64)
        dZ = np.zeros_like(x_obs, dtype=np.float64)
        n_sides = len(x) - 1
        y_avg = 0.5 * (abs(y1) + abs(y2))

        for i in range(n_sides):
            x1, z1 = x[i], z[i]
            x2, z2 = x[i + 1], z[i + 1]
            dx, dz = x2 - x1, z2 - z1

            if dx == 0 and dz == 0:
                continue

            phi = np.arctan2(dz, dx)

            for j, (xo, zo) in enumerate(zip(x_obs, z_obs)):
                rx1, rz1 = x1 - xo, z1 - zo
                rx2, rz2 = x2 - xo, z2 - zo
                r1, r2 = np.hypot(rx1, rz1), np.hypot(rx2, rz2)

                if r1 < 1e-6 or r2 < 1e-6:
                    continue

                theta1, theta2 = np.arctan2(rz1, rx1), np.arctan2(rz2, rx2)
                dtheta = theta2 - theta1
                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi

                ln_r = np.log(r2 / r1)
                P_i = dtheta * np.sin(phi) * np.cos(phi) + (np.cos(phi) ** 2) * ln_r
                Q_i = dtheta * (np.cos(phi) ** 2) - np.sin(phi) * np.cos(phi) * ln_r

                r_mid = 0.5 * (r1 + r2)
                strike_factor = y_avg / np.sqrt(r_mid ** 2 + y_avg ** 2) if y_avg > 0 else 1.0

                dX[j] += (P_i * np.sin(I_prime) - Q_i * np.cos(I_prime)) * strike_factor
                dZ[j] += (P_i * np.cos(I_prime) + Q_i * np.sin(I_prime)) * strike_factor

        scale = 200.0 * J_prime
        dX_nT = dX * scale
        dZ_nT = dZ * scale

        cos_I_prime_f = np.cos(np.arctan(np.tan(I_f_r) / np.cos(D_f_r - alpha_r)))
        dT_nT = dX_nT * cos_I_prime_f * np.cos(D_f_r - alpha_r) + dZ_nT * np.sin(I_f_r)

        return dX_nT, dZ_nT, dT_nT

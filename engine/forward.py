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

    def calculate_gravity(self, x_obs, z_obs):
        gz_total = np.zeros_like(x_obs, dtype=np.float64)
        for body in self.bodies:
            gz_total += self._calc_single_gravity_25d(
                x_obs, z_obs, body.x, body.z, body.density, body.y_min, body.y_max
            )
        return gz_total

    def _calc_single_gravity_25d(self, x_obs, z_obs, x, z, density, y1, y2):
        gz = np.zeros_like(x_obs, dtype=np.float64)
        n_sides = len(x) - 1

        # Finite strike end-effect correction multiplier relative to infinite 2D
        # Ratio based on average depth/distance vs strike length (Rasmussen & Pedersen, 1979)
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

                a = rx1 + rz1 * np.tan(phi) if np.abs(np.cos(phi)) > 1e-6 else rx1
                term_2d = np.sin(phi) * (-dtheta) + np.cos(phi) * np.log(r2 / r1)

                # 2.5D strike correction factor
                r_mid = 0.5 * (r1 + r2)
                strike_factor = y_avg / np.sqrt(r_mid ** 2 + y_avg ** 2) if y_avg > 0 else 1.0

                gz[j] += a * term_2d * strike_factor

        return 2.0 * self.G * density * self.SI2MGAL * gz

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
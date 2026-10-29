import numpy as np
from scipy.optimize import least_squares
from engine.forward import PotentialField2D


def invert_body_parameters(target_body, x_obs, z_obs, obs_data, all_bodies=None, field_type='gravity',
                           field_params=None):
    """
    Performs Levenberg-Marquardt inversion to optimize target_body density/susceptibility
    and vertex depths against observed data, preserving multi-body context and topology closure.
    """
    if all_bodies is None:
        all_bodies = [target_body]

    engine = PotentialField2D(all_bodies)

    x_orig = target_body.x.copy()
    z_orig = target_body.z.copy()
    rho_orig = target_body.density
    k_orig = target_body.k_succ

    # Number of unique vertices (excluding closing vertex at -1)
    n_verts = len(z_orig) - 1

    if field_type == 'gravity':
        def objective(params):
            target_body.density = params[0]
            target_body.z[:n_verts] = params[1:]
            target_body.z[-1] = params[1]  # Enforce closure: z[-1] == z[0]
            calc = engine.calculate_gravity(x_obs, z_obs)
            return calc - obs_data

        p0 = [rho_orig] + list(z_orig[:n_verts])
    else:
        F = field_params.get('F', 50000.0) if field_params else 50000.0
        Inc = field_params.get('Inc', 60.0) if field_params else 60.0
        Dec = field_params.get('Dec', 0.0) if field_params else 0.0
        alpha = field_params.get('alpha', 0.0) if field_params else 0.0

        def objective(params):
            target_body.k_succ = params[0]
            target_body.z[:n_verts] = params[1:]
            target_body.z[-1] = params[1]  # Enforce closure: z[-1] == z[0]
            _, _, dT = engine.calculate_magnetics(x_obs, z_obs, F, Inc, Dec, alpha)
            return dT - obs_data

        p0 = [k_orig] + list(z_orig[:n_verts])

    try:
        res = least_squares(objective, p0, method='lm', max_nfev=100)

        # Apply optimized parameters
        if field_type == 'gravity':
            target_body.density = res.x[0]
        else:
            target_body.k_succ = res.x[0]

        target_body.z[:n_verts] = res.x[1:]
        target_body.z[-1] = res.x[1]
        return True
    except Exception:
        # Revert on failure
        target_body.density = rho_orig
        target_body.k_succ = k_orig
        target_body.update_polygon(x_orig, z_orig)
        return False
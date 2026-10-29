import numpy as np
from scipy.optimize import least_squares
from engine.forward import PotentialField2D


def invert_body_parameters(body, x_obs, z_obs, obs_data, field_type='gravity', field_params=None):
    """
    Performs least-squares inversion (Levenberg-Marquardt) to optimize
    a body's density contrast or vertex depths to fit observed data.
    """
    engine = PotentialField2D([body])

    # Optimization vector: [density, *z_vertices]
    # We keep x coordinates fixed and optimize depth (z) and density
    x_orig = body.x.copy()
    z_orig = body.z.copy()
    rho_orig = body.density

    if field_type == 'gravity':
        def objective(params):
            body.density = params[0]
            body.z[1:-1] = params[1:]  # Keep closure vertex fixed
            body.z[-1] = params[1]
            calc = engine.calculate_gravity(x_obs, z_obs)
            return calc - obs_data
    else:
        F = field_params.get('F', 48000)
        Inc = field_params.get('Inc', 55)

        def objective(params):
            body.k_succ = params[0]
            body.z[1:-1] = params[1:]
            body.z[-1] = params[1]
            _, _, dT = engine.calculate_magnetics(x_obs, z_obs, F, Inc, 0.0, 0.0)
            return dT - obs_data

    # Initial guess vector
    p0 = [rho_orig if field_type == 'gravity' else body.k_succ] + list(z_orig[1:-1])

    try:
        res = least_squares(objective, p0, method='lm', max_nfev=50)
        # Apply optimized parameters
        if field_type == 'gravity':
            body.density = res.x[0]
        else:
            body.k_succ = res.x[0]
        body.z[1:-1] = res.x[1:]
        body.z[-1] = res.x[1]
        success = True
    except Exception as e:
        # Revert on failure
        body.density = rho_orig
        body.k_succ = rho_orig if field_type != 'gravity' else body.k_succ
        body.update_polygon(x_orig, z_orig)
        success = False

    return success
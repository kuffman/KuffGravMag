import numpy as np
from scipy.interpolate import griddata


def extract_profile_from_xyz(x, y, v, x1, y1, x2, y2, n_points=250):
    """
    Extracts a 2D line profile from 3D XYZ scattered map data.
    """
    # Define the coordinates of the profile line
    prof_x = np.linspace(x1, x2, n_points)
    prof_y = np.linspace(y1, y2, n_points)

    # Calculate distance along the profile
    # Center the profile distance relative to the middle of the line
    dist = np.hypot(prof_x - x1, prof_y - y1)
    dist = dist - (dist.max() / 2.0)

    # Interpolate the Z-value data along the profile line
    prof_v = griddata((x, y), v, (prof_x, prof_y), method='linear')

    # Replace NaNs (points outside the data convex hull) with nearest neighbor
    if np.isnan(prof_v).any():
        prof_v_nearest = griddata((x, y), v, (prof_x, prof_y), method='nearest')
        prof_v[np.isnan(prof_v)] = prof_v_nearest[np.isnan(prof_v)]

    return dist, prof_v
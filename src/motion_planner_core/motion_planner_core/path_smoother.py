"""Path smoothing using cubic spline interpolation.

Takes discrete waypoints and produces a smooth, continuous path
using scipy's cubic spline. The result is C2 continuous (continuous
position, velocity, and acceleration).
"""

import numpy as np
from scipy.interpolate import CubicSpline


def smooth_path(waypoints: np.ndarray, num_samples: int = 200) -> np.ndarray:
    """Smooth a set of 2D waypoints using cubic spline interpolation.

    Args:
        waypoints: (N, 2) array of [x, y] waypoints.
        num_samples: Number of points to sample on the smoothed path.

    Returns:
        (num_samples, 2) array of smoothed [x, y] points.

    Raises:
        ValueError: If fewer than 2 waypoints are provided.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints for smoothing")

    # Parameterize by cumulative chord length for natural spacing
    diffs = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    t = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    # Fit cubic spline for x(t) and y(t) independently
    cs_x = CubicSpline(t, waypoints[:, 0], bc_type='natural')
    cs_y = CubicSpline(t, waypoints[:, 1], bc_type='natural')

    t_smooth = np.linspace(t[0], t[-1], num_samples)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)

    return np.column_stack([x_smooth, y_smooth])


def compute_path_curvature(path: np.ndarray) -> np.ndarray:
    """Compute curvature at each point of a 2D path.

    Uses finite differences: kappa = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)

    Args:
        path: (N, 2) array of [x, y] points.

    Returns:
        (N,) array of curvature values.
    """
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature


def compute_path_headings(path: np.ndarray) -> np.ndarray:
    """Compute heading angle at each point of a 2D path.

    Args:
        path: (N, 2) array of [x, y] points.

    Returns:
        (N,) array of heading angles in radians.
    """
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    return np.arctan2(dy, dx)

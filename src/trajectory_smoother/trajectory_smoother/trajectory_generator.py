"""Trajectory generation with trapezoidal velocity profile.

Takes a smooth path and generates a time-parameterized trajectory
with acceleration, cruise, and deceleration phases.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """A single point on the trajectory."""
    x: float
    y: float
    heading: float  # radians
    time: float     # seconds
    velocity: float # m/s


def generate_trajectory(
    path: np.ndarray,
    max_vel: float = 0.22,
    max_accel: float = 0.5,
    max_decel: float = 0.5,
    dt: float = 0.05,
) -> list[TrajectoryPoint]:
    """Generate a time-parameterized trajectory with trapezoidal velocity profile.

    Args:
        path: (N, 2) array of smoothed [x, y] points.
        max_vel: Maximum linear velocity (m/s).
        max_accel: Maximum acceleration (m/s^2).
        max_decel: Maximum deceleration (m/s^2).
        dt: Time step for trajectory sampling (s).

    Returns:
        List of TrajectoryPoint with time stamps and velocities.

    Raises:
        ValueError: If path has fewer than 2 points.
    """
    if len(path) < 2:
        raise ValueError("Need at least 2 path points for trajectory generation")

    # Compute cumulative arc length along the path
    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_dist = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_dist = cumulative_dist[-1]

    if total_dist < 1e-9:
        raise ValueError("Path has zero length")

    # Compute headings along the path
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.append(headings, headings[-1])  # repeat last heading

    # Build trapezoidal velocity profile over arc length
    vel_profile = _trapezoidal_profile(total_dist, max_vel, max_accel, max_decel, dt)

    # Map velocity profile (which gives us s(t)) back to path x,y
    trajectory = []
    for t, s, v in vel_profile:
        # Interpolate x, y, heading at arc length s
        x = np.interp(s, cumulative_dist, path[:, 0])
        y = np.interp(s, cumulative_dist, path[:, 1])
        heading = np.interp(s, cumulative_dist, headings)
        trajectory.append(TrajectoryPoint(x=x, y=y, heading=heading, time=t, velocity=v))

    return trajectory


def _trapezoidal_profile(
    total_dist: float,
    max_vel: float,
    max_accel: float,
    max_decel: float,
    dt: float,
) -> list[tuple[float, float, float]]:
    """Compute trapezoidal velocity profile.

    Returns list of (time, distance, velocity) tuples.
    Handles the case where the path is too short to reach max velocity
    (triangular profile).
    """
    # Distance to accelerate to max_vel
    d_accel = max_vel**2 / (2 * max_accel)
    # Distance to decelerate from max_vel
    d_decel = max_vel**2 / (2 * max_decel)

    if d_accel + d_decel > total_dist:
        # Triangular profile — can't reach max_vel
        # Peak velocity: v_peak = sqrt(2 * total_dist * a1 * a2 / (a1 + a2))
        v_peak = np.sqrt(2 * total_dist * max_accel * max_decel / (max_accel + max_decel))
        d_accel = v_peak**2 / (2 * max_accel)
        d_decel = v_peak**2 / (2 * max_decel)
        d_cruise = 0.0
        actual_max_vel = v_peak
    else:
        d_cruise = total_dist - d_accel - d_decel
        actual_max_vel = max_vel

    result = []
    t = 0.0
    s = 0.0
    v = 0.0

    while s < total_dist - 1e-6:
        result.append((t, s, v))

        if s < d_accel:
            v = min(v + max_accel * dt, actual_max_vel)
        elif s < d_accel + d_cruise:
            v = actual_max_vel
        else:
            v = max(v - max_decel * dt, 0.0)

        # If velocity hits zero before reaching the end, snap to end
        if v < 1e-9:
            break

        s += v * dt
        s = min(s, total_dist)
        t += dt

    # Final point
    result.append((t + dt, total_dist, 0.0))
    return result


def trajectory_to_arrays(trajectory: list[TrajectoryPoint]) -> dict[str, np.ndarray]:
    """Convert trajectory list to numpy arrays for plotting/analysis.

    Returns dict with keys: x, y, heading, time, velocity.
    """
    return {
        'x': np.array([p.x for p in trajectory]),
        'y': np.array([p.y for p in trajectory]),
        'heading': np.array([p.heading for p in trajectory]),
        'time': np.array([p.time for p in trajectory]),
        'velocity': np.array([p.velocity for p in trajectory]),
    }

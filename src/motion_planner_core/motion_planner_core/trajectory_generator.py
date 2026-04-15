"""Trajectory generation with trapezoidal velocity profile.

Takes a smooth path and generates a time-parameterized trajectory
with acceleration, cruise, and deceleration phases.
"""

import numpy as np
from dataclasses import dataclass
from motion_planner_core.types import Trajectory


@dataclass
class TrajectoryPoint:
    x: float
    y: float
    heading: float
    time: float
    velocity: float


def generate_trajectory(
    path: np.ndarray,
    max_vel: float = 0.22,
    max_accel: float = 0.5,
    max_decel: float = 0.5,
    dt: float = 0.05,
) -> Trajectory:
    """Generate a time-parameterized trajectory with trapezoidal velocity profile.

    Returns a Trajectory object.
    """
    if len(path) < 2:
        raise ValueError("Need at least 2 path points for trajectory generation")

    diffs = np.diff(path, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative_dist = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_dist = cumulative_dist[-1]

    if total_dist < 1e-9:
        raise ValueError("Path has zero length")

    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    headings = np.append(headings, headings[-1])

    vel_profile = _trapezoidal_profile(total_dist, max_vel, max_accel, max_decel, dt)

    xs, ys, hs, ts, vs = [], [], [], [], []
    for t, s, v in vel_profile:
        xs.append(np.interp(s, cumulative_dist, path[:, 0]))
        ys.append(np.interp(s, cumulative_dist, path[:, 1]))
        hs.append(np.interp(s, cumulative_dist, headings))
        ts.append(t)
        vs.append(v)

    return Trajectory(
        x=np.array(xs), y=np.array(ys), heading=np.array(hs),
        time=np.array(ts), velocity=np.array(vs),
    )


def _trapezoidal_profile(total_dist, max_vel, max_accel, max_decel, dt):
    d_accel = max_vel**2 / (2 * max_accel)
    d_decel = max_vel**2 / (2 * max_decel)

    if d_accel + d_decel > total_dist:
        v_peak = np.sqrt(2 * total_dist * max_accel * max_decel / (max_accel + max_decel))
        d_accel = v_peak**2 / (2 * max_accel)
        d_decel = v_peak**2 / (2 * max_decel)
        d_cruise = 0.0
        actual_max_vel = v_peak
    else:
        d_cruise = total_dist - d_accel - d_decel
        actual_max_vel = max_vel

    result = []
    t, s, v = 0.0, 0.0, 0.0

    while s < total_dist - 1e-6:
        result.append((t, s, v))
        if s < d_accel:
            v = min(v + max_accel * dt, actual_max_vel)
        elif s < d_accel + d_cruise:
            v = actual_max_vel
        else:
            v = max(v - max_decel * dt, 0.0)
        if v < 1e-9:
            break
        s = min(s + v * dt, total_dist)
        t += dt

    result.append((t + dt, total_dist, 0.0))
    return result


def trajectory_to_arrays(trajectory: Trajectory) -> dict[str, np.ndarray]:
    """Convenience: convert Trajectory to dict of arrays."""
    return trajectory.to_dict()

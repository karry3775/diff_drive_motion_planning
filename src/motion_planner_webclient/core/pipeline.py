"""End-to-end motion planning pipeline.

Orchestrates: costmap → plan path → smooth → generate trajectory.
Single entry point for the core library. No ROS dependency.
"""

import numpy as np
from motion_planner_core.costmap import Costmap, CostmapConfig
from motion_planner_core.path_planner import plan_path
from motion_planner_core.path_smoother import smooth_path
from motion_planner_core.trajectory_generator import generate_trajectory, TrajectoryPoint


def build_trajectory(
    waypoints: np.ndarray,
    costmap: Costmap = None,
    num_smooth_samples: int = 200,
    max_vel: float = 0.22,
    max_accel: float = 0.5,
    max_decel: float = 0.5,
    dt: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, list[TrajectoryPoint]]:
    """Run the full pipeline: plan → smooth → generate trajectory.

    Args:
        waypoints: (N, 2) array of [x, y] waypoints.
        costmap: Costmap with obstacles. None or empty = no obstacles.
        num_smooth_samples: Points on the smoothed path.
        max_vel: Max linear velocity (m/s).
        max_accel: Max acceleration (m/s^2).
        max_decel: Max deceleration (m/s^2).
        dt: Trajectory sample period (s).

    Returns:
        (planned_path, smoothed_path, trajectory) tuple.
        planned_path: (M, 2) obstacle-free waypoints from A*.
        smoothed_path: (num_smooth_samples, 2) cubic spline result.
        trajectory: list of TrajectoryPoint with time and velocity.
    """
    # Step 1: Plan obstacle-free path
    if costmap is not None and not costmap.is_empty():
        planned = plan_path(waypoints, costmap)
    else:
        planned = waypoints

    # Step 2: Smooth
    smoothed = smooth_path(planned, num_smooth_samples)

    # Step 3: Generate time-parameterized trajectory
    trajectory = generate_trajectory(smoothed, max_vel, max_accel, max_decel, dt)

    return planned, smoothed, trajectory


def build_trajectory_from_config(config: dict) -> tuple[np.ndarray, np.ndarray, list[TrajectoryPoint], Costmap]:
    """Build trajectory from a YAML config dict.

    Args:
        config: Parsed YAML config with waypoints, obstacles, etc.

    Returns:
        (planned_path, smoothed_path, trajectory, costmap) tuple.
    """
    waypoints = np.array(config['waypoints'], dtype=float)
    smooth_cfg = config.get('smoothing', {})
    traj_cfg = config.get('trajectory', {})

    # Build costmap from obstacles if present
    obstacles = config.get('obstacles', [])
    if obstacles:
        costmap = Costmap.from_obstacles(obstacles)
    else:
        costmap = None

    planned, smoothed, trajectory = build_trajectory(
        waypoints=waypoints,
        costmap=costmap,
        num_smooth_samples=smooth_cfg.get('num_samples', 200),
        max_vel=traj_cfg.get('max_velocity', 0.22),
        max_accel=traj_cfg.get('max_acceleration', 0.5),
        max_decel=traj_cfg.get('max_deceleration', 0.5),
        dt=traj_cfg.get('dt', 0.05),
    )

    return planned, smoothed, trajectory, costmap

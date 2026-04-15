"""End-to-end motion planning pipeline.

Orchestrates: costmap -> plan path -> smooth -> generate trajectory.
Single entry point for the core library. No ROS dependency.
"""

import numpy as np
from motion_planner_core.costmap import Costmap, CostmapConfig
from motion_planner_core.path_planner import plan_path
from motion_planner_core.path_smoother import smooth_path
from motion_planner_core.trajectory_generator import generate_trajectory
from motion_planner_core.types import Trajectory


class PlanningError(Exception):
    """Raised when the smoothed path violates the costmap."""
    pass


def _check_path_clearance(smoothed: np.ndarray, costmap: Costmap) -> list[int]:
    """Return indices of smoothed path points that are in occupied space."""
    violations = []
    for i in range(len(smoothed)):
        if not costmap.is_free_world(smoothed[i, 0], smoothed[i, 1]):
            violations.append(i)
    return violations


def build_trajectory(
    waypoints: np.ndarray,
    costmap: Costmap = None,
    num_smooth_samples: int = 200,
    max_vel: float = 0.22,
    max_accel: float = 0.5,
    max_decel: float = 0.5,
    dt: float = 0.05,
    strict: bool = False,
) -> tuple[np.ndarray, np.ndarray, Trajectory]:
    """Run the full pipeline: plan -> smooth -> generate trajectory.

    If strict=True and the smoothed path cuts through obstacles,
    raises PlanningError. If strict=False, logs a warning but continues.
    """
    if costmap is not None and not costmap.is_empty():
        planned = plan_path(waypoints, costmap)
    else:
        planned = waypoints

    smoothed = smooth_path(planned, num_smooth_samples)

    if costmap is not None and not costmap.is_empty():
        violations = _check_path_clearance(smoothed, costmap)
        if violations and strict:
            raise PlanningError(
                f"Smoothed path has {len(violations)} points in obstacle space. "
                f"Adjust waypoints or obstacle positions."
            )

    trajectory = generate_trajectory(smoothed, max_vel, max_accel, max_decel, dt)
    return planned, smoothed, trajectory


def build_trajectory_from_config(config: dict) -> tuple[np.ndarray, np.ndarray, Trajectory, Costmap]:
    """Build trajectory from a YAML config dict."""
    waypoints = np.array(config['waypoints'], dtype=float)
    smooth_cfg = config.get('smoothing', {})
    traj_cfg = config.get('trajectory', {})

    obstacles = config.get('obstacles', [])
    costmap = Costmap.from_obstacles(obstacles) if obstacles else None

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

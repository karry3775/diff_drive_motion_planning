"""motion_planner_core — Pure Python library for robot motion planning.

No ROS dependency. All algorithms are independently testable.

Modules:
    costmap             — 2D occupancy grid abstraction
    path_planner        — A* obstacle-aware path planning
    path_smoother       — Cubic spline smoothing
    trajectory_generator — Trapezoidal velocity profiling
    pure_pursuit        — Trajectory tracking controller
    potential_field     — Reactive obstacle avoidance (safety net)
    pipeline            — End-to-end orchestration
"""

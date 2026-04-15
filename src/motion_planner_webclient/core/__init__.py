"""motion_planner_core — Pure Python motion planning library.

Modules:
    types                  — Shared types: RobotState, VelocityCommand, Trajectory
    costmap                — 2D occupancy grid abstraction
    path_planner           — A* obstacle-aware path planning
    path_smoother          — Cubic spline smoothing
    trajectory_generator   — Trapezoidal velocity profiling
    controller             — Controller interface and factory
    pure_pursuit           — Geometric path tracking (for real robots)
    pid_controller         — Time-based trajectory tracking with fault detection
    feedforward_controller — Direct trajectory replay (for kinematic simulation)
    potential_field        — Reactive obstacle avoidance (safety net)
    pipeline               — End-to-end orchestration
"""

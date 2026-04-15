"""Feedforward trajectory controller.

Replays the trajectory directly — at time t, outputs the reference
linear velocity and computes angular velocity from path curvature.
Perfect tracking in a kinematic simulation.

Only useful for simulation/visualization. For real robots, use
Pure Pursuit or PID which handle disturbances.
"""

import math
import numpy as np
from motion_planner_core.types import RobotState, VelocityCommand, Trajectory
from motion_planner_core.controller import Controller


class FeedforwardController(Controller):

    def __init__(self, goal_tolerance: float = 0.1):
        self.goal_tolerance = goal_tolerance
        self._goal_reached = False
        self._curvatures = None

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    def _compute_curvatures(self, trajectory: Trajectory) -> np.ndarray:
        """Compute curvature at each trajectory point from the path geometry."""
        dx = np.gradient(trajectory.x)
        dy = np.gradient(trajectory.y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = (dx**2 + dy**2)**1.5
        denom[denom < 1e-12] = 1e-12
        return (dx * ddy - dy * ddx) / denom

    def compute_command(self, state: RobotState, trajectory: Trajectory,
                        elapsed_time: float) -> VelocityCommand:
        if self._goal_reached:
            return VelocityCommand(0.0, 0.0)

        goal_dist = math.hypot(trajectory.x[-1] - state.x, trajectory.y[-1] - state.y)
        if goal_dist < self.goal_tolerance or elapsed_time >= trajectory.duration:
            self._goal_reached = True
            return VelocityCommand(0.0, 0.0)

        if self._curvatures is None:
            self._curvatures = self._compute_curvatures(trajectory)

        # Find the trajectory index closest to current time
        idx = int(np.searchsorted(trajectory.time, elapsed_time))
        idx = min(idx, len(trajectory) - 1)

        linear = float(trajectory.velocity[idx])
        angular = float(linear * self._curvatures[idx])

        return VelocityCommand(linear=max(linear, 0.0), angular=angular)

    def reset(self):
        self._goal_reached = False
        self._curvatures = None

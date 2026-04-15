"""Pure Pursuit trajectory tracking controller for differential drive robots.

Geometric controller — steers toward a lookahead point on the path.
Ignores the time dimension of the trajectory; uses positions and velocities only.
"""

import math
import numpy as np
from motion_planner_core.types import RobotState, VelocityCommand, Trajectory
from motion_planner_core.controller import Controller


class PurePursuitController(Controller):

    def __init__(
        self,
        lookahead_distance: float = 0.3,
        goal_tolerance: float = 0.1,
        max_linear_vel: float = 0.22,
        max_angular_vel: float = 2.84,
        min_linear_vel: float = 0.05,
    ):
        self.lookahead_distance = lookahead_distance
        self.goal_tolerance = goal_tolerance
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.min_linear_vel = min_linear_vel
        self._goal_reached = False
        self._last_closest_idx = 0

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    def compute_command(self, state: RobotState, trajectory: Trajectory,
                        elapsed_time: float) -> VelocityCommand:
        if self._goal_reached:
            return VelocityCommand(0.0, 0.0)

        goal_dist = math.hypot(trajectory.x[-1] - state.x, trajectory.y[-1] - state.y)
        if goal_dist < self.goal_tolerance:
            self._goal_reached = True
            return VelocityCommand(0.0, 0.0)

        # Find closest point (allow backward search for recovery)
        search_start = max(0, self._last_closest_idx - 20)
        dx = trajectory.x[search_start:] - state.x
        dy = trajectory.y[search_start:] - state.y
        distances = np.sqrt(dx**2 + dy**2)
        closest_idx = search_start + int(np.argmin(distances))
        if closest_idx >= self._last_closest_idx:
            self._last_closest_idx = closest_idx

        # Find lookahead point
        all_dx = trajectory.x - state.x
        all_dy = trajectory.y - state.y
        all_distances = np.sqrt(all_dx**2 + all_dy**2)

        lookahead_idx = len(trajectory) - 1
        for i in range(closest_idx, len(trajectory)):
            if all_distances[i] >= self.lookahead_distance:
                lookahead_idx = i
                break

        # Transform to robot local frame
        dx_l = trajectory.x[lookahead_idx] - state.x
        dy_l = trajectory.y[lookahead_idx] - state.y
        local_x = math.cos(-state.theta) * dx_l - math.sin(-state.theta) * dy_l
        local_y = math.sin(-state.theta) * dx_l + math.cos(-state.theta) * dy_l

        l_sq = local_x**2 + local_y**2
        if l_sq < 1e-9:
            return VelocityCommand(0.0, 0.0)

        curvature = 2.0 * local_y / l_sq

        vel_idx = min(lookahead_idx, len(trajectory) - 1)
        linear = np.clip(max(trajectory.velocity[vel_idx], self.min_linear_vel),
                         0.0, self.max_linear_vel)
        angular = np.clip(linear * curvature, -self.max_angular_vel, self.max_angular_vel)

        return VelocityCommand(linear=float(linear), angular=float(angular))

    def reset(self):
        self._goal_reached = False
        self._last_closest_idx = 0

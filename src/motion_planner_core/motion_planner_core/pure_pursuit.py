"""Pure Pursuit trajectory tracking controller for differential drive robots.

The controller computes linear and angular velocity commands to follow
a reference trajectory by steering toward a lookahead point on the path.
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class RobotState:
    """Current state of the differential drive robot."""
    x: float
    y: float
    theta: float  # heading in radians


@dataclass
class VelocityCommand:
    """Velocity command for a differential drive robot."""
    linear: float   # m/s
    angular: float  # rad/s


class PurePursuitController:
    """Pure Pursuit controller for trajectory tracking.

    Finds a lookahead point on the trajectory and computes the curvature
    needed to reach it, then converts to (v, omega) commands.
    """

    def __init__(
        self,
        lookahead_distance: float = 0.3,
        min_lookahead: float = 0.15,
        max_lookahead: float = 0.6,
        goal_tolerance: float = 0.1,
        max_linear_vel: float = 0.22,
        max_angular_vel: float = 2.84,
        min_linear_vel: float = 0.05,
    ):
        self.lookahead_distance = lookahead_distance
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.goal_tolerance = goal_tolerance
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.min_linear_vel = min_linear_vel
        self._goal_reached = False
        self._last_closest_idx = 0  # monotonic progress along trajectory

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    def compute_command(
        self,
        state: RobotState,
        trajectory_x: np.ndarray,
        trajectory_y: np.ndarray,
        trajectory_vel: np.ndarray,
    ) -> VelocityCommand:
        """Compute velocity command to follow the trajectory.

        Args:
            state: Current robot state.
            trajectory_x: Array of trajectory x coordinates.
            trajectory_y: Array of trajectory y coordinates.
            trajectory_vel: Array of reference velocities along trajectory.

        Returns:
            VelocityCommand with linear and angular velocities.
        """
        if self._goal_reached:
            return VelocityCommand(0.0, 0.0)

        # Check if we've reached the goal
        goal_dist = math.hypot(
            trajectory_x[-1] - state.x,
            trajectory_y[-1] - state.y,
        )
        if goal_dist < self.goal_tolerance:
            self._goal_reached = True
            return VelocityCommand(0.0, 0.0)

        # Find the closest point on the trajectory
        # Allow some backward search to recover from obstacle avoidance deviations
        search_start = max(0, self._last_closest_idx - 20)
        dx = trajectory_x[search_start:] - state.x
        dy = trajectory_y[search_start:] - state.y
        distances = np.sqrt(dx**2 + dy**2)
        closest_idx = search_start + int(np.argmin(distances))
        # Only advance the index, never go backward more than the search window
        if closest_idx >= self._last_closest_idx:
            self._last_closest_idx = closest_idx

        # Find the lookahead point: first point ahead of closest that is >= lookahead_distance away
        all_dx = trajectory_x - state.x
        all_dy = trajectory_y - state.y
        all_distances = np.sqrt(all_dx**2 + all_dy**2)

        lookahead_idx = len(trajectory_x) - 1
        for i in range(closest_idx, len(trajectory_x)):
            if all_distances[i] >= self.lookahead_distance:
                lookahead_idx = i
                break

        # Lookahead point in global frame
        lx = trajectory_x[lookahead_idx]
        ly = trajectory_y[lookahead_idx]

        # Transform lookahead point to robot's local frame
        dx_local = lx - state.x
        dy_local = ly - state.y
        local_x = math.cos(-state.theta) * dx_local - math.sin(-state.theta) * dy_local
        local_y = math.sin(-state.theta) * dx_local + math.cos(-state.theta) * dy_local

        # Compute curvature: gamma = 2 * y_local / L^2
        L_sq = local_x**2 + local_y**2
        if L_sq < 1e-9:
            return VelocityCommand(0.0, 0.0)

        curvature = 2.0 * local_y / L_sq

        # Reference velocity from trajectory — use at least min_linear_vel to avoid stalling
        ref_vel = trajectory_vel[min(lookahead_idx, len(trajectory_vel) - 1)]
        linear_vel = max(ref_vel, self.min_linear_vel)
        linear_vel = min(linear_vel, self.max_linear_vel)

        # Angular velocity = v * curvature
        angular_vel = linear_vel * curvature
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)

        return VelocityCommand(linear=linear_vel, angular=float(angular_vel))

    def reset(self):
        """Reset controller state for a new trajectory."""
        self._goal_reached = False
        self._last_closest_idx = 0

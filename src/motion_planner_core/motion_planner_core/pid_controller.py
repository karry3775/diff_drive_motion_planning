"""Time-based PID trajectory tracking controller.

Tracks where the robot SHOULD be at the current time, computes
lateral/longitudinal error, applies PID corrections on top of
the feedforward velocity from the trajectory.

Faults if cross-track error exceeds threshold.
"""

import math
import numpy as np
from motion_planner_core.types import RobotState, VelocityCommand, Trajectory
from motion_planner_core.controller import Controller


class PIDController(Controller):

    def __init__(
        self,
        kp_lateral: float = 2.0,
        kp_longitudinal: float = 1.0,
        kd_lateral: float = 0.1,
        kd_longitudinal: float = 0.0,
        max_linear_vel: float = 0.22,
        max_angular_vel: float = 2.84,
        max_cross_track_error: float = 0.5,
        goal_tolerance: float = 0.1,
    ):
        self.kp_lat = kp_lateral
        self.kp_lon = kp_longitudinal
        self.kd_lat = kd_lateral
        self.kd_lon = kd_longitudinal
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.max_cross_track_error = max_cross_track_error
        self.goal_tolerance = goal_tolerance
        self._goal_reached = False
        self._faulted = False
        self._prev_error_lat = 0.0
        self._prev_error_lon = 0.0

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    @property
    def faulted(self) -> bool:
        return self._faulted

    def compute_command(self, state: RobotState, trajectory: Trajectory,
                        elapsed_time: float) -> VelocityCommand:
        if self._goal_reached or self._faulted:
            return VelocityCommand(0.0, 0.0)

        goal_dist = math.hypot(trajectory.x[-1] - state.x, trajectory.y[-1] - state.y)
        if goal_dist < self.goal_tolerance:
            self._goal_reached = True
            return VelocityCommand(0.0, 0.0)

        # Reference state at current time
        ref_x, ref_y, ref_heading, ref_vel = trajectory.at_time(elapsed_time)
        ref_vel = max(ref_vel, 0.03)

        # Error in world frame
        dx = ref_x - state.x
        dy = ref_y - state.y

        # Check fault
        cross_track = math.hypot(dx, dy)
        if cross_track > self.max_cross_track_error:
            self._faulted = True
            return VelocityCommand(0.0, 0.0)

        # Transform error to robot frame
        cos_t = math.cos(-state.theta)
        sin_t = math.sin(-state.theta)
        error_lon = cos_t * dx - sin_t * dy
        error_lat = sin_t * dx + cos_t * dy

        # PD lateral → angular
        d_lat = error_lat - self._prev_error_lat
        self._prev_error_lat = error_lat
        angular = self.kp_lat * error_lat + self.kd_lat * d_lat

        # PD longitudinal → linear (feedforward + correction)
        d_lon = error_lon - self._prev_error_lon
        self._prev_error_lon = error_lon
        linear = ref_vel + self.kp_lon * error_lon + self.kd_lon * d_lon

        linear = float(np.clip(linear, 0.0, self.max_linear_vel))
        angular = float(np.clip(angular, -self.max_angular_vel, self.max_angular_vel))

        return VelocityCommand(linear=linear, angular=angular)

    def reset(self):
        self._goal_reached = False
        self._faulted = False
        self._prev_error_lat = 0.0
        self._prev_error_lon = 0.0

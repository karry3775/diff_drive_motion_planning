"""Time-based PID trajectory tracking controller.

Tracks where the robot SHOULD be at the current time according to
the trajectory, and uses PID corrections to minimize position error.
Faults if cross-track error exceeds a threshold.
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class RobotState:
    x: float
    y: float
    theta: float


@dataclass
class VelocityCommand:
    linear: float
    angular: float


class TrajectoryPIDController:
    """PID controller that tracks a time-parameterized trajectory.

    At each timestep, looks up the reference (x, y, heading, velocity)
    at the current elapsed time, computes lateral and longitudinal error,
    and applies PID corrections on top of the feedforward velocity.
    """

    def __init__(
        self,
        kp_lateral: float = 2.0,
        kp_longitudinal: float = 1.0,
        ki_lateral: float = 0.0,
        ki_longitudinal: float = 0.0,
        kd_lateral: float = 0.1,
        kd_longitudinal: float = 0.0,
        max_linear_vel: float = 0.22,
        max_angular_vel: float = 2.84,
        max_cross_track_error: float = 0.5,
        goal_tolerance: float = 0.1,
    ):
        self.kp_lat = kp_lateral
        self.ki_lat = ki_lateral
        self.kd_lat = kd_lateral
        self.kp_lon = kp_longitudinal
        self.ki_lon = ki_longitudinal
        self.kd_lon = kd_longitudinal
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.max_cross_track_error = max_cross_track_error
        self.goal_tolerance = goal_tolerance

        self._start_time = None
        self._goal_reached = False
        self._faulted = False
        self._integral_lat = 0.0
        self._integral_lon = 0.0
        self._prev_error_lat = 0.0
        self._prev_error_lon = 0.0

    @property
    def goal_reached(self) -> bool:
        return self._goal_reached

    @property
    def faulted(self) -> bool:
        return self._faulted

    def compute_command(
        self,
        state: RobotState,
        traj_x: np.ndarray,
        traj_y: np.ndarray,
        traj_vel: np.ndarray,
        traj_heading: np.ndarray,
        traj_time: np.ndarray,
        current_time: float,
    ) -> VelocityCommand:
        """Compute velocity command based on time-referenced trajectory.

        Args:
            state: Current robot state.
            traj_x, traj_y: Trajectory positions.
            traj_vel: Reference velocities.
            traj_heading: Reference headings.
            traj_time: Timestamps for each trajectory point.
            current_time: Elapsed time since trajectory start.
        """
        if self._goal_reached or self._faulted:
            return VelocityCommand(0.0, 0.0)

        # Check goal
        goal_dist = math.hypot(traj_x[-1] - state.x, traj_y[-1] - state.y)
        if goal_dist < self.goal_tolerance:
            self._goal_reached = True
            return VelocityCommand(0.0, 0.0)

        # Find reference point at current time
        if current_time >= traj_time[-1]:
            # Past end of trajectory — just go to goal
            ref_x, ref_y = traj_x[-1], traj_y[-1]
            ref_heading = traj_heading[-1]
            ref_vel = 0.05
        else:
            ref_x = np.interp(current_time, traj_time, traj_x)
            ref_y = np.interp(current_time, traj_time, traj_y)
            ref_heading = np.interp(current_time, traj_time, traj_heading)
            ref_vel = np.interp(current_time, traj_time, traj_vel)
            ref_vel = max(ref_vel, 0.03)

        # Error in world frame
        dx = ref_x - state.x
        dy = ref_y - state.y

        # Transform to robot frame
        cos_t = math.cos(-state.theta)
        sin_t = math.sin(-state.theta)
        error_lon = cos_t * dx - sin_t * dy  # along robot heading
        error_lat = sin_t * dx + cos_t * dy  # perpendicular

        cross_track = math.hypot(dx, dy)
        if cross_track > self.max_cross_track_error:
            self._faulted = True
            return VelocityCommand(0.0, 0.0)

        # PID lateral → angular correction
        self._integral_lat += error_lat
        d_lat = error_lat - self._prev_error_lat
        self._prev_error_lat = error_lat
        angular = (self.kp_lat * error_lat +
                   self.ki_lat * self._integral_lat +
                   self.kd_lat * d_lat)

        # PID longitudinal → linear velocity correction
        self._integral_lon += error_lon
        d_lon = error_lon - self._prev_error_lon
        self._prev_error_lon = error_lon
        linear = ref_vel + (self.kp_lon * error_lon +
                            self.ki_lon * self._integral_lon +
                            self.kd_lon * d_lon)

        linear = np.clip(linear, 0.0, self.max_linear_vel)
        angular = np.clip(angular, -self.max_angular_vel, self.max_angular_vel)

        return VelocityCommand(linear=float(linear), angular=float(angular))

    def reset(self):
        self._goal_reached = False
        self._faulted = False
        self._integral_lat = 0.0
        self._integral_lon = 0.0
        self._prev_error_lat = 0.0
        self._prev_error_lon = 0.0

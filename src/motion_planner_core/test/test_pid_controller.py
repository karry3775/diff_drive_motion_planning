"""Tests for PID trajectory tracking controller."""

import math
import numpy as np
import pytest
from motion_planner_core.pid_controller import PIDController
from motion_planner_core.types import RobotState, Trajectory


def make_straight_trajectory(length=5.0, n=200, vel=0.2):
    x = np.linspace(0, length, n)
    y = np.zeros(n)
    heading = np.zeros(n)
    velocity = np.full(n, vel)
    time = np.linspace(0, length / vel, n)
    return Trajectory(x, y, heading, velocity, time)


class TestPIDController:

    @pytest.fixture
    def controller(self):
        return PIDController(goal_tolerance=0.1, max_linear_vel=0.22, max_cross_track_error=0.5)

    @pytest.fixture
    def straight_traj(self):
        return make_straight_trajectory()

    def test_on_track_produces_forward_motion(self, controller, straight_traj):
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        cmd = controller.compute_command(state, straight_traj, 0.0)
        assert cmd.linear > 0.0
        assert abs(cmd.angular) < 0.5

    def test_corrects_lateral_offset(self, controller, straight_traj):
        state = RobotState(x=0.0, y=0.3, theta=0.0)
        cmd = controller.compute_command(state, straight_traj, 0.0)
        # Should steer toward path (negative angular for positive y offset)
        assert cmd.angular < 0.0

    def test_goal_reached(self, controller, straight_traj):
        state = RobotState(x=5.0, y=0.0, theta=0.0)
        cmd = controller.compute_command(state, straight_traj, straight_traj.duration)
        assert controller.goal_reached
        assert cmd.linear == 0.0

    def test_faults_on_large_error(self, controller, straight_traj):
        state = RobotState(x=0.0, y=2.0, theta=0.0)  # way off track
        cmd = controller.compute_command(state, straight_traj, 0.0)
        assert controller.faulted
        assert cmd.linear == 0.0

    def test_reset_clears_fault(self, controller, straight_traj):
        state = RobotState(x=0.0, y=2.0, theta=0.0)
        controller.compute_command(state, straight_traj, 0.0)
        assert controller.faulted
        controller.reset()
        assert not controller.faulted
        assert not controller.goal_reached

    def test_respects_max_velocity(self, controller):
        traj = make_straight_trajectory(vel=10.0)
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        cmd = controller.compute_command(state, traj, 0.0)
        assert cmd.linear <= controller.max_linear_vel + 1e-6

    def test_tracks_straight_line(self):
        controller = PIDController(goal_tolerance=0.15, max_linear_vel=0.22)
        traj = make_straight_trajectory()
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        dt = 0.05

        for step in range(5000):
            t = step * dt
            cmd = controller.compute_command(state, traj, t)
            if controller.goal_reached:
                break
            state.x += cmd.linear * math.cos(state.theta) * dt
            state.y += cmd.linear * math.sin(state.theta) * dt
            state.theta += cmd.angular * dt

        assert controller.goal_reached

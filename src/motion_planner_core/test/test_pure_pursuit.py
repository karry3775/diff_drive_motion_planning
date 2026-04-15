"""Tests for Pure Pursuit controller and full pipeline integration."""

import math
import numpy as np
import pytest
from motion_planner_core.pure_pursuit import PurePursuitController
from motion_planner_core.types import RobotState, VelocityCommand, Trajectory


def make_straight_trajectory(length=5.0, n=200, vel=0.2):
    x = np.linspace(0, length, n)
    y = np.zeros(n)
    heading = np.zeros(n)
    velocity = np.full(n, vel)
    time = np.linspace(0, length / vel, n)
    return Trajectory(x, y, heading, velocity, time)


class TestPurePursuitController:

    @pytest.fixture
    def controller(self):
        return PurePursuitController(lookahead_distance=0.3, goal_tolerance=0.1, max_linear_vel=0.22)

    @pytest.fixture
    def straight_traj(self):
        return make_straight_trajectory()

    def test_straight_line_low_angular(self, controller, straight_traj):
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        cmd = controller.compute_command(state, straight_traj, 0.0)
        assert abs(cmd.angular) < 0.1
        assert cmd.linear > 0.0

    def test_turns_toward_path(self, controller, straight_traj):
        state = RobotState(x=0.0, y=0.5, theta=0.0)
        cmd = controller.compute_command(state, straight_traj, 0.0)
        assert cmd.angular < 0.0

    def test_goal_reached(self, controller, straight_traj):
        state = RobotState(x=5.0, y=0.0, theta=0.0)
        cmd = controller.compute_command(state, straight_traj, 0.0)
        assert controller.goal_reached
        assert cmd.linear == 0.0
        assert cmd.angular == 0.0

    def test_respects_max_velocity(self, controller):
        traj = make_straight_trajectory(vel=10.0)
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        cmd = controller.compute_command(state, traj, 0.0)
        assert cmd.linear <= controller.max_linear_vel + 1e-6

    def test_respects_max_angular_velocity(self, controller):
        x = np.linspace(0, 5, 200)
        traj = Trajectory(x, x, np.zeros(200), np.full(200, 0.22), np.linspace(0, 25, 200))
        state = RobotState(x=0.0, y=0.0, theta=math.pi)
        cmd = controller.compute_command(state, traj, 0.0)
        assert abs(cmd.angular) <= controller.max_angular_vel + 1e-6

    def test_reset(self, controller, straight_traj):
        controller.compute_command(RobotState(5.0, 0.0, 0.0), straight_traj, 0.0)
        assert controller.goal_reached
        controller.reset()
        assert not controller.goal_reached

    def test_after_goal_reached_returns_zero(self, controller, straight_traj):
        state = RobotState(x=5.0, y=0.0, theta=0.0)
        controller.compute_command(state, straight_traj, 0.0)
        cmd = controller.compute_command(state, straight_traj, 0.0)
        assert cmd.linear == 0.0
        assert cmd.angular == 0.0


class TestFullPipelineIntegration:

    def test_robot_reaches_goal(self):
        from motion_planner_core.path_smoother import smooth_path
        from motion_planner_core.trajectory_generator import generate_trajectory

        waypoints = np.array([[0, 0], [1, 0.5], [2, 1.5], [3, 1], [4, 2], [5, 2.5]])
        smoothed = smooth_path(waypoints, num_samples=200)
        trajectory = generate_trajectory(smoothed, max_vel=0.22, max_accel=0.5)

        controller = PurePursuitController(lookahead_distance=0.3, goal_tolerance=0.15, max_linear_vel=0.22)
        init_heading = math.atan2(waypoints[1, 1], waypoints[1, 0])
        state = RobotState(x=0.0, y=0.0, theta=init_heading)
        dt = 0.05

        for step in range(5000):
            cmd = controller.compute_command(state, trajectory, step * dt)
            if controller.goal_reached:
                break
            state.x += cmd.linear * math.cos(state.theta) * dt
            state.y += cmd.linear * math.sin(state.theta) * dt
            state.theta += cmd.angular * dt

        assert controller.goal_reached
        final_dist = math.hypot(state.x - waypoints[-1, 0], state.y - waypoints[-1, 1])
        assert final_dist < 0.3

    def test_tracking_error_bounded(self):
        from motion_planner_core.path_smoother import smooth_path
        from motion_planner_core.trajectory_generator import generate_trajectory

        waypoints = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
        smoothed = smooth_path(waypoints, num_samples=200)
        trajectory = generate_trajectory(smoothed, max_vel=0.22, max_accel=0.5)

        controller = PurePursuitController(lookahead_distance=0.3, goal_tolerance=0.1, max_linear_vel=0.22)
        init_heading = math.atan2(1.0, 1.0)
        state = RobotState(x=0.0, y=0.0, theta=init_heading)
        dt = 0.05
        max_error = 0.0

        for step in range(5000):
            cmd = controller.compute_command(state, trajectory, step * dt)
            if controller.goal_reached:
                break
            state.x += cmd.linear * math.cos(state.theta) * dt
            state.y += cmd.linear * math.sin(state.theta) * dt
            state.theta += cmd.angular * dt

            dists = np.sqrt((trajectory.x - state.x)**2 + (trajectory.y - state.y)**2)
            max_error = max(max_error, float(np.min(dists)))

        assert max_error < 0.2

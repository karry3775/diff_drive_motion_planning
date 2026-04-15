"""Tests for Pure Pursuit controller."""

import math
import numpy as np
import pytest
from trajectory_smoother.pure_pursuit import (
    PurePursuitController,
    RobotState,
    VelocityCommand,
)


class TestPurePursuitController:
    """Tests for the Pure Pursuit trajectory tracking controller."""

    @pytest.fixture
    def controller(self):
        return PurePursuitController(
            lookahead_distance=0.3,
            goal_tolerance=0.1,
            max_linear_vel=0.22,
            max_angular_vel=2.84,
        )

    @pytest.fixture
    def straight_trajectory(self):
        x = np.linspace(0, 5, 200)
        y = np.zeros(200)
        vel = np.full(200, 0.2)
        return x, y, vel

    def test_straight_line_low_angular(self, controller, straight_trajectory):
        """Robot on a straight path should have near-zero angular velocity."""
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        x, y, vel = straight_trajectory
        cmd = controller.compute_command(state, x, y, vel)
        assert abs(cmd.angular) < 0.1
        assert cmd.linear > 0.0

    def test_turns_toward_path(self, controller, straight_trajectory):
        """Robot offset from path should turn toward it."""
        state = RobotState(x=0.0, y=0.5, theta=0.0)  # offset above path
        x, y, vel = straight_trajectory
        cmd = controller.compute_command(state, x, y, vel)
        # Should turn clockwise (negative angular) to reach the path below
        assert cmd.angular < 0.0

    def test_goal_reached(self, controller, straight_trajectory):
        """Controller should report goal reached when near the end."""
        x, y, vel = straight_trajectory
        state = RobotState(x=5.0, y=0.0, theta=0.0)
        cmd = controller.compute_command(state, x, y, vel)
        assert controller.goal_reached
        assert cmd.linear == 0.0
        assert cmd.angular == 0.0

    def test_respects_max_velocity(self, controller):
        x = np.linspace(0, 5, 200)
        y = np.zeros(200)
        vel = np.full(200, 10.0)  # way above max
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        cmd = controller.compute_command(state, x, y, vel)
        assert cmd.linear <= controller.max_linear_vel + 1e-6

    def test_respects_max_angular_velocity(self, controller):
        x = np.linspace(0, 5, 200)
        y = np.linspace(0, 5, 200)
        vel = np.full(200, 0.22)
        # Robot facing wrong direction
        state = RobotState(x=0.0, y=0.0, theta=math.pi)
        cmd = controller.compute_command(state, x, y, vel)
        assert abs(cmd.angular) <= controller.max_angular_vel + 1e-6

    def test_reset(self, controller, straight_trajectory):
        x, y, vel = straight_trajectory
        state = RobotState(x=5.0, y=0.0, theta=0.0)
        controller.compute_command(state, x, y, vel)
        assert controller.goal_reached
        controller.reset()
        assert not controller.goal_reached

    def test_after_goal_reached_returns_zero(self, controller, straight_trajectory):
        x, y, vel = straight_trajectory
        state = RobotState(x=5.0, y=0.0, theta=0.0)
        controller.compute_command(state, x, y, vel)
        # Subsequent calls should also return zero
        cmd = controller.compute_command(state, x, y, vel)
        assert cmd.linear == 0.0
        assert cmd.angular == 0.0


class TestFullPipelineIntegration:
    """Integration test: smooth → generate trajectory → track with pure pursuit."""

    def test_robot_reaches_goal(self):
        from trajectory_smoother.path_smoother import smooth_path
        from trajectory_smoother.trajectory_generator import (
            generate_trajectory,
            trajectory_to_arrays,
        )

        waypoints = np.array([
            [0.0, 0.0], [1.0, 0.5], [2.0, 1.5],
            [3.0, 1.0], [4.0, 2.0], [5.0, 2.5],
        ])

        smoothed = smooth_path(waypoints, num_samples=200)
        trajectory = generate_trajectory(smoothed, max_vel=0.22, max_accel=0.5)
        arrays = trajectory_to_arrays(trajectory)

        controller = PurePursuitController(
            lookahead_distance=0.3, goal_tolerance=0.15, max_linear_vel=0.22,
        )

        # Set initial heading toward first segment
        init_heading = math.atan2(waypoints[1, 1] - waypoints[0, 1],
                                  waypoints[1, 0] - waypoints[0, 0])
        state = RobotState(x=0.0, y=0.0, theta=init_heading)
        dt = 0.05
        max_steps = 5000

        for _ in range(max_steps):
            cmd = controller.compute_command(
                state, arrays['x'], arrays['y'], arrays['velocity']
            )
            if controller.goal_reached:
                break
            state.x += cmd.linear * math.cos(state.theta) * dt
            state.y += cmd.linear * math.sin(state.theta) * dt
            state.theta += cmd.angular * dt

        assert controller.goal_reached, "Robot did not reach the goal"
        final_dist = math.hypot(state.x - waypoints[-1, 0], state.y - waypoints[-1, 1])
        assert final_dist < 0.3, f"Robot ended too far from goal: {final_dist:.3f}m"

    def test_tracking_error_bounded(self):
        """Cross-track error should stay small throughout the trajectory."""
        from trajectory_smoother.path_smoother import smooth_path
        from trajectory_smoother.trajectory_generator import (
            generate_trajectory,
            trajectory_to_arrays,
        )

        waypoints = np.array([
            [0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0],
        ])

        smoothed = smooth_path(waypoints, num_samples=200)
        trajectory = generate_trajectory(smoothed, max_vel=0.22, max_accel=0.5)
        arrays = trajectory_to_arrays(trajectory)

        controller = PurePursuitController(
            lookahead_distance=0.3, goal_tolerance=0.1, max_linear_vel=0.22,
        )

        init_heading = math.atan2(waypoints[1, 1] - waypoints[0, 1],
                                  waypoints[1, 0] - waypoints[0, 0])
        state = RobotState(x=0.0, y=0.0, theta=init_heading)
        dt = 0.05
        max_error = 0.0
        max_steps = 5000

        for _ in range(max_steps):
            cmd = controller.compute_command(
                state, arrays['x'], arrays['y'], arrays['velocity']
            )
            if controller.goal_reached:
                break
            state.x += cmd.linear * math.cos(state.theta) * dt
            state.y += cmd.linear * math.sin(state.theta) * dt
            state.theta += cmd.angular * dt

            dists = np.sqrt(
                (arrays['x'] - state.x)**2 + (arrays['y'] - state.y)**2
            )
            max_error = max(max_error, float(np.min(dists)))

        assert max_error < 0.2, f"Tracking error too large: {max_error:.3f}m"

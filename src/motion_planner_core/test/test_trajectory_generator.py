"""Tests for trajectory generation module."""

import numpy as np
import pytest
from motion_planner_core.trajectory_generator import (
    generate_trajectory,
    trajectory_to_arrays,
)


class TestGenerateTrajectory:
    """Tests for trapezoidal velocity profile trajectory generation."""

    @pytest.fixture
    def straight_path(self):
        return np.column_stack([np.linspace(0, 5, 100), np.zeros(100)])

    @pytest.fixture
    def curved_path(self):
        t = np.linspace(0, np.pi, 100)
        return np.column_stack([np.cos(t), np.sin(t)])

    def test_output_format(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert len(traj) > 0
        assert hasattr(traj[0], 'x')
        assert hasattr(traj[0], 'y')
        assert hasattr(traj[0], 'time')
        assert hasattr(traj[0], 'velocity')

    def test_starts_at_path_start(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert abs(traj[0].x - straight_path[0, 0]) < 1e-6
        assert abs(traj[0].y - straight_path[0, 1]) < 1e-6

    def test_ends_at_path_end(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert abs(traj[-1].x - straight_path[-1, 0]) < 0.1
        assert abs(traj[-1].y - straight_path[-1, 1]) < 0.1

    def test_time_monotonically_increasing(self, straight_path):
        traj = generate_trajectory(straight_path)
        times = [p.time for p in traj]
        assert all(t2 >= t1 for t1, t2 in zip(times, times[1:]))

    def test_velocity_respects_max(self, straight_path):
        max_vel = 0.22
        traj = generate_trajectory(straight_path, max_vel=max_vel)
        for p in traj:
            assert p.velocity <= max_vel + 1e-6

    def test_starts_and_ends_at_zero_velocity(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert traj[0].velocity == pytest.approx(0.0, abs=1e-6)
        assert traj[-1].velocity == pytest.approx(0.0, abs=1e-6)

    def test_trapezoidal_shape(self, straight_path):
        """Verify the velocity profile has accel, cruise, decel phases."""
        traj = generate_trajectory(straight_path, max_vel=0.22, max_accel=0.5)
        arrays = trajectory_to_arrays(traj)
        vel = arrays['velocity']
        # Should have increasing, constant, and decreasing sections
        max_v = np.max(vel)
        assert max_v > 0.0
        # Find where velocity first reaches max
        at_max = np.where(np.abs(vel - max_v) < 0.01)[0]
        assert len(at_max) > 1, "Should have a cruise phase"

    def test_short_path_triangular_profile(self):
        """Very short path should produce triangular (not trapezoidal) profile."""
        short_path = np.column_stack([np.linspace(0, 0.05, 20), np.zeros(20)])
        traj = generate_trajectory(short_path, max_vel=0.22, max_accel=0.5)
        arrays = trajectory_to_arrays(traj)
        # Peak velocity should be less than max
        assert np.max(arrays['velocity']) < 0.22

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            generate_trajectory(np.array([[0, 0]]))

    def test_zero_length_path_raises(self):
        with pytest.raises(ValueError, match="zero length"):
            generate_trajectory(np.array([[1, 1], [1, 1]]))

    def test_trajectory_to_arrays(self, straight_path):
        traj = generate_trajectory(straight_path)
        arrays = trajectory_to_arrays(traj)
        assert 'x' in arrays
        assert 'y' in arrays
        assert 'time' in arrays
        assert 'velocity' in arrays
        assert len(arrays['x']) == len(traj)

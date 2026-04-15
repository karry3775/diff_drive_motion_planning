"""Tests for trajectory generation module."""

import numpy as np
import pytest
from motion_planner_core.trajectory_generator import generate_trajectory, trajectory_to_arrays


class TestGenerateTrajectory:

    @pytest.fixture
    def straight_path(self):
        return np.column_stack([np.linspace(0, 5, 100), np.zeros(100)])

    def test_output_has_arrays(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert len(traj) > 0
        assert len(traj.x) == len(traj.time)

    def test_starts_at_path_start(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert abs(traj.x[0] - straight_path[0, 0]) < 1e-6
        assert abs(traj.y[0] - straight_path[0, 1]) < 1e-6

    def test_ends_at_path_end(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert abs(traj.x[-1] - straight_path[-1, 0]) < 0.1
        assert abs(traj.y[-1] - straight_path[-1, 1]) < 0.1

    def test_time_monotonically_increasing(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert all(traj.time[i+1] >= traj.time[i] for i in range(len(traj)-1))

    def test_velocity_respects_max(self, straight_path):
        traj = generate_trajectory(straight_path, max_vel=0.22)
        assert np.all(traj.velocity <= 0.22 + 1e-6)

    def test_starts_and_ends_at_zero_velocity(self, straight_path):
        traj = generate_trajectory(straight_path)
        assert traj.velocity[0] == pytest.approx(0.0, abs=1e-6)
        assert traj.velocity[-1] == pytest.approx(0.0, abs=1e-6)

    def test_trapezoidal_shape(self, straight_path):
        traj = generate_trajectory(straight_path, max_vel=0.22, max_accel=0.5)
        max_v = np.max(traj.velocity)
        at_max = np.sum(np.abs(traj.velocity - max_v) < 0.01)
        assert max_v > 0.0
        assert at_max > 1  # cruise phase exists

    def test_short_path_triangular_profile(self):
        short_path = np.column_stack([np.linspace(0, 0.05, 20), np.zeros(20)])
        traj = generate_trajectory(short_path, max_vel=0.22, max_accel=0.5)
        assert np.max(traj.velocity) < 0.22

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
        assert 'time' in arrays
        assert len(arrays['x']) == len(traj)

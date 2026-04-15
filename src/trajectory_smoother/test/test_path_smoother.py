"""Tests for path smoothing module."""

import numpy as np
import pytest
from trajectory_smoother.path_smoother import (
    smooth_path,
    compute_path_curvature,
    compute_path_headings,
)


class TestSmoothPath:
    """Tests for the cubic spline path smoothing function."""

    def test_output_shape(self):
        waypoints = np.array([[0, 0], [1, 1], [2, 0], [3, 1]])
        result = smooth_path(waypoints, num_samples=100)
        assert result.shape == (100, 2)

    def test_passes_through_endpoints(self):
        waypoints = np.array([[0, 0], [1, 2], [3, 1], [5, 3]])
        result = smooth_path(waypoints, num_samples=500)
        np.testing.assert_allclose(result[0], waypoints[0], atol=1e-6)
        np.testing.assert_allclose(result[-1], waypoints[-1], atol=1e-6)

    def test_passes_through_all_waypoints(self):
        waypoints = np.array([[0, 0], [1, 1], [2, 0.5], [3, 2]])
        result = smooth_path(waypoints, num_samples=1000)
        for wp in waypoints:
            dists = np.linalg.norm(result - wp, axis=1)
            assert np.min(dists) < 0.01, f"Smoothed path doesn't pass near waypoint {wp}"

    def test_smoothness_c2_continuous(self):
        """Verify the path is smooth by checking curvature doesn't have jumps."""
        waypoints = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
        result = smooth_path(waypoints, num_samples=500)
        curvature = compute_path_curvature(result)
        # Curvature should change gradually (no huge jumps)
        curvature_diff = np.abs(np.diff(curvature))
        assert np.max(curvature_diff) < 1.0, "Curvature has discontinuities"

    def test_straight_line_stays_straight(self):
        waypoints = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        result = smooth_path(waypoints, num_samples=100)
        # Y values should all be ~0
        np.testing.assert_allclose(result[:, 1], 0.0, atol=1e-10)

    def test_minimum_waypoints(self):
        waypoints = np.array([[0, 0], [1, 1]])
        result = smooth_path(waypoints, num_samples=50)
        assert result.shape == (50, 2)

    def test_too_few_waypoints_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            smooth_path(np.array([[0, 0]]))

    def test_num_samples_respected(self):
        waypoints = np.array([[0, 0], [1, 1], [2, 0]])
        for n in [10, 50, 500]:
            result = smooth_path(waypoints, num_samples=n)
            assert len(result) == n


class TestComputePathHeadings:
    def test_straight_east(self):
        path = np.array([[0, 0], [1, 0], [2, 0]])
        headings = compute_path_headings(path)
        np.testing.assert_allclose(headings, 0.0, atol=1e-10)

    def test_straight_north(self):
        path = np.array([[0, 0], [0, 1], [0, 2]])
        headings = compute_path_headings(path)
        np.testing.assert_allclose(headings, np.pi / 2, atol=1e-10)

    def test_diagonal(self):
        path = np.array([[0, 0], [1, 1], [2, 2]])
        headings = compute_path_headings(path)
        np.testing.assert_allclose(headings, np.pi / 4, atol=1e-10)

"""Tests for costmap and path planner."""

import numpy as np
import pytest
from motion_planner_core.costmap import Costmap, CostmapConfig
from motion_planner_core.path_planner import plan_path


class TestCostmap:

    def test_empty_costmap(self):
        cm = Costmap(CostmapConfig())
        assert cm.is_empty()
        assert cm.is_free_world(0.0, 0.0)

    def test_add_circle_obstacle(self):
        cm = Costmap(CostmapConfig())
        cm.add_circle_obstacle(2.0, 1.0, 0.3)
        assert not cm.is_empty()
        assert not cm.is_free_world(2.0, 1.0)
        assert cm.is_free_world(4.0, 3.0)

    def test_world_grid_roundtrip(self):
        cm = Costmap(CostmapConfig(origin_x=0.0, origin_y=0.0, resolution=0.1))
        row, col = cm.world_to_grid(1.0, 2.0)
        x, y = cm.grid_to_world(row, col)
        assert abs(x - 1.0) < 0.1
        assert abs(y - 2.0) < 0.1

    def test_out_of_bounds_is_occupied(self):
        cm = Costmap(CostmapConfig(width=5.0, height=5.0))
        assert cm.is_occupied(-1, -1)
        assert cm.is_occupied(9999, 9999)

    def test_from_obstacles(self):
        cm = Costmap.from_obstacles([[2.0, 1.0, 0.3]], inflation_radius=0.0)
        assert not cm.is_free_world(2.0, 1.0)
        assert cm.is_free_world(4.0, 3.0)

    def test_from_obstacles_empty(self):
        cm = Costmap.from_obstacles([])
        assert cm.is_empty()

    def test_inflate(self):
        cm = Costmap(CostmapConfig(resolution=0.05))
        cm.add_circle_obstacle(2.0, 1.0, 0.1)
        # 0.15m from surface = 0.25m from center — should be free before inflate
        assert cm.is_free_world(2.25, 1.0)
        cm.inflate(0.2)
        # After inflation, 0.25m from center should be occupied (0.1 radius + 0.2 inflate)
        assert not cm.is_free_world(2.25, 1.0)

    def test_from_occupancy_grid(self):
        data = np.zeros((100, 100), dtype=np.int8)
        data[50, 50] = 100
        cm = Costmap.from_occupancy_grid(data, 0.0, 0.0, 0.05)
        assert not cm.is_empty()


class TestPathPlanner:

    def test_no_obstacles_returns_waypoints(self):
        wp = np.array([[0, 0], [1, 1], [2, 0]], dtype=float)
        cm = Costmap.from_obstacles([])
        result = plan_path(wp, cm)
        np.testing.assert_array_equal(result, wp)

    def test_plans_around_obstacle(self):
        wp = np.array([[0, 0], [4, 0]], dtype=float)
        cm = Costmap.from_obstacles([[2.0, 0.0, 0.5]], inflation_radius=0.15)
        result = plan_path(wp, cm)
        # Should have more points than just start/end
        assert len(result) > 2
        # All points should be in free space
        for p in result:
            row, col = cm.world_to_grid(p[0], p[1])
            assert not cm.is_occupied(row, col)

    def test_clear_path_stays_direct(self):
        wp = np.array([[0, 0], [1, 0]], dtype=float)
        cm = Costmap.from_obstacles([[5.0, 5.0, 0.3]])
        result = plan_path(wp, cm)
        # Obstacle is far away, should be direct
        assert len(result) == 2

    def test_single_waypoint(self):
        wp = np.array([[0, 0]], dtype=float)
        cm = Costmap.from_obstacles([])
        result = plan_path(wp, cm)
        assert len(result) == 1

    def test_preserves_start_and_end(self):
        wp = np.array([[0, 0], [4, 0]], dtype=float)
        cm = Costmap.from_obstacles([[2.0, 0.0, 0.5]], inflation_radius=0.15)
        result = plan_path(wp, cm)
        assert np.linalg.norm(result[0] - wp[0]) < 0.3
        assert np.linalg.norm(result[-1] - wp[-1]) < 0.3

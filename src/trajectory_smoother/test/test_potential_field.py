"""Tests for potential field obstacle avoidance."""

import math
import numpy as np
import pytest
from trajectory_smoother.potential_field import PotentialField, Obstacle


class TestPotentialField:

    @pytest.fixture
    def single_obstacle(self):
        return PotentialField(
            obstacles=[Obstacle(x=2.0, y=0.0, radius=0.3)],
            influence_distance=0.5,
            repulsive_gain=0.5,
        )

    def test_no_force_far_away(self, single_obstacle):
        """No repulsive force when far from obstacle."""
        fx, fy = single_obstacle.compute_repulsive_force(10.0, 10.0)
        assert abs(fx) < 1e-9
        assert abs(fy) < 1e-9

    def test_force_pushes_away(self, single_obstacle):
        """Force should push robot away from obstacle."""
        # Robot to the left of obstacle at (2.0, 0.0)
        fx, fy = single_obstacle.compute_repulsive_force(1.5, 0.0)
        assert fx < 0.0  # pushed left (away from obstacle)

    def test_force_increases_closer(self, single_obstacle):
        """Force should be stronger closer to obstacle."""
        # Obstacle at (2.0, 0.0) radius 0.3, surface at 1.7
        # Both points outside obstacle but within influence zone
        fx_far, _ = single_obstacle.compute_repulsive_force(1.0, 0.0)
        fx_near, _ = single_obstacle.compute_repulsive_force(1.5, 0.0)
        assert abs(fx_near) > abs(fx_far)

    def test_force_inside_obstacle(self, single_obstacle):
        """Strong force when inside obstacle."""
        fx, fy = single_obstacle.compute_repulsive_force(2.0, 0.1)
        force_mag = math.hypot(fx, fy)
        assert force_mag > 1.0

    def test_no_obstacles_no_force(self):
        pf = PotentialField(obstacles=[], influence_distance=0.5)
        fx, fy = pf.compute_repulsive_force(0.0, 0.0)
        assert abs(fx) < 1e-9
        assert abs(fy) < 1e-9

    def test_multiple_obstacles(self):
        pf = PotentialField(
            obstacles=[
                Obstacle(x=1.0, y=0.0, radius=0.2),
                Obstacle(x=-1.0, y=0.0, radius=0.2),
            ],
            influence_distance=0.5,
        )
        # At origin, equidistant — forces should roughly cancel in x
        fx, fy = pf.compute_repulsive_force(0.0, 0.0)
        assert abs(fx) < 0.1

    def test_adjust_velocity_no_obstacle(self):
        pf = PotentialField(obstacles=[], influence_distance=0.5)
        lin, ang = pf.adjust_velocity(0.0, 0.0, 0.0, 0.2, 0.0)
        assert lin == pytest.approx(0.2)
        assert ang == pytest.approx(0.0)

    def test_adjust_velocity_slows_near_obstacle(self):
        pf = PotentialField(
            obstacles=[Obstacle(x=1.0, y=0.0, radius=0.2)],
            influence_distance=0.5,
            repulsive_gain=0.5,
        )
        # Robot heading toward obstacle
        lin, ang = pf.adjust_velocity(0.7, 0.0, 0.0, 0.2, 0.0)
        assert lin < 0.2  # should slow down

    def test_adjust_velocity_steers_away(self):
        pf = PotentialField(
            obstacles=[Obstacle(x=1.0, y=0.3, radius=0.2)],
            influence_distance=0.5,
            repulsive_gain=0.5,
        )
        # Robot near obstacle, heading east
        _, ang = pf.adjust_velocity(0.7, 0.2, 0.0, 0.2, 0.0)
        assert ang < 0.0  # should steer away (south)
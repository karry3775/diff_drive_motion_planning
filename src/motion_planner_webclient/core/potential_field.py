"""Artificial Potential Field for obstacle avoidance.

Computes a repulsive velocity adjustment that pushes the robot away
from obstacles. Designed to overlay on Pure Pursuit commands.
"""

import math
import numpy as np
from dataclasses import dataclass


@dataclass
class Obstacle:
    """Circular obstacle defined by center and radius."""
    x: float
    y: float
    radius: float


class PotentialField:
    """Computes repulsive forces from obstacles.

    Uses a simple linear repulsion model within the influence zone.
    Gentle enough to not overpower the trajectory tracker.
    """

    def __init__(
        self,
        obstacles: list[Obstacle],
        influence_distance: float = 0.5,
        repulsive_gain: float = 0.3,
    ):
        self.obstacles = obstacles
        self.influence_distance = influence_distance
        self.repulsive_gain = repulsive_gain

    def compute_repulsive_force(self, x: float, y: float) -> tuple[float, float]:
        """Compute total repulsive force at position (x, y).

        Returns (fx, fy) force vector in world frame.
        """
        fx, fy = 0.0, 0.0

        for obs in self.obstacles:
            dx = x - obs.x
            dy = y - obs.y
            dist_to_center = math.hypot(dx, dy)
            dist_to_surface = dist_to_center - obs.radius

            if dist_to_center < 1e-6:
                fx += self.repulsive_gain * 5.0
                continue

            unit_x = dx / dist_to_center
            unit_y = dy / dist_to_center

            if dist_to_surface <= 0:
                # Inside obstacle — strong but bounded push
                magnitude = self.repulsive_gain * 5.0
            elif dist_to_surface < self.influence_distance:
                # Linear falloff within influence zone
                magnitude = self.repulsive_gain * (
                    1.0 - dist_to_surface / self.influence_distance
                )
            else:
                continue

            fx += magnitude * unit_x
            fy += magnitude * unit_y

        return fx, fy

    def adjust_velocity(
        self,
        x: float,
        y: float,
        theta: float,
        linear: float,
        angular: float,
        max_angular: float = 2.84,
    ) -> tuple[float, float]:
        """Adjust velocity command based on repulsive forces.

        Converts the repulsive force into a gentle steering correction.
        Returns adjusted (linear, angular) velocities.
        """
        fx, fy = self.compute_repulsive_force(x, y)
        force_mag = math.hypot(fx, fy)

        if force_mag < 1e-6:
            return linear, angular

        # Project force into robot frame
        local_x = math.cos(-theta) * fx - math.sin(-theta) * fy
        local_y = math.sin(-theta) * fx + math.cos(-theta) * fy

        # Lateral force → angular correction (gentle)
        angular_correction = local_y * 1.5
        angular = np.clip(angular + angular_correction, -max_angular, max_angular)

        # If force is pushing against us, slow down a bit (but not too much)
        if local_x < -0.1:
            linear = max(linear * 0.5, 0.05)

        return linear, angular

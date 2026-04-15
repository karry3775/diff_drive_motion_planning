"""Shared types for the motion planner."""

import numpy as np
from dataclasses import dataclass


@dataclass
class RobotState:
    x: float
    y: float
    theta: float  # radians


@dataclass
class VelocityCommand:
    linear: float   # m/s
    angular: float  # rad/s


class Trajectory:
    """Time-parameterized trajectory. Consumed by all controllers."""

    def __init__(self, x: np.ndarray, y: np.ndarray, heading: np.ndarray,
                 velocity: np.ndarray, time: np.ndarray):
        self.x = x
        self.y = y
        self.heading = heading
        self.velocity = velocity
        self.time = time

    def __len__(self):
        return len(self.x)

    def at_time(self, t: float) -> tuple[float, float, float, float]:
        """Interpolate reference (x, y, heading, velocity) at time t."""
        x = float(np.interp(t, self.time, self.x))
        y = float(np.interp(t, self.time, self.y))
        h = float(np.interp(t, self.time, self.heading))
        v = float(np.interp(t, self.time, self.velocity))
        return x, y, h, v

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            'x': self.x, 'y': self.y, 'heading': self.heading,
            'velocity': self.velocity, 'time': self.time,
        }

"""2D Costmap for obstacle representation.

A resolution-based occupancy grid that abstracts away the obstacle source.
Can be populated from a list of geometric obstacles, an occupancy grid array,
or any other source. The planner just sees free/occupied cells.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class CostmapConfig:
    """Configuration for costmap dimensions and resolution."""
    origin_x: float = -1.0
    origin_y: float = -1.0
    width: float = 10.0
    height: float = 6.0
    resolution: float = 0.05


class Costmap:
    """2D occupancy grid. Values: 0 = free, 100 = occupied."""

    def __init__(self, config: CostmapConfig):
        self.config = config
        self.cols = int(config.width / config.resolution)
        self.rows = int(config.height / config.resolution)
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int((x - self.config.origin_x) / self.config.resolution)
        row = int((y - self.config.origin_y) / self.config.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = self.config.origin_x + (col + 0.5) * self.config.resolution
        y = self.config.origin_y + (row + 0.5) * self.config.resolution
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_occupied(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return True
        return self.grid[row, col] > 50

    def is_free_world(self, x: float, y: float) -> bool:
        row, col = self.world_to_grid(x, y)
        return not self.is_occupied(row, col)

    def is_empty(self) -> bool:
        return np.all(self.grid == 0)

    def add_circle_obstacle(self, cx: float, cy: float, radius: float):
        r_cells = int(radius / self.config.resolution) + 1
        center_row, center_col = self.world_to_grid(cx, cy)
        for dr in range(-r_cells, r_cells + 1):
            for dc in range(-r_cells, r_cells + 1):
                r, c = center_row + dr, center_col + dc
                if not self.in_bounds(r, c):
                    continue
                wx, wy = self.grid_to_world(r, c)
                if (wx - cx)**2 + (wy - cy)**2 <= radius**2:
                    self.grid[r, c] = 100

    def inflate(self, inflation_radius: float):
        """Inflate occupied cells by robot footprint radius."""
        r_cells = int(inflation_radius / self.config.resolution) + 1
        occupied = np.argwhere(self.grid >= 100)
        inflated = self.grid.copy()
        for row, col in occupied:
            for dr in range(-r_cells, r_cells + 1):
                for dc in range(-r_cells, r_cells + 1):
                    r, c = row + dr, col + dc
                    if not self.in_bounds(r, c):
                        continue
                    if np.sqrt(dr**2 + dc**2) * self.config.resolution <= inflation_radius:
                        inflated[r, c] = 100
        self.grid = inflated

    @classmethod
    def from_obstacles(cls, obstacles: list, config: CostmapConfig = None,
                       inflation_radius: float = 0.3) -> 'Costmap':
        """Build from list of [x, y, radius] obstacles."""
        cfg = config or CostmapConfig()
        costmap = cls(cfg)
        if not obstacles:
            return costmap
        for obs in obstacles:
            costmap.add_circle_obstacle(obs[0], obs[1], obs[2])
        if inflation_radius > 0:
            costmap.inflate(inflation_radius)
        return costmap

    @classmethod
    def from_occupancy_grid(cls, data: np.ndarray, origin_x: float,
                            origin_y: float, resolution: float) -> 'Costmap':
        """Build from a 2D occupancy grid array (nav_msgs/OccupancyGrid compatible)."""
        config = CostmapConfig(
            origin_x=origin_x, origin_y=origin_y,
            width=data.shape[1] * resolution,
            height=data.shape[0] * resolution,
            resolution=resolution,
        )
        costmap = cls(config)
        costmap.grid = data.astype(np.int8)
        return costmap

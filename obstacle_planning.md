# How Obstacle-Aware Path Planning Works

## Pipeline

```
Waypoints -> Costmap -> A* (per segment) -> Downsample -> Cubic Spline -> Trajectory
```

## Step 1: Costmap

The obstacle list `[x, y, radius]` gets rasterized onto a 2D grid (default 0.05m resolution). Each obstacle marks cells within its radius as occupied. Then the grid is inflated by the robot's footprint radius (0.3m default) so the planner can treat the robot as a point.

## Step 2: A* Search

For each consecutive pair of waypoints, check if a straight line is clear. If blocked, run A* on the grid with 8-connected neighbors (diagonal cost 1.414, cardinal cost 1.0) and Euclidean heuristic.

## Step 3: Downsample

Raw A* output has one point per grid cell. Downsample to keep points at least `5 * resolution` apart.

## Step 4: Cubic Spline

Downsampled A* waypoints go through the same cubic spline smoother. Produces a C2-continuous curve.

## Known Limitations

The A* planner produces grid-aligned paths, and the cubic spline can overshoot at sharp turns introduced by the grid discretization. See Future Work in README for planned improvements.

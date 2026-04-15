# Path Smoothing and Trajectory Control for Differential Drive Robots

A ROS2 Python package implementing path smoothing, trajectory generation with trapezoidal velocity profiling, and Pure Pursuit trajectory tracking for a TurtleBot3 Burger (differential drive robot).

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│  Path Smoother   │────▶│ Trajectory Generator  │────▶│  Pure Pursuit     │
│  (Cubic Spline)  │     │ (Trapezoidal Profile) │     │  Controller       │
└─────────────────┘     └──────────────────────┘     └──────────────────┘
  waypoints.yaml          /smoothed_path               /trajectory
                                                        /cmd_vel ──▶ Robot
```

The system is split into three decoupled modules, each usable as a standalone library or as a ROS2 node:

| Module | File | Responsibility |
|--------|------|----------------|
| **Path Smoother** | `path_smoother.py` | Cubic spline interpolation of discrete waypoints into a C2-continuous path |
| **Trajectory Generator** | `trajectory_generator.py` | Time-parameterization with trapezoidal velocity profile (accel → cruise → decel) |
| **Pure Pursuit Controller** | `pure_pursuit.py` | Geometric trajectory tracking for differential drive robots |

### Design Choices

**Cubic Spline Smoothing** — Chosen because it guarantees the path passes through all waypoints (unlike B-splines), provides C2 continuity (continuous position, velocity, acceleration), and is parameterized by arc length for natural spacing.

**Trapezoidal Velocity Profile** — Provides realistic acceleration/deceleration phases. Automatically degrades to a triangular profile when the path is too short to reach max velocity. Respects TurtleBot3 Burger limits (0.22 m/s max).

**Pure Pursuit Controller** — The standard geometric controller for differential drive robots. It finds a lookahead point on the trajectory and computes the curvature needed to reach it. Key features:
- Monotonic progress tracking prevents the robot from backtracking
- Minimum velocity prevents stalling near trajectory endpoints
- Configurable lookahead distance for tuning responsiveness vs. smoothness

## Quick Start

### Prerequisites

```bash
# Python dependencies (no ROS2 needed for standalone simulation)
pip install numpy scipy pyyaml matplotlib pytest
```

### Run Standalone Simulation (No ROS2)

```bash
cd <project_root>
PYTHONPATH=src/trajectory_smoother:$PYTHONPATH python3 -m trajectory_smoother.simulation \
    --config src/trajectory_smoother/config/waypoints.yaml
```

This produces `simulation_results.png` with four plots:
1. Path comparison (raw waypoints vs. smoothed path vs. robot trajectory)
2. Trapezoidal velocity profile
3. Cross-track error over time
4. Path curvature

### Run Tests

```bash
PYTHONPATH=src/trajectory_smoother:$PYTHONPATH python3 -m pytest src/trajectory_smoother/test/ -v
```

### Run with ROS2 + TurtleBot3 Gazebo

```bash
# 1. Source ROS2 and build
source /opt/ros/humble/setup.bash
cd <workspace_root>
colcon build --packages-select trajectory_smoother
source install/setup.bash

# 2. Launch TurtleBot3 in Gazebo
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 3. Launch the trajectory tracking pipeline
ros2 launch trajectory_smoother trajectory_tracking.launch.py
```

## Configuration

All parameters are in `config/waypoints.yaml`:

```yaml
waypoints:           # List of [x, y] waypoints in meters
smoothing:
  num_samples: 200   # Points on the smoothed path
trajectory:
  max_velocity: 0.22      # m/s (TurtleBot3 Burger limit)
  max_acceleration: 0.5   # m/s^2
  max_deceleration: 0.5   # m/s^2
  dt: 0.05                # Sample period
controller:
  lookahead_distance: 0.3 # meters
  goal_tolerance: 0.1     # meters
```

## Test Results

31 tests covering:
- **Path Smoother**: shape, endpoint/waypoint interpolation, C2 continuity, edge cases
- **Trajectory Generator**: format, timing, velocity limits, trapezoidal/triangular profiles, error handling
- **Pure Pursuit**: straight-line tracking, turning behavior, velocity limits, goal detection, reset
- **Integration**: full pipeline end-to-end (smooth → generate → track), tracking error bounds

Simulation performance with default waypoints:
- **Max cross-track error**: 0.028 m
- **Mean cross-track error**: 0.009 m
- **Simulation duration**: 44.6 s

## Extending to a Real Robot

1. **Replace `/odom` source** — Use robot_localization (EKF) fusing wheel odometry + IMU instead of raw odometry for drift correction.
2. **Tune controller gains** — Real robots have latency, wheel slip, and uneven terrain. The lookahead distance and velocity limits need tuning on hardware.
3. **Add safety** — Implement a watchdog that publishes zero velocity if no controller output is received within a timeout. Add bumper/cliff sensor integration.
4. **Localization** — For longer paths, integrate AMCL or SLAM for global localization instead of relying on dead-reckoning odometry.
5. **Velocity smoothing** — Add a low-pass filter on cmd_vel to avoid jerky motion from discrete controller updates.

## Obstacle Avoidance (Extra Credit — Planned)

The planned approach uses **Artificial Potential Fields**:
- Attractive potential toward the next trajectory point
- Repulsive potential from obstacles detected via LiDAR
- The combined gradient modifies the velocity command from Pure Pursuit
- This overlays on the existing controller without replacing it

Implementation would add a `potential_field.py` module that subscribes to `/scan` (LaserScan) and modifies the cmd_vel output from the Pure Pursuit node.

## AI Tools Used

This project was developed with assistance from Claude (Anthropic) via Amazon's Kiro AI agent for:
- Initial project scaffolding and ROS2 package structure
- Algorithm implementation guidance
- Test case design
- Bug identification (infinite loop in velocity profile, controller convergence)
- Documentation generation

## Project Structure

```
src/trajectory_smoother/
├── config/
│   └── waypoints.yaml          # Waypoint and parameter configuration
├── launch/
│   └── trajectory_tracking.launch.py  # ROS2 launch file
├── trajectory_smoother/
│   ├── __init__.py
│   ├── path_smoother.py        # Cubic spline smoothing (pure Python)
│   ├── trajectory_generator.py # Trapezoidal velocity profile (pure Python)
│   ├── pure_pursuit.py         # Pure Pursuit controller (pure Python)
│   ├── simulation.py           # Standalone 2D simulation + plotting
│   ├── path_smoother_node.py   # ROS2 node wrapper
│   ├── trajectory_generator_node.py  # ROS2 node wrapper
│   └── pure_pursuit_node.py    # ROS2 node wrapper
├── test/
│   ├── test_path_smoother.py
│   ├── test_trajectory_generator.py
│   └── test_pure_pursuit.py
├── package.xml
├── setup.py
└── setup.cfg
```

Core algorithms are in pure Python files with no ROS2 dependency, making them independently testable. ROS2 node wrappers are thin layers that handle pub/sub and parameter loading.

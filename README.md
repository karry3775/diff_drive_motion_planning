# Differential Drive Motion Planner

Path planning, smoothing, and trajectory tracking for a TurtleBot3 Burger in ROS2 Humble + Gazebo. Includes a browser-based interactive demo.

## Architecture

```
                    motion_planner_core (pure Python, no ROS)
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Costmap ──> Path Planner (A*) ──> Smoother (Cubic Spline)   │
│                                         |                    │
│                              Trajectory Generator            │
│                              (Trapezoidal Profile)           │
│                                         |                    │
│                                    Controller                │
│                          ┌─────────────┼────────────┐        │
│                     Pure Pursuit      PID      Feedforward   │
│                                                              │
└──────────────────────────────────────────────────────────────┘

                    motion_planner_service (ROS2 nodes)
┌──────────────────────────────────────────────────────────────┐
│  planner_node      — runs core pipeline, publishes paths     │
│  controller_node   — odom -> controller -> cmd_vel           │
│  recorder_node     — logs odom, publishes trail, saves plots │
│  obstacle_node     — spawns obstacles in Gazebo + RViz       │
└──────────────────────────────────────────────────────────────┘

                    motion_planner_webclient
┌──────────────────────────────────────────────────────────────┐
│  Browser demo — runs core library via Pyodide (WASM)         │
│  Drop waypoints + obstacles, pick controller, simulate       │
└──────────────────────────────────────────────────────────────┘
```

The core library has zero ROS dependency. The service nodes are thin wrappers that handle pub/sub. The web client runs the same core library in the browser.

## Controllers

Three controller strategies, same interface:

- **Pure Pursuit** — geometric, steers toward a lookahead point on the path. Best for real robots where disturbances exist.
- **PID** — time-based, tracks where the robot should be at time t. Feedforward velocity from trajectory + PID corrections. Faults if cross-track error exceeds threshold. Gains are tunable via config.
- **Feedforward** — direct trajectory replay using `omega = v * curvature`. Drifts over time due to open-loop numerical integration error.
**Path Planning** — A* finds obstacle-free routes between consecutive waypoints on the costmap grid. If no obstacles, waypoints pass through directly. If the smoothed path cuts through an obstacle, planning reports an error.
### Production Readiness
- **Hardware abstraction layer** — currently the service nodes talk directly to ROS2 topics. A driver interface between the core library and the robot (or simulator) would make the stack portable. Switching from Gazebo to a real TurtleBot3 should only require swapping the driver, not touching the planner or controller.
- **Comms layer independence** — the core library is already ROS-free, but the service layer is tightly coupled to ROS2. Abstracting the comms interface would allow swapping ROS2 for another middleware (DDS, ZeroMQ, etc.) without rewriting the nodes.
- **C++ or Rust port** — Python is fine for prototyping but not suitable for real-time control loops. A C++ port is the natural next step since ROS2 has native C++ support. Rust with rclrs is another option.
- **Perception integration** — currently obstacles are static and configured upfront. A real system needs to ingest sensor data (LiDAR, depth camera), maintain a local costmap in memory (ring buffer or sliding window), and replan dynamically.
- **Local planner (TEB)** — the current A* + cubic spline approach is a global planner. A local planner like Timed Elastic Band would optimize the trajectory in real-time considering kinematics, dynamic obstacles, and smoothness simultaneously.
- **CI/CD** — no pipeline exists yet. Jenkins or GitHub Actions running the test suite, linting, and building on every push.
- **Code formatting** — no formatter configured. Black + flake8 (or ruff) should be added.

### Algorithmic Improvements
- Theta* or Any-Angle A* for smoother obstacle avoidance paths
- Path pruning after A* to remove redundant intermediate waypoints before smoothing
- B-spline as alternative to cubic spline (naturally smooths corners)
- Double-S (S-curve) velocity profile for smoother acceleration
- PID controller tuning guide with recommended gains for TurtleBot3
- Dynamic replanning when new obstacles appear mid-execution

### Web Client
- Tracking error and velocity profile plots after simulation (done)
- Step-by-step animation speed control
- Export waypoints as YAML
- Support for non-circular obstacle geometries

## AI Tools Used

Built collaboratively with Claude (Anthropic). The AI generated most of the implementation code, while I drove the architecture decisions, reviewed the output, caught design issues (controller coupling, obstacle avoidance approach, feedforward limitations), and iterated on the design through back-and-forth discussion.
"""Standalone 2D simulation of the full pipeline.

Runs path smoothing, trajectory generation, and pure pursuit tracking
in a simple kinematic simulation of a differential drive robot.
Produces matplotlib plots showing results — no ROS2 required.

Usage:
    python3 -m trajectory_smoother.simulation [--config path/to/waypoints.yaml]
"""

import argparse
import math
import numpy as np
import yaml

from trajectory_smoother.path_smoother import smooth_path, compute_path_curvature
from trajectory_smoother.trajectory_generator import generate_trajectory, trajectory_to_arrays
from trajectory_smoother.pure_pursuit import PurePursuitController, RobotState


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def simulate(config: dict) -> dict:
    """Run the full pipeline and return results for plotting.

    Returns dict with keys: waypoints, smoothed_path, trajectory,
    robot_x, robot_y, robot_theta, tracking_error, time.
    """
    waypoints = np.array(config['waypoints'], dtype=float)
    smooth_cfg = config.get('smoothing', {})
    traj_cfg = config.get('trajectory', {})
    ctrl_cfg = config.get('controller', {})

    # 1. Smooth the path
    num_samples = smooth_cfg.get('num_samples', 200)
    smoothed = smooth_path(waypoints, num_samples)

    # 2. Generate trajectory
    trajectory = generate_trajectory(
        smoothed,
        max_vel=traj_cfg.get('max_velocity', 0.22),
        max_accel=traj_cfg.get('max_acceleration', 0.5),
        max_decel=traj_cfg.get('max_deceleration', 0.5),
        dt=traj_cfg.get('dt', 0.05),
    )
    traj_arrays = trajectory_to_arrays(trajectory)

    # 3. Simulate robot with pure pursuit
    controller = PurePursuitController(
        lookahead_distance=ctrl_cfg.get('lookahead_distance', 0.3),
        min_lookahead=ctrl_cfg.get('min_lookahead', 0.15),
        max_lookahead=ctrl_cfg.get('max_lookahead', 0.6),
        goal_tolerance=ctrl_cfg.get('goal_tolerance', 0.1),
        max_linear_vel=traj_cfg.get('max_velocity', 0.22),
        max_angular_vel=ctrl_cfg.get('max_angular_velocity', 2.84),
    )

    dt = traj_cfg.get('dt', 0.05)
    state = RobotState(x=waypoints[0, 0], y=waypoints[0, 1], theta=0.0)

    robot_x, robot_y, robot_theta = [state.x], [state.y], [state.theta]
    tracking_errors = [0.0]
    sim_time = [0.0]
    t = 0.0
    max_sim_time = traj_arrays['time'][-1] * 2  # safety limit

    while not controller.goal_reached and t < max_sim_time:
        cmd = controller.compute_command(
            state, traj_arrays['x'], traj_arrays['y'], traj_arrays['velocity']
        )

        # Kinematic update (differential drive)
        state.x += cmd.linear * math.cos(state.theta) * dt
        state.y += cmd.linear * math.sin(state.theta) * dt
        state.theta += cmd.angular * dt
        # Normalize theta to [-pi, pi]
        state.theta = math.atan2(math.sin(state.theta), math.cos(state.theta))
        t += dt

        robot_x.append(state.x)
        robot_y.append(state.y)
        robot_theta.append(state.theta)
        sim_time.append(t)

        # Cross-track error: distance to nearest trajectory point
        dists = np.sqrt(
            (traj_arrays['x'] - state.x)**2 + (traj_arrays['y'] - state.y)**2
        )
        tracking_errors.append(float(np.min(dists)))

    return {
        'waypoints': waypoints,
        'smoothed_path': smoothed,
        'trajectory': traj_arrays,
        'curvature': compute_path_curvature(smoothed),
        'robot_x': np.array(robot_x),
        'robot_y': np.array(robot_y),
        'robot_theta': np.array(robot_theta),
        'tracking_error': np.array(tracking_errors),
        'time': np.array(sim_time),
    }


def plot_results(results: dict):
    """Generate matplotlib plots of the simulation results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Path comparison
    ax = axes[0, 0]
    ax.plot(results['waypoints'][:, 0], results['waypoints'][:, 1],
            'ro-', label='Raw Waypoints', markersize=8)
    ax.plot(results['smoothed_path'][:, 0], results['smoothed_path'][:, 1],
            'b-', label='Smoothed Path', linewidth=2)
    ax.plot(results['robot_x'], results['robot_y'],
            'g--', label='Robot Trajectory', linewidth=1.5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Path Comparison')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # 2. Velocity profile
    ax = axes[0, 1]
    ax.plot(results['trajectory']['time'], results['trajectory']['velocity'],
            'b-', label='Reference Velocity')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Trapezoidal Velocity Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Tracking error
    ax = axes[1, 0]
    ax.plot(results['time'], results['tracking_error'], 'r-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-track Error (m)')
    ax.set_title(f'Tracking Error (max={np.max(results["tracking_error"]):.4f}m, '
                 f'mean={np.mean(results["tracking_error"]):.4f}m)')
    ax.grid(True, alpha=0.3)

    # 4. Path curvature
    ax = axes[1, 1]
    t_param = np.linspace(0, 1, len(results['curvature']))
    ax.plot(t_param, results['curvature'], 'purple')
    ax.set_xlabel('Path Parameter')
    ax.set_ylabel('Curvature (1/m)')
    ax.set_title('Path Curvature')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simulation_results.png', dpi=150)
    print('Saved simulation_results.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Standalone trajectory tracking simulation')
    parser.add_argument(
        '--config',
        default='src/trajectory_smoother/config/waypoints.yaml',
        help='Path to waypoints YAML config',
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print('Running simulation...')
    results = simulate(config)
    print(f'Simulation complete: {len(results["robot_x"])} steps, '
          f'duration={results["time"][-1]:.1f}s')
    print(f'Max tracking error: {np.max(results["tracking_error"]):.4f}m')
    print(f'Mean tracking error: {np.mean(results["tracking_error"]):.4f}m')

    plot_results(results)


if __name__ == '__main__':
    main()

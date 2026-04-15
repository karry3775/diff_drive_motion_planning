"""Standalone 2D simulation of the full pipeline.

Runs costmap → path planning → smoothing → trajectory generation → controller
in a kinematic simulation. Produces matplotlib plots. No ROS2 required.

Usage:
    python3 -m motion_planner_core.simulation [--config path/to/waypoints.yaml]
"""

import argparse
import math
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from motion_planner_core.pipeline import build_trajectory_from_config
from motion_planner_core.controller import create_controller
from motion_planner_core.types import RobotState
from motion_planner_core.path_smoother import compute_path_curvature


def simulate(config: dict) -> dict:
    waypoints = np.array(config['waypoints'], dtype=float)
    traj_cfg = config.get('trajectory', {})

    planned, smoothed, trajectory, costmap = build_trajectory_from_config(config)
    controller = create_controller(config)

    dt = traj_cfg.get('dt', 0.05)
    init_heading = math.atan2(
        waypoints[1, 1] - waypoints[0, 1],
        waypoints[1, 0] - waypoints[0, 0],
    )
    state = RobotState(x=waypoints[0, 0], y=waypoints[0, 1], theta=init_heading)

    robot_x, robot_y, robot_theta = [state.x], [state.y], [state.theta]
    tracking_errors = [0.0]
    sim_time = [0.0]
    t = 0.0
    max_sim_time = trajectory.duration * 2

    while not controller.goal_reached and not controller.faulted and t < max_sim_time:
        cmd = controller.compute_command(state, trajectory, t)
        state.x += cmd.linear * math.cos(state.theta) * dt
        state.y += cmd.linear * math.sin(state.theta) * dt
        state.theta += cmd.angular * dt
        state.theta = math.atan2(math.sin(state.theta), math.cos(state.theta))
        t += dt

        robot_x.append(state.x)
        robot_y.append(state.y)
        robot_theta.append(state.theta)
        sim_time.append(t)

        dists = np.sqrt((trajectory.x - state.x)**2 + (trajectory.y - state.y)**2)
        tracking_errors.append(float(np.min(dists)))

    return {
        'waypoints': waypoints,
        'planned_path': planned,
        'smoothed_path': smoothed,
        'trajectory': trajectory.to_dict(),
        'curvature': compute_path_curvature(smoothed),
        'costmap': costmap,
        'obstacles': config.get('obstacles', []),
        'robot_x': np.array(robot_x),
        'robot_y': np.array(robot_y),
        'robot_theta': np.array(robot_theta),
        'tracking_error': np.array(tracking_errors),
        'time': np.array(sim_time),
        'goal_reached': controller.goal_reached,
        'faulted': controller.faulted,
        'controller_type': config.get('controller', {}).get('type', 'pure_pursuit'),
    }


def plot_results(results: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(results['waypoints'][:, 0], results['waypoints'][:, 1],
            'ro-', label='Waypoints', markersize=8)
    if len(results['planned_path']) != len(results['waypoints']):
        ax.plot(results['planned_path'][:, 0], results['planned_path'][:, 1],
                'm.-', label='Planned (A*)', markersize=4, alpha=0.5)
    ax.plot(results['smoothed_path'][:, 0], results['smoothed_path'][:, 1],
            'b-', label='Smoothed', linewidth=2)
    ax.plot(results['robot_x'], results['robot_y'],
            'g--', label='Robot', linewidth=1.5)
    for obs in results['obstacles']:
        circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.4)
        ax.add_patch(circle)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ctrl_label = results['controller_type'].replace('_', ' ').title()
    ax.set_title(f'Path Comparison ({ctrl_label})')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(results['trajectory']['time'], results['trajectory']['velocity'], 'b-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity Profile')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(results['time'], results['tracking_error'], 'r-')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Cross-track Error (m)')
    ax.set_title(f'Tracking Error (max={np.max(results["tracking_error"]):.4f}m, '
                 f'mean={np.mean(results["tracking_error"]):.4f}m)')
    ax.grid(True, alpha=0.3)

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
    parser = argparse.ArgumentParser(description='Standalone motion planning simulation')
    parser.add_argument('--config', default='config/waypoints.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print('Running simulation...')
    results = simulate(config)
    status = 'Goal reached' if results['goal_reached'] else ('FAULTED' if results['faulted'] else 'Timeout')
    print(f'{status} | {len(results["robot_x"])} steps, {results["time"][-1]:.1f}s')
    print(f'Controller: {results["controller_type"]}')
    print(f'Max error: {np.max(results["tracking_error"]):.4f}m')
    print(f'Mean error: {np.mean(results["tracking_error"]):.4f}m')
    plot_results(results)


if __name__ == '__main__':
    main()

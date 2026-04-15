"""
ROS2 node that records robot odometry and trajectory data during execution.
Subscribes to /odom, /smoothed_path, /raw_waypoints, and /trajectory_velocities.
When the robot reaches the goal (or on shutdown), saves plots to simulation_results.png.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Float64MultiArray


class DataRecorderNode(Node):
    def __init__(self):
        super().__init__('data_recorder_node')

        self.robot_x = []
        self.robot_y = []
        self.robot_theta = []
        self.robot_time = []
        self.smoothed_path = None
        self.waypoints = None
        self.traj_vel = None
        self.traj_time = None
        self.start_time = None
        self.goal_reached = False

        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        if not config_file:
            self.get_logger().error('No config_file parameter provided')
            return

        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        waypoints = config['waypoints']
        last_wp = waypoints[-1]
        self.goal_x = float(last_wp[0])
        self.goal_y = float(last_wp[1])
        self.goal_tol = config.get('controller', {}).get('goal_tolerance', 0.1)

        self.odom_sub = self.create_subscription(Odometry, '/odom', self._on_odom, 10)
        self.path_sub = self.create_subscription(Path, '/smoothed_path', self._on_path, 10)
        self.wp_sub = self.create_subscription(Path, '/raw_waypoints', self._on_waypoints, 10)
        self.vel_sub = self.create_subscription(
            Float64MultiArray, '/trajectory_velocities', self._on_vel, 10
        )

        self.get_logger().info('Data recorder started — will save plots on goal or shutdown')

    def _on_odom(self, msg: Odometry):
        if self.goal_reached:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            self.start_time = now

        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        self.robot_x.append(msg.pose.pose.position.x)
        self.robot_y.append(msg.pose.pose.position.y)
        self.robot_theta.append(yaw)
        self.robot_time.append(now - self.start_time)

        # Check goal
        dist = math.hypot(msg.pose.pose.position.x - self.goal_x,
                          msg.pose.pose.position.y - self.goal_y)
        if dist < self.goal_tol and len(self.robot_x) > 50:
            self.goal_reached = True
            self.get_logger().info(f'Goal reached! Recorded {len(self.robot_x)} odom samples')
            self._save_plots()

    def _on_path(self, msg: Path):
        if self.smoothed_path is None:
            self.smoothed_path = np.array([
                [p.pose.position.x, p.pose.position.y] for p in msg.poses
            ])

    def _on_waypoints(self, msg: Path):
        if self.waypoints is None:
            self.waypoints = np.array([
                [p.pose.position.x, p.pose.position.y] for p in msg.poses
            ])

    def _on_vel(self, msg: Float64MultiArray):
        if self.traj_vel is None:
            self.traj_vel = np.array(msg.data)

    def _save_plots(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        rx = np.array(self.robot_x)
        ry = np.array(self.robot_y)
        rt = np.array(self.robot_time)
        rtheta = np.array(self.robot_theta)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Path comparison
        ax = axes[0, 0]
        if self.waypoints is not None:
            ax.plot(self.waypoints[:, 0], self.waypoints[:, 1],
                    'ro-', label='Raw Waypoints', markersize=8)
        if self.smoothed_path is not None:
            ax.plot(self.smoothed_path[:, 0], self.smoothed_path[:, 1],
                    'b-', label='Smoothed Path', linewidth=2)
        ax.plot(rx, ry, 'g--', label='Robot (actual)', linewidth=1.5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Path Comparison (Gazebo)')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 2. Velocity profile (from odom — compute actual velocity)
        ax = axes[0, 1]
        if len(rt) > 1:
            dx = np.diff(rx)
            dy = np.diff(ry)
            dt = np.diff(rt)
            dt[dt < 1e-9] = 1e-9
            actual_vel = np.sqrt(dx**2 + dy**2) / dt
            ax.plot(rt[1:], actual_vel, 'g-', label='Actual Velocity', alpha=0.7)
        if self.traj_vel is not None:
            traj_t = np.linspace(0, rt[-1], len(self.traj_vel))
            ax.plot(traj_t, self.traj_vel, 'b--', label='Reference Velocity', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Tracking error
        ax = axes[1, 0]
        if self.smoothed_path is not None:
            errors = []
            for i in range(len(rx)):
                dists = np.sqrt(
                    (self.smoothed_path[:, 0] - rx[i])**2 +
                    (self.smoothed_path[:, 1] - ry[i])**2
                )
                errors.append(np.min(dists))
            errors = np.array(errors)
            ax.plot(rt, errors, 'r-')
            ax.set_title(f'Tracking Error (max={np.max(errors):.4f}m, '
                         f'mean={np.mean(errors):.4f}m)')
        else:
            ax.set_title('Tracking Error (no path data)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cross-track Error (m)')
        ax.grid(True, alpha=0.3)

        # 4. Robot heading
        ax = axes[1, 1]
        ax.plot(rt, np.degrees(rtheta), 'purple')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Heading (deg)')
        ax.set_title('Robot Heading')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=150)
        self.get_logger().info('Saved simulation_results.png')
        plt.close()

    def destroy_node(self):
        if not self.goal_reached and len(self.robot_x) > 10:
            self.get_logger().info('Shutdown — saving plots with data collected so far')
            self._save_plots()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DataRecorderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
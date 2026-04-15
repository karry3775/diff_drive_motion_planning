"""ROS2 node for trajectory generation.

Subscribes to the smoothed path, generates a time-parameterized trajectory
with trapezoidal velocity profile, and publishes it.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Float64MultiArray
import numpy as np
import yaml
import math

from trajectory_smoother.trajectory_generator import generate_trajectory, trajectory_to_arrays


class TrajectoryGeneratorNode(Node):
    def __init__(self):
        super().__init__('trajectory_generator_node')

        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        if not config_file:
            self.get_logger().error('No config_file parameter provided')
            return

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        traj_cfg = config.get('trajectory', {})
        self.max_vel = traj_cfg.get('max_velocity', 0.22)
        self.max_accel = traj_cfg.get('max_acceleration', 0.5)
        self.max_decel = traj_cfg.get('max_deceleration', 0.5)
        self.dt = traj_cfg.get('dt', 0.05)

        self.traj_pub = self.create_publisher(Path, '/trajectory', 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, '/trajectory_velocities', 10)

        self.path_sub = self.create_subscription(Path, '/smoothed_path', self._on_path, 10)
        self.get_logger().info('Waiting for smoothed path...')

    def _on_path(self, msg: Path):
        path = np.array([
            [p.pose.position.x, p.pose.position.y] for p in msg.poses
        ])

        if len(path) < 2:
            return

        trajectory = generate_trajectory(
            path, self.max_vel, self.max_accel, self.max_decel, self.dt
        )
        arrays = trajectory_to_arrays(trajectory)

        # Publish trajectory as Path
        traj_msg = Path()
        traj_msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='odom')
        for i in range(len(trajectory)):
            pose = PoseStamped()
            pose.header = traj_msg.header
            pose.pose.position.x = float(arrays['x'][i])
            pose.pose.position.y = float(arrays['y'][i])
            yaw = float(arrays['heading'][i])
            pose.pose.orientation.z = math.sin(yaw / 2)
            pose.pose.orientation.w = math.cos(yaw / 2)
            traj_msg.poses.append(pose)
        self.traj_pub.publish(traj_msg)

        # Publish velocity profile
        vel_msg = Float64MultiArray()
        vel_msg.data = arrays['velocity'].tolist()
        self.vel_pub.publish(vel_msg)

        self.get_logger().info(
            f'Generated trajectory: {len(trajectory)} points, '
            f'duration={arrays["time"][-1]:.1f}s'
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

"""ROS2 node for path smoothing.

Loads waypoints from YAML config, smooths them using cubic spline,
and publishes the smoothed path for visualization and downstream use.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
import yaml
import math

from trajectory_smoother.path_smoother import smooth_path, compute_path_headings


class PathSmootherNode(Node):
    def __init__(self):
        super().__init__('path_smoother_node')

        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        if not config_file:
            self.get_logger().error('No config_file parameter provided')
            return

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        waypoints = np.array(config['waypoints'], dtype=float)
        num_samples = config.get('smoothing', {}).get('num_samples', 200)

        # Smooth the path
        smoothed = smooth_path(waypoints, num_samples)
        headings = compute_path_headings(smoothed)

        # Publish smoothed path
        self.path_pub = self.create_publisher(Path, '/smoothed_path', 10)
        self.waypoints_pub = self.create_publisher(Path, '/raw_waypoints', 10)

        # Publish once per second for visualization
        self.smoothed = smoothed
        self.headings = headings
        self.waypoints = waypoints
        self.timer = self.create_timer(1.0, self._publish)
        self.get_logger().info(
            f'Smoothed {len(waypoints)} waypoints into {len(smoothed)} points'
        )

    def _publish(self):
        stamp = self.get_clock().now().to_msg()

        # Smoothed path
        path_msg = Path()
        path_msg.header = Header(stamp=stamp, frame_id='odom')
        for i in range(len(self.smoothed)):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(self.smoothed[i, 0])
            pose.pose.position.y = float(self.smoothed[i, 1])
            yaw = self.headings[i]
            pose.pose.orientation.z = math.sin(yaw / 2)
            pose.pose.orientation.w = math.cos(yaw / 2)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

        # Raw waypoints as path
        wp_msg = Path()
        wp_msg.header = Header(stamp=stamp, frame_id='odom')
        for wp in self.waypoints:
            pose = PoseStamped()
            pose.header = wp_msg.header
            pose.pose.position.x = float(wp[0])
            pose.pose.position.y = float(wp[1])
            wp_msg.poses.append(pose)
        self.waypoints_pub.publish(wp_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathSmootherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

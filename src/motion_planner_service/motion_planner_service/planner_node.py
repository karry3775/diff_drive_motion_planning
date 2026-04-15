"""ROS2 node: runs core pipeline, publishes paths and trajectory."""

import math
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header, Float64MultiArray

from motion_planner_core.pipeline import build_trajectory_from_config


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        if not config_file:
            self.get_logger().error('No config_file parameter')
            return

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        planned, smoothed, trajectory, costmap = build_trajectory_from_config(config)
        waypoints = np.array(config['waypoints'], dtype=float)

        self.waypoints = waypoints
        self.smoothed = smoothed
        self.trajectory = trajectory

        self.wp_pub = self.create_publisher(Path, '/raw_waypoints', 10)
        self.path_pub = self.create_publisher(Path, '/smoothed_path', 10)
        self.traj_pub = self.create_publisher(Path, '/trajectory', 10)
        self.vel_pub = self.create_publisher(Float64MultiArray, '/trajectory_velocities', 10)

        self.timer = self.create_timer(1.0, self._publish)
        self.get_logger().info(
            f'Pipeline: {len(waypoints)} waypoints -> {len(smoothed)} smoothed -> '
            f'{len(trajectory)} trajectory points'
        )

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        frame = 'odom'

        self.wp_pub.publish(self._make_path(self.waypoints, stamp, frame))
        self.path_pub.publish(self._make_path(self.smoothed, stamp, frame))

        traj_points = np.column_stack([self.trajectory.x, self.trajectory.y])
        self.traj_pub.publish(self._make_path(traj_points, stamp, frame, self.trajectory.heading))

        vel_msg = Float64MultiArray()
        vel_msg.data = self.trajectory.velocity.tolist()
        self.vel_pub.publish(vel_msg)

    def _make_path(self, points, stamp, frame, headings=None):
        msg = Path()
        msg.header = Header(stamp=stamp, frame_id=frame)
        for i, pt in enumerate(points):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(pt[0])
            pose.pose.position.y = float(pt[1])
            if headings is not None:
                yaw = float(headings[i])
                pose.pose.orientation.z = math.sin(yaw / 2)
                pose.pose.orientation.w = math.cos(yaw / 2)
            msg.poses.append(pose)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

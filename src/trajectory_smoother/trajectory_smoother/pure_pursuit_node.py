"""ROS2 node for Pure Pursuit trajectory tracking.

Subscribes to the trajectory and robot odometry, computes velocity
commands using the Pure Pursuit controller, and publishes cmd_vel.
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import numpy as np
import math
import yaml

from trajectory_smoother.pure_pursuit import PurePursuitController, RobotState


class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        if not config_file:
            self.get_logger().error('No config_file parameter provided')
            return

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        ctrl_cfg = config.get('controller', {})
        traj_cfg = config.get('trajectory', {})

        self.controller = PurePursuitController(
            lookahead_distance=ctrl_cfg.get('lookahead_distance', 0.3),
            min_lookahead=ctrl_cfg.get('min_lookahead', 0.15),
            max_lookahead=ctrl_cfg.get('max_lookahead', 0.6),
            goal_tolerance=ctrl_cfg.get('goal_tolerance', 0.1),
            max_linear_vel=traj_cfg.get('max_velocity', 0.22),
            max_angular_vel=ctrl_cfg.get('max_angular_velocity', 2.84),
        )

        self.traj_x = None
        self.traj_y = None
        self.traj_vel = None

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.traj_sub = self.create_subscription(Path, '/trajectory', self._on_trajectory, 10)
        self.vel_sub = self.create_subscription(
            Float64MultiArray, '/trajectory_velocities', self._on_velocities, 10
        )
        self.odom_sub = self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        self.get_logger().info('Pure Pursuit controller ready, waiting for trajectory...')

    def _on_trajectory(self, msg: Path):
        self.traj_x = np.array([p.pose.position.x for p in msg.poses])
        self.traj_y = np.array([p.pose.position.y for p in msg.poses])
        self.controller.reset()
        self.get_logger().info(f'Received trajectory with {len(msg.poses)} points')

    def _on_velocities(self, msg: Float64MultiArray):
        self.traj_vel = np.array(msg.data)

    def _on_odom(self, msg: Odometry):
        if self.traj_x is None or self.traj_vel is None:
            return

        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        state = RobotState(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            theta=yaw,
        )

        cmd = self.controller.compute_command(
            state, self.traj_x, self.traj_y, self.traj_vel
        )

        twist = Twist()
        twist.linear.x = cmd.linear
        twist.angular.z = cmd.angular
        self.cmd_pub.publish(twist)

        if self.controller.goal_reached:
            self.get_logger().info('Goal reached!')


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

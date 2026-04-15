"""ROS2 node: Pure Pursuit controller. Subscribes odom + trajectory, publishes cmd_vel."""

import math
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from motion_planner_core.pure_pursuit import PurePursuitController, RobotState
from motion_planner_core.potential_field import PotentialField, Obstacle


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        if not config_file:
            self.get_logger().error('No config_file parameter')
            return

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        ctrl_cfg = config.get('controller', {})
        traj_cfg = config.get('trajectory', {})

        self.controller = PurePursuitController(
            lookahead_distance=ctrl_cfg.get('lookahead_distance', 0.3),
            goal_tolerance=ctrl_cfg.get('goal_tolerance', 0.1),
            max_linear_vel=traj_cfg.get('max_velocity', 0.22),
            max_angular_vel=ctrl_cfg.get('max_angular_velocity', 2.84),
        )
        self.max_angular_vel = ctrl_cfg.get('max_angular_velocity', 2.84)

        # Optional potential field safety net
        obs_cfg = config.get('obstacles', [])
        if obs_cfg:
            obstacles = [Obstacle(x=o[0], y=o[1], radius=o[2]) for o in obs_cfg]
            pf_cfg = config.get('potential_field', {})
            self.potential_field = PotentialField(
                obstacles=obstacles,
                influence_distance=pf_cfg.get('influence_distance', 0.6),
                repulsive_gain=pf_cfg.get('repulsive_gain', 0.3),
            )
        else:
            self.potential_field = None

        self.traj_x = None
        self.traj_y = None
        self.traj_vel = None

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Path, '/trajectory', self._on_trajectory, 10)
        self.create_subscription(Float64MultiArray, '/trajectory_velocities', self._on_vel, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        self.get_logger().info('Controller ready')

    def _on_trajectory(self, msg):
        self.traj_x = np.array([p.pose.position.x for p in msg.poses])
        self.traj_y = np.array([p.pose.position.y for p in msg.poses])
        self.controller.reset()

    def _on_vel(self, msg):
        self.traj_vel = np.array(msg.data)

    def _on_odom(self, msg):
        if self.traj_x is None or self.traj_vel is None:
            return

        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        state = RobotState(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            theta=yaw,
        )

        cmd = self.controller.compute_command(state, self.traj_x, self.traj_y, self.traj_vel)
        linear, angular = cmd.linear, cmd.angular

        if self.potential_field and not self.controller.goal_reached:
            linear, angular = self.potential_field.adjust_velocity(
                state.x, state.y, state.theta, linear, angular, self.max_angular_vel,
            )

        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.cmd_pub.publish(twist)

        if self.controller.goal_reached:
            self.get_logger().info('Goal reached!')


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

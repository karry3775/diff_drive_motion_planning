"""ROS2 node: Pure Pursuit controller. Subscribes odom + trajectory, publishes cmd_vel."""

import math
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from motion_planner_core.controller import create_controller
from motion_planner_core.types import RobotState, Trajectory
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

        self.controller = create_controller(config)
        self.max_angular_vel = config.get('controller', {}).get('max_angular_velocity', 2.84)
        self.start_time = None

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

        self.trajectory = None

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Path, '/trajectory', self._on_trajectory, 10)
        self.create_subscription(Float64MultiArray, '/trajectory_velocities', self._on_vel, 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 10)

        # Temp storage until we have both path and velocities
        self._traj_x = None
        self._traj_y = None
        self._traj_vel = None

        self.get_logger().info('Controller ready')

    def _on_trajectory(self, msg):
        self._traj_x = np.array([p.pose.position.x for p in msg.poses])
        self._traj_y = np.array([p.pose.position.y for p in msg.poses])
        headings = []
        for p in msg.poses:
            q = p.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            headings.append(yaw)
        self._traj_heading = np.array(headings)
        self._build_trajectory()

    def _on_vel(self, msg):
        self._traj_vel = np.array(msg.data)
        self._build_trajectory()

    def _build_trajectory(self):
        if self._traj_x is None or self._traj_vel is None:
            return
        if len(self._traj_x) != len(self._traj_vel):
            return
        dt = 0.05
        time = np.arange(len(self._traj_vel)) * dt
        self.trajectory = Trajectory(
            self._traj_x, self._traj_y, self._traj_heading,
            self._traj_vel, time,
        )
        self.start_time = None
        self.controller.reset()
        self.get_logger().info(f'Trajectory loaded: {len(self.trajectory)} points')

    def _on_odom(self, msg):
        if self.trajectory is None:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.start_time is None:
            self.start_time = now
        elapsed = now - self.start_time

        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        state = RobotState(
            x=msg.pose.pose.position.x,
            y=msg.pose.pose.position.y,
            theta=yaw,
        )

        cmd = self.controller.compute_command(state, self.trajectory, elapsed)
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
        if self.controller.faulted:
            self.get_logger().error('Controller faulted — cross-track error too large')


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

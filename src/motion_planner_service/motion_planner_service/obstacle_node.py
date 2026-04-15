"""ROS2 node: spawns obstacles in Gazebo and publishes RViz markers."""

import os
import yaml
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from gazebo_msgs.srv import SpawnEntity
from ament_index_python.packages import get_package_share_directory


class ObstacleNode(Node):
    def __init__(self):
        super().__init__('obstacle_node')

        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value
        if not config_file:
            self.get_logger().error('No config_file parameter')
            return

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.obstacles = config.get('obstacles', [])
        if not self.obstacles:
            self.get_logger().info('No obstacles configured')
            return

        pkg_dir = get_package_share_directory('motion_planner_service')
        template_path = os.path.join(pkg_dir, 'assets', 'obstacle.sdf.template')
        with open(template_path, 'r') as f:
            self.sdf_template = f.read()

        self.marker_pub = self.create_publisher(MarkerArray, '/obstacles', 10)
        self.timer = self.create_timer(1.0, self._publish_markers)
        self._spawn_in_gazebo()
        self.get_logger().info(f'Spawned {len(self.obstacles)} obstacles')

    def _spawn_in_gazebo(self):
        client = self.create_client(SpawnEntity, '/spawn_entity')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Gazebo spawn service unavailable — markers only')
            return

        for i, obs in enumerate(self.obstacles):
            x, y, radius = obs[0], obs[1], obs[2]
            name = f'obstacle_{i}'
            sdf = self.sdf_template.format(
                name=name, radius=radius, height=0.4,
                r=1, g=0, b=0,
            )

            req = SpawnEntity.Request()
            req.name = name
            req.xml = sdf
            req.initial_pose.position.x = x
            req.initial_pose.position.y = y
            req.initial_pose.position.z = 0.2
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

    def _publish_markers(self):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, obs in enumerate(self.obstacles):
            m = Marker()
            m.header = Header(stamp=stamp, frame_id='odom')
            m.ns = 'obstacles'
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = obs[0]
            m.pose.position.y = obs[1]
            m.pose.position.z = 0.2
            m.scale.x = obs[2] * 2
            m.scale.y = obs[2] * 2
            m.scale.z = 0.4
            m.color.r = 1.0
            m.color.a = 0.8
            marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

"""ROS2 node that spawns obstacles in Gazebo and publishes RViz markers.

Reads obstacle positions from the config YAML and:
1. Spawns cylinder models in Gazebo via /spawn_entity service
2. Publishes MarkerArray on /obstacles for RViz visualization
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from gazebo_msgs.srv import SpawnEntity
import yaml


class ObstacleSpawnerNode(Node):
    def __init__(self):
        super().__init__('obstacle_spawner_node')

        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').get_parameter_value().string_value

        if not config_file:
            self.get_logger().error('No config_file parameter provided')
            return

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.obstacles = config.get('obstacles', [])
        if not self.obstacles:
            self.get_logger().info('No obstacles configured')
            return

        # Publish markers for RViz
        self.marker_pub = self.create_publisher(MarkerArray, '/obstacles', 10)
        self.timer = self.create_timer(1.0, self._publish_markers)

        # Spawn in Gazebo
        self._spawn_in_gazebo()

        self.get_logger().info(f'Spawned {len(self.obstacles)} obstacles')

    def _spawn_in_gazebo(self):
        client = self.create_client(SpawnEntity, '/spawn_entity')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn('Gazebo spawn service not available — markers only')
            return

        for i, obs in enumerate(self.obstacles):
            x, y, radius = obs[0], obs[1], obs[2]
            name = f'obstacle_{i}'
            sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>0.4</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
          <diffuse>1 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""
            req = SpawnEntity.Request()
            req.name = name
            req.xml = sdf
            req.initial_pose.position.x = x
            req.initial_pose.position.y = y
            req.initial_pose.position.z = 0.2
            future = client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            if future.result() is not None:
                self.get_logger().info(f'Spawned {name} at ({x}, {y}) r={radius}')
            else:
                self.get_logger().warn(f'Failed to spawn {name}')

    def _publish_markers(self):
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i, obs in enumerate(self.obstacles):
            x, y, radius = obs[0], obs[1], obs[2]
            marker = Marker()
            marker.header = Header(stamp=stamp, frame_id='odom')
            marker.ns = 'obstacles'
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.2
            marker.scale.x = radius * 2  # diameter
            marker.scale.y = radius * 2
            marker.scale.z = 0.4
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleSpawnerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
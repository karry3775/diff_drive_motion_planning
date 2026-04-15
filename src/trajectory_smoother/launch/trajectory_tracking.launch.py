"""Launch file for the full trajectory tracking pipeline.

Launches:
  1. path_smoother_node — loads waypoints, publishes smoothed path
  2. trajectory_generator_node — generates time-parameterized trajectory
  3. pure_pursuit_node — tracks trajectory, publishes cmd_vel
  4. data_recorder_node -- records data
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('trajectory_smoother')
    config_file = os.path.join(pkg_dir, 'config', 'waypoints.yaml')

    return LaunchDescription([
        Node(
            package='trajectory_smoother',
            executable='path_smoother_node',
            name='path_smoother',
            parameters=[{'config_file': config_file}],
            output='screen',
        ),
        Node(
            package='trajectory_smoother',
            executable='trajectory_generator_node',
            name='trajectory_generator',
            parameters=[{'config_file': config_file}],
            output='screen',
        ),
        Node(
            package='trajectory_smoother',
            executable='pure_pursuit_node',
            name='pure_pursuit',
            parameters=[{'config_file': config_file}],
            output='screen',
        ),
        Node(
            package='trajectory_smoother',
            executable='data_recorder_node',
            name='data_recorder',
            parameters=[{'config_file': config_file}],
            output='screen',
        ),
    ])

"""Launch: obstacle spawner, planner, controller, recorder."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('motion_planner_service')
    config_file = os.path.join(pkg_dir, 'config', 'waypoints.yaml')

    common_params = [{'config_file': config_file}]

    return LaunchDescription([
        Node(
            package='motion_planner_service',
            executable='obstacle_node',
            name='obstacle_node',
            parameters=common_params,
            output='screen',
        ),
        Node(
            package='motion_planner_service',
            executable='planner_node',
            name='planner_node',
            parameters=common_params,
            output='screen',
        ),
        Node(
            package='motion_planner_service',
            executable='controller_node',
            name='controller_node',
            parameters=common_params,
            output='screen',
        ),
        Node(
            package='motion_planner_service',
            executable='recorder_node',
            name='recorder_node',
            parameters=common_params,
            output='screen',
        ),
    ])

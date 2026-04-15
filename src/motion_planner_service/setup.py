from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'motion_planner_service'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'assets'), glob('assets/*')),
    ],
    install_requires=['setuptools', 'motion_planner_core'],
    zip_safe=True,
    maintainer='kartik',
    maintainer_email='kartikprakash3775@gmail.com',
    description='ROS2 service layer for motion_planner_core',
    license='MIT',
    entry_points={
        'console_scripts': [
            'planner_node = motion_planner_service.planner_node:main',
            'controller_node = motion_planner_service.controller_node:main',
            'recorder_node = motion_planner_service.recorder_node:main',
            'obstacle_node = motion_planner_service.obstacle_node:main',
        ],
    },
)

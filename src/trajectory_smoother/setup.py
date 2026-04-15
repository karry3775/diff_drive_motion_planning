from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'trajectory_smoother'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='kartik',
    maintainer_email='kartikprakash3775@gmail.com',
    description='Path smoothing and trajectory tracking for differential drive robots',
    license='MIT',
    entry_points={
        'console_scripts': [
            'path_smoother_node = trajectory_smoother.path_smoother_node:main',
            'trajectory_generator_node = trajectory_smoother.trajectory_generator_node:main',
            'pure_pursuit_node = trajectory_smoother.pure_pursuit_node:main',
            'data_recorder_node = trajectory_smoother.data_recorder_node:main',
            'obstacle_spawner_node = trajectory_smoother.obstacle_spawner_node:main'
        ],
    },
)

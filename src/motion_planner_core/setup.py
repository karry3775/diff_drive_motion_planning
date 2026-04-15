from setuptools import find_packages, setup

setup(
    name='motion_planner_core',
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    install_requires=['numpy', 'scipy', 'pyyaml', 'matplotlib'],
    python_requires='>=3.10',
    description='Pure Python motion planning library: path planning, smoothing, trajectory generation, tracking',
    license='MIT',
)

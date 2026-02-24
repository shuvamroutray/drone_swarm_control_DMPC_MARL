from setuptools import setup
import glob
import os

package_name = 'control_swarm_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        # Install launch files
        (os.path.join('share', package_name, 'launch'),
            glob.glob('launch/*.launch.py')),

        # Install models
        (os.path.join('share', package_name, 'models'),
            glob.glob('models/*.sdf')),

        (os.path.join('share', package_name, 'models/simple_drone'),
            glob.glob('models/simple_drone/*.sdf')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shuvam',
    maintainer_email='your@email.com',
    description='ROS2 integration for DMPC swarm control',
    license='MIT',

    entry_points={
        'console_scripts': [
            'swarm_central_agent_node = control_swarm_ros.swarm_central_agent_node:main',
            'swarm_decentral_agent_node = control_swarm_ros.swarm_decentral_agent_node:main',
            'swarm_decentral_agent_node_crazy_SIM = control_swarm_ros.swarm_decentral_agent_node_crazy_SIM:main',
        ],
    },
)
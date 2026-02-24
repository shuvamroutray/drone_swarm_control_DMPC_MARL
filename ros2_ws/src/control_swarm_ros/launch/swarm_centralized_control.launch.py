from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='control_swarm_ros',
            executable='swarm_central_agent_node',
            output='screen'
        )
    ])


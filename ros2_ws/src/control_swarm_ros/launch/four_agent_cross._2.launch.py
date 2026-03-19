from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    drone_names = ["cf231", "cf232", "cf233", "cf234"]

    targets = [
        [0.0, -1.5, 1.0],
        [0.0, 1.5, 1.0],
        [-1.5, 0.0, 1.0],
        [1.5, 0.0, 1.0]
    ]

    nodes = []

    for i, name in enumerate(drone_names):

        neighbors = [n for n in drone_names if n != name]

        nodes.append(
            Node(
                package='control_swarm_ros',
                executable='four_agent_cross_2',
                namespace=name,
                parameters=[{
                    'drone_name': name,
                    'neighbor_names': neighbors,
                    'num_drones': 4,
                    'target': targets[i]
                }],
                output='screen'
            )
        )

    return LaunchDescription(nodes)
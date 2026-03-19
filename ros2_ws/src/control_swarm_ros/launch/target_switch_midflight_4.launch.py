from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    drones = ["cf231", "cf232"]

    nodes = []

    for name in drones:

        neighbor = ["cf232"] if name == "cf231" else ["cf231"]

        nodes.append(
            Node(
                package='control_swarm_ros',
                executable='target_switch_midflight_4',
                namespace=name,
                parameters=[{
                    'drone_name': name,
                    'neighbor_names': neighbor,
                    'num_drones': 2,
                    'switch_time': 20.0
                }],
                output='screen'
            )
        )

    return LaunchDescription(nodes)
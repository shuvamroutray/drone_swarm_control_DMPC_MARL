from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # ============================================================
    # EXPERIMENT CONFIG
    # ============================================================

    drone_names = ["cf231", "cf232"]
    num_drones = 2

    # Initial targets (before switch)
    targets_1 = {
        "cf231": [0.0, -1.0, 1.0],
        "cf232": [0.0,  1.0, 1.0],
    }

    # Step targets (after switch)
    targets_2 = {
        "cf231": [1.5, -1.0, 1.0],
        "cf232": [-1.5,  1.0, 1.0],
    }

    switch_time = 10.0   # seconds

    # ============================================================
    # NODE CREATION
    # ============================================================

    nodes = []

    for name in drone_names:

        neighbor_names = [n for n in drone_names if n != name]

        nodes.append(
            Node(
                package='control_swarm_ros',
                executable='mission_change_midflight_3',
                namespace=name,
                parameters=[{
                    'drone_name': name,
                    'num_drones': num_drones,
                    'neighbor_names': neighbor_names,
                    'target': targets_1[name],
                    'target_2': targets_2[name],
                    'switch_time': switch_time
                }],
                output='screen'
            )
        )

    return LaunchDescription(nodes)
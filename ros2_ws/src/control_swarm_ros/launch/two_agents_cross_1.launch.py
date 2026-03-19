from launch import LaunchDescription
from launch_ros.actions import Node
import yaml
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    #Relevant ROS Parameters
    # horizon_len = 10
    # control_freq = 1.0/10.0   

    # Path to crazyflies.yaml
    crazy_pkg_path = get_package_share_directory("crazyflie")
    yaml_path = os.path.join(crazy_pkg_path, "config", "crazyflies.yaml")

    # Load YAML
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    robots = config.get("robots", {})

    # Extract only enabled drones
    drone_names = [
        name for name, data in robots.items()
        if data.get("enabled", False)
    ]

    num_drones = len(drone_names)

    print(f"Detected drones: {drone_names}")

    # Target Generation
    targets = []

    #*******Auto generation of targets: Symmetric Targets
    # spacing = 1.5
    # for i in range(num_drones):
    #     targets.append([0.0, (i - num_drones/2) * spacing, 1.0])

    
    #*******Manual Target Generation************
    targets = [
        [0,  -1.5, 1.0],
        [0,  1.5, 1.0],
        [-1.5, 0, 1.0],
        [1.5, 0, 1.0]
    ]



    nodes = []

    for i, name in enumerate(drone_names):

        neighbor_names = [
            n for n in drone_names if n != name
        ]

        nodes.append(
            Node(
                package='control_swarm_ros',
                executable='two_agents_cross_1',
                namespace=name,
                parameters=[{
                    'drone_name': name,
                    'num_drones': num_drones,
                    'neighbor_names': neighbor_names,
                    'target': targets[i]
                    # 'Np': horizon_len,
                    # 'dt':control_freq
                }],
                output='screen'
            )
        )

    return LaunchDescription(nodes)
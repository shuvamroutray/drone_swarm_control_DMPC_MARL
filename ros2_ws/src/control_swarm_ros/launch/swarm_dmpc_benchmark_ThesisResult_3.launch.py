import yaml
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

from ament_index_python.packages import (
    get_package_share_directory
)


def generate_launch_description():


    config_path = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'crazyflies.yaml'
    )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    drone_names = [
        name for name, cfg
        in config['robots'].items()
        if cfg.get('enabled', False)
    ]

    num_drones = len(drone_names)

    nodes = []


    scenario_seed = LaunchConfiguration(
        "scenario_seed"
    )


    for i, name in enumerate(drone_names):

        neighbors = [
            n for n in drone_names
            if n != name
        ]



        node = Node(
            package="control_swarm_ros",
            executable="swarm_node_crazy_SIM_MARL_PeriodicTrigger_4c_ThesisResults",
            namespace=name,
            name=f"{name}_dmpc",
            parameters=[{
                "drone_name": name,
                "neighbor_names": neighbors,
                "num_drones": num_drones,
                "drone_id": i,
                "Np": 10,
                "dt": 0.1,
                "scenario_seed": scenario_seed,

            }],
            output="screen",
        )

        nodes.append(node)

    return LaunchDescription([

        DeclareLaunchArgument(
            "scenario_seed",
            default_value="0"
        ),

        *nodes
    ])
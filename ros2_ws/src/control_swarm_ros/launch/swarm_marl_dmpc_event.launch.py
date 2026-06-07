import yaml
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ============================================================
    # LOAD DRONE CONFIG
    # ============================================================
    config_path = os.path.join(
        get_package_share_directory('crazyflie'),
        'config',
        'crazyflies.yaml'
    )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    drone_names = [
        name for name, cfg in config['robots'].items()
        if cfg.get('enabled', False)
    ]

    num_drones = len(drone_names)

    nodes = []

    # ============================================================
    # CREATE NODES PER DRONE
    # ============================================================
    for i, name in enumerate(drone_names):

        neighbors = [n for n in drone_names if n != name]

        # ========================================================
        # MARL NODE
        # ========================================================
        marl_node = Node(
            package="control_swarm_ros",
            executable="marl_decentral_agent_node",
            name=f"{name}_marl",
            namespace=name,
            parameters=[{
                "drone_id": i,
                "num_drones": num_drones,
                "drone_name": name,
                "drone_names": drone_names
            }],
            output="screen",
        )

        # ========================================================
        # DMPC NODE (EVENT TRIGGERED)
        # ========================================================
        dmpc_node = Node(
            package="control_swarm_ros",
            executable="swarm_node_crazy_SIM_MARL_EventTrigger",
            name=f"{name}_dmpc",
            namespace=name,
            parameters=[{
                "drone_name": name,
                "neighbor_names": neighbors,
                "num_drones": num_drones,
                "Np": 10,        # match your controller
                "dt": 0.1,
            }],
            output="screen",
        )

        nodes.append(marl_node)
        nodes.append(dmpc_node)

    return LaunchDescription(nodes)
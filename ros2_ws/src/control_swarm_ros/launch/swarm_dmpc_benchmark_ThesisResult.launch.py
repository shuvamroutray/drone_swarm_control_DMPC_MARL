import yaml
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import SetEnvironmentVariable


def generate_launch_description():

    # ============================================================
    # LOAD DRONE CONFIG
    # ============================================================

    # env_pythonpath = SetEnvironmentVariable(
    #     name='PYTHONPATH',
    #     value=
    #         '/home/shuvam/.local/lib/python3.10/site-packages:'
    #         '/home/shuvam/6_Drone_Swarm_DMPC_MARL/Control_Swarm/src:'
    #         '/home/shuvam/6_Drone_Swarm_DMPC_MARL/ros2_ws/install/control_swarm_ros/lib/python3.10/site-packages:'
    #         '$PYTHONPATH'
    # )


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
    # DMPC NODES ONLY
    # ============================================================

    for i, name in enumerate(drone_names):

        neighbors = [
            n for n in drone_names
            if n != name
        ]

        dmpc_node = Node(
            package="control_swarm_ros",
            executable="swarm_node_crazy_SIM_MARL_EventTrigger_4a_ThesisResults",
            name=f"{name}_dmpc",
            namespace=name,
            parameters=[{
                "drone_name": name,
                "neighbor_names": neighbors,
                "num_drones": num_drones,
                "drone_id": i,
                "Np": 10,
                "dt": 0.1,
            }],
            output="screen",
        )

        nodes.append(dmpc_node)

    # ============================================================
    # BENCHMARK MANAGER
    # ============================================================

    benchmark_node = Node(
        package="control_swarm_ros",
        executable="dmpc_benchmark_manager_ThesisResults",
        name="dmpc_benchmark_manager",
        output="screen",
    )

    nodes.append(benchmark_node)

    return LaunchDescription(nodes)
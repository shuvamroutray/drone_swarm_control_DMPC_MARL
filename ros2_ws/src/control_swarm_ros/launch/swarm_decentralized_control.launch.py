from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np


def generate_launch_description():

    no_of_drones = 2

    # targets = [
    #     [2.0,  0.0, 1.0],
    #     [-2.0,  0.0, 1.0]
    # ]

    targets = [
        [0,  -1.5, 1.0],
        [0,  1.5, 1.0]
    ]

    

    nodes = []

    for i in range(no_of_drones):

        nodes.append(

            Node(
                package='control_swarm_ros',
                executable='swarm_decentral_agent_node',
                namespace=f'cf{i+1}',
                parameters=[{'drone_id': i, 'num_drones': no_of_drones, 'target': targets[i]}], # Note: The drone id is starting from 0 but the namespace is starting from 1, i.e: drone0 will have namespace /cf1

            )
        )   

    return LaunchDescription(nodes)
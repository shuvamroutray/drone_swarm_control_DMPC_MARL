from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os


def generate_launch_description():

    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    spawn_cf1 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'cf1',
            '-file', os.path.join(
                os.getenv('AMENT_PREFIX_PATH').split(':')[0],
                'share/control_swarm_ros/models/simple_drone/simple_drone.sdf'
            ),
            '-x', '0', '-y', '1.5', '-z', '0.5',
            '-robot_namespace', 'cf1'
        ],
        output='screen'
    )

    spawn_cf2 = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'cf2',
            '-file', os.path.join(
                os.getenv('AMENT_PREFIX_PATH').split(':')[0],
                'share/control_swarm_ros/models/simple_drone/simple_drone.sdf'
            ),
            '-x', '0', '-y', '-1.5', '-z', '0.5',
            '-robot_namespace', 'cf2'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        spawn_cf1,
        spawn_cf2
    ])
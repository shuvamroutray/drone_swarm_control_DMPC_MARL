#!/bin/bash

cd ~/6_Drone_Swarm_DMPC_MARL/ros2_ws

export PYTHONPATH=$PYTHONPATH:/home/shuvam/.local/lib/python3.10/site-packages
export PYTHONPATH=$PYTHONPATH:~/6_Drone_Swarm_DMPC_MARL/Control_Swarm/src

source /opt/ros/humble/setup.bash
source ~/6_Drone_Swarm_DMPC_MARL/ros2_ws/install/setup.bash

ros2 launch crazyflie launch.py backend:=sim gui:=False

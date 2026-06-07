#!/bin/bash

cd ~/6_Drone_Swarm_DMPC_MARL

export PYTHONPATH=$PYTHONPATH:/home/shuvam/.local/lib/python3.10/site-packages
export PYTHONPATH=$PYTHONPATH:~/6_Drone_Swarm_DMPC_MARL/Control_Swarm/src

source /opt/ros/humble/setup.bash
source ~/6_Drone_Swarm_DMPC_MARL/ros2_ws/install/setup.bash

python3 ~/6_Drone_Swarm_DMPC_MARL/ros2_ws/src/control_swarm_ros/control_swarm_ros/Scripts_MARL_DMPC/swarm_node_crazy_SIM_MARL_EventTrigger_4.py \
--ros-args \
-p drone_name:=cf232 \
-p neighbor_names:="[cf231]" \
-p drone_id:=1 \
-p num_drones:=2

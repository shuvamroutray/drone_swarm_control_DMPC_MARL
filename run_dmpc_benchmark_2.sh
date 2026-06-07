#!/bin/bash

NUM_SCENARIOS=3

RESULTS_DIR=~/6_Drone_Swarm_DMPC_MARL/dmpc_results

mkdir -p $RESULTS_DIR

for ((seed=0; seed<NUM_SCENARIOS; seed++))
do

    echo "====================================="
    echo "Running Scenario $seed"
    echo "====================================="

    cd ~/6_Drone_Swarm_DMPC_MARL

    export PYTHONPATH=$PYTHONPATH:/home/shuvam/.local/lib/python3.10/site-packages
    export PYTHONPATH=$PYTHONPATH:/home/shuvam/6_Drone_Swarm_DMPC_MARL/Control_Swarm/src

    source /opt/ros/humble/setup.bash
    source ~/6_Drone_Swarm_DMPC_MARL/ros2_ws/install/setup.bash

    rm -f /tmp/benchmark_done

	ros2 launch control_swarm_ros \
	    swarm_dmpc_benchmark_ThesisResult.launch.py \
	    scenario_seed:=$seed &

	LAUNCH_PID=$!

	echo "Launch PID = $LAUNCH_PID"

	while [ ! -f /tmp/benchmark_done ]
	do
	    sleep 1
	done

	echo "Scenario $seed completed"

	cp benchmark_results.csv \
	   $RESULTS_DIR/scenario_${seed}.csv

	kill -INT $LAUNCH_PID

	kill -INT $LAUNCH_PID

    wait $LAUNCH_PID 2>/dev/null

    sleep 2
            
    



    if [ -f benchmark_results.csv ]; then

        cp benchmark_results.csv \
           $RESULTS_DIR/scenario_${seed}.csv

    fi

    pkill -f ros2
    pkill -f swarm_node_crazy
    pkill -f benchmark_manager

    sleep 10

done

#!/bin/bash

NUM_SCENARIOS=3

RESULTS_DIR=~/6_Drone_Swarm_DMPC_MARL/dmpc_results

mkdir -p $RESULTS_DIR

for ((seed=0; seed<NUM_SCENARIOS; seed++))
do

    echo ""
    echo "====================================="
    echo "Running Scenario $seed"
    echo "====================================="

    #################################################
    # CLEAN PREVIOUS FILES
    #################################################

    rm -f /tmp/cf*_done
    rm -f /tmp/cf*_benchmark.txt

    #################################################
    # ENVIRONMENT
    #################################################

    export PYTHONPATH=$PYTHONPATH:/home/shuvam/.local/lib/python3.10/site-packages
    export PYTHONPATH=$PYTHONPATH:/home/shuvam/6_Drone_Swarm_DMPC_MARL/Control_Swarm/src

    source /opt/ros/humble/setup.bash
    source ~/6_Drone_Swarm_DMPC_MARL/ros2_ws/install/setup.bash

    #################################################
    # START SIMULATOR
    #################################################

    echo "Starting simulator..."

    ros2 launch crazyflie launch.py \
        backend:=sim \
        gui:=False \
        > /tmp/sim_${seed}.log 2>&1 &

    SIM_PID=$!

    echo "Simulator PID = $SIM_PID"

    sleep 20

    #################################################
    # START DMPC
    #################################################

    echo "Starting DMPC..."

    ros2 launch control_swarm_ros \
        swarm_dmpc_benchmark_ThesisResult_3.launch.py \
        scenario_seed:=$seed \
        > /tmp/dmpc_${seed}.log 2>&1 &

    DMPC_PID=$!

    echo "DMPC PID = $DMPC_PID"

    #################################################
    # WAIT FOR MISSION COMPLETE
    #################################################

    TIMEOUT=180

    START_TIME=$(date +%s)

    while true
    do

        DONE_COUNT=$(ls /tmp/cf*_done 2>/dev/null | wc -l)

        if [ "$DONE_COUNT" -eq 2 ]; then

            echo "Mission completed."

            break

        fi

        NOW=$(date +%s)

        ELAPSED=$((NOW - START_TIME))

        if [ "$ELAPSED" -gt "$TIMEOUT" ]; then

            echo "TIMEOUT"

            break

        fi

        sleep 1

    done

    #################################################
    # SAVE RESULTS
    #################################################

    mkdir -p \
        $RESULTS_DIR/scenario_${seed}

    cp /tmp/cf*_benchmark.txt \
       $RESULTS_DIR/scenario_${seed}/ \
       2>/dev/null

    echo "$seed" > \
       $RESULTS_DIR/scenario_${seed}/seed.txt

    #################################################
    # CLEANUP
    #################################################

    echo "Stopping DMPC..."

    kill -INT $DMPC_PID 2>/dev/null

    sleep 3

    echo "Stopping simulator..."

    kill -INT $SIM_PID 2>/dev/null

    sleep 5

    pkill -f swarm_node_crazy
    pkill -f control_swarm_ros
    pkill -f crazyflie
    pkill -f ros2

    sleep 10

done

echo ""
echo "====================================="
echo "ALL SCENARIOS COMPLETE"
echo "====================================="

#!/bin/bash

source /opt/ros/humble/setup.bash
source ../../../../install/local_setup.bash

# Name of the package and node(s)
PACKAGE_NAME="cbm_pop"
NODE_NAME="cbm_population_agent_online"
NODE_NAME_LLM="cbm_population_agent_llm"
LLM_INTERFACE_NODE="llm_interface_agent"

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <INSTANCE_COUNT> <TRACKING_TIMEOUT> <LEARNING_METHOD>"
    exit 1
fi

INSTANCE_COUNT=$1
TRACKING_TIMEOUT=$2
USER_LEARNING_METHOD=$3

PROBLEM_FILENAME="resources/150_Task_Problem.csv"
START_AGENT_ID=1
USE_LLM_NODE=false
REAL_LEARNING_METHOD="$USER_LEARNING_METHOD"

case "$USER_LEARNING_METHOD" in
    "FEA_LLM")
        REAL_LEARNING_METHOD="Ferreira_et_al."
        USE_LLM_NODE=true
        ;;
    "QL_LLM")
        REAL_LEARNING_METHOD="Q-Learning"
        USE_LLM_NODE=true
        ;;
esac

cleanup() {
    echo "Stopping all cbm-population-agent processes..."
    pkill -f "cbm_population_agent" 2>/dev/null
    pkill -f "cbm_population_agent_llm" 2>/dev/null
    pkill -f "cbm_fitness_logger" 2>/dev/null
    pkill -f "$LLM_INTERFACE_NODE" 2>/dev/null
    sleep 1
    echo "All processes have been stopped."
}

trap cleanup SIGINT SIGTERM EXIT

# Run the loop 20 times
for RUN in {1..5}; do
    echo "Starting iteration $RUN of 20..."

    # Array to track process IDs
    PIDS=()

    # Launch multiple agents
    for i in $(seq 0 $((INSTANCE_COUNT-1))); do
        AGENT_ID=$((START_AGENT_ID + i))
        echo "Launching instance with agent_id=$AGENT_ID, learning_method=$REAL_LEARNING_METHOD, and problem_filename=$PROBLEM_FILENAME..."

        if [ "$USE_LLM_NODE" = true ]; then
            ros2 run "$PACKAGE_NAME" "$NODE_NAME_LLM" --ros-args \
                -p agent_id:=$AGENT_ID \
                -p problem_filename:=$PROBLEM_FILENAME \
                -p runtime:=$TRACKING_TIMEOUT \
                -p learning_method:="$REAL_LEARNING_METHOD" &

            PIDS+=($!)  # Store process ID

            ros2 run "$PACKAGE_NAME" "$LLM_INTERFACE_NODE" --ros-args \
                -p runtime:=$TRACKING_TIMEOUT \
                -p agent_id:=$AGENT_ID &

            PIDS+=($!)  # Store process ID

        else
            ros2 run "$PACKAGE_NAME" "$NODE_NAME" --ros-args \
                -p agent_id:=$AGENT_ID \
                -p problem_filename:=$PROBLEM_FILENAME \
                -p runtime:=$TRACKING_TIMEOUT \
                -p learning_method:="$REAL_LEARNING_METHOD" &

            PIDS+=($!)  # Store process ID
        fi
    done

    # Launch cbm_fitness_logger as a foreground process
    ros2 run "$PACKAGE_NAME" cbm_fitness_logger --ros-args -p timeout:=$TRACKING_TIMEOUT &
    PIDS+=($!)

    echo "Launched $INSTANCE_COUNT agent(s) using learning_method=$USER_LEARNING_METHOD and fitness_logger."
    echo "Waiting for all ROS processes to finish..."

    while true; do
        ACTIVE_PROCESSES=$(pgrep -f "cbm_population_agent|cbm_fitness_logger|llm_interface_agent")

        if [ -z "$ACTIVE_PROCESSES" ]; then
            break  # Exit loop when all processes are gone
        fi

        sleep 1
    done

    # Ensure all related processes are actually stopped
    echo "Ensuring no ROS processes are left running..."
    pkill -f "cbm_population_agent" 2>/dev/null
    pkill -f "cbm_population_agent_llm" 2>/dev/null
    pkill -f "cbm_fitness_logger" 2>/dev/null
    pkill -f "$LLM_INTERFACE_NODE" 2>/dev/null
    sleep 1  # Give some time for cleanup

    # Final process check before proceeding
    if pgrep -f "cbm_population_agent" || pgrep -f "cbm_fitness_logger"; then
        echo "WARNING: Some processes are still running! Manually killing them..."
        pkill -9 -f "cbm_population_agent"
        pkill -9 -f "cbm_population_agent_llm"
        pkill -9 -f "cbm_fitness_logger"
        pkill -9 -f "$LLM_INTERFACE_NODE"
    fi

    echo "Iteration $RUN complete. Proceeding to next run..."
done

echo "All 20 iterations completed. Performing post-processing..."
# Example: python3 process_results.py

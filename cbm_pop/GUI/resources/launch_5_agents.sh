#!/bin/bash

source /opt/ros/humble/setup.bash
source ../../../../install/local_setup.bash

# Name of the package and node(s)
PACKAGE_NAME="cbm_pop"
NODE_NAME="cbm_population_agent_online"
NODE_NAME_LLM="cbm_population_agent_llm"
LLM_INTERFACE_NODE="llm_interface_agent"

# Check if INSTANCE_COUNT, TRACKING_TIMEOUT, and LEARNING_METHOD are passed as arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <INSTANCE_COUNT> <TRACKING_TIMEOUT> <LEARNING_METHOD>"
    exit 1
fi

INSTANCE_COUNT=$1
TRACKING_TIMEOUT=$2
USER_LEARNING_METHOD=$3

# For example, we'll hardcode your problem file
PROBLEM_FILENAME="resources/150_Task_Problem.csv"

# Base agent_id to start from
START_AGENT_ID=1

# Determine which node to use and which parameter value to pass
# If it's an LLM-based method, we launch the LLM node with the "real" param
# Otherwise, we launch the original node with the param as-is.
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
    # If it's something else (Ferreira_et_al., Q-Learning, etc.),
    # we'll leave USE_LLM_NODE=false (original node) and
    # REAL_LEARNING_METHOD=$USER_LEARNING_METHOD
esac

cleanup() {
    echo "Stopping all cbm-population-agent processes..."

    # Terminate all processes that match these
    pkill -f "cbm_population_agent" 2>/dev/null
    pkill -f "cbm_population_agent_llm" 2>/dev/null
    pkill -f "cbm_fitness_logger" 2>/dev/null
    pkill -f "$LLM_INTERFACE_NODE" 2>/dev/null

    # Wait for all processes to terminate
    sleep 1
    echo "All processes have been stopped."
    exit
}

trap cleanup SIGINT SIGTERM EXIT

# Launch multiple agents
for i in $(seq 0 $((INSTANCE_COUNT-1))); do
    AGENT_ID=$((START_AGENT_ID + i))
    echo "Launching instance with agent_id=$AGENT_ID, learning_method=$REAL_LEARNING_METHOD, and problem_filename=$PROBLEM_FILENAME..."

    if [ "$USE_LLM_NODE" = true ]; then
        # Use the LLM-enabled node
        ros2 run "$PACKAGE_NAME" "$NODE_NAME_LLM" --ros-args \
            -p agent_id:=$AGENT_ID \
            -p problem_filename:=$PROBLEM_FILENAME \
            -p runtime:=$TRACKING_TIMEOUT \
            -p learning_method:="$REAL_LEARNING_METHOD" &

        # Also launch the llm_interface_agent
        ros2 run "$PACKAGE_NAME" "$LLM_INTERFACE_NODE" --ros-args \
            -p runtime:=$TRACKING_TIMEOUT \
            -p agent_id:=$AGENT_ID &

    else
        # Otherwise, use the original node
        ros2 run "$PACKAGE_NAME" "$NODE_NAME" --ros-args \
            -p agent_id:=$AGENT_ID \
            -p problem_filename:=$PROBLEM_FILENAME \
            -p runtime:=$TRACKING_TIMEOUT \
            -p learning_method:="$REAL_LEARNING_METHOD" &
    fi
done

# Launch cbm_fitness_logger as a foreground process
ros2 run "$PACKAGE_NAME" cbm_fitness_logger --ros-args -p timeout:=$TRACKING_TIMEOUT

echo "Launched $INSTANCE_COUNT agent(s) using learning_method=$USER_LEARNING_METHOD and fitness_logger."
echo "Press Ctrl+C to stop all instances."

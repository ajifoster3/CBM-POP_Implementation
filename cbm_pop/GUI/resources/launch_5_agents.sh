#!/bin/bash

source /opt/ros/humble/setup.bash
source ../../../../install/local_setup.bash

# Name of the package and node
PACKAGE_NAME="cbm_pop"
NODE_NAME="cbm_population_agent"

# Check if INSTANCE_COUNT is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <INSTANCE_COUNT>"
    exit 1
fi

# Number of instances to run
INSTANCE_COUNT=$1
PROBLEM_FILENAME="resources/150_Task_Problem.csv"

# Base agent_id to start from
START_AGENT_ID=1

# Loop to start multiple instances
for i in $(seq 0 $((INSTANCE_COUNT - 1))); do
    AGENT_ID=$((START_AGENT_ID + i))  # Increment agent_id
    echo "Launching instance with agent_id=$AGENT_ID and problem_filename=$PROBLEM_FILENAME..."
    # Run each instance with parameters
    ros2 run $PACKAGE_NAME $NODE_NAME --ros-args -p agent_id:=$AGENT_ID -p problem_filename:=$PROBLEM_FILENAME &
done

echo "Launched $INSTANCE_COUNT instances of $NODE_NAME."
echo "Press Ctrl+C to stop all instances."

# Wait for all background processes
wait

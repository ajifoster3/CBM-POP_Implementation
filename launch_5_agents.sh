#!/bin/bash

# Name of the package and node
PACKAGE_NAME="cbm_pop"
NODE_NAME="cbm_population_agent"

# Number of instances to run
INSTANCE_COUNT=5
PROBLEM_FILENAME="src/cbm_pop/150_Task_Problem.csv"

# Base agent_id to start from
START_AGENT_ID=1

# Loop to start multiple instances
for i in $(seq 0 $((INSTANCE_COUNT - 1))); do
    AGENT_ID=$((START_AGENT_ID + i))  # Increment agent_id
    echo "Launching instance with agent_id=$AGENT_ID and problem_filename=$PROBLEM_FILENAME..."
    # Run each instance with parameters
    ros2 run $PACKAGE_NAME $NODE_NAME --ros-args -p agent_id:=$AGENT_ID -p problem_filename:=$PROBLEM_FILENAME &
    ros2 run $PACKAGE_NAME $NODE_NAME fitness_logger
done

echo "Launched $INSTANCE_COUNT instances of $NODE_NAME."
echo "Press Ctrl+C to stop all instances."

# Wait for all background processes
wait

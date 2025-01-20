#!/bin/bash

source /opt/ros/humble/setup.bash
source ../../../../install/local_setup.bash

# Name of the package and node
PACKAGE_NAME="cbm_pop"
NODE_NAME="cbm_population_agent"

# Check if INSTANCE_COUNT is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <INSTANCE_COUNT> <TRACKING_TIMEOUT> <LEARNING_METHOD>"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Usage: $0 <INSTANCE_COUNT> <TRACKING_TIMEOUT> <LEARNING_METHOD>"
    exit 1
fi
if [ -z "$3" ]; then
    echo "Usage: $0 <INSTANCE_COUNT> <TRACKING_TIMEOUT> <LEARNING_METHOD>"
    exit 1
fi

# Number of instances to run
INSTANCE_COUNT=$1
TRACKING_TIMEOUT=$2
LEARNING_METHOD=$3
PROBLEM_FILENAME="resources/150_Task_Problem.csv"

# Base agent_id to start from
START_AGENT_ID=1

# Function to clean up background processes
cleanup() {
    echo "Stopping all cbm-population-agent processes..."

    # Terminate all `cbm-population-agent` processes
    pkill -f "cbm-population-agent" 2>/dev/null

    # Also ensure the fitness_logger is stopped
    pkill -f "cbm_fitness_logger" 2>/dev/null

    # Wait for all processes to terminate
    sleep 1
    echo "All processes have been stopped."
    exit
}

# Set up a trap to handle script termination
trap cleanup SIGINT SIGTERM EXIT

# Loop to start multiple instances
for i in $(seq 0 $((INSTANCE_COUNT - 1))); do
    AGENT_ID=$((START_AGENT_ID + i))  # Increment agent_id
    echo "Launching instance with agent_id=$AGENT_ID and problem_filename=$PROBLEM_FILENAME..."
    # Run each instance with parameters
    ros2 run $PACKAGE_NAME $NODE_NAME --ros-args -p agent_id:=$AGENT_ID -p problem_filename:=$PROBLEM_FILENAME -p runtime:=$TRACKING_TIMEOUT -p learning_method:=$LEARNING_METHOD &
done

# Launch cbm_fitness_logger as a background process
ros2 run $PACKAGE_NAME cbm_fitness_logger --ros-args -p timeout:=$TRACKING_TIMEOUT

echo "Launched $INSTANCE_COUNT instances of $NODE_NAME and fitness_logger."
echo "Press Ctrl+C to stop all instances."

#!/bin/bash

# Configuration
NUM_AGENTS=20
PACKAGE_NAME="cbm_pop"
LOGGER_EXECUTABLE="simple_fitness_logger"
AGENT_EXECUTABLE="cbm_population_agent_online_simple_simulation"
SIM_EXECUTABLE="simple_simulator"
RUNTIME=-1.0  # seconds
LEARNING_METHOD="Ferreira_et_al."

# Optional: activate your ROS 2 workspace environment
# source ~/ros_ws/install/setup.bash

# Launch the fitness logger
echo "[INFO] Launching SimpleFitnessLogger..."
ros2 run $PACKAGE_NAME $LOGGER_EXECUTABLE &
LOGGER_PID=$!

# Launch the simulator (must support --num_robots or similar arg)
echo "[INFO] Launching SimpleSimulator..."
ros2 run $PACKAGE_NAME $SIM_EXECUTABLE --num_robots $NUM_AGENTS &
SIM_PID=$!

# Launch each agent
AGENT_PIDS=()
for ((i=0; i<NUM_AGENTS; i++)); do
  echo "[INFO] Launching Agent $i..."
  ros2 run $PACKAGE_NAME $AGENT_EXECUTABLE \
    --ros-args \
    -p agent_id:=$i \
    -p runtime:=$RUNTIME \
    -p learning_method:="$LEARNING_METHOD" \
    -p num_tsp_agents:=$NUM_AGENTS&
  AGENT_PIDS+=($!)
done

# Wait for all processes to finish
wait $LOGGER_PID
wait $SIM_PID
for pid in "${AGENT_PIDS[@]}"; do
  wait $pid
done

echo "[INFO] All nodes have completed."

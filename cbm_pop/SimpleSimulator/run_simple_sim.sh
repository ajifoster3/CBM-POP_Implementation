#!/bin/bash

# Configuration
NUM_AGENTS=10
NUM_RUNS=10  # <-- Number of times to repeat the whole process
PACKAGE_NAME="cbm_pop"
LOGGER_EXECUTABLE="simple_fitness_logger"
AGENT_EXECUTABLE="cbm_population_agent_online_simple_simulation_lock"
SIM_EXECUTABLE="simple_simulator"
RUNTIME=-1.0  # seconds
LEARNING_METHOD="Q-Learning"
LR=0.5
GAMMA_DECAY=0.5
POSITIVE_REWARD=1.0
NEGATIVE_REWARD=-0.5


# Optional: source your ROS 2 workspace
# source ~/ros_ws/install/setup.bash

for ((run=1; run<=NUM_RUNS; run++)); do
  echo "============================="
  echo "[INFO] Starting run $run/$NUM_RUNS"
  echo "============================="

  # Launch the fitness logger
  echo "[INFO] Launching SimpleFitnessLogger..."
  ros2 run $PACKAGE_NAME $LOGGER_EXECUTABLE &
  LOGGER_PID=$!

  # Launch the simulator
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
      -p num_tsp_agents:=$NUM_AGENTS \
      -p lr:=$LR \
      -p gamma_decay:=$GAMMA_DECAY \
      -p positive_reward:=$POSITIVE_REWARD \
      -p negative_reward:=$NEGATIVE_REWARD&
    AGENT_PIDS+=($!)
  done

  # Wait for all processes to finish
  wait $LOGGER_PID
  wait $SIM_PID
  for pid in "${AGENT_PIDS[@]}"; do
    wait $pid
  done

  echo "[INFO] Completed run $run"
  echo
done

echo "[INFO] All $NUM_RUNS runs completed."

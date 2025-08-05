#!/bin/bash
source /opt/ros/humble/setup.bash
source ../../../../install/local_setup.bash

# Configuration
NUM_AGENTS=10
NUM_RUNS=50
PACKAGE_NAME="cbm_pop"
LOGGER_EXECUTABLE="simple_fitness_logger"
AGENT_EXECUTABLE="cbm_population_agent_online_simple_simulation_lock"
SIM_EXECUTABLE="simple_simulator"
RUNTIME=-1.0
LEARNING_METHOD="Q-Learning"
TIMEOUT_SECONDS=300  # 5 minutes

RESULTS_ROOT="resources/run_logs"
mkdir -p "$RESULTS_ROOT"

# Grid: 4 values for each
LR_VALUES=(0.1 0.01 0.001)
POSITIVE_REWARD_VALUES=(1.0 3.0)
NEGATIVE_REWARD_VALUES=(-0.25 -2.0)
GAMMA_DECAYS=(0.2 0.3 0.4 0.5)

for GAMMA_DECAY in "${GAMMA_DECAYS[@]}"; do
  for LR in "${LR_VALUES[@]}"; do
    for POSITIVE_REWARD in "${POSITIVE_REWARD_VALUES[@]}"; do
      for NEGATIVE_REWARD in "${NEGATIVE_REWARD_VALUES[@]}"; do

        PARAM_DIR="$RESULTS_ROOT/gamma_${GAMMA_DECAY}_lr_${LR}_pos_${POSITIVE_REWARD}_neg_${NEGATIVE_REWARD}"
        mkdir -p "$PARAM_DIR"

        EXISTING_RUNS=$(find "$PARAM_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l)
        START_RUN=$((EXISTING_RUNS + 1))

        if [ "$EXISTING_RUNS" -ge "$NUM_RUNS" ]; then
          echo "[INFO] Already completed $EXISTING_RUNS runs for gamma=$GAMMA_DECAY, lr=$LR, pos=$POSITIVE_REWARD, neg=$NEGATIVE_REWARD. Skipping."
          continue
        fi

        run=$START_RUN
        while [ $run -le $NUM_RUNS ]; do
          echo "============================="
          echo "[INFO] Starting run $run/$NUM_RUNS with gamma=$GAMMA_DECAY, lr=$LR, pos=$POSITIVE_REWARD, neg=$NEGATIVE_REWARD"
          echo "============================="

          CONFIG_DIR="$PARAM_DIR/run_${run}"
          mkdir -p "$CONFIG_DIR"

          # Start logger
          ros2 run $PACKAGE_NAME $LOGGER_EXECUTABLE \
            --ros-args -p parent_log_dir:="$CONFIG_DIR" &
          LOGGER_PID=$!

          # Start simulator
          ros2 run $PACKAGE_NAME $SIM_EXECUTABLE --num_robots $NUM_AGENTS &
          SIM_PID=$!

          # Start agents
          AGENT_PIDS=()
          for ((i=0; i<NUM_AGENTS; i++)); do
            ros2 run $PACKAGE_NAME $AGENT_EXECUTABLE \
              --ros-args \
              -p agent_id:=$i \
              -p runtime:=$RUNTIME \
              -p learning_method:="$LEARNING_METHOD" \
              -p num_tsp_agents:=$NUM_AGENTS \
              -p lr:=$LR \
              -p gamma_decay:=$GAMMA_DECAY \
              -p positive_reward:=$POSITIVE_REWARD \
              -p negative_reward:=$NEGATIVE_REWARD &
            AGENT_PIDS+=($!)
          done

          # Wait with timeout
          START_TIME=$(date +%s)
          RUN_TIMEOUT=0

          while true; do
            sleep 5
            RUNNING=$(ps -p $LOGGER_PID $SIM_PID "${AGENT_PIDS[@]}" > /dev/null && echo "yes" || echo "no")
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))

            if [ "$RUNNING" == "no" ]; then
              echo "[INFO] Run $run completed in $ELAPSED seconds."
              break
            fi

            if [ "$ELAPSED" -gt "$TIMEOUT_SECONDS" ]; then
              echo "[WARNING] Run $run exceeded $TIMEOUT_SECONDS seconds. Killing processes and retrying..."
              RUN_TIMEOUT=1
              break
            fi
          done

          # Cleanup
          pkill -P $LOGGER_PID 2>/dev/null
          pkill -P $SIM_PID 2>/dev/null
          for pid in "${AGENT_PIDS[@]}"; do
            pkill -P $pid 2>/dev/null
          done
          kill $LOGGER_PID $SIM_PID "${AGENT_PIDS[@]}" 2>/dev/null

          if [ "$RUN_TIMEOUT" -eq 1 ]; then
            echo "[CLEANUP] Deleting failed run directory: $CONFIG_DIR"
            rm -rf "$CONFIG_DIR"
            echo "[RETRY] Repeating run $run"
            continue
          fi

          echo "[INFO] Completed run $run for gamma=$GAMMA_DECAY, lr=$LR, pos=$POSITIVE_REWARD, neg=$NEGATIVE_REWARD"
          echo
          run=$((run + 1))
        done

      done
    done
  done
done

echo "[INFO] All 256 parameter settings completed or skipped if up to date."

#!/usr/bin/env bash
set -Eeuo pipefail
ulimit -c 0 || true   # don't leave "core dumped" files
set +m                # quiet job-control messages

########################################
# Safe source (works with set -u)
########################################
_safesource() {
  local f="$1"; local was_nounset=0
  if [[ -o nounset ]]; then was_nounset=1; set +u; fi
  # shellcheck source=/dev/null
  source "$f"
  (( was_nounset )) && set -u
}
_safesource /opt/ros/humble/setup.bash
_safesource ../../../../install/local_setup.bash

########################################
# Helpers
########################################
# Start a command in a new session/process group and log stdout/stderr to a file.
# Args: <logfile> <ros_log_dir> <cmd...>
start_in_group_to () {
  local logfile="$1"; local roslogdir="$2"; shift 2
  mkdir -p "$(dirname "$logfile")" "$roslogdir"
  # Prefer line-buffered logging if stdbuf is available
  if command -v stdbuf >/dev/null 2>&1; then
    ROS_LOG_DIR="$roslogdir" PYTHONUNBUFFERED=1 setsid stdbuf -oL -eL "$@" >>"$logfile" 2>&1 &
  else
    ROS_LOG_DIR="$roslogdir" PYTHONUNBUFFERED=1 setsid "$@" >>"$logfile" 2>&1 &
  fi
  echo $!
}

is_alive () { kill -0 "$1" 2>/dev/null; }
wait_silently () { wait "$1" 2>/dev/null || true; }

# Recursively collect descendants (handles multi-level trees)
_collect_descendants() {
  local root="$1"
  local kids
  kids=$(pgrep -P "$root" || true)
  for c in $kids; do
    echo "$c"
    _collect_descendants "$c"
  done
}

kill_tree() {
  local pid="$1"
  # Gather all descendants first (depth-first)
  local all=()
  mapfile -t all < <(_collect_descendants "$pid")
  # INT -> TERM -> KILL across the tree
  for p in "${all[@]}" "$pid"; do /bin/kill -s INT "$p" 2>/dev/null || true; done
  sleep 1
  for p in "${all[@]}" "$pid"; do is_alive "$p" && /bin/kill -s TERM "$p" 2>/dev/null || true; done
  sleep 1
  for p in "${all[@]}" "$pid"; do is_alive "$p" && /bin/kill -s KILL "$p" 2>/dev/null || true; done
  wait_silently "$pid"
}

# Strong final sweep by executable name, not "ros2 run ..."
sweep_by_name() {
  local patterns=("$@")
  # TERM
  for pat in "${patterns[@]}"; do pkill -f "$pat" 2>/dev/null || true; done
  sleep 1
  # KILL anything stubborn
  for pat in "${patterns[@]}"; do pkill -9 -f "$pat" 2>/dev/null || true; done
}

########################################
# Config
########################################
NUM_AGENTS=10
NUM_RUNS=25
PACKAGE_NAME="cbm_pop"
LOGGER_EXECUTABLE="simple_fitness_logger"
AGENT_EXECUTABLE="cbm_population_agent_online_simple_simulation_offline"
SIM_EXECUTABLE="simple_simulator"
TRACKER_PATTERN="tracker"   # adjust if your tracker binary has a different name

RUNTIME=-1.0
LEARNING_METHOD="Q-Learning"
RUN_DURATION_SECONDS=60

RESULTS_ROOT="resources/run_logs"
mkdir -p "$RESULTS_ROOT"

# Patterns that match the **actual** node processes once launched
SWEEP_PATTERNS=(
  "(^|/)$SIM_EXECUTABLE( |$)"
  "(^|/)$LOGGER_EXECUTABLE( |$)"
  "(^|/)$AGENT_EXECUTABLE( |$)"
  "$TRACKER_PATTERN"
)

# Parameter values
LR_VALUES=(0.2)
POSITIVE_REWARD_VALUES=(1)
NEGATIVE_REWARD_VALUES=(-0.5)
GAMMA_DECAYS=(0.2)

########################################
# Cleanup on exit/Ctrl-C
########################################
LOGGER_PID=
SIM_PID=
AGENT_PIDS=()

cleanup () {
  [[ -n "${LOGGER_PID:-}" ]] && kill_tree "$LOGGER_PID"
  [[ -n "${SIM_PID:-}"    ]] && kill_tree "$SIM_PID"
  if ((${#AGENT_PIDS[@]})); then
    for pid in "${AGENT_PIDS[@]}"; do
      [[ -n "${pid:-}" ]] && kill_tree "$pid"
    done
  fi
  sweep_by_name "${SWEEP_PATTERNS[@]}"
}
trap cleanup EXIT INT TERM

########################################
# Main loop
########################################
for GAMMA_DECAY in "${GAMMA_DECAYS[@]}"; do
  for LR in "${LR_VALUES[@]}"; do
    for POSITIVE_REWARD in "${POSITIVE_REWARD_VALUES[@]}"; do
      for NEGATIVE_REWARD in "${NEGATIVE_REWARD_VALUES[@]}"; do

        PARAM_DIR="$RESULTS_ROOT/gamma_${GAMMA_DECAY}_lr_${LR}_pos_${POSITIVE_REWARD}_neg_${NEGATIVE_REWARD}"
        mkdir -p "$PARAM_DIR"

        EXISTING_RUNS=$(find "$PARAM_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l || echo 0)
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

          # A central "run log" for the launcher itself (optional)
          RUN_LOG="$CONFIG_DIR/run_launcher.log"
          {
            echo "[INFO] $(date -Iseconds) Launching processes..."

            # Set a per-run ROS 2 log directory (all nodes will write under here)
            ROS2_LOG_DIR="$CONFIG_DIR/ros_logs"

            # Start logger
            LOGGER_LOG="$CONFIG_DIR/logger.log"
            LOGGER_PID=$(start_in_group_to "$LOGGER_LOG" "$ROS2_LOG_DIR" \
              ros2 run "$PACKAGE_NAME" "$LOGGER_EXECUTABLE" \
              --ros-args -p parent_log_dir:="$CONFIG_DIR")

            # Start simulator
            SIM_LOG="$CONFIG_DIR/simulator.log"
            SIM_PID=$(start_in_group_to "$SIM_LOG" "$ROS2_LOG_DIR" \
              ros2 run "$PACKAGE_NAME" "$SIM_EXECUTABLE" --num_robots "$NUM_AGENTS")

            # Start agents (each to its own log)
            AGENT_PIDS=()
            for ((i=0; i<NUM_AGENTS; i++)); do
              AGENT_LOG="$CONFIG_DIR/agent_${i}.log"
              pid=$(start_in_group_to "$AGENT_LOG" "$ROS2_LOG_DIR" \
                ros2 run "$PACKAGE_NAME" "$AGENT_EXECUTABLE" \
                  --ros-args \
                  -p agent_id:=$i \
                  -p runtime:=$RUNTIME \
                  -p learning_method:="$LEARNING_METHOD" \
                  -p num_tsp_agents:=$NUM_AGENTS \
                  -p lr:=$LR \
                  -p gamma_decay:=$GAMMA_DECAY \
                  -p positive_reward:=$POSITIVE_REWARD \
                  -p negative_reward:=$NEGATIVE_REWARD)
              AGENT_PIDS+=("$pid")
            done

            echo "[INFO] $(date -Iseconds) All processes started."
          } >>"$RUN_LOG" 2>&1

          echo "[INFO] Letting run $run execute for $RUN_DURATION_SECONDS seconds..."
          sleep "$RUN_DURATION_SECONDS"

          echo "[INFO] Stopping processes for run $run..."
          # Kill full trees (handles children that moved to new process groups)
          kill_tree "$LOGGER_PID"
          kill_tree "$SIM_PID"
          for pid in "${AGENT_PIDS[@]}"; do kill_tree "$pid"; done

          # Final sweep by **executable name** (not "ros2 run ...")
          sweep_by_name "${SWEEP_PATTERNS[@]}"

          # Optional: warn if anything survived
          if pgrep -f "${SIM_EXECUTABLE}|${AGENT_EXECUTABLE}|${LOGGER_EXECUTABLE}|${TRACKER_PATTERN}" >/dev/null 2>&1; then
            echo "[WARN] Some processes still appear alive:"
            pgrep -f -l "${SIM_EXECUTABLE}|${AGENT_EXECUTABLE}|${LOGGER_EXECUTABLE}|${TRACKER_PATTERN}" || true
          fi

          echo "[INFO] Completed run $run (stopped after fixed duration)."
          echo
          run=$((run + 1))
        done

      done
    done
  done
done

echo "[INFO] All parameter settings completed or skipped if up to date."

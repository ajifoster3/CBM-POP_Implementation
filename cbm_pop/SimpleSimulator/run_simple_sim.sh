#!/bin/bash

# --- allow unset vars while sourcing ROS/your overlay ---
set -e
set -o pipefail
set +u
source /opt/ros/humble/setup.bash
source ../../../../install/local_setup.bash
# --- restore nounset after sourcing ---
set -u

# ===== Better diagnostics =====
export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity} {time} {name}({pid})] {message}'
export RCUTILS_LOGGING_USE_STDOUT=1
export RCUTILS_LOGGING_BUFFERED_STREAM=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONASYNCIODEBUG=1
ulimit -c unlimited || true

# ===== Configuration =====
AGENT_COUNTS=(10)
PROBLEM_SIZES=(15)

NUM_AGENTS="${AGENT_COUNTS[0]}"
PROBLEM_SIZE="${PROBLEM_SIZES[0]}"
INIT_WITH_HEURISTIC=false
NUM_RUNS=200

PACKAGE_NAME="cbm_pop"
LOGGER_EXECUTABLE="simple_fitness_logger"
AGENT_EXECUTABLE="cbm_population_agent_online_simple_simulation"
SIM_EXECUTABLE="simple_simulator"

RUNTIME=-1.0
TIMEOUT_SECONDS=300 # 5 minutes

RESULTS_ROOT="resources/run_logs"
mkdir -p "$RESULTS_ROOT"

# Hyper-parameter grids
# Q-Learning params
LR_VALUES=(0.5)
POSITIVE_REWARD_VALUES=(10.0)
NEGATIVE_REWARD_VALUES=(-1.0)
GAMMA_DECAYS=(0.25)

# ρ is used by BOTH Q-Learning (mimetism) and Ferreira
RHO_VALUES=(0.20)

# Ferreira-only η
ETA_VALUES=(0.75)

# Booleans must be lowercase YAML to be parsed as bool
LOCK_MODES=(false)
PRESERVE_NEXT_TASKS=(true)

# UCB sweep (applies to any learning method)
USE_UCBS=(false)
UCB_C_VALUES=(0.0)

# Methods to sweep
LEARNING_METHODS=('Double-Deep-Q') # 'Ferreira_et_al.'

# NEW: DQN/Double DQN specific parameters
#REPLAY_BUFFER_SIZES=(50)
#TAU_VALUES=(0.01)
#BATCH_SIZES=(10)
REPLAY_BUFFER_SIZES=(25 50 75)
TAU_VALUES=(0.01 0.05 0.1)
BATCH_SIZES=()

# ===== NEW: inject coalition-best at DI-cycle end =====
INJECT_BEST_ON_CYCLE=(false)
# Only used when INJECT_BEST_ON_CYCLE is true
INJECT_BEST_PROBS=(0.90)

# ===== Helpers =====
unique_id () {
  printf "%(%Y%m%d_%H%M%S)T_%04x" -1 "$((RANDOM%65536))"
}

# Robust kill that doesn't assume pkill exists
kill_pid_tree () {
  local pid="$1"
  [ -z "${pid:-}" ] && return 0

  # Recursively terminate children first
  local child
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null || true); do
    kill_pid_tree "$child"
  done

  # Send TERM then KILL
  kill -TERM "$pid" 2>/dev/null || true
  sleep 0.5
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup_run () {
  local logger_pid="$1"; shift
  local sim_pid="$1"; shift
  local -a agent_pids=("$@")

  kill_pid_tree "$logger_pid"
  kill_pid_tree "$sim_pid"
  local p
  for p in "${agent_pids[@]}"; do
    kill_pid_tree "$p"
  done
}

# Use exec so the PID we get is the real ROS2 process, not the stdbuf wrapper
start_logged () {
  local logfile="$1"
  local roslogdir="$2"
  shift 2

  mkdir -p "$(dirname "$logfile")" "$roslogdir"

  RCUTILS_LOGGING_DIRECTORY="$roslogdir" \
  ROS_LOG_DIR="$roslogdir" \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PYTHONASYNCIODEBUG=1 \
  bash -c 'exec stdbuf -oL -eL "$@" >>"$0" 2>&1' "$logfile" "$@" &
  echo $!
}

decode_status () {
  local status="$1"
  if (( status > 128 )); then
    echo "terminated by signal $((status - 128))"
  else
    echo "exited with code $status"
  fi
}

# Zombie-aware liveness check
check_first_dead () {
  FIRST_DEAD_PID=""
  local pid stat
  for pid in "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"; do
    [ -z "$pid" ] && continue

    # If ps can't find the pid, it's dead
    if ! ps -p "$pid" >/dev/null 2>&1; then
      FIRST_DEAD_PID="$pid"
      return 1
    fi

    # If it's a zombie, treat as dead
    stat="$(ps -o stat= -p "$pid" 2>/dev/null | awk '{print $1}')"
    if [ -n "$stat" ] && [[ "$stat" == Z* ]]; then
      FIRST_DEAD_PID="$pid"
      return 1
    fi

    # As a fallback, if SIG 0 fails, it's dead
    if ! kill -0 "$pid" 2>/dev/null; then
      FIRST_DEAD_PID="$pid"
      return 1
    fi
  done
  return 0
}

slug () {
  local s="$1"
  s="${s//[^A-Za-z0-9._-]/_}"
  echo "$s"
}

write_param_tag_file () {
  # PARAM_DIR, METHOD, LOCK, PRESERVE, USE_UCB, UCB_C, INJECT, INJECT_PROB,
  # [gamma lr pos neg rho if Q-Learning] OR [eta rho if Ferreira]
  local _dir="$1"; shift
  local _method="$1"; shift
  local _lock="$1"; shift
  local _preserve="$1"; shift
  local _use_ucb="$1"; shift
  local _ucb_c="$1"; shift
  local _inject="$1"; shift
  local _inject_prob="$1"; shift

  local line="method=${_method} lock=${_lock} preserve_next_task=${_preserve} use_ucb=${_use_ucb} ucb_c=${_ucb_c} inject_best_on_cycle=${_inject} inject_best_prob=${_inject_prob} num_agents=${NUM_AGENTS} problem_size=${PROBLEM_SIZE}"

  if [[ "$_method" == "Q-Learning" ]]; then
    local _gamma="$1"; local _lr="$2"; local _pos="$3"; local _neg="$4"; local _rho="$5"
    line+=" gamma_decay=${_gamma} lr=${_lr} pos=${_pos} neg=${_neg} rho=${_rho}"
  elif [[ "$_method" == "Ferreira_et_al." ]]; then
    local _eta="$1"; local _rho="$2"
    line+=" eta=${_eta} rho=${_rho}"
  fi

  echo "$line" > "$_dir/setting_tag.txt"
}

write_run_settings () {
  # CONFIG_DIR, METHOD, LOCK, PRESERVE, RUN_INDEX, USE_UCB, UCB_C, INJECT, INJECT_PROB,
  # [gamma lr pos neg rho if Q-Learning] OR [eta rho if Ferreira]
  local _cfg="$1"; shift
  local _method="$1"; shift
  local _lock="$1"; shift
  local _preserve="$1"; shift
  local _uid="$1"; shift || true
  local _use_ucb="$1"; shift
  local _ucb_c="$1"; shift
  local _inject="$1"; shift
  local _inject_prob="$1"; shift

  {
    echo "method=${_method}"
    echo "lock_mode=${_lock}"
    echo "preserve_next_task=${_preserve}"
    echo "use_ucb=${_use_ucb}"
    echo "ucb_c=${_ucb_c}"
    echo "inject_best_on_cycle=${_inject}"
    echo "inject_best_prob=${_inject_prob}"
    echo "num_agents=${NUM_AGENTS}"
    echo "problem_size=${PROBLEM_SIZE}"
    echo "runtime=${RUNTIME}"
    echo "timeout_seconds=${TIMEOUT_SECONDS}"
    echo "package=${PACKAGE_NAME}"
    echo "logger_exec=${LOGGER_EXECUTABLE}"
    echo "agent_exec=${AGENT_EXECUTABLE}"
    echo "sim_exec=${SIM_EXECUTABLE}"
    echo "start_iso=$(date -Is)"

    if [[ "$_method" == "Q-Learning" ]]; then
      local _gamma="$1"; local _lr="$2"; local _pos="$3"; local _neg="$4"; local _rho="$5"
      echo "gamma_decay=${_gamma}"
      echo "lr=${_lr}"
      echo "positive_reward=${_pos}"
      echo "negative_reward=${_neg}"
      echo "rho=${_rho}"
    elif [[ "$_method" == "Ferreira_et_al." ]]; then
      local _eta="$1"; local _rho="$2"
      echo "eta=${_eta}"
      echo "rho=${_rho}"
    fi
  } > "$_cfg/run_settings.txt"
}

declare -A PID_ROLE
declare -A PID_LOG
AGENT_PIDS=()

start_all_processes () {
  # $1 LOCK, $2 PRESERVE, $3 INJECT, $4 INJECT_PROB, $5 METHOD,
  # $6..$9 QL params (LR, GAMMA, POS, NEG),
  # $10..$11 Ferreira params (ETA, RHO_F),
  # $12 RHO_SHARED (QL mimetism),
  # $13 USE_UCB (bool), $14 UCB_C (float),
  # $15 REPLAY_BUFFER_SIZE, $16 TAU, $17 BATCH_SIZE
  local CUR_LOCK="$1"; shift
  local CUR_PRESERVE="$1"; shift
  local CUR_INJECT="$1"; shift
  local CUR_INJECT_PROB="$1"; shift
  local CUR_METHOD="$1"; shift
  local CUR_LR="$1"; shift
  local CUR_GAMMA="$1"; shift
  local CUR_POS="$1"; shift
  local CUR_NEG="$1"; shift
  local CUR_ETA="$1"; shift
  local CUR_RHO_F="$1"; shift
  local CUR_RHO_SHARED="$1"; shift
  local CUR_USE_UCB="$1"; shift
  local CUR_UCB_C="$1"; shift
  local CUR_REPLAY_BUFFER_SIZE="$1"; shift
  local CUR_TAU="$1"; shift
  local CUR_BATCH_SIZE="$1"; shift

  # Logger
  LOGGER_LOG="$CONFIG_DIR/logger.log"
  LOGGER_PID=$(start_logged "$LOGGER_LOG" "$ROS2_LOG_DIR" \
    ros2 run "$PACKAGE_NAME" "$LOGGER_EXECUTABLE" \
      --ros-args \
      -p parent_log_dir:="'$CONFIG_DIR'" \
      -p num_tsp_agents:="$NUM_AGENTS" \
      -p problem_size:="$PROBLEM_SIZE")
  
  if [[ -z "$LOGGER_PID" ]]; then
    echo "Error: LOGGER_PID is unbound."
    exit 1
  fi

  PID_ROLE["$LOGGER_PID"]="logger"; PID_LOG["$LOGGER_PID"]="$LOGGER_LOG"

  # Simulator
  SIM_LOG="$CONFIG_DIR/simulator.log"
  SIM_PID=$(start_logged "$SIM_LOG" "$ROS2_LOG_DIR" \
    ros2 run "$PACKAGE_NAME" "$SIM_EXECUTABLE" \
      --num_robots "$NUM_AGENTS" \
      --problem_size "$PROBLEM_SIZE")
  
  if [[ -z "$SIM_PID" ]]; then
    echo "Error: SIM_PID is unbound."
    exit 1
  fi

  PID_ROLE["$SIM_PID"]="simulator"; PID_LOG["$SIM_PID"]="$SIM_LOG"

  # Agents
  AGENT_PIDS=()
  for ((i=0; i<NUM_AGENTS; i++)); do
    local AGENT_LOG="$CONFIG_DIR/agent_${i}.log"
    local -a CMD=(
      ros2 run "$PACKAGE_NAME" "$AGENT_EXECUTABLE"
      --ros-args
      -p agent_id:="$i"
      -p runtime:="$RUNTIME"
      -p learning_method:="'${CUR_METHOD}'"
      -p num_tsp_agents:="$NUM_AGENTS"
      -p problem_size:="$PROBLEM_SIZE"
      -p lock_mode:=${CUR_LOCK}
      -p preserve_next_task:=${CUR_PRESERVE}
      -p use_ucb:=${CUR_USE_UCB}
      -p ucb_c:="$CUR_UCB_C"
      -p inject_best_on_cycle:=${CUR_INJECT}
      -p inject_best_prob:="$CUR_INJECT_PROB"
      -p initialise_with_heuristic:=$INIT_WITH_HEURISTIC
    )

    if [[ "$CUR_METHOD" == "Q-Learning" || "$CUR_METHOD" == "Double-Deep-Q" ]]; then
      CMD+=(
        -p lr:="$CUR_LR"
        -p gamma_decay:="$CUR_GAMMA"
        -p positive_reward:="$CUR_POS"
        -p negative_reward:="$CUR_NEG"
        -p rho:="$CUR_RHO_SHARED"
        -p replay_buffer_size:="$CUR_REPLAY_BUFFER_SIZE"
        -p tau:="$CUR_TAU"
        -p batch_size:="$CUR_BATCH_SIZE"
      )
    elif [[ "$CUR_METHOD" == "Ferreira_et_al." ]]; then
      CMD+=(
        -p eta:="$CUR_ETA"
        -p rho:="$CUR_RHO_F"
      )
    fi

    local pid
    pid=$(start_logged "$AGENT_LOG" "$ROS2_LOG_DIR" "${CMD[@]}")
    AGENT_PIDS+=("$pid")
    PID_ROLE["$pid"]="agent[$i]"; PID_LOG["$pid"]="$AGENT_LOG"
  done
}

# ===== Traps for cluster signals =====
on_sigint () {
  echo "[SIGINT] Cleaning up processes..."
  cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
  exit 130
}

on_sigterm () {
  echo "[SIGTERM] Cleaning up processes..."
  cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
  exit 143
}

on_exit () {
  cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
}

trap on_sigint INT
trap on_sigterm TERM
trap on_exit EXIT

# ===== Main sweep =====
for PROBLEM_SIZE in "${PROBLEM_SIZES[@]}"; do
  for NUM_AGENTS in "${AGENT_COUNTS[@]}"; do
    COMBO_TAG="size_${PROBLEM_SIZE}_agents_${NUM_AGENTS}"
    COMBO_ROOT="$RESULTS_ROOT/$COMBO_TAG"
    mkdir -p "$COMBO_ROOT"
    echo "[INFO] Sweeping problem_size=$PROBLEM_SIZE num_agents=$NUM_AGENTS"
    
    for METHOD in "${LEARNING_METHODS[@]}"; do
      METHOD_TAG="$(slug "$METHOD")"
      
      for INJECT in "${INJECT_BEST_ON_CYCLE[@]}"; do
        # Use probabilities only when INJECT is true
        if [[ "$INJECT" == true ]]; then
          _PINJ_SET=("${INJECT_BEST_PROBS[@]}")
        else
          _PINJ_SET=("0.0")
        fi

        for PINJ in "${_PINJ_SET[@]}"; do
          for LOCK in "${LOCK_MODES[@]}"; do
            for PRESERVE in "${PRESERVE_NEXT_TASKS[@]}"; do
              for USE_UCB in "${USE_UCBS[@]}"; do
                for UCB_C in "${UCB_C_VALUES[@]}"; do
                  # Declare EARLY_FAILURE globally
                  EARLY_FAILURE=false

                  if [[ "$METHOD" == "Q-Learning" || "$METHOD" == "Double-Deep-Q" ]]; then
                    for GAMMA_DECAY in "${GAMMA_DECAYS[@]}"; do
                      for LR in "${LR_VALUES[@]}"; do
                        for POSITIVE_REWARD in "${POSITIVE_REWARD_VALUES[@]}"; do
                          for NEGATIVE_REWARD in "${NEGATIVE_REWARD_VALUES[@]}"; do
                            for RHO in "${RHO_VALUES[@]}"; do
                              for REPLAY_BUFFER_SIZE in "${REPLAY_BUFFER_SIZES[@]}"; do
                                for TAU in "${TAU_VALUES[@]}"; do
                                  for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                                    PARAM_DIR="$COMBO_ROOT/method_${METHOD_TAG}_gamma_${GAMMA_DECAY}_lr_${LR}_pos_${POSITIVE_REWARD}_neg_${NEGATIVE_REWARD}_rho_${RHO}_lock_${LOCK}_preserve_${PRESERVE}_inject_${INJECT}_pinj_${PINJ}_ucb_${USE_UCB}_ucbc_${UCB_C}_replay_buffer_size_${REPLAY_BUFFER_SIZE}_tau_${TAU}_batch_size_${BATCH_SIZE}"
                                    mkdir -p "$PARAM_DIR"
                                    
                                    write_param_tag_file "$PARAM_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$USE_UCB" "$UCB_C" "$INJECT" "$PINJ" "$GAMMA_DECAY" "$LR" "$POSITIVE_REWARD" "$NEGATIVE_REWARD" "$RHO" "$REPLAY_BUFFER_SIZE" "$TAU" "$BATCH_SIZE"
                                    
                                    EXISTING_RUNS=$(find "$PARAM_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l | tr -d ' ')
                                    START_RUN=$((EXISTING_RUNS + 1))

                                    if [ "$EXISTING_RUNS" -ge "$NUM_RUNS" ]; then
                                      echo "[INFO] Already completed $EXISTING_RUNS runs for $PARAM_DIR. Skipping."
                                      continue
                                    fi

                                    run=$START_RUN
                                    while [ $run -le $NUM_RUNS ]; do
                                      echo "============================="
                                      echo "[INFO] Run $run/$NUM_RUNS :: method=$METHOD lock=$LOCK preserve=$PRESERVE inject=$INJECT pinj=$PINJ gamma=$GAMMA_DECAY lr=$LR pos=$POSITIVE_REWARD neg=$NEGATIVE_REWARD rho=$RHO replay_buffer_size=$REPLAY_BUFFER_SIZE tau=$TAU batch_size=$BATCH_SIZE ucb=$USE_UCB ucb_c=$UCB_C num_agents=$NUM_AGENTS problem_size=$PROBLEM_SIZE"
                                      echo "============================="

                                      CONFIG_DIR="$PARAM_DIR/run_${run}"
                                      ROS2_LOG_DIR="$CONFIG_DIR/ros_logs"
                                      mkdir -p "$ROS2_LOG_DIR"

                                      write_run_settings "$CONFIG_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$run" "$USE_UCB" "$UCB_C" "$INJECT" "$PINJ" "$GAMMA_DECAY" "$LR" "$POSITIVE_REWARD" "$NEGATIVE_REWARD" "$RHO" "$REPLAY_BUFFER_SIZE" "$TAU" "$BATCH_SIZE"

                                      start_all_processes "$LOCK" "$PRESERVE" "$INJECT" "$PINJ" "$METHOD" "$LR" "$GAMMA_DECAY" "$POSITIVE_REWARD" "$NEGATIVE_REWARD" 0 0 "$RHO" "$USE_UCB" "$UCB_C" "$REPLAY_BUFFER_SIZE" "$TAU" "$BATCH_SIZE"

                                      START_TIME=$(date +%s)
                                      RUN_TIMEOUT=0
                                      FIRST_DEAD_PID=""

                                      # Polling loop with zombie-aware detection and timeout
                                      while true; do
                                        sleep 2
                                        CURRENT_TIME=$(date +%s)
                                        ELAPSED=$((CURRENT_TIME - START_TIME))

                                        if ! check_first_dead; then
                                          role="${PID_ROLE[$FIRST_DEAD_PID]}"
                                          logf="${PID_LOG[$FIRST_DEAD_PID]}"
                                          if wait "$FIRST_DEAD_PID"; then
                                            STATUS=0
                                          else
                                            STATUS=$?
                                          fi

                                          echo "[timeouttime]: $ELAPSED"
                                          if [ "$ELAPSED" -le 30 ]; then
                                            EARLY_FAILURE=true
                                            echo "[EARLY FAILURE DETECTED] $role (pid=$FIRST_DEAD_PID) exited with code $STATUS after ${ELAPSED}s"
                                          fi

                                          echo "[DETECT] $role (pid=$FIRST_DEAD_PID) $(decode_status "$STATUS") after ${ELAPSED}s"
                                          echo "--------- last 120 lines of $logf ---------"
                                          tail -n 120 "$logf" || true
                                          echo "-------------------------------------------"
                                          break
                                        fi

                                        if [ "$ELAPSED" -ge "$TIMEOUT_SECONDS" ]; then
                                          echo "[TIMEOUT] Run $run exceeded ${TIMEOUT_SECONDS}s."
                                          RUN_TIMEOUT=1
                                          break
                                        fi
                                      done

                                      if [ "$EARLY_FAILURE" == true ]; then
                                        echo "[CLEANUP] Deleting failed run directory: $CONFIG_DIR"
                                        rm -rf "$CONFIG_DIR"
                                        echo "[RETRY] Repeating run $run"
                                        EARLY_FAILURE=false
                                        cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
                                        continue
                                      fi

                                      cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
                                      echo "end_iso=$(date -Is)" >> "$CONFIG_DIR/run_settings.txt"
                                      echo "[INFO] Completed run $run for $CONFIG_DIR"
                                      run=$((run + 1))
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  else
                    # Non-Q-Learning (Ferreira)
                    if [[ "$METHOD" == "Ferreira_et_al." ]]; then
                      for ETA in "${ETA_VALUES[@]}"; do
                        for RHO in "${RHO_VALUES[@]}"; do
                          PARAM_DIR="$COMBO_ROOT/method_${METHOD_TAG}_eta_${ETA}_rho_${RHO}_lock_${LOCK}_preserve_${PRESERVE}_inject_${INJECT}_pinj_${PINJ}_ucb_${USE_UCB}_ucbc_${UCB_C}"
                          mkdir -p "$PARAM_DIR"
                          write_param_tag_file "$PARAM_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$USE_UCB" "$UCB_C" "$INJECT" "$PINJ" "$ETA" "$RHO"
                          EXISTING_RUNS=$(find "$PARAM_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l | tr -d ' ')
                          START_RUN=$((EXISTING_RUNS + 1))

                          if [ "$EXISTING_RUNS" -ge "$NUM_RUNS" ]; then
                            echo "[INFO] Already completed $EXISTING_RUNS runs for $PARAM_DIR. Skipping."
                            continue
                          fi

                          run=$START_RUN
                          while [ $run -le $NUM_RUNS ]; do
                            echo "============================="
                            echo "[INFO] Run $run/$NUM_RUNS :: method=$METHOD lock=$LOCK preserve=$PRESERVE inject=$INJECT pinj=$PINJ eta=${ETA} rho=${RHO} ucb=$USE_UCB ucb_c=${UCB_C} num_agents=${NUM_AGENTS} problem_size=${PROBLEM_SIZE}"
                            echo "============================="

                            CONFIG_DIR="$PARAM_DIR/run_${run}"
                            ROS2_LOG_DIR="$CONFIG_DIR/ros_logs"
                            mkdir -p "$ROS2_LOG_DIR"

                            write_run_settings "$CONFIG_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$run" "$USE_UCB" "$UCB_C" "$INJECT" "$PINJ" "$ETA" "$RHO"

                            # placeholders for QL params: LR,GAMMA,POS,NEG = 0; pass ETA,RHO and no shared rho
                            start_all_processes "$LOCK" "$PRESERVE" "$INJECT" "$PINJ" "$METHOD" 0 0 0 0 "$ETA" "$RHO" 0 "$USE_UCB" "$UCB_C"

                            START_TIME=$(date +%s)
                            RUN_TIMEOUT=0
                            FIRST_DEAD_PID=""

                            while true; do
                              sleep 2
                              CURRENT_TIME=$(date +%s)
                              ELAPSED=$((CURRENT_TIME - START_TIME))

                              if ! check_first_dead; then
                                role="${PID_ROLE[$FIRST_DEAD_PID]}"
                                logf="${PID_LOG[$FIRST_DEAD_PID]}"

                                if wait "$FIRST_DEAD_PID"; then
                                  STATUS=0
                                else
                                  STATUS=$?
                                fi

                                if [ "$ELAPSED" -le 30 ]; then
                                  EARLY_FAILURE=true
                                fi

                                echo "[DETECT] $role (pid=$FIRST_DEAD_PID) $(decode_status "$STATUS") after ${ELAPSED}s"
                                echo "--------- last 120 lines of $logf ---------"
                                tail -n 120 "$logf" || true
                                echo "-------------------------------------------"
                                break
                              fi

                              if [ "$ELAPSED" -ge "$TIMEOUT_SECONDS" ]; then
                                echo "[TIMEOUT] Run $run exceeded ${TIMEOUT_SECONDS}s."
                                RUN_TIMEOUT=1
                                break
                              fi
                            done

                            if [ "$EARLY_FAILURE" == true ]; then
                              echo "[CLEANUP] Deleting failed run directory: $CONFIG_DIR"
                              rm -rf "$CONFIG_DIR"
                              EARLY_FAILURE=false
                              cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
                              echo "[RETRY] Repeating run $run"
                              continue
                            fi

                            cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
                            echo "end_iso=$(date -Is)" >> "$CONFIG_DIR/run_settings.txt"
                            echo "[INFO] Completed run $run for $CONFIG_DIR"
                            run=$((run + 1))
                          done
                        done
                      done
                    else
                      # Any other non-QL method without eta/rho
                      PARAM_DIR="$COMBO_ROOT/method_${METHOD_TAG}_lock_${LOCK}_preserve_${PRESERVE}_inject_${INJECT}_pinj_${PINJ}_ucb_${USE_UCB}_ucbc_${UCB_C}"
                      mkdir -p "$PARAM_DIR"
                      write_param_tag_file "$PARAM_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$USE_UCB" "$UCB_C" "$INJECT" "$PINJ"
                      EXISTING_RUNS=$(find "$PARAM_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l | tr -d ' ')
                      START_RUN=$((EXISTING_RUNS + 1))

                      if [ "$EXISTING_RUNS" -ge "$NUM_RUNS" ]; then
                        echo "[INFO] Already completed $EXISTING_RUNS runs for $PARAM_DIR. Skipping."
                        continue
                      fi

                      run=$START_RUN
                      while [ $run -le $NUM_RUNS ]; do
                        echo "============================="
                        echo "[INFO] Run $run/$NUM_RUNS :: method=$METHOD lock=$LOCK preserve=$PRESERVE inject=$INJECT pinj=$PINJ ucb=$USE_UCB ucb_c=${UCB_C} num_agents=${NUM_AGENTS} problem_size=${PROBLEM_SIZE}"
                        echo "============================="

                        CONFIG_DIR="$PARAM_DIR/run_${run}"
                        ROS2_LOG_DIR="$CONFIG_DIR/ros_logs"
                        mkdir -p "$ROS2_LOG_DIR"

                        write_run_settings "$CONFIG_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$run" "$USE_UCB" "$UCB_C" "$INJECT" "$PINJ"

                        # placeholders for both families
                        start_all_processes "$LOCK" "$PRESERVE" "$INJECT" "$PINJ" "$METHOD" 0 0 0 0 0 0 0 "$USE_UCB" "$UCB_C"

                        START_TIME=$(date +%s)
                        RUN_TIMEOUT=0
                        FIRST_DEAD_PID=""

                        while true; do
                          sleep 2
                          CURRENT_TIME=$(date +%s)
                          ELAPSED=$((CURRENT_TIME - START_TIME))

                          if ! check_first_dead; then
                            role="${PID_ROLE[$FIRST_DEAD_PID]}"
                            logf="${PID_LOG[$FIRST_DEAD_PID]}"

                            if wait "$FIRST_DEAD_PID"; then
                              STATUS=0
                            else
                              STATUS=$?
                            fi

                            echo "[timeouttime]: $ELAPSED"
                            if [ "$ELAPSED" -le 30 ]; then
                              EARLY_FAILURE=true
                              echo "[EARLY FAILURE DETECTED] $role (pid=$FIRST_DEAD_PID) exited with code $STATUS after ${ELAPSED}s"
                            fi

                            echo "[DETECT] $role (pid=$FIRST_DEAD_PID) $(decode_status "$STATUS") after ${ELAPSED}s"
                            echo "--------- last 120 lines of $logf ---------"
                            tail -n 120 "$logf" || true
                            echo "-------------------------------------------"
                            break
                          fi

                          if [ "$ELAPSED" -ge "$TIMEOUT_SECONDS" ]; then
                            echo "[TIMEOUT] Run $run exceeded ${TIMEOUT_SECONDS}s."
                            RUN_TIMEOUT=1
                            break
                          fi
                        done

                        if [ "$EARLY_FAILURE" == true ]; then
                          echo "[CLEANUP] Deleting failed run directory: $CONFIG_DIR"
                          rm -rf "$CONFIG_DIR"
                          echo "[RETRY] Repeating run $run"
                          EARLY_FAILURE=false
                          cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
                          continue
                        fi

                        cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
                        echo "end_iso=$(date -Is)" >> "$CONFIG_DIR/run_settings.txt"
                        echo "[INFO] Completed run $run for $CONFIG_DIR"
                        run=$((run + 1))
                      done
                    fi
                  fi
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "[INFO] All parameter settings completed or skipped if up to date."



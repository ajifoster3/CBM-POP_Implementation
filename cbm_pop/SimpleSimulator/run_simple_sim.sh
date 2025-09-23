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
AGENT_COUNTS=(4)
PROBLEM_SIZES=(8)
NUM_AGENTS="${AGENT_COUNTS[0]}"
PROBLEM_SIZE="${PROBLEM_SIZES[0]}"
NUM_RUNS=100
PACKAGE_NAME="cbm_pop"
LOGGER_EXECUTABLE="simple_fitness_logger"
AGENT_EXECUTABLE="cbm_population_agent_online_simple_simulation"
SIM_EXECUTABLE="simple_simulator"
RUNTIME=-1.0
TIMEOUT_SECONDS=300  # 5 minutes

RESULTS_ROOT="resources/run_logs"
mkdir -p "$RESULTS_ROOT"

# Hyper-parameter grids
# Q-Learning params
LR_VALUES=(0.1 0.25 0.5 0.75 0.9)
POSITIVE_REWARD_VALUES=(3.0)
NEGATIVE_REWARD_VALUES=(-1.0)
GAMMA_DECAYS=(0.1 0.25 0.5 0.75)
# ρ is used by BOTH Q-Learning (mimetism) and Ferreira
RHO_VALUES=(0.20)

# Ferreira-only η
ETA_VALUES=(0.20)

# Booleans must be lowercase YAML to be parsed as bool
LOCK_MODES=(false true)
PRESERVE_NEXT_TASKS=(true)

# UCB sweep (applies to any learning method)
USE_UCBS=(false)
UCB_C_VALUES=(0.25)

# Methods to sweep
LEARNING_METHODS=('Q-Learning' 'Ferreira_et_al.')
# LEARNING_METHODS=('Q-Learning' 'Ferreira_et_al.')

# ===== Helpers =====
unique_id () {
  printf "%(%Y%m%d_%H%M%S)T_%04x" -1 "$((RANDOM%65536))"
}

kill_pid_tree () {
  local pid="$1"
  [ -z "${pid:-}" ] && return 0
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill -TERM "$pid" 2>/dev/null || true
  sleep 1
  pkill -KILL -P "$pid" 2>/dev/null || true
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup_run () {
  local logger_pid="$1"; shift
  local sim_pid="$1"; shift
  local -a agent_pids=("$@")
  kill_pid_tree "$logger_pid"
  kill_pid_tree "$sim_pid"
  local p
  for p in "${agent_pids[@]}"; do kill_pid_tree "$p"; done
}

start_logged () {
  local logfile="$1"; local roslogdir="$2"; shift 2
  mkdir -p "$(dirname "$logfile")" "$roslogdir"
  RCUTILS_LOGGING_DIRECTORY="$roslogdir" \
  ROS_LOG_DIR="$roslogdir" \
  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  PYTHONASYNCIODEBUG=1 \
  stdbuf -oL -eL "$@" >>"$logfile" 2>&1 & echo $!
}

decode_status () {
  local status="$1"
  if (( status > 128 )); then echo "terminated by signal $((status - 128))"
  else echo "exited with code $status"; fi
}

check_first_dead () {
  FIRST_DEAD_PID=""
  local pid
  for pid in "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"; do
    [ -z "$pid" ] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      FIRST_DEAD_PID="$pid"
      return 1
    fi
  done
  return 0
}

slug() {
  local s="$1"; s="${s//[^A-Za-z0-9._-]/_}"; echo "$s";
}

write_param_tag_file () {
  # PARAM_DIR, METHOD, LOCK, PRESERVE, USE_UCB, UCB_C,
  # [gamma lr pos neg rho if Q-Learning] OR [eta rho if Ferreira]
  local _dir="$1"; shift
  local _method="$1"; shift
  local _lock="$1"; shift
  local _preserve="$1"; shift
  local _use_ucb="$1"; shift
  local _ucb_c="$1"; shift

  local line="method=${_method} lock=${_lock} preserve_next_task=${_preserve} use_ucb=${_use_ucb} ucb_c=${_ucb_c} num_agents=${NUM_AGENTS} problem_size=${PROBLEM_SIZE}"

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
  # CONFIG_DIR, METHOD, LOCK, PRESERVE, RUN_INDEX, USE_UCB, UCB_C,
  # [gamma lr pos neg rho if Q-Learning] OR [eta rho if Ferreira]
  local _cfg="$1"; shift
  local _method="$1"; shift
  local _lock="$1"; shift
  local _preserve="$1"; shift
  local _uid="$1"; shift || true
  local _use_ucb="$1"; shift
  local _ucb_c="$1"; shift

  {
    echo "method=${_method}"
    echo "lock_mode=${_lock}"
    echo "preserve_next_task=${_preserve}"
    echo "use_ucb=${_use_ucb}"
    echo "ucb_c=${_ucb_c}"
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
  # $1 LOCK, $2 PRESERVE, $3 METHOD,
  # $4..$7 QL params (LR, GAMMA, POS, NEG),
  # $8..$9 Ferreira params (ETA, RHO),
  # $10 RHO (shared for QL),
  # $11 USE_UCB (bool), $12 UCB_C (float)
  local CUR_LOCK="$1"; shift
  local CUR_PRESERVE="$1"; shift
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

  # Logger
  LOGGER_LOG="$CONFIG_DIR/logger.log"
  LOGGER_PID=$(start_logged "$LOGGER_LOG" "$ROS2_LOG_DIR" \
    ros2 run "$PACKAGE_NAME" "$LOGGER_EXECUTABLE" \
      --ros-args \
        -p parent_log_dir:="'$CONFIG_DIR'" \
        -p num_tsp_agents:="$NUM_AGENTS" \
        -p problem_size:="$PROBLEM_SIZE")
  PID_ROLE["$LOGGER_PID"]="logger"; PID_LOG["$LOGGER_PID"]="$LOGGER_LOG"

  # Simulator
  SIM_LOG="$CONFIG_DIR/simulator.log"
  SIM_PID=$(start_logged "$SIM_LOG" "$ROS2_LOG_DIR" \
    ros2 run "$PACKAGE_NAME" "$SIM_EXECUTABLE" \
      --num_robots "$NUM_AGENTS" \
      --problem_size "$PROBLEM_SIZE")
  PID_ROLE["$SIM_PID"]="simulator"; PID_LOG["$SIM_PID"]="$SIM_LOG"

  # Agents
  AGENT_PIDS=()
  for ((i=0; i<NUM_AGENTS; i++)); do
    local AGENT_LOG="$CONFIG_DIR/agent_${i}.log"
    local -a CMD=(ros2 run "$PACKAGE_NAME" "$AGENT_EXECUTABLE"
                  --ros-args
                  -p agent_id:="$i"
                  -p runtime:="$RUNTIME"
                  -p learning_method:="'${CUR_METHOD}'"
                  -p num_tsp_agents:="$NUM_AGENTS"
                  -p problem_size:="$PROBLEM_SIZE"
                  -p lock_mode:=${CUR_LOCK}
                  -p preserve_next_task:=${CUR_PRESERVE}
                  -p use_ucb:=${CUR_USE_UCB}
                  -p ucb_c:="$CUR_UCB_C")
    if [[ "$CUR_METHOD" == "Q-Learning" ]]; then
      CMD+=( -p lr:="$CUR_LR"
             -p gamma_decay:="$CUR_GAMMA"
             -p positive_reward:="$CUR_POS"
             -p negative_reward:="$CUR_NEG"
             -p rho:="$CUR_RHO_SHARED" )
    elif [[ "$CUR_METHOD" == "Ferreira_et_al." ]]; then
      CMD+=( -p eta:="$CUR_ETA"
             -p rho:="$CUR_RHO_F" )
    fi

    local pid
    pid=$(start_logged "$AGENT_LOG" "$ROS2_LOG_DIR" "${CMD[@]}")
    AGENT_PIDS+=("$pid")
    PID_ROLE["$pid"]="agent[$i]"; PID_LOG["$pid"]="$AGENT_LOG"
  done
}

on_sigint () {
  echo "[INTERRUPT] Cleaning up processes..."
  cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"
  exit 130
}
trap on_sigint INT

# ===== Main sweep =====
for PROBLEM_SIZE in "${PROBLEM_SIZES[@]}"; do
  for NUM_AGENTS in "${AGENT_COUNTS[@]}"; do
    COMBO_TAG="size_${PROBLEM_SIZE}_agents_${NUM_AGENTS}"
    COMBO_ROOT="$RESULTS_ROOT/$COMBO_TAG"
    mkdir -p "$COMBO_ROOT"

    echo "[INFO] Sweeping problem_size=$PROBLEM_SIZE num_agents=$NUM_AGENTS"

    for METHOD in "${LEARNING_METHODS[@]}"; do
      METHOD_TAG="$(slug "$METHOD")"
      for LOCK in "${LOCK_MODES[@]}"; do
        for PRESERVE in "${PRESERVE_NEXT_TASKS[@]}"; do
          for USE_UCB in "${USE_UCBS[@]}"; do
            for UCB_C in "${UCB_C_VALUES[@]}"; do

              if [[ "$METHOD" == "Q-Learning" ]]; then
                for GAMMA_DECAY in "${GAMMA_DECAYS[@]}"; do
                  for LR in "${LR_VALUES[@]}"; do
                    for POSITIVE_REWARD in "${POSITIVE_REWARD_VALUES[@]}"; do
                      for NEGATIVE_REWARD in "${NEGATIVE_REWARD_VALUES[@]}"; do
                        for RHO in "${RHO_VALUES[@]}"; do
                          PARAM_DIR="$COMBO_ROOT/method_${METHOD_TAG}_gamma_${GAMMA_DECAY}_lr_${LR}_pos_${POSITIVE_REWARD}_neg_${NEGATIVE_REWARD}_rho_${RHO}_lock_${LOCK}_preserve_${PRESERVE}_ucb_${USE_UCB}_ucbc_${UCB_C}"
                          mkdir -p "$PARAM_DIR"
                          write_param_tag_file "$PARAM_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$USE_UCB" "$UCB_C" "$GAMMA_DECAY" "$LR" "$POSITIVE_REWARD" "$NEGATIVE_REWARD" "$RHO"

                          EXISTING_RUNS=$(find "$PARAM_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l | tr -d ' ')
                          START_RUN=$((EXISTING_RUNS + 1))
                          if [ "$EXISTING_RUNS" -ge "$NUM_RUNS" ]; then
                            echo "[INFO] Already completed $EXISTING_RUNS runs for $PARAM_DIR. Skipping."
                            continue
                          fi

                          run=$START_RUN
                          while [ $run -le $NUM_RUNS ]; do
                            echo "============================="
                            echo "[INFO] Run $run/$NUM_RUNS :: method=$METHOD lock=$LOCK preserve=$PRESERVE gamma=$GAMMA_DECAY lr=$LR pos=$POSITIVE_REWARD neg=$NEGATIVE_REWARD rho=$RHO ucb=$USE_UCB ucb_c=$UCB_C num_agents=$NUM_AGENTS problem_size=$PROBLEM_SIZE"
                            echo "============================="

                            CONFIG_DIR="$PARAM_DIR/run_${run}"
                            ROS2_LOG_DIR="$CONFIG_DIR/ros_logs"
                            mkdir -p "$ROS2_LOG_DIR"

                            write_run_settings "$CONFIG_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$run" "$USE_UCB" "$UCB_C" "$GAMMA_DECAY" "$LR" "$POSITIVE_REWARD" "$NEGATIVE_REWARD" "$RHO"

                            # placeholders: ETA=0, RHO_F=0 (Ferreira), RHO_SHARED=$RHO (QL)
                            start_all_processes "$LOCK" "$PRESERVE" "$METHOD" "$LR" "$GAMMA_DECAY" "$POSITIVE_REWARD" "$NEGATIVE_REWARD" 0 0 "$RHO" "$USE_UCB" "$UCB_C"

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
                                if wait "$FIRST_DEAD_PID"; then STATUS=0; else STATUS=$?; fi
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

                            cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"

                            if [ "$RUN_TIMEOUT" -eq 1 ]; then
                              echo "[CLEANUP] Deleting failed run directory: $CONFIG_DIR"
                              rm -rf "$CONFIG_DIR"
                              echo "[RETRY] Repeating run $run"
                              continue
                            fi

                            echo "end_iso=$(date -Is)" >> "$CONFIG_DIR/run_settings.txt"
                            echo "[INFO] Completed run $run for $CONFIG_DIR"
                            run=$((run + 1))
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
                      PARAM_DIR="$COMBO_ROOT/method_${METHOD_TAG}_eta_${ETA}_rho_${RHO}_lock_${LOCK}_preserve_${PRESERVE}_ucb_${USE_UCB}_ucbc_${UCB_C}"
                      mkdir -p "$PARAM_DIR"
                      write_param_tag_file "$PARAM_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$USE_UCB" "$UCB_C" "$ETA" "$RHO"

                      EXISTING_RUNS=$(find "$PARAM_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l | tr -d ' ')
                      START_RUN=$((EXISTING_RUNS + 1))
                      if [ "$EXISTING_RUNS" -ge "$NUM_RUNS" ]; then
                        echo "[INFO] Already completed $EXISTING_RUNS runs for $PARAM_DIR. Skipping."
                        continue
                      fi

                      run=$START_RUN
                      while [ $run -le $NUM_RUNS ]; do
                        echo "============================="
                        echo "[INFO] Run $run/$NUM_RUNS :: method=$METHOD lock=$LOCK preserve=$PRESERVE eta=${ETA} rho=${RHO} ucb=$USE_UCB ucb_c=$UCB_C num_agents=$NUM_AGENTS problem_size=$PROBLEM_SIZE"
                        echo "============================="

                        CONFIG_DIR="$PARAM_DIR/run_${run}"
                        ROS2_LOG_DIR="$CONFIG_DIR/ros_logs"
                        mkdir -p "$ROS2_LOG_DIR"

                        write_run_settings "$CONFIG_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$run" "$USE_UCB" "$UCB_C" "$ETA" "$RHO"

                        # placeholders for QL params: LR,GAMMA,POS,NEG = 0; pass ETA,RHO and no shared rho
                        start_all_processes "$LOCK" "$PRESERVE" "$METHOD" 0 0 0 0 "$ETA" "$RHO" 0 "$USE_UCB" "$UCB_C"

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
                            if wait "$FIRST_DEAD_PID"; then STATUS=0; else STATUS=$?; fi
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

                        cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"

                        if [ "$RUN_TIMEOUT" -eq 1 ]; then
                          echo "[CLEANUP] Deleting failed run directory: $CONFIG_DIR"
                          rm -rf "$CONFIG_DIR"
                          echo "[RETRY] Repeating run $run"
                          continue
                        fi

                        echo "end_iso=$(date -Is)" >> "$CONFIG_DIR/run_settings.txt"
                        echo "[INFO] Completed run $run for $CONFIG_DIR"
                        run=$((run + 1))
                      done
                    done
                  done

                else
                  # Any other non-QL method without eta/rho
                  PARAM_DIR="$COMBO_ROOT/method_${METHOD_TAG}_lock_${LOCK}_preserve_${PRESERVE}_ucb_${USE_UCB}_ucbc_${UCB_C}"
                  mkdir -p "$PARAM_DIR"
                  write_param_tag_file "$PARAM_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$USE_UCB" "$UCB_C"

                  EXISTING_RUNS=$(find "$PARAM_DIR" -maxdepth 1 -type d -name 'run_*' | wc -l | tr -d ' ')
                  START_RUN=$((EXISTING_RUNS + 1))
                  if [ "$EXISTING_RUNS" -ge "$NUM_RUNS" ]; then
                    echo "[INFO] Already completed $EXISTING_RUNS runs for $PARAM_DIR. Skipping."
                    continue
                  fi

                  run=$START_RUN
                  while [ $run -le $NUM_RUNS ]; do
                    echo "============================="
                    echo "[INFO] Run $run/$NUM_RUNS :: method=$METHOD lock=$LOCK preserve=$PRESERVE ucb=$USE_UCB ucb_c=$UCB_C num_agents=$NUM_AGENTS problem_size=$PROBLEM_SIZE"
                    echo "============================="

                    CONFIG_DIR="$PARAM_DIR/run_${run}"
                    ROS2_LOG_DIR="$CONFIG_DIR/ros_logs"
                    mkdir -p "$ROS2_LOG_DIR"

                    write_run_settings "$CONFIG_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$run" "$USE_UCB" "$UCB_C"

                    # placeholders for both families
                    start_all_processes "$LOCK" "$PRESERVE" "$METHOD" 0 0 0 0 0 0 0 "$USE_UCB" "$UCB_C"

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
                        if wait "$FIRST_DEAD_PID"; then STATUS=0; else STATUS=$?; fi
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

                    cleanup_run "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"

                    if [ "$RUN_TIMEOUT" -eq 1 ]; then
                      echo "[CLEANUP] Deleting failed run directory: $CONFIG_DIR"
                      rm -rf "$CONFIG_DIR"
                      echo "[RETRY] Repeating run $run"
                      continue
                    fi

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

echo "[INFO] All parameter settings completed or skipped if up to date."

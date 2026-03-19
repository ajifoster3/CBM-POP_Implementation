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
export RCUTILS_LOGGING_BUFFERED_STREAM=0
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONASYNCIODEBUG=1
ulimit -c unlimited || true

gen_seed() {
  if command -v od >/dev/null 2>&1; then
    printf '%d\n' $(( 10000 + ( $(od -An -N2 -tu2 /dev/urandom | tr -d ' ') % 90000 ) ))
  elif command -v shuf >/dev/null 2>&1; then
    shuf -i 10000-99999 -n1
  else
    printf '%d\n' $(( 10000 + ( ( (RANDOM << 15) | RANDOM ) % 90000 ) ))
  fi
}

# ===== Configuration =====
NUM_RUNS=200

PACKAGE_NAME="cbm_pop"
LOGGER_EXECUTABLE="simple_fitness_logger"
AGENT_EXECUTABLE="cbm_population_agent_online_simple_simulation"
SIM_EXECUTABLE="simple_simulator"

AGENT_INIT_MSG="["
AGENT_STARTUP_TIMEOUT=30

RUNTIME=-1.0
TIMEOUT_SECONDS=300

RESULTS_ROOT="resources/run_logs"
mkdir -p "$RESULTS_ROOT"

# ===== Parse CLI (short + long) =====
PARSED=$(getopt -o p:a:m:l:t:r:L:P:N:G:R:E:U:C:B:Q:T:S:H:V: \
  -l p:,problem-size:,a:,agents:,m:,method:,l:,lock:,t:,preserve:,r:,inject:,\
L:,lr:,P:,positive-reward:,N:,negative-reward:,G:,gamma-decay:,R:,rho:,E:,eta:,\
U:,use-ucb:,C:,ucb-c:,B:,replay-buffer-size:,Q:,tau:,T:,batch-size:,S:,inject-best-prob:,\
H:,init-with-heuristic:,V:,problem-class:,problem-seed:,kill-thresholds:,enable-kill:,num-to-kill:,\
enable-revive:,revive-threshold:,\
ucb-window:,is-free-weight-matrix:,is-inject-best-on-cycle:,\
is-append-first-task:,is-knn-enabled:,is-mimetism-enabled: -- "$@") || {
  echo "Invalid options"; exit 1;
}
eval set -- "$PARSED"

while true; do
  case "$1" in
    -p|--p|--problem-size)        PROBLEM_SIZE="$2"; shift 2 ;;
    -a|--a|--agents)              NUM_AGENTS="$2"; shift 2 ;;
    -m|--m|--method)              METHOD="$2"; shift 2 ;;
    -l|--l|--lock)                LOCK="$2"; shift 2 ;;
    -t|--t|--preserve)            PRESERVE="$2"; shift 2 ;;
    -r|--r|--inject)              INJECT="$2"; shift 2 ;;
    -L|--L|--lr)                  LR="$2"; shift 2 ;;
    -P|--P|--positive-reward)     POSITIVE_REWARD="$2"; shift 2 ;;
    -N|--N|--negative-reward)     NEGATIVE_REWARD="$2"; shift 2 ;;
    -G|--G|--gamma-decay)         GAMMA_DECAY="$2"; shift 2 ;;
    -R|--R|--rho)                 RHO="$2"; shift 2 ;;
    -E|--E|--eta)                 ETA="$2"; shift 2 ;;
    -U|--U|--use-ucb)             USE_UCB="$2"; shift 2 ;;
    -C|--C|--ucb-c)               UCB_C="$2"; shift 2 ;;
    -B|--B|--replay-buffer-size)  REPLAY_BUFFER_SIZE="$2"; shift 2 ;;
    -Q|--Q|--tau)                 TAU="$2"; shift 2 ;;
    -T|--T|--batch-size)          BATCH_SIZE="$2"; shift 2 ;;
    -S|--S|--inject-best-prob)    INJECT_BEST_PROB="$2"; shift 2 ;;
    -H|--H|--init-with-heuristic) INIT_WITH_HEURISTIC="$2"; shift 2 ;;
    --problem-class)              PROBLEM_CLASS="$2"; shift 2 ;;
    --problem-seed)               PROBLEM_SEED="$2"; shift 2 ;;
    --kill-thresholds)            KILL_THRESHOLDS_STR="$2"; shift 2 ;;
    --enable-kill)                ENABLE_KILL="$2"; shift 2 ;;
    --num-to-kill)                NUM_TO_KILL="$2"; shift 2 ;;
    --enable-revive)              ENABLE_REVIVE="$2"; shift 2 ;;
    --revive-threshold)           REVIVE_THRESHOLDS_STR="$2"; shift 2 ;;
    --ucb-window)                 UCB_WINDOW="$2"; shift 2 ;;
    --is-free-weight-matrix)      IS_FREE_WEIGHT_MATRIX="$2"; shift 2 ;;
    --is-inject-best-on-cycle)    IS_INJECT_BEST_ON_CYCLE="$2"; shift 2 ;;
    --is-append-first-task)       IS_APPEND_FIRST_TASK="$2"; shift 2 ;;
    --is-knn-enabled)             IS_KNN_ENABLED="$2"; shift 2 ;;
    --is-mimetism-enabled)        IS_MIMETISM_ENABLED="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unexpected option: $1"; exit 1 ;;
  esac
done

# ===== Defaults =====
PROBLEM_SIZE=${PROBLEM_SIZE:-15}
NUM_AGENTS=${NUM_AGENTS:-10}
METHOD=${METHOD:-"Double-Deep-Q"}
LOCK=${LOCK:-"false"}
PRESERVE=${PRESERVE:-"true"}
INJECT=${INJECT:-"false"}
LR=${LR:-0.5}
POSITIVE_REWARD=${POSITIVE_REWARD:-10.0}
NEGATIVE_REWARD=${NEGATIVE_REWARD:-"-1.0"}
GAMMA_DECAY=${GAMMA_DECAY:-0.25}
RHO=${RHO:-0.50}
ETA=${ETA:-0.75}
USE_UCB=${USE_UCB:-"false"}
UCB_C=${UCB_C:-0.0}
UCB_WINDOW=${UCB_WINDOW:-200}
REPLAY_BUFFER_SIZE=${REPLAY_BUFFER_SIZE:-50}
TAU=${TAU:-0.01}
BATCH_SIZE=${BATCH_SIZE:-10}
INJECT_BEST_PROB=${INJECT_BEST_PROB:-0.90}
INIT_WITH_HEURISTIC=${INIT_WITH_HEURISTIC:-"false"}
PROBLEM_CLASS=${PROBLEM_CLASS:-"Simple_Grid"}
PROBLEM_SEED=${PROBLEM_SEED:-}
KILL_THRESHOLDS_STR=${KILL_THRESHOLDS_STR:-"1.0"}
ENABLE_KILL=${ENABLE_KILL:-"false"}
NUM_TO_KILL=${NUM_TO_KILL:-"10"}
ENABLE_REVIVE=${ENABLE_REVIVE:-"false"}
REVIVE_THRESHOLDS_STR=${REVIVE_THRESHOLDS_STR:-"1.0"}
IS_FREE_WEIGHT_MATRIX=${IS_FREE_WEIGHT_MATRIX:-"false"}
IS_INJECT_BEST_ON_CYCLE=${IS_INJECT_BEST_ON_CYCLE:-"false"}
IS_APPEND_FIRST_TASK=${IS_APPEND_FIRST_TASK:-"true"}
IS_KNN_ENABLED=${IS_KNN_ENABLED:-"false"}
IS_MIMETISM_ENABLED=${IS_MIMETISM_ENABLED:-"true"}

# turn thresholds string into arrays
if [[ "$ENABLE_KILL" == "true" ]]; then
  read -r -a KILL_THRESHOLDS <<< "$KILL_THRESHOLDS_STR"
else
  KILL_THRESHOLDS=("disabled")
fi

if [[ "$ENABLE_REVIVE" == "true" ]]; then
  read -r -a REVIVE_THRESHOLDS <<< "$REVIVE_THRESHOLDS_STR"
else
  REVIVE_THRESHOLDS=("disabled")
fi

# ===== Helpers =====

log_node_diagnostics() {
    local out_dir="$1"
    {
        echo "=== NODE DIAGNOSTICS ==="
        echo "Hostname: $(hostname)"
        echo "Date: $(date)"
        echo "Ulimit: $(ulimit -a)"
        echo "--- Memory ---"
        free -h
        echo "--- Network Interfaces ---"
        ip addr || ifconfig || echo "ip command missing"
        echo "--- Env Vars (ROS specific) ---"
        env | grep -E "ROS|RMW|PYTHON"
        echo "--- Load Average ---"
        uptime
    } > "$out_dir/node_diagnostics.txt"
}

check_coverage_complete() {
  local run_dir="$1"
  local missing_coverage=false
  for agent_log in "$run_dir"/agent_*.log; do
    if ! grep -q "Coverage Complete" "$agent_log"; then
      echo "Coverage not complete in $agent_log"
      missing_coverage=true
    fi
  done
  $missing_coverage && return 1 || return 0
}

unique_id () {
  printf "%(%Y%m%d_%H%M%S)T_%04x" -1 "$((RANDOM%65536))"
}

kill_pid_tree () {
  local pid="$1"
  [ -z "${pid:-}" ] && return 0
  local child
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null || true); do
    kill_pid_tree "$child"
  done
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

check_first_dead () {
  FIRST_DEAD_PID=""
  local pid stat
  for pid in "$LOGGER_PID" "$SIM_PID" "${AGENT_PIDS[@]}"; do
    [ -z "$pid" ] && continue
    if ! ps -p "$pid" >/dev/null 2>&1; then
      FIRST_DEAD_PID="$pid"
      return 1
    fi
    stat="$(ps -o stat= -p "$pid" 2>/dev/null | awk '{print $1}')"
    if [ -n "$stat" ] && [[ "$stat" == Z* ]]; then
      FIRST_DEAD_PID="$pid"
      return 1
    fi
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

write_run_settings () {
  local _cfg="$1"; shift
  local _method="$1"; shift
  local _lock="$1"; shift
  local _preserve="$1"; shift
  local _uid="$1"; shift
  local _use_ucb="$1"; shift
  local _ucb_c="$1"; shift
  local _ucb_window="$1"; shift
  local _inject="$1"; shift
  local _inject_prob="$1"; shift
  local _problem_class="$1"; shift
  local _kill_th="$1"; shift
  local _revive_th="$1"; shift
  local _is_free_wm="$1"; shift
  local _is_inject_cycle="$1"; shift
  local _is_append="$1"; shift
  local _is_knn="$1"; shift
  local _is_mimetism="$1"; shift
  {
    echo "method=${_method}"
    echo "lock_mode=${_lock}"
    echo "preserve_next_task=${_preserve}"
    echo "use_ucb=${_use_ucb}"
    echo "ucb_c=${_ucb_c}"
    echo "ucb_window=${_ucb_window}"
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
    echo "initialise_with_heuristic=${INIT_WITH_HEURISTIC}"
    echo "start_iso=$(date -Is)"
    echo "problem_class=${_problem_class}"
    echo "kill_threshold=${_kill_th}"
    echo "revive_threshold=${_revive_th}"
    echo "is_free_weight_matrix=${_is_free_wm}"
    echo "is_inject_best_on_cycle=${_is_inject_cycle}"
    echo "is_append_first_task=${_is_append}"
    echo "is_knn_enabled=${_is_knn}"
    echo "is_mimetism_enabled=${_is_mimetism}"
    if [[ "$_method" == "Q-Learning" || "$_method" == "Double-Deep-Q" || "$_method" == "UCB" ]]; then
      echo "gamma_decay=${GAMMA_DECAY}"
      echo "lr=${LR}"
      echo "positive_reward=${POSITIVE_REWARD}"
      echo "negative_reward=${NEGATIVE_REWARD}"
      echo "rho=${RHO}"
      echo "replay_buffer_size=${REPLAY_BUFFER_SIZE}"
      echo "tau=${TAU}"
      echo "batch_size=${BATCH_SIZE}"
    elif [[ "$_method" == "Ferreira_et_al." ]]; then
      echo "eta=${ETA}"
      echo "rho=${RHO}"
    fi
  } > "$_cfg/run_settings.txt"
}

write_param_tag_file () {
  local _dir="$1"; shift
  local _method="$1"; shift
  local _lock="$1"; shift
  local _preserve="$1"; shift
  local _use_ucb="$1"; shift
  local _ucb_c="$1"; shift
  local _ucb_window="$1"; shift
  local _inject="$1"; shift
  local _pinj="$1"; shift
  local _gamma="$1"; shift
  local _lr="$1"; shift
  local _pos="$1"; shift
  local _neg="$1"; shift
  local _rho="$1"; shift
  local _eta="$1"; shift
  local _replay="$1"; shift
  local _tau="$1"; shift
  local _batch="$1"; shift
  local _init_heur="$1"; shift
  local _n_agents="$1"; shift
  local _p_size="$1"; shift
  local _problem_class="$1"; shift
  local _enable_kill="$1"; shift
  local _kill_th="$1"; shift
  local _enable_revive="$1"; shift
  local _revive_th="$1"; shift
  local _is_free_wm="$1"; shift
  local _is_inject_cycle="$1"; shift
  local _is_append="$1"; shift
  local _is_knn="$1"; shift
  local _is_mimetism="$1"; shift
  {
    echo "method=${_method}"
    echo "lock=${_lock}"
    echo "preserve_next_task=${_preserve}"
    echo "use_ucb=${_use_ucb}"
    echo "ucb_c=${_ucb_c}"
    echo "ucb_window=${_ucb_window}"
    echo "inject_best_on_cycle=${_inject}"
    echo "inject_best_prob=${_pinj}"
    echo "num_agents=${_n_agents}"
    echo "problem_size=${_p_size}"
    echo "initialise_with_heuristic=${_init_heur}"
    echo "gamma_decay=${_gamma}"
    echo "lr=${_lr}"
    echo "positive_reward=${_pos}"
    echo "negative_reward=${_neg}"
    echo "rho=${_rho}"
    echo "eta=${_eta}"
    echo "replay_buffer_size=${_replay}"
    echo "tau=${_tau}"
    echo "batch_size=${_batch}"
    echo "created_iso=$(date -Is)"
    echo "problem_class=${_problem_class}"
    echo "enable_kill=${_enable_kill}"
    echo "kill_threshold=${_kill_th}"
    echo "enable_revive=${_enable_revive}"
    echo "revive_threshold=${_revive_th}"
    echo "is_free_weight_matrix=${_is_free_wm}"
    echo "is_inject_best_on_cycle=${_is_inject_cycle}"
    echo "is_append_first_task=${_is_append}"
    echo "is_knn_enabled=${_is_knn}"
    echo "is_mimetism_enabled=${_is_mimetism}"
  } > "$_dir/setting_tag.txt"
}

declare -A PID_ROLE
declare -A PID_LOG
LOGGER_PID=""
SIM_PID=""
AGENT_PIDS=()

start_all_processes () {
  export ROS_LOCALHOST_ONLY=1
  export ROS_DOMAIN_ID=$(( (RANDOM % 100) + 1 ))

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
  local CUR_UCB_WINDOW="$1"; shift
  local CUR_REPLAY_BUFFER_SIZE="$1"; shift
  local CUR_TAU="$1"; shift
  local CUR_BATCH_SIZE="$1"; shift
  local CUR_PROBLEM_CLASS="$1"; shift
  local CUR_PROBLEM_SEED="$1"; shift
  local CUR_ENABLE_KILL="$1"; shift
  local CUR_KILL_TH="$1"; shift
  local CUR_NUM_TO_KILL="$1"; shift
  local CUR_ENABLE_REVIVE="$1"; shift
  local CUR_REVIVE_TH="$1"; shift
  local CUR_IS_FREE_WEIGHT_MATRIX="$1"; shift
  local CUR_IS_INJECT_BEST_ON_CYCLE="$1"; shift
  local CUR_IS_APPEND_FIRST_TASK="$1"; shift
  local CUR_IS_KNN_ENABLED="$1"; shift
  local CUR_IS_MIMETISM_ENABLED="$1"; shift

  LOGGER_LOG="$CONFIG_DIR/logger.log"
  LOGGER_PID=$(start_logged "$LOGGER_LOG" "$ROS2_LOG_DIR" \
    ros2 run "$PACKAGE_NAME" "$LOGGER_EXECUTABLE" \
      --ros-args \
      -p parent_log_dir:="'$CONFIG_DIR'" \
      -p num_tsp_agents:="$NUM_AGENTS" \
      -p problem_size:="$PROBLEM_SIZE")
  PID_ROLE["$LOGGER_PID"]="logger"; PID_LOG["$LOGGER_PID"]="$LOGGER_LOG"

  SIM_LOG="$CONFIG_DIR/simulator.log"
  local -a SIM_CMD=(
    ros2 run "$PACKAGE_NAME" "$SIM_EXECUTABLE"
    --num_robots "$NUM_AGENTS"
    --problem_size "$PROBLEM_SIZE"
    --problem_class "$CUR_PROBLEM_CLASS"
    --problem_seed "$CUR_PROBLEM_SEED"
  )
  if [[ "$CUR_ENABLE_KILL" == "true" ]]; then
    SIM_CMD+=( --enable_kill --kill_threshold "$CUR_KILL_TH" --num_to_kill "$CUR_NUM_TO_KILL" )
  fi
  if [[ "$CUR_ENABLE_REVIVE" == "true" ]]; then
    SIM_CMD+=( --enable_revive --revive_threshold "$CUR_REVIVE_TH" )
  fi
  SIM_PID=$(start_logged "$SIM_LOG" "$ROS2_LOG_DIR" "${SIM_CMD[@]}")
  PID_ROLE["$SIM_PID"]="simulator"; PID_LOG["$SIM_PID"]="$SIM_LOG"

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
      -p ucb_window:="$CUR_UCB_WINDOW"
      -p inject_best_on_cycle:=${CUR_INJECT}
      -p inject_best_prob:="$CUR_INJECT_PROB"
      -p initialise_with_heuristic:="$INIT_WITH_HEURISTIC"
      -p problem_class:="$CUR_PROBLEM_CLASS"
      -p problem_seed:="$CUR_PROBLEM_SEED"
      -p is_free_weight_matrix:=${CUR_IS_FREE_WEIGHT_MATRIX}
      -p is_inject_best_on_cycle:=${CUR_IS_INJECT_BEST_ON_CYCLE}
      -p is_append_first_task:=${CUR_IS_APPEND_FIRST_TASK}
      -p is_knn_enabled:=${CUR_IS_KNN_ENABLED}
      -p is_mimetism_enabled:=${CUR_IS_MIMETISM_ENABLED}
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
    elif [[ "$CUR_METHOD" == "UCB" ]]; then
      CMD+=(
        -p rho:="$CUR_RHO_SHARED"
      )
    fi

    local pid
    pid=$(start_logged "$AGENT_LOG" "$ROS2_LOG_DIR" "${CMD[@]}")
    AGENT_PIDS+=("$pid")
    PID_ROLE["$pid"]="agent[$i]"; PID_LOG["$pid"]="$AGENT_LOG"
  done

  echo "   [STARTUP] Waiting for $NUM_AGENTS agents to initialize (Parallel check)..."
  local start_wait=$(date +%s)

  local -a is_ready
  for ((i=0; i<NUM_AGENTS; i++)); do is_ready[$i]=0; done

  local pending_count=$NUM_AGENTS

  while [ "$pending_count" -gt 0 ]; do
    local now=$(date +%s)

    if (( now - start_wait > AGENT_STARTUP_TIMEOUT )); then
      echo "[ERROR] Startup Timeout! The following agents failed to init within ${AGENT_STARTUP_TIMEOUT}s:"

      echo "--- DIAGNOSTICS FOR FAILED AGENTS ---"
      dmesg | tail -n 20 | grep -i "kill" || echo "No recent kernel kills found in dmesg"

      for ((i=0; i<NUM_AGENTS; i++)); do
        if [ "${is_ready[$i]}" -eq 0 ]; then
             local pid=${AGENT_PIDS[$i]}
             local logf="${PID_LOG[$pid]}"
             echo ">>> AGENT $i (PID $pid) LOG DUMP START <<<"
             if [ -f "$logf" ]; then
                 echo "--- HEAD (First 20 lines) ---"
                 head -n 20 "$logf"
                 echo "--- TAIL (Last 50 lines) ---"
                 tail -n 50 "$logf"
             else
                 echo "Log file not found: $logf"
             fi
             echo ">>> AGENT $i LOG DUMP END <<<"
        fi
      done
      return 1
    fi

    for ((i=0; i<NUM_AGENTS; i++)); do
      if [ "${is_ready[$i]}" -eq 0 ]; then
        local pid=${AGENT_PIDS[$i]}
        local logf="${PID_LOG[$pid]}"

        if ! kill -0 "$pid" 2>/dev/null; then
             echo "[ERROR] Agent $i (PID $pid) crashed immediately! (Process gone)"
             return 1
        fi

        if [ -f "$logf" ] && grep -Fq "$AGENT_INIT_MSG" "$logf" 2>/dev/null; then
             is_ready[$i]=1
             pending_count=$((pending_count - 1))
        fi
      fi
    done
    sleep 1
  done

  echo "   [STARTUP] All $NUM_AGENTS agents initialized successfully."
  return 0
}

on_sigint () {
  echo "[SIGINT] Cleaning up processes..."
  cleanup_run "${LOGGER_PID:-}" "${SIM_PID:-}" "${AGENT_PIDS[@]:-}"
  exit 130
}
on_sigterm () {
  echo "[SIGTERM] Cleaning up processes..."
  cleanup_run "${LOGGER_PID:-}" "${SIM_PID:-}" "${AGENT_PIDS[@]:-}"
  exit 143
}
on_exit () {
  cleanup_run "${LOGGER_PID:-}" "${SIM_PID:-}" "${AGENT_PIDS[@]:-}"
}
trap on_sigint INT
trap on_sigterm TERM
trap on_exit EXIT

LAST_RUN_NUM=""
SAME_RUN_COUNT=0

# ===== Main: iterate kill & revive thresholds, then runs =====
for KILL_TH in "${KILL_THRESHOLDS[@]}"; do
  for REVIVE_TH in "${REVIVE_THRESHOLDS[@]}"; do

    if [[ "$ENABLE_KILL" == "true" ]]; then
      BASE_ROOT="$RESULTS_ROOT/size_${PROBLEM_SIZE}_agents_${NUM_AGENTS}_${PROBLEM_CLASS}_kill_${KILL_TH}_killnum_${NUM_TO_KILL}"
    else
      BASE_ROOT="$RESULTS_ROOT/size_${PROBLEM_SIZE}_agents_${NUM_AGENTS}_${PROBLEM_CLASS}_kill_disabled"
    fi

    if [[ "$ENABLE_REVIVE" == "true" ]]; then
      COMBO_ROOT="${BASE_ROOT}_revive_${REVIVE_TH}"
    else
      COMBO_ROOT="${BASE_ROOT}_revive_disabled"
    fi

    mkdir -p "$COMBO_ROOT"

    METHOD_TAG="$(slug "$METHOD")"
    PARAM_DIR="$COMBO_ROOT/method_${METHOD_TAG}_lock_${LOCK}_preserve_${PRESERVE}_inject_${INJECT}_pinj_${INJECT_BEST_PROB}_ucb_${USE_UCB}_ucbc_${UCB_C}_ucbw_${UCB_WINDOW}_replay_${REPLAY_BUFFER_SIZE}_tau_${TAU}_batch_${BATCH_SIZE}_gamma_${GAMMA_DECAY}_lr_${LR}_pos_${POSITIVE_REWARD}_neg_${NEGATIVE_REWARD}_rho_${RHO}_eta_${ETA}_heur_${INIT_WITH_HEURISTIC}_freewm_${IS_FREE_WEIGHT_MATRIX}_injectcyc_${IS_INJECT_BEST_ON_CYCLE}_append_${IS_APPEND_FIRST_TASK}_knn_${IS_KNN_ENABLED}_mimetism_${IS_MIMETISM_ENABLED}"
    mkdir -p "$PARAM_DIR"

    write_param_tag_file "$PARAM_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$USE_UCB" "$UCB_C" "$UCB_WINDOW" \
                          "$INJECT" "$INJECT_BEST_PROB" "$GAMMA_DECAY" "$LR" "$POSITIVE_REWARD" \
                          "$NEGATIVE_REWARD" "$RHO" "$ETA" "$REPLAY_BUFFER_SIZE" "$TAU" \
                          "$BATCH_SIZE" "$INIT_WITH_HEURISTIC" "$NUM_AGENTS" "$PROBLEM_SIZE" "$PROBLEM_CLASS" \
                          "$ENABLE_KILL" "$KILL_TH" "$ENABLE_REVIVE" "$REVIVE_TH" \
                          "$IS_FREE_WEIGHT_MATRIX" "$IS_INJECT_BEST_ON_CYCLE" \
                          "$IS_APPEND_FIRST_TASK" "$IS_KNN_ENABLED" "$IS_MIMETISM_ENABLED"

    LAST_RUN_DIR=""
    LAST_RUN_NUM_LOCAL=0
    if ls -d "$PARAM_DIR"/run_* >/dev/null 2>&1; then
      LAST_RUN_DIR=$(ls -d "$PARAM_DIR"/run_* 2>/dev/null | sort -V | tail -n 1)
      LAST_RUN_NUM_LOCAL=$(basename "$LAST_RUN_DIR" | sed 's/run_//')
      if ! check_coverage_complete "$LAST_RUN_DIR"; then
        echo "[RECOVER] Last run $LAST_RUN_NUM_LOCAL in $PARAM_DIR was incomplete. Deleting and re-running it."
        rm -rf "$LAST_RUN_DIR"
        START_RUN=$LAST_RUN_NUM_LOCAL
      else
        START_RUN=$((LAST_RUN_NUM_LOCAL + 1))
      fi
    else
      START_RUN=1
    fi

    if [ "$START_RUN" -gt "$NUM_RUNS" ]; then
      echo "[INFO] Already completed $((START_RUN-1)) runs for $PARAM_DIR. Skipping."
      continue
    fi

    run=$START_RUN
    while [ "$run" -le "$NUM_RUNS" ]; do

      RUN_SEED="${PROBLEM_SEED:-$(gen_seed)}"
      echo "============================="
      echo "[INFO] Run $run/$NUM_RUNS :: kill_enabled=$ENABLE_KILL :: kill_th=$KILL_TH :: revive_enabled=$ENABLE_REVIVE :: revive_th=$REVIVE_TH :: method=$METHOD lock=$LOCK preserve=$PRESERVE inject=$INJECT pinj=$INJECT_BEST_PROB gamma=$GAMMA_DECAY lr=$LR pos=$POSITIVE_REWARD neg=$NEGATIVE_REWARD rho=$RHO eta=$ETA replay=$REPLAY_BUFFER_SIZE tau=$TAU batch=$BATCH_SIZE ucb=$USE_UCB ucb_c=$UCB_C ucb_window=$UCB_WINDOW num_agents=$NUM_AGENTS problem_size=$PROBLEM_SIZE heur=$INIT_WITH_HEURISTIC problem_class=$PROBLEM_CLASS seed=$RUN_SEED free_wm=$IS_FREE_WEIGHT_MATRIX inject_cyc=$IS_INJECT_BEST_ON_CYCLE append=$IS_APPEND_FIRST_TASK knn=$IS_KNN_ENABLED mimetism=$IS_MIMETISM_ENABLED"
      echo "============================="

      CONFIG_DIR="$PARAM_DIR/run_${run}"
      ROS2_LOG_DIR="$CONFIG_DIR/ros_logs"
      mkdir -p "$ROS2_LOG_DIR"

      log_node_diagnostics "$CONFIG_DIR"

      write_run_settings "$CONFIG_DIR" "$METHOD" "$LOCK" "$PRESERVE" "$(unique_id)" \
                          "$USE_UCB" "$UCB_C" "$UCB_WINDOW" "$INJECT" "$INJECT_BEST_PROB" \
                          "$PROBLEM_CLASS" "$KILL_TH" "$REVIVE_TH" \
                          "$IS_FREE_WEIGHT_MATRIX" "$IS_INJECT_BEST_ON_CYCLE" \
                          "$IS_APPEND_FIRST_TASK" "$IS_KNN_ENABLED" "$IS_MIMETISM_ENABLED"

      if ! start_all_processes \
            "$LOCK" "$PRESERVE" "$INJECT" "$INJECT_BEST_PROB" "$METHOD" \
            "$LR" "$GAMMA_DECAY" "$POSITIVE_REWARD" "$NEGATIVE_REWARD" \
            "$ETA" "$RHO" \
            "$RHO" \
            "$USE_UCB" "$UCB_C" "$UCB_WINDOW" \
            "$REPLAY_BUFFER_SIZE" "$TAU" "$BATCH_SIZE" "$PROBLEM_CLASS" "$RUN_SEED" \
            "$ENABLE_KILL" "$KILL_TH" "$NUM_TO_KILL" \
            "$ENABLE_REVIVE" "$REVIVE_TH" \
            "$IS_FREE_WEIGHT_MATRIX" "$IS_INJECT_BEST_ON_CYCLE" \
            "$IS_APPEND_FIRST_TASK" "$IS_KNN_ENABLED" "$IS_MIMETISM_ENABLED"; then

        echo "[FAIL] Start-up failed. Cleaning up and retrying run $run..."
        cleanup_run "${LOGGER_PID:-}" "${SIM_PID:-}" "${AGENT_PIDS[@]:-}"
        rm -rf "$CONFIG_DIR"

        if [[ "$LAST_RUN_NUM" == "$run" ]]; then
          SAME_RUN_COUNT=$((SAME_RUN_COUNT + 1))
        else
          LAST_RUN_NUM="$run"
          SAME_RUN_COUNT=1
        fi
        if [ "$SAME_RUN_COUNT" -ge 5 ]; then
          echo "[FATAL] Run $run has failed start-up 5 times. Exiting."
          exit 1
        fi
        continue
      fi

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
          echo "[ELAPSED]: ${ELAPSED}s"
          if [ "$ELAPSED" -le 30 ]; then
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

      cleanup_run "${LOGGER_PID:-}" "${SIM_PID:-}" "${AGENT_PIDS[@]:-}"

      if ! check_coverage_complete "$CONFIG_DIR"; then
        echo "[CLEANUP] Deleting failed run directory: $CONFIG_DIR"
        rm -rf "$CONFIG_DIR"
        echo "[RETRY] Repeating run $run (kill_th=$KILL_TH, revive_th=$REVIVE_TH)"

        if [[ "$LAST_RUN_NUM" == "$run" ]]; then
          SAME_RUN_COUNT=$((SAME_RUN_COUNT + 1))
        else
          LAST_RUN_NUM="$run"
          SAME_RUN_COUNT=1
        fi
        if [ "$SAME_RUN_COUNT" -ge 5 ]; then
          echo "[FATAL] Run $run has failed $SAME_RUN_COUNT times. Exiting to avoid infinite loop."
          exit 1
        fi
        continue
      fi

      LAST_RUN_NUM="$run"
      SAME_RUN_COUNT=0

      echo "end_iso=$(date -Is)" >> "$CONFIG_DIR/run_settings.txt"
      echo "[INFO] Completed run $run for $CONFIG_DIR (timeout=${RUN_TIMEOUT})"
      run=$((run + 1))
    done
  done
done

echo "[INFO] All simulations completed."
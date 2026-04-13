#!/bin/bash
# run_des_sims.sh — run multiple CBM-POP DES simulations (no ROS required).
#
# Local multi-run usage:
#   ./run_des_sims.sh --agents 5 --problem-size 10 --num-runs 50
#
# SLURM array usage (one run per array task):
#   ./run_des_sims.sh --agents 5 --problem-size 10 --run-id $SLURM_ARRAY_TASK_ID

set -e
set -o pipefail

PYTHON="$(command -v python3)"

# Add project root to PYTHONPATH so cbm_pop is always importable
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"   # CBM-POP_Implementation/
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

# ===== Globals (initialised before any trap) =====
SIM_PID=""

# ===== Seed generator =====
gen_seed() {
  if command -v od >/dev/null 2>&1; then
    printf '%d\n' $(( 10000 + ( $(od -An -N2 -tu2 /dev/urandom | tr -d ' ') % 90000 ) ))
  elif command -v shuf >/dev/null 2>&1; then
    shuf -i 10000-99999 -n1
  else
    printf '%d\n' $(( 10000 + ( ( (RANDOM << 15) | RANDOM ) % 90000 ) ))
  fi
}

unique_id() {
  printf "%(%Y%m%d_%H%M%S)T_%04x" -1 "$((RANDOM % 65536))"
}

slug() {
  local s="$1"; s="${s//[^A-Za-z0-9._-]/_}"; echo "$s"
}

# ===== Configuration =====
MAX_ATTEMPTS=3          # retries per run before giving up
TIMEOUT_SECONDS=1800    # wall-clock timeout per run (seconds)
RESULTS_ROOT="resources/run_logs"
mkdir -p "$RESULTS_ROOT"

# ===== CLI parsing =====
PARSED=$(getopt \
  -o p:a: \
  -l problem-size:,agents:,problem-class:,problem-seed:,run-id:,num-runs:,\
method:,speed:,max-sim-time:,pop-size:,di-cycle-length:,num-solution-attempts:,\
lr:,gamma-decay:,positive-reward:,negative-reward:,rho:,eta:,\
ucb-c:,ucb-window:,is-free-weight-matrix:,\
is-knn-enabled:,is-mimetism-enabled:,is-inject-best-on-cycle:,inject-best-prob:,\
is-append-first-task:,\
progress-interval:,output-root: \
  -- "$@") || { echo "Invalid options"; exit 1; }
eval set -- "$PARSED"

while true; do
  case "$1" in
    -p|--problem-size)            PROBLEM_SIZE="$2";            shift 2 ;;
    -a|--agents)                  NUM_AGENTS="$2";              shift 2 ;;
    --problem-class)              PROBLEM_CLASS="$2";           shift 2 ;;
    --problem-seed)               PROBLEM_SEED="$2";            shift 2 ;;
    --run-id)                     RUN_ID="$2";                  shift 2 ;;
    --num-runs)                   NUM_RUNS="$2";                shift 2 ;;
    --method)                     METHOD="$2";                  shift 2 ;;
    --speed)                      SPEED="$2";                   shift 2 ;;
    --max-sim-time)               MAX_SIM_TIME="$2";            shift 2 ;;
    --pop-size)                   POP_SIZE="$2";                shift 2 ;;
    --di-cycle-length)            DI_CYCLE_LENGTH="$2";         shift 2 ;;
    --num-solution-attempts)      NUM_SOLUTION_ATTEMPTS="$2";   shift 2 ;;
    --lr)                         LR="$2";                      shift 2 ;;
    --gamma-decay)                GAMMA_DECAY="$2";             shift 2 ;;
    --positive-reward)            POSITIVE_REWARD="$2";         shift 2 ;;
    --negative-reward)            NEGATIVE_REWARD="$2";         shift 2 ;;
    --rho)                        RHO="$2";                     shift 2 ;;
    --eta)                        ETA="$2";                     shift 2 ;;
    --ucb-c)                      UCB_C="$2";                   shift 2 ;;
    --ucb-window)                 UCB_WINDOW="$2";              shift 2 ;;
    --is-free-weight-matrix)      IS_FREE_WEIGHT_MATRIX="$2";   shift 2 ;;
    --is-knn-enabled)             IS_KNN_ENABLED="$2";          shift 2 ;;
    --is-mimetism-enabled)        IS_MIMETISM_ENABLED="$2";     shift 2 ;;
    --is-inject-best-on-cycle)    IS_INJECT_BEST_ON_CYCLE="$2"; shift 2 ;;
    --inject-best-prob)           INJECT_BEST_PROB="$2";        shift 2 ;;
    --is-append-first-task)       IS_APPEND_FIRST_TASK="$2";    shift 2 ;;
    --progress-interval)          PROGRESS_INTERVAL="$2";       shift 2 ;;
    --output-root)                RESULTS_ROOT="$2";            shift 2 ;;
    --) shift; break ;;
    *) echo "Unexpected option: $1"; exit 1 ;;
  esac
done

# ===== Defaults =====
PROBLEM_SIZE=${PROBLEM_SIZE:-15}
NUM_AGENTS=${NUM_AGENTS:-5}
PROBLEM_CLASS=${PROBLEM_CLASS:-"Simple_Grid"}
PROBLEM_SEED=${PROBLEM_SEED:-}
RUN_ID=${RUN_ID:-${SLURM_ARRAY_TASK_ID:-}}   # empty = local multi-run mode
NUM_RUNS=${NUM_RUNS:-1}
SPEED=${SPEED:-1.0}
MAX_SIM_TIME=${MAX_SIM_TIME:-"inf"}
POP_SIZE=${POP_SIZE:-10}
DI_CYCLE_LENGTH=${DI_CYCLE_LENGTH:-10}
NUM_SOLUTION_ATTEMPTS=${NUM_SOLUTION_ATTEMPTS:-21}
LR=${LR:-0.22}
GAMMA_DECAY=${GAMMA_DECAY:-0.95}
POSITIVE_REWARD=${POSITIVE_REWARD:-7.0}
NEGATIVE_REWARD=${NEGATIVE_REWARD:--8.0}
RHO=${RHO:-0.5}
METHOD=${METHOD:-"Q-Learning"}
ETA=${ETA:-0.1}
UCB_C=${UCB_C:-1.414}
UCB_WINDOW=${UCB_WINDOW:-200}
IS_FREE_WEIGHT_MATRIX=${IS_FREE_WEIGHT_MATRIX:-"false"}
IS_KNN_ENABLED=${IS_KNN_ENABLED:-"false"}
IS_MIMETISM_ENABLED=${IS_MIMETISM_ENABLED:-"true"}
IS_INJECT_BEST_ON_CYCLE=${IS_INJECT_BEST_ON_CYCLE:-"false"}
INJECT_BEST_PROB=${INJECT_BEST_PROB:-0.9}
IS_APPEND_FIRST_TASK=${IS_APPEND_FIRST_TASK:-"true"}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-0}

# ===== Directory layout =====
PARAM_DIR="$RESULTS_ROOT/size_${PROBLEM_SIZE}_agents_${NUM_AGENTS}_$(slug "$PROBLEM_CLASS")/method_$(slug "$METHOD")_lr_${LR}_gamma_${GAMMA_DECAY}_pos_${POSITIVE_REWARD}_neg_${NEGATIVE_REWARD}_rho_${RHO}_eta_${ETA}_knn_${IS_KNN_ENABLED}_mimetism_${IS_MIMETISM_ENABLED}_inject_${IS_INJECT_BEST_ON_CYCLE}_pinj_${INJECT_BEST_PROB}_speed_${SPEED}_pop_${POP_SIZE}_di_${DI_CYCLE_LENGTH}_freewm_${IS_FREE_WEIGHT_MATRIX}"
mkdir -p "$PARAM_DIR"

# ===== Helpers =====

write_param_tag() {
  local _dir="$1"
  {
    echo "simulator=DES"
    echo "method=${METHOD}"
    echo "num_agents=${NUM_AGENTS}"
    echo "problem_size=${PROBLEM_SIZE}"
    echo "problem_class=${PROBLEM_CLASS}"
    echo "speed=${SPEED}"
    echo "max_sim_time=${MAX_SIM_TIME}"
    echo "pop_size=${POP_SIZE}"
    echo "di_cycle_length=${DI_CYCLE_LENGTH}"
    echo "num_solution_attempts=${NUM_SOLUTION_ATTEMPTS}"
    echo "lr=${LR}"
    echo "gamma_decay=${GAMMA_DECAY}"
    echo "positive_reward=${POSITIVE_REWARD}"
    echo "negative_reward=${NEGATIVE_REWARD}"
    echo "rho=${RHO}"
    echo "eta=${ETA}"
    echo "ucb_c=${UCB_C}"
    echo "ucb_window=${UCB_WINDOW}"
    echo "is_free_weight_matrix=${IS_FREE_WEIGHT_MATRIX}"
    echo "is_knn_enabled=${IS_KNN_ENABLED}"
    echo "is_mimetism_enabled=${IS_MIMETISM_ENABLED}"
    echo "is_inject_best_on_cycle=${IS_INJECT_BEST_ON_CYCLE}"
    echo "inject_best_prob=${INJECT_BEST_PROB}"
    echo "is_append_first_task=${IS_APPEND_FIRST_TASK}"
    echo "created_iso=$(date -Is)"
  } > "$_dir/setting_tag.txt"
}

write_run_settings() {
  local _cfg="$1" _seed="$2" _uid="$3"
  {
    echo "simulator=DES"
    echo "method=${METHOD}"
    echo "num_agents=${NUM_AGENTS}"
    echo "problem_size=${PROBLEM_SIZE}"
    echo "problem_class=${PROBLEM_CLASS}"
    echo "problem_seed=${_seed}"
    echo "speed=${SPEED}"
    echo "max_sim_time=${MAX_SIM_TIME}"
    echo "pop_size=${POP_SIZE}"
    echo "di_cycle_length=${DI_CYCLE_LENGTH}"
    echo "num_solution_attempts=${NUM_SOLUTION_ATTEMPTS}"
    echo "lr=${LR}"
    echo "gamma_decay=${GAMMA_DECAY}"
    echo "positive_reward=${POSITIVE_REWARD}"
    echo "negative_reward=${NEGATIVE_REWARD}"
    echo "rho=${RHO}"
    echo "eta=${ETA}"
    echo "ucb_c=${UCB_C}"
    echo "ucb_window=${UCB_WINDOW}"
    echo "is_free_weight_matrix=${IS_FREE_WEIGHT_MATRIX}"
    echo "is_knn_enabled=${IS_KNN_ENABLED}"
    echo "is_mimetism_enabled=${IS_MIMETISM_ENABLED}"
    echo "is_inject_best_on_cycle=${IS_INJECT_BEST_ON_CYCLE}"
    echo "inject_best_prob=${INJECT_BEST_PROB}"
    echo "is_append_first_task=${IS_APPEND_FIRST_TASK}"
    echo "timeout_seconds=${TIMEOUT_SECONDS}"
    echo "run_uid=${_uid}"
    echo "start_iso=$(date -Is)"
  } > "$_cfg/run_settings.txt"
}

check_des_complete() {
  local _dir="$1"
  local _csv="$_dir/coverage_log.csv"
  [ -f "$_csv" ] || { echo "No coverage_log.csv in $_dir"; return 1; }
  # Last data row: check coverage_fraction == 1.0000
  local _last
  _last=$(tail -n 1 "$_csv" | awk -F',' '{print $4}' | tr -d '[:space:]')
  if [[ "$_last" == "1.0000" ]]; then
    return 0
  fi
  echo "Coverage incomplete in $_csv (last fraction=${_last})"
  return 1
}

build_des_cmd() {
  local _seed="$1" _outdir="$2"
  local -a CMD=(
    "$PYTHON" -m cbm_pop.DESSimulator.run_des_sim
    --method              "$METHOD"
    --num_agents          "$NUM_AGENTS"
    --problem_size        "$PROBLEM_SIZE"
    --problem_class       "$PROBLEM_CLASS"
    --problem_seed        "$_seed"
    --speed               "$SPEED"
    --pop_size            "$POP_SIZE"
    --di_cycle_length     "$DI_CYCLE_LENGTH"
    --num_solution_attempts "$NUM_SOLUTION_ATTEMPTS"
    --lr                  "$LR"
    --gamma_decay         "$GAMMA_DECAY"
    --positive_reward     "$POSITIVE_REWARD"
    --negative_reward     "$NEGATIVE_REWARD"
    --rho                 "$RHO"
    --eta                 "$ETA"
    --ucb_c               "$UCB_C"
    --ucb_window          "$UCB_WINDOW"
    --inject_best_prob    "$INJECT_BEST_PROB"
    --output_dir          "$_outdir"
  )
  [[ "$MAX_SIM_TIME"         != "inf"   ]] && CMD+=( --max_sim_time "$MAX_SIM_TIME" )
  [[ "$IS_FREE_WEIGHT_MATRIX" == "true" ]] && CMD+=( --is_free_weight_matrix )
  [[ "$IS_KNN_ENABLED"        == "true" ]] && CMD+=( --is_knn_enabled )
  [[ "$IS_MIMETISM_ENABLED"   == "false" ]] && CMD+=( --no_mimetism )
  [[ "$IS_INJECT_BEST_ON_CYCLE" == "true" ]] && CMD+=( --inject_best )
  [[ "$IS_APPEND_FIRST_TASK"  == "false" ]] && CMD+=( --no_append_first_task )
  [[ "$PROGRESS_INTERVAL"    != "0"     ]] && CMD+=( --progress_interval "$PROGRESS_INTERVAL" )
  echo "${CMD[@]}"
}

kill_pid_tree() {
  local pid="$1"
  [ -z "${pid:-}" ] && return 0
  local child
  for child in $(ps -o pid= --ppid "$pid" 2>/dev/null || true); do
    kill_pid_tree "$child"
  done
  kill -TERM "$pid" 2>/dev/null || true
  sleep 0.3
  kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  kill_pid_tree "${SIM_PID:-}"
}

on_sigint()  { echo "[SIGINT] Cleaning up...";  cleanup; exit 130; }
on_sigterm() { echo "[SIGTERM] Cleaning up..."; cleanup; exit 143; }
on_exit()    { cleanup; }
trap on_sigint  INT
trap on_sigterm TERM
trap on_exit    EXIT

# ===== Run a single simulation =====
# Returns 0 on success, 1 on failure/timeout.
run_one() {
  local _run_id="$1" _seed="$2"
  local _cfg="$PARAM_DIR/run_${_run_id}"
  mkdir -p "$_cfg"

  write_run_settings "$_cfg" "$_seed" "$(unique_id)"

  local _log="$_cfg/des_sim.log"
  local -a _cmd
  read -r -a _cmd <<< "$(build_des_cmd "$_seed" "$_cfg")"

  echo "   [START] run_id=${_run_id} seed=${_seed} agents=${NUM_AGENTS} size=${PROBLEM_SIZE} class=${PROBLEM_CLASS}"

  stdbuf -oL -eL "${_cmd[@]}" >> "$_log" 2>&1 &
  SIM_PID=$!

  local _start
  _start=$(date +%s)
  local _last_hb=0

  while kill -0 "$SIM_PID" 2>/dev/null; do
    sleep 2
    local _elapsed=$(( $(date +%s) - _start ))

    if (( _elapsed - _last_hb >= 30 )); then
      echo "   [HEARTBEAT] run_id=${_run_id} elapsed=${_elapsed}s"
      _last_hb=$_elapsed
    fi

    if (( _elapsed >= TIMEOUT_SECONDS )); then
      echo "   [TIMEOUT] run_id=${_run_id} exceeded ${TIMEOUT_SECONDS}s — killing."
      echo "[FAILURE_REASON] timeout after ${TIMEOUT_SECONDS}s" >> "$_log"
      kill_pid_tree "$SIM_PID"
      SIM_PID=""
      return 1
    fi
  done

  # Collect exit code
  local _status=0
  wait "$SIM_PID" 2>/dev/null || _status=$?
  SIM_PID=""

  local _elapsed=$(( $(date +%s) - _start ))

  if [ "$_status" -ne 0 ]; then
    echo "   [FAIL] run_id=${_run_id} exited with code ${_status} after ${_elapsed}s"
    echo "[FAILURE_REASON] python exited with code ${_status}" >> "$_log"
    echo "   --- last 30 lines of $_log ---"
    tail -n 30 "$_log" || true
    return 1
  fi

  if ! check_des_complete "$_cfg"; then
    echo "   [FAIL] run_id=${_run_id} coverage incomplete after ${_elapsed}s"
    echo "[FAILURE_REASON] coverage incomplete at exit (no timeout)" >> "$_log"
    return 1
  fi

  echo "end_iso=$(date -Is)" >> "$_cfg/run_settings.txt"
  echo "   [SUCCESS] run_id=${_run_id} completed in ${_elapsed}s"
  return 0
}

# ===== Write param tag once =====
write_param_tag "$PARAM_DIR"

# ===== Determine run range =====
if [ -n "${RUN_ID:-}" ]; then
  # SLURM array mode: run exactly this one ID
  START_RUN=$RUN_ID
  END_RUN=$RUN_ID
else
  # Local multi-run mode: resume from last completed
  START_RUN=1
  END_RUN=$NUM_RUNS

  if ls -d "$PARAM_DIR"/run_[0-9]* >/dev/null 2>&1; then
    LAST_DIR=$(ls -d "$PARAM_DIR"/run_[0-9]* 2>/dev/null \
               | grep -E '/run_[0-9]+$' | sort -V | tail -n 1)
    LAST_NUM=$(basename "$LAST_DIR" | sed 's/run_//')
    if check_des_complete "$LAST_DIR" 2>/dev/null; then
      START_RUN=$(( LAST_NUM + 1 ))
    else
      echo "[RECOVER] Last run $LAST_NUM was incomplete — preserving logs and re-running."
      mv "$LAST_DIR" "${LAST_DIR}_recovered_$(date +%s)"
      START_RUN=$LAST_NUM
    fi
  fi

  if [ "$START_RUN" -gt "$END_RUN" ]; then
    echo "[INFO] All $NUM_RUNS runs already complete in $PARAM_DIR."
    exit 0
  fi
fi

# ===== Main loop =====
STUCK_RUN=""
STUCK_COUNT=0

run=$START_RUN
while [ "$run" -le "$END_RUN" ]; do
  RUN_SEED="${PROBLEM_SEED:-$(gen_seed)}"
  echo "============================="
  echo "[INFO] Run ${run}/${END_RUN} | agents=${NUM_AGENTS} size=${PROBLEM_SIZE} class=${PROBLEM_CLASS} seed=${RUN_SEED} method=${METHOD} speed=${SPEED} max_sim_time=${MAX_SIM_TIME} pop_size=${POP_SIZE} di_cycle=${DI_CYCLE_LENGTH} solution_attempts=${NUM_SOLUTION_ATTEMPTS} lr=${LR} gamma=${GAMMA_DECAY} pos_reward=${POSITIVE_REWARD} neg_reward=${NEGATIVE_REWARD} rho=${RHO} eta=${ETA} ucb_c=${UCB_C} ucb_window=${UCB_WINDOW} free_wm=${IS_FREE_WEIGHT_MATRIX} knn=${IS_KNN_ENABLED} mimetism=${IS_MIMETISM_ENABLED} inject_best=${IS_INJECT_BEST_ON_CYCLE} inject_prob=${INJECT_BEST_PROB} append_first=${IS_APPEND_FIRST_TASK}"
  echo "============================="

  attempt=0
  success=false
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$(( attempt + 1 ))
    if run_one "$run" "$RUN_SEED"; then
      success=true
      break
    fi
    # Preserve failed attempt logs for post-mortem inspection
    _failed_dir="$PARAM_DIR/run_${run}_failed_attempt_${attempt}"
    rm -rf "$_failed_dir"
    if [ -d "$PARAM_DIR/run_${run}" ]; then
      mv "$PARAM_DIR/run_${run}" "$_failed_dir"
      echo "   [PRESERVED] Failed logs saved to $_failed_dir"
    fi
    if [ $attempt -lt $MAX_ATTEMPTS ]; then
      echo "   [RETRY] attempt $attempt/$MAX_ATTEMPTS failed — retrying run $run"
      sleep 2
    fi
  done

  if ! $success; then
    # Stuck detection
    if [[ "${STUCK_RUN:-}" == "$run" ]]; then
      STUCK_COUNT=$(( STUCK_COUNT + 1 ))
    else
      STUCK_RUN="$run"
      STUCK_COUNT=1
    fi
    if [ "$STUCK_COUNT" -ge 3 ]; then
      echo "[FATAL] Run $run has failed $STUCK_COUNT times consecutively. Exiting."
      exit 1
    fi
    echo "[WARN] Run $run failed all $MAX_ATTEMPTS attempts — moving on."
    # In SLURM mode, exit 1 so the array retry can try again
    [ -n "${RUN_ID:-}" ] && exit 1
  else
    STUCK_RUN="$run"
    STUCK_COUNT=0
    run=$(( run + 1 ))
  fi
done

echo "[INFO] All runs complete."
exit 0

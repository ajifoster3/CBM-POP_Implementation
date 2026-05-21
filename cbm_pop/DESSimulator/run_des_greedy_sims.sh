#!/bin/bash
# run_des_greedy_sims.sh — run multiple greedy DES simulations (no ROS required).
#
# Usage:
#   ./run_des_greedy_sims.sh --agents 5 --problem-size 10 --num-runs 50

set -e
set -o pipefail

PYTHON="$(command -v python3)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

SIM_PID=""

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

MAX_ATTEMPTS=3
TIMEOUT_SECONDS=3600
RESULTS_ROOT="resources/run_logs"
mkdir -p "$RESULTS_ROOT"

PARSED=$(getopt \
  -o p:a: \
  -l problem-size:,agents:,problem-class:,problem-seed:,run-id:,num-runs:,start-run:,end-run:,\
speed:,max-sim-time:,progress-interval:,output-root:,\
enable-kill:,kill-threshold:,num-to-kill:,enable-revive:,revive-threshold: \
  -- "$@") || { echo "Invalid options"; exit 1; }
eval set -- "$PARSED"

while true; do
  case "$1" in
    -p|--problem-size)    PROBLEM_SIZE="$2";     shift 2 ;;
    -a|--agents)          NUM_AGENTS="$2";       shift 2 ;;
    --problem-class)      PROBLEM_CLASS="$2";    shift 2 ;;
    --problem-seed)       PROBLEM_SEED="$2";     shift 2 ;;
    --run-id)             RUN_ID="$2";           shift 2 ;;
    --num-runs)           NUM_RUNS="$2";         shift 2 ;;
    --start-run)          START_RUN_ARG="$2";    shift 2 ;;
    --end-run)            END_RUN_ARG="$2";      shift 2 ;;
    --speed)              SPEED="$2";            shift 2 ;;
    --max-sim-time)       MAX_SIM_TIME="$2";     shift 2 ;;
    --progress-interval)  PROGRESS_INTERVAL="$2"; shift 2 ;;
    --output-root)        RESULTS_ROOT="$2";     shift 2 ;;
    --enable-kill)        ENABLE_KILL="$2";      shift 2 ;;
    --kill-threshold)     KILL_THRESHOLD="$2";   shift 2 ;;
    --num-to-kill)        NUM_TO_KILL="$2";      shift 2 ;;
    --enable-revive)      ENABLE_REVIVE="$2";    shift 2 ;;
    --revive-threshold)   REVIVE_THRESHOLD="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "Unexpected option: $1"; exit 1 ;;
  esac
done

PROBLEM_SIZE=${PROBLEM_SIZE:-15}
NUM_AGENTS=${NUM_AGENTS:-5}
PROBLEM_CLASS=${PROBLEM_CLASS:-"Simple_Grid"}
PROBLEM_SEED=${PROBLEM_SEED:-}
RUN_ID=${RUN_ID:-${SLURM_ARRAY_TASK_ID:-}}
NUM_RUNS=${NUM_RUNS:-1}
START_RUN_ARG=${START_RUN_ARG:-}
END_RUN_ARG=${END_RUN_ARG:-}
SPEED=${SPEED:-1.0}
MAX_SIM_TIME=${MAX_SIM_TIME:-"inf"}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-0}
ENABLE_KILL=${ENABLE_KILL:-"false"}
KILL_THRESHOLD=${KILL_THRESHOLD:-0.2}
NUM_TO_KILL=${NUM_TO_KILL:-1}
ENABLE_REVIVE=${ENABLE_REVIVE:-"false"}
REVIVE_THRESHOLD=${REVIVE_THRESHOLD:-0.8}

require_bool() {
  local _name="$1" _value="$2"
  case "$_value" in
    true|false) ;;
    *)
      echo "Invalid boolean for ${_name}: ${_value} (expected true or false)" >&2
      exit 1
      ;;
  esac
}
require_bool "enable_kill"   "$ENABLE_KILL"
require_bool "enable_revive" "$ENABLE_REVIVE"

ENV_SUFFIX=""
[[ "$ENABLE_KILL"   == "true" ]] && ENV_SUFFIX="_kill_kth_${KILL_THRESHOLD}_ntk_${NUM_TO_KILL}"
[[ "$ENABLE_REVIVE" == "true" ]] && ENV_SUFFIX="${ENV_SUFFIX}_rev_rth_${REVIVE_THRESHOLD}"
ENV_DIR="$RESULTS_ROOT/size_${PROBLEM_SIZE}_agents_${NUM_AGENTS}_$(slug "$PROBLEM_CLASS")${ENV_SUFFIX}"
PARAM_DIR="$ENV_DIR/method_greedy_speed_${SPEED}"
mkdir -p "$PARAM_DIR"

write_param_tag() {
  local _dir="$1"
  {
    echo "simulator=DES_greedy"
    echo "method=greedy"
    echo "num_agents=${NUM_AGENTS}"
    echo "problem_size=${PROBLEM_SIZE}"
    echo "problem_class=${PROBLEM_CLASS}"
    echo "speed=${SPEED}"
    echo "max_sim_time=${MAX_SIM_TIME}"
    echo "enable_kill=${ENABLE_KILL}"
    echo "kill_threshold=${KILL_THRESHOLD}"
    echo "num_to_kill=${NUM_TO_KILL}"
    echo "enable_revive=${ENABLE_REVIVE}"
    echo "revive_threshold=${REVIVE_THRESHOLD}"
    echo "created_iso=$(date -Is)"
  } > "$_dir/setting_tag.txt"
}

write_run_settings() {
  local _cfg="$1" _seed="$2" _uid="$3"
  {
    echo "simulator=DES_greedy"
    echo "method=greedy"
    echo "num_agents=${NUM_AGENTS}"
    echo "problem_size=${PROBLEM_SIZE}"
    echo "problem_class=${PROBLEM_CLASS}"
    echo "problem_seed=${_seed}"
    echo "speed=${SPEED}"
    echo "max_sim_time=${MAX_SIM_TIME}"
    echo "enable_kill=${ENABLE_KILL}"
    echo "kill_threshold=${KILL_THRESHOLD}"
    echo "num_to_kill=${NUM_TO_KILL}"
    echo "enable_revive=${ENABLE_REVIVE}"
    echo "revive_threshold=${REVIVE_THRESHOLD}"
    echo "timeout_seconds=${TIMEOUT_SECONDS}"
    echo "run_uid=${_uid}"
    echo "start_iso=$(date -Is)"
  } > "$_cfg/run_settings.txt"
}

check_des_complete() {
  local _dir="$1"
  local _log="$_dir/des_sim.log"
  if [ -f "$_log" ] && grep -q '\[DES_GREEDY_EXIT\] reason=complete' "$_log"; then
    return 0
  fi
  local _csv="$_dir/coverage_log.csv"
  if [ -f "$_csv" ]; then
    local _last
    _last=$(tail -n 1 "$_csv" | awk -F',' '{print $4}' | tr -d '[:space:]')
    if [[ "$_last" == "1.0000" ]]; then
      return 0
    fi
    echo "Coverage incomplete in $_csv (last fraction=${_last})"
  else
    echo "No coverage_log.csv in $_dir"
  fi
  return 1
}

build_des_cmd() {
  local _seed="$1" _outdir="$2"
  local -a CMD=(
    "$PYTHON" -m cbm_pop.DESSimulator.run_des_greedy_sim
    --num_agents    "$NUM_AGENTS"
    --problem_size  "$PROBLEM_SIZE"
    --problem_class "$PROBLEM_CLASS"
    --problem_seed  "$_seed"
    --speed         "$SPEED"
    --output_dir    "$_outdir"
  )
  [[ "$MAX_SIM_TIME"      != "inf"   ]] && CMD+=( --max_sim_time "$MAX_SIM_TIME" )
  [[ "$PROGRESS_INTERVAL" != "0"     ]] && CMD+=( --progress_interval "$PROGRESS_INTERVAL" )
  [[ "$ENABLE_KILL"       == "true"  ]] && CMD+=( --enable_kill )
  [[ "$ENABLE_KILL"       == "true"  ]] && CMD+=( --kill_threshold  "$KILL_THRESHOLD" )
  [[ "$ENABLE_KILL"       == "true"  ]] && CMD+=( --num_to_kill     "$NUM_TO_KILL" )
  [[ "$ENABLE_REVIVE"     == "true"  ]] && CMD+=( --enable_revive )
  [[ "$ENABLE_REVIVE"     == "true"  ]] && CMD+=( --revive_threshold "$REVIVE_THRESHOLD" )
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

cleanup()    { kill_pid_tree "${SIM_PID:-}"; }
on_sigint()  { echo "[SIGINT] Cleaning up...";  cleanup; exit 130; }
on_sigterm() { echo "[SIGTERM] Cleaning up..."; cleanup; exit 143; }
on_exit()    { cleanup; }
trap on_sigint  INT
trap on_sigterm TERM
trap on_exit    EXIT

run_one() {
  local _run_id="$1" _seed="$2"
  local _cfg="$PARAM_DIR/run_${_run_id}"
  mkdir -p "$_cfg"

  write_run_settings "$_cfg" "$_seed" "$(unique_id)"

  local _log="$_cfg/des_sim.log"
  local -a _cmd
  read -r -a _cmd <<< "$(build_des_cmd "$_seed" "$_cfg")"

  echo "   [START] run_id=${_run_id} seed=${_seed} agents=${NUM_AGENTS} size=${PROBLEM_SIZE} class=${PROBLEM_CLASS}"

  PYTHONUNBUFFERED=1 "${_cmd[@]}" >> "$_log" 2>&1 &
  SIM_PID=$!

  local _start _last_hb=0
  _start=$(date +%s)

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

  echo "end_iso=$(date -Is)" >> "$_cfg/run_settings.txt"
  echo "   [SUCCESS] run_id=${_run_id} completed in ${_elapsed}s"
  return 0
}

write_param_tag "$PARAM_DIR"

find_last_run_in_range() {
  local lo="$1" hi="$2"
  RESUME_LAST_NUM=0
  RESUME_LAST_DIR=""
  local _d _n
  for _d in $(ls -d "$PARAM_DIR"/run_[0-9]* 2>/dev/null | grep -E '/run_[0-9]+$' | sort -V); do
    _n=$(basename "$_d" | sed 's/run_//')
    if [ "$_n" -ge "$lo" ] && [ "$_n" -le "$hi" ]; then
      RESUME_LAST_NUM=$_n
      RESUME_LAST_DIR=$_d
    fi
  done
}

if [ -n "${RUN_ID:-}" ]; then
  START_RUN=$RUN_ID
  END_RUN=$RUN_ID
elif [ -n "${START_RUN_ARG:-}" ]; then
  START_RUN=${START_RUN_ARG}
  END_RUN=${END_RUN_ARG}

  find_last_run_in_range "$START_RUN" "$END_RUN"
  if [ -n "$RESUME_LAST_DIR" ]; then
    if check_des_complete "$RESUME_LAST_DIR" 2>/dev/null; then
      START_RUN=$(( RESUME_LAST_NUM + 1 ))
    else
      echo "[RECOVER] Last run $RESUME_LAST_NUM (range ${START_RUN_ARG}-${END_RUN_ARG}) was incomplete — re-running."
      if mv "$RESUME_LAST_DIR" "${RESUME_LAST_DIR}_recovered_$(date +%s)" 2>/dev/null; then
        START_RUN=$RESUME_LAST_NUM
      else
        START_RUN=$RESUME_LAST_NUM
      fi
    fi
  fi

  if [ "$START_RUN" -gt "$END_RUN" ]; then
    echo "[INFO] All runs ${START_RUN_ARG}-${END_RUN_ARG} already complete in $PARAM_DIR."
    exit 0
  fi
else
  START_RUN=1
  END_RUN=$NUM_RUNS

  find_last_run_in_range "$START_RUN" "$END_RUN"
  if [ -n "$RESUME_LAST_DIR" ]; then
    if check_des_complete "$RESUME_LAST_DIR" 2>/dev/null; then
      START_RUN=$(( RESUME_LAST_NUM + 1 ))
    else
      echo "[RECOVER] Last run $RESUME_LAST_NUM was incomplete — re-running."
      mv "$RESUME_LAST_DIR" "${RESUME_LAST_DIR}_recovered_$(date +%s)"
      START_RUN=$RESUME_LAST_NUM
    fi
  fi

  if [ "$START_RUN" -gt "$END_RUN" ]; then
    echo "[INFO] All $NUM_RUNS runs already complete in $PARAM_DIR."
    exit 0
  fi
fi

STUCK_RUN=""
STUCK_COUNT=0

run=$START_RUN
while [ "$run" -le "$END_RUN" ]; do
  RUN_SEED="${PROBLEM_SEED:-$(gen_seed)}"
  echo "============================="
  echo "[INFO] Run ${run}/${END_RUN} | agents=${NUM_AGENTS} size=${PROBLEM_SIZE} class=${PROBLEM_CLASS} seed=${RUN_SEED} speed=${SPEED} kill=${ENABLE_KILL} ntk=${NUM_TO_KILL} revive=${ENABLE_REVIVE}"
  echo "============================="

  attempt=0
  success=false
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$(( attempt + 1 ))
    if run_one "$run" "$RUN_SEED"; then
      success=true
      break
    fi
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
    [ -n "${RUN_ID:-}" ] && exit 1
  else
    STUCK_RUN="$run"
    STUCK_COUNT=0
    run=$(( run + 1 ))
  fi
done

echo "[INFO] All runs complete."
exit 0
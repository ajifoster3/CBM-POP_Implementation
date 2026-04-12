#!/bin/bash
###############################################################################
#  gen_full_benchmark_des.sh
#
#  DES-simulator counterpart to gen_full_benchmark_v2.sh.
#  Generates sbatch files for the Q-CBM benchmarking suite using the
#  Discrete-Event Simulator (no ROS required).
#
#  Each sbatch job calls run_des_sims.sh with --num-runs N_REPEATS; the DES
#  script handles per-run retry / resume logic internally.
#
#  Experiment groups
#  -----------------
#  2  Scale agents        4 methods × 4 environments × {5..30} agents × 400 tasks
#  3a Scale tasks         4 methods × 3 environments × 5 agents × {25..400} tasks
#  3b Scale tasks (large) 4 methods × 3 environments × 15 agents × {400..1600} tasks
#  6  Learning dynamics   Q-CBM only × 4 environments × 10 agents × ~200 tasks
#  7  Full method × env   All 11 methods × all 9 environments × 10 agents × ~200 tasks
#
#  NOTE: Kill/revive experiments (4a/4b/4c/5) are omitted — the DES
#        simulator does not currently support robot-failure injection.
#
#  Problem-size mapping (for grid-based environments):
#    5→25  8→64  10→100  14→196≈200  20→400  28→784≈800  40→1600
###############################################################################

set -euo pipefail

# ── Cluster / container settings ──────────────────────────────────────────────
PARTITION="cpu_shared"
MEM_PER_TASK="60G"
TIME_LIMIT="72:00:00"
WS_ROOT="/scratch/p48/afoster2/ros_ws"
CONTAINER="/scratch/p48/afoster2/ros2_humble.sif"
DATE="10-04-2026"
N_REPEATS=10     # passed to run_des_sims.sh --num-runs; retry/resume handled internally

# ── Output directory ──────────────────────────────────────────────────────────
OUTDIR="sbatches_des_${DATE}"
mkdir -p "$OUTDIR"

# ── Default Q-CBM hyperparameters ─────────────────────────────────────────────
DEF_LR=0.22
DEF_POS=7.0
DEF_NEG=-8.0
DEF_GAMMA=0.95
DEF_RHO=0.5
DEF_ETA=0.0
DEF_INJECT_PROB=0.9

# ── All environments ──────────────────────────────────────────────────────────
ALL_ENVS=(
    "Simple_Grid"
    "Random_Spread"
    "Linear_Rows"
    "Density_Gradient"
    "Wind_Farm_Grid"
    "Disjoint_Regions"
    "Bottleneck_Corridor"
    "Concentric_Rings"
    "Hierarchical_Clusters"
)

###############################################################################
#  make_sbatch  –  emit one sbatch file
#
#  Usage: make_sbatch  EXPERIMENT  LABEL  JOB_NAME  PROBLEM_SIZE  AGENTS  \
#                      PROBLEM_CLASS  METHOD_ARGS
###############################################################################
make_sbatch() {
    local experiment="$1"
    local label="$2"
    local job_name="$3"
    local problem_size="$4"
    local agents="$5"
    local problem_class="$6"
    local args="$7"

    # DES runs one simulation at a time — a few extra CPUs suffice
    local cpus=$(( agents + 2 ))
    local sbatch_file="${OUTDIR}/${DATE}_exp${experiment}_${label}_${problem_class}_size_${problem_size}_agents_${agents}.sbatch"

    # Skip if already generated (dedup across experiment groups)
    [[ -f "$sbatch_file" ]] && return

    cat <<EOL > "$sbatch_file"
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -p ${PARTITION}
#SBATCH -t ${TIME_LIMIT}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${MEM_PER_TASK}
#SBATCH --hint=nomultithread
#SBATCH --open-mode=append
#SBATCH -o /scratch/p48/afoster2/sbatches/logs/slurm-%j.out
#SBATCH -e /scratch/p48/afoster2/sbatches/logs/slurm-%j.err

set -euo pipefail

WS_ROOT="${WS_ROOT}"
CONTAINER="${CONTAINER}"

export OMP_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="\${SLURM_CPUS_PER_TASK}"

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

chmod +x \${WS_ROOT}/src/CBM-POP_Implementation/cbm_pop/DESSimulator/run_des_sims.sh

singularity exec --cleanenv \\
  --bind "\$WS_ROOT:\$WS_ROOT" --bind "/scratch:/scratch" \\
  --env OMP_NUM_THREADS="\$OMP_NUM_THREADS" \\
  --env MKL_NUM_THREADS="\$MKL_NUM_THREADS" \\
  --env OPENBLAS_NUM_THREADS="\$OPENBLAS_NUM_THREADS" \\
  --env NUMEXPR_NUM_THREADS="\$NUMEXPR_NUM_THREADS" \\
  --env PYTHONUNBUFFERED="\$PYTHONUNBUFFERED" \\
  --env PYTHONFAULTHANDLER="\$PYTHONFAULTHANDLER" \\
  "\$CONTAINER" bash -lc '
    set -euo pipefail
    cd "${WS_ROOT}/src/CBM-POP_Implementation/cbm_pop/DESSimulator"
    ./run_des_sims.sh ${args} --num-runs ${N_REPEATS}
  '
EOL
    echo "Created: ${sbatch_file}"
}


###############################################################################
#  Method argument builders
#  (DES flags only — no --lock/--preserve/--use-ucb/--tau/--batch-size/etc.)
###############################################################################

args_qcbm() {
    echo "--method 'Q-Learning' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled true --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle true \
--is-append-first-task true"
}

args_uniform() {
    echo "--method 'Uniform' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled false --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle true \
--is-append-first-task true"
}

args_individual() {
    echo "--method 'Q-Learning' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled false --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle true \
--is-append-first-task true"
}

args_free_cycle() {
    echo "--method 'Q-Learning' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled true --is-knn-enabled true \
--is-free-weight-matrix true --is-inject-best-on-cycle true \
--is-append-first-task true"
}

args_knn_less() {
    echo "--method 'Q-Learning' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled true --is-knn-enabled false \
--is-free-weight-matrix false --is-inject-best-on-cycle true \
--is-append-first-task true"
}

args_greedy_less() {
    echo "--method 'Q-Learning' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob 0.0 \
--is-mimetism-enabled true --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle false \
--is-append-first-task true"
}

args_no_append() {
    echo "--method 'Q-Learning' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled true --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle true \
--is-append-first-task false"
}

args_cbm_pop() {
    echo "--method 'Ferreira_et_al.' \
--lr 0.0 --positive-reward 0.0 --negative-reward 0.0 \
--gamma-decay 0.0 --rho 0.5 --eta 20.0 \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob 0.0 \
--is-mimetism-enabled true --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle false \
--is-append-first-task true"
}

args_ucb() {
    echo "--method 'UCB' \
--lr 0.0 --positive-reward 0.0 --negative-reward 0.0 \
--gamma-decay 0.0 --rho 0.5 --eta ${DEF_ETA} \
--ucb-c 1.414 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled true --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle true \
--is-append-first-task true"
}

args_ql_step() {
    echo "--method 'Q-Learning-Step' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled true --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle true \
--is-append-first-task true"
}

args_ql_separate() {
    echo "--method 'Q-Learning-Separate' \
--lr ${DEF_LR} --positive-reward ${DEF_POS} --negative-reward ${DEF_NEG} \
--gamma-decay ${DEF_GAMMA} --rho ${DEF_RHO} --eta ${DEF_ETA} \
--ucb-c 0.0 --ucb-window 200 \
--inject-best-prob ${DEF_INJECT_PROB} \
--is-mimetism-enabled true --is-knn-enabled true \
--is-free-weight-matrix false --is-inject-best-on-cycle true \
--is-append-first-task true"
}


###############################################################################
#  Helper: emit one method across environments/sizes/agents
###############################################################################
emit_method() {
    local experiment="$1"; shift
    local label="$1";      shift
    local job_name="$1";   shift
    local args_func="$1";  shift
    local -n _envs=$1;     shift
    local -n _sizes=$1;    shift
    local -n _agents=$1;   shift

    local base_args
    base_args=$($args_func)

    for ps in "${_sizes[@]}"; do
      for ag in "${_agents[@]}"; do
        for env in "${_envs[@]}"; do
            local full_args="--problem-size ${ps} --agents ${ag} --problem-class '${env}' ${base_args}"
            make_sbatch "$experiment" "$label" "$job_name" "$ps" "$ag" "$env" "$full_args"
        done
      done
    done
}


###############################################################################
#  EXPERIMENT 2 — Scalability over agents (extended to 30)
###############################################################################
echo ""
echo "===== EXPERIMENT 2: Scalability over agents ====="

EXP2_ENVS=("Simple_Grid" "Random_Spread" "Density_Gradient" "Wind_Farm_Grid")
EXP2_SIZES=(20)
EXP2_AGENTS=(5 10 15 20 25 30 40 50 75 100)

emit_method "2" "qcbm"    "qcbm"   args_qcbm    EXP2_ENVS EXP2_SIZES EXP2_AGENTS
emit_method "2" "cbm_pop" "cbmpop" args_cbm_pop  EXP2_ENVS EXP2_SIZES EXP2_AGENTS
emit_method "2" "ucb"     "ucb"    args_ucb      EXP2_ENVS EXP2_SIZES EXP2_AGENTS
emit_method "2" "uniform" "unifrm" args_uniform  EXP2_ENVS EXP2_SIZES EXP2_AGENTS


###############################################################################
#  EXPERIMENT 3a — Scalability over tasks (5 agents)
###############################################################################
echo ""
echo "===== EXPERIMENT 3a: Scalability over tasks (5 agents) ====="

EXP3A_GRID_ENVS=("Simple_Grid" "Random_Spread")
EXP3A_WIND_ENVS=("Wind_Farm_Grid")
EXP3A_GRID_SIZES=(5 8 10 14 20)
EXP3A_WIND_SIZES=(10 14 20 25 30)
EXP3A_AGENTS=(5)

for METHOD_ARGS in "qcbm qcbm args_qcbm" "cbm_pop cbmpop args_cbm_pop" "ucb ucb args_ucb" "uniform unifrm args_uniform"; do
  read -r mname mshort margs <<< "$METHOD_ARGS"
  emit_method "3a" "$mname" "$mshort" "$margs" EXP3A_GRID_ENVS EXP3A_GRID_SIZES EXP3A_AGENTS
  emit_method "3a" "$mname" "$mshort" "$margs" EXP3A_WIND_ENVS EXP3A_WIND_SIZES EXP3A_AGENTS
done


###############################################################################
#  EXPERIMENT 3b — Scalability over tasks — large scale (15 agents)
###############################################################################
echo ""
echo "===== EXPERIMENT 3b: Scalability over tasks — large scale (15 agents) ====="

EXP3B_GRID_ENVS=("Simple_Grid" "Random_Spread")
EXP3B_WIND_ENVS=("Wind_Farm_Grid")
EXP3B_GRID_SIZES=(20 28 40)
EXP3B_WIND_SIZES=(30 40 50)
EXP3B_AGENTS=(15)

for METHOD_ARGS in "qcbm qcbm args_qcbm" "cbm_pop cbmpop args_cbm_pop" "ucb ucb args_ucb" "uniform unifrm args_uniform"; do
  read -r mname mshort margs <<< "$METHOD_ARGS"
  emit_method "3b" "$mname" "$mshort" "$margs" EXP3B_GRID_ENVS EXP3B_GRID_SIZES EXP3B_AGENTS
  emit_method "3b" "$mname" "$mshort" "$margs" EXP3B_WIND_ENVS EXP3B_WIND_SIZES EXP3B_AGENTS
done


###############################################################################
#  EXPERIMENT 6 — Learning dynamics (Q-CBM only)
###############################################################################
echo ""
echo "===== EXPERIMENT 6: Learning dynamics (Q-CBM only) ====="

EXP6_ENVS=("Simple_Grid" "Density_Gradient" "Bottleneck_Corridor" "Hierarchical_Clusters")
EXP6_SIZES=(14)
EXP6_AGENTS=(10)

emit_method "6" "qcbm" "qcbm" args_qcbm EXP6_ENVS EXP6_SIZES EXP6_AGENTS


###############################################################################
#  EXPERIMENT 7 — Full method × environment matrix
#
#  All 11 methods × all 9 environments × 10 agents × size 14 (~200 tasks).
#  Ablation study is a slice of this matrix: report {Simple_Grid,
#  Random_Spread, Hierarchical_Clusters} for the ablation table in the paper.
###############################################################################
echo ""
echo "===== EXPERIMENT 7: Full method × environment matrix ====="

EXP7_SIZES=(14)
EXP7_AGENTS=(10)

emit_method "7" "qcbm"          "qcbm"    args_qcbm          ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "uniform"        "unifrm"  args_uniform        ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "individual"     "individ" args_individual     ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "free_cycle"     "freecyc" args_free_cycle     ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "knn_less"       "knnles"  args_knn_less       ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "greedy_less"    "grdyls"  args_greedy_less    ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "no_append"      "noappd"  args_no_append      ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "cbm_pop"        "cbmpop"  args_cbm_pop        ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "ucb"            "ucb"     args_ucb            ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "ql_step"        "qlstep"  args_ql_step        ALL_ENVS EXP7_SIZES EXP7_AGENTS
emit_method "7" "ql_separate"    "qlsep"   args_ql_separate    ALL_ENVS EXP7_SIZES EXP7_AGENTS


###############################################################################
#  Summary
###############################################################################
echo ""
echo "==========================================="
TOTAL=$(find "$OUTDIR" -name '*.sbatch' | wc -l)
echo "  Total sbatch files generated: ${TOTAL}"
echo "  Output directory: ${OUTDIR}/"
echo "  Each job runs ${N_REPEATS} iterations via run_des_sims.sh --num-runs"
echo "  Retry/resume handled internally by run_des_sims.sh"
echo "==========================================="
echo ""
echo "To submit all jobs:"
echo "  for f in ${OUTDIR}/*.sbatch; do sbatch \"\$f\"; done"
echo ""
echo "To submit a specific experiment (e.g. Experiment 2):"
echo "  for f in ${OUTDIR}/${DATE}_exp2_*.sbatch; do sbatch \"\$f\"; done"
echo ""
echo "To count jobs per experiment:"
echo "  for e in 2 3a 3b 6 7; do"
echo "    n=\$(ls ${OUTDIR}/${DATE}_exp\${e}_*.sbatch 2>/dev/null | wc -l)"
echo "    echo \"  Experiment \${e}: \${n} jobs\""
echo "  done"
echo ""
echo "Paper notes:"
echo "  - Ablation table:  filter Exp 7 to {Simple_Grid, Random_Spread, Hierarchical_Clusters}"
echo "  - Kill/revive (4a/4b/4c/5): not available in DES simulator"

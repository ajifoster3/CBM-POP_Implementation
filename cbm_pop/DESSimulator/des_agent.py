"""
DES-compatible CBM-POP agent.

All ROS infrastructure is removed.  The public interface is:

  initialise(robot_poses)               — call once before the DES loop
  compute_step(robot_poses) -> StepData — runs one operator, measures wall time
  apply_step_result(step_data) -> bool  — apply result when its event fires
  receive_coalition_best(sol, sender)   — ingest a peer's better solution
  handle_task_covered(task_id)          — react to coverage events
"""

import random
import time as _wall
import traceback
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

import numpy as np

from cbm_pop.Condition import ConditionFunctions
from cbm_pop.Operator import Operator
from cbm_pop.Operator_Fuctions import OperatorFunctions
from cbm_pop.SimpleSimulator.simple_fitness import SimpleFitness
from cbm_pop.SimpleSimulator.simple_problem import SimpleProblem
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.ucb_bandit import UCBBandit


class LearningMethod(Enum):
    Q_LEARNING          = 'Q-Learning'
    Q_LEARNING_STEP     = 'Q-Learning-Step'
    Q_LEARNING_SEPARATE = 'Q-Learning-Separate'
    Q_LEARNING_GAIN          = 'Q-Learning-improveoncurrent'            # reward = improvement over current solution (per-step, moving target)
    Q_LEARNING_STEP_GAIN     = 'Q-Learning-Step-improveoncurrent'       # reward = improvement over DI-cycle start (per-step, fixed baseline)
    Q_LEARNING_SEPARATE_GAIN = 'Q-Learning-Separate-improveoncurrent'   # Separate variant with immediate-current reward
    FERREIRA            = 'Ferreira_et_al.'
    UCB                 = 'UCB'
    UNIFORM             = 'Uniform'


@dataclass
class StepData:
    """Carries the output of one compute_step() call through the event queue."""
    result:    Any            # candidate solution tuple, or None
    operator:  Operator
    condition: int
    wall_time: float          # measured operator wall time (seconds)


class DESAgent:
    def __init__(
        self,
        agent_id:              int,
        num_agents:            int,
        problem:               SimpleProblem,
        pop_size:              int   = 10,
        di_cycle_length:       int   = 10,
        num_solution_attempts: int   = 21,
        lr:                    float = 0.22,
        gamma_decay:           float = 0.95,
        positive_reward:       float = 7.0,
        negative_reward:       float = -8.0,
        rho:                   float = 0.5,
        eta:                   float = 0.1,
        is_knn_enabled:        bool  = False,
        is_mimetism_enabled:   bool  = True,
        is_append_first_task:  bool  = True,
        is_inject_best_on_cycle: bool  = False,
        inject_best_prob:      float = 0.9,
        is_free_weight_matrix:    bool  = False,
        initialise_with_heuristic: bool = True,
        init_method:           str   = 'voronoi',
        method:                str   = 'Q-Learning',
        ucb_c:                 float = 1.414,
        ucb_window:            int   = 200,
        time_discount:         bool  = False,
        time_discount_lambda:  float = 0.1,
        is_relative_reward:    bool  = False,
        reward_ema_alpha:      float = 0.05,
        logger=None,
    ):
        self.agent_id   = agent_id
        self.num_agents = num_agents
        self.problem    = problem
        self._logger    = logger
        self.eta        = eta

        self.pop_size              = pop_size
        self.di_cycle_length       = di_cycle_length
        self.no_improvement_attempts = num_solution_attempts
        self.lr                    = lr
        self.gamma_decay           = gamma_decay
        self.positive_reward       = positive_reward
        self.negative_reward       = negative_reward
        self.rho                   = rho
        self.is_mimetism_enabled   = is_mimetism_enabled
        self.is_append_first_task  = is_append_first_task
        self.is_inject_best_on_cycle      = is_inject_best_on_cycle
        self.inject_best_prob             = inject_best_prob
        self.initialise_with_heuristic    = initialise_with_heuristic
        self.init_method                  = init_method

        if is_knn_enabled:
            self.intensifiers = [
                Operator.ONE_MOVE,
                Operator.TWO_SWAP,
                Operator.TWO_OPT_INTRA,
                Operator.NEAREST_K_RELOCATION,
            ]
        else:
            self.intensifiers = [
                Operator.ONE_MOVE,
                Operator.TWO_SWAP,
                Operator.TWO_OPT_INTRA,
            ]
        self.diversifiers = [
            Operator.BEST_COST_ROUTE_CROSSOVER,
            Operator.INTRA_DEPOT_REMOVAL,
            Operator.INTRA_DEPOT_SWAPPING,
            Operator.SINGLE_ACTION_REROUTING,
        ]
        self.weight_matrix = WeightMatrix(
            len(self.intensifiers), len(self.diversifiers), is_free_weight_matrix
        )
        self.operator_admissibility = self._build_classical_admissibility_matrix()

        try:
            self.learning_method = LearningMethod(method)
        except ValueError:
            raise ValueError(
                f"Unknown method '{method}'. Valid options: "
                f"{[m.value for m in LearningMethod]}"
            )

        self.ucb_bandit = (
            UCBBandit(len(self.intensifiers) + len(self.diversifiers), ucb_c, ucb_window)
            if self.learning_method == LearningMethod.UCB
            else None
        )

        self.time_discount        = time_discount
        self.time_discount_lambda = time_discount_lambda
        self.is_relative_reward   = is_relative_reward
        self._reward_ema_alpha    = reward_ema_alpha
        # Per-step EMA (intensifiers)
        self._reward_ema_mean     = 0.0   # EMA of improvement (-gain)
        self._reward_ema_var      = 1.0   # EMA variance (initialised to 1 to avoid early div-by-zero)
        self._reward_ema_n        = 0     # number of updates so far
        # Per-cycle EMA (diversifiers) — updated once per DI cycle with total cycle gain
        self._div_ema_mean        = 0.0
        self._div_ema_var         = 1.0
        self._div_ema_n           = 0

        # Solution state
        self.population:              Optional[list]          = None
        self.current_solution:        Optional[tuple]         = None
        self.local_best_solution:     Optional[tuple]         = None
        self.coalition_best_solution: Optional[tuple]         = None
        self.coalition_best_agent:    Optional[int]           = None

        # Liveness
        self.is_alive:      bool       = True
        self.failed_agents: List[bool] = [False] * num_agents

        # Coverage
        self.is_covered: List[bool] = [False] * problem.num_tasks

        # Robot poses (this agent tracks all robots)
        self.robot_poses:         List[Optional[Tuple]] = [None] * num_agents
        self.initial_robot_poses: List[Optional[Tuple]] = [None] * num_agents

        # Task assignment
        self.current_task:        Optional[int]  = None
        self._last_raw_candidate: Any            = object()
        self._uniq_hist:          deque          = deque(maxlen=3)
        self.is_task_locked:      bool           = False
        self.locked_task:         Optional[int]  = None
        self.osc_count:           int            = 0
        self.max_osc_before_lock: int            = 5

        # Learning
        self.previous_experience:      list = []
        self.received_weight_matrices: list = []
        self.di_cycle_start_solution:  Optional[tuple] = None
        self.di_cycle_count:           int  = 0
        self.di_cycle_total:           int  = 0
        self.iteration_count:          int  = 0
        self.no_improvement_attempt_count: int = 0
        self.best_local_improved:        bool = False
        self.best_coalition_improved:    bool = False
        self.best_cycle_start_improved:  bool = False
        self.current_parent_idx:       Optional[int] = None
        self._current_sim_time:        float = 0.0

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def initialise(self, robot_poses: List[Tuple[float, float]]) -> None:
        """Call once with all starting positions before the DES loop."""
        for i, pos in enumerate(robot_poses):
            self.robot_poses[i]         = pos
            self.initial_robot_poses[i] = pos

        self.problem.initialize_robot_initial_pose_cost_matrix(self.initial_robot_poses)
        self.problem.update_robot_cost_matrix(self.robot_poses)

        self.population = self._generate_population()
        self.current_parent_idx, self.current_solution = self._select_solution()
        self.coalition_best_solution = deepcopy(self.current_solution)
        self._assign_next_task(self.coalition_best_solution)

    # ------------------------------------------------------------------ #
    # Core DES interface                                                   #
    # ------------------------------------------------------------------ #

    def compute_step(self, robot_poses: List[Tuple[float, float]]) -> StepData:
        """
        Run one optimization step and measure its wall time.

        Called at the moment the *previous* event fires so that the measured
        wall time can be used to schedule this step's completion event in the
        future.  The result is NOT applied here; that happens in
        apply_step_result() when the event fires.
        """
        for i, pos in enumerate(robot_poses):
            if pos is not None:
                self.robot_poses[i] = pos
        self.problem.update_robot_cost_matrix(self.robot_poses)

        self._strip_covered_solutions()

        # Re-seed population if empty or if current solution is missing uncovered tasks
        num_uncovered = sum(1 for c in self.is_covered if not c)
        _sol_incomplete = (
            num_uncovered > 0
            and self.current_solution
            and self.current_solution[0]
            and len(self.current_solution[0]) < num_uncovered
        )
        if not self.current_solution or not self.current_solution[0] or _sol_incomplete:
            self.population = self._generate_population()
            self.current_parent_idx, self.current_solution = self._select_solution()

        # Random restart on stagnation
        if self.no_improvement_attempt_count >= self.no_improvement_attempts:
            self.current_parent_idx, self.current_solution = self._select_random_solution()
            self.no_improvement_attempt_count = 0

        if self.di_cycle_count == 0 and self.di_cycle_start_solution is None:
            self.di_cycle_start_solution = deepcopy(self.current_solution)

        condition = ConditionFunctions.perceive_condition_row(
            self.previous_experience, self.intensifiers, self.diversifiers
        )
        enabled = self.intensifiers + self.diversifiers
        if self.learning_method == LearningMethod.UCB:
            admissible = self._admissible_operator_indices(condition)
            operator = enabled[self.ucb_bandit.select(admissible)]
        else:
            operator = OperatorFunctions.choose_operator(
                self.weight_matrix.weights, condition, enabled
            )

        t0        = _wall.process_time()
        result    = self._apply_operator(operator)
        wall_time = max(_wall.process_time() - t0, 1e-9)

        return StepData(result=result, operator=operator,
                        condition=condition, wall_time=wall_time)

    def apply_step_result(self, step: StepData, sim_time: float = 0.0) -> bool:
        """
        Apply a previously computed StepData to this agent's state.
        Called when the OPERATOR_COMPLETE event fires (sim_time = start + wall_time).
        Returns True if the coalition best was improved.
        """
        self._current_sim_time = sim_time
        self.iteration_count += 1

        if step is None or step.result is None:
            self.no_improvement_attempt_count += 1
            self._update_reward_ema(0.0)   # zero improvement for null result
            return False

        result = self._remove_covered(step.result)
        if not result or not result[0]:
            self.no_improvement_attempt_count += 1
            self._update_reward_ema(0.0)   # zero improvement for empty result
            return False

        num_uncovered = sum(1 for c in self.is_covered if not c)
        new_f         = self.fitness(result)
        cur_f         = self.fitness(self.current_solution)
        if self.di_cycle_start_solution is None:
            self.di_cycle_start_solution = deepcopy(self.current_solution)
        cycle_start_f = self.fitness(self.di_cycle_start_solution)
        loc_f         = self.fitness(self.local_best_solution)
        coal_f        = self.fitness(self.coalition_best_solution)

        gain       = new_f - cur_f
        cycle_gain = new_f - cycle_start_f
        self._update_reward_ema(gain)
        op_idx  = self._op_index(step.operator)
        self.previous_experience.append([step.condition, op_idx, gain])

        step_improved_local       = (loc_f == float('inf')) or (new_f < loc_f)
        step_improved_current     = gain < 0
        step_improved_cycle_start = cycle_gain < 0

        # Step-level learning
        if self.learning_method == LearningMethod.Q_LEARNING_STEP:
            self._learning_step(step.condition, op_idx, step_improved_local, gain, step.wall_time)
        elif (self.learning_method == LearningMethod.Q_LEARNING_SEPARATE
              and step.operator in self.intensifiers):
            self._learning_step(step.condition, op_idx, step_improved_current, gain, step.wall_time)
        elif self.learning_method == LearningMethod.Q_LEARNING_GAIN:
            self._learning_step(step.condition, op_idx, step_improved_current, gain, step.wall_time)
        elif self.learning_method == LearningMethod.Q_LEARNING_STEP_GAIN:
            self._learning_step(step.condition, op_idx, step_improved_current, gain, step.wall_time)
        elif (self.learning_method == LearningMethod.Q_LEARNING_SEPARATE_GAIN
              and step.operator in self.intensifiers):
            self._learning_step(step.condition, op_idx, step_improved_current, gain, step.wall_time)
        elif self.learning_method == LearningMethod.UCB:
            self.ucb_bandit.update(op_idx, self._apply_time_discount(-gain, gain, step.wall_time))
        if step_improved_cycle_start:
            self.best_cycle_start_improved = True
        if step_improved_local:
            self.local_best_solution  = deepcopy(result)
            self.best_local_improved  = True
        if step_improved_current:
            self.no_improvement_attempt_count = 0
        else:
            self.no_improvement_attempt_count += 1

        # A complete result always beats an incomplete coalition_best.
        # The fitness function has no penalty for missing tasks, so an incomplete
        # solution can have artificially low fitness and block a complete one.
        _coal_incomplete = (
            num_uncovered > 0
            and self.coalition_best_solution is not None
            and len(self.coalition_best_solution[0]) < num_uncovered
        )
        _result_complete = (num_uncovered == 0 or len(result[0]) >= num_uncovered)
        coalition_improved = (
            (coal_f == float('inf'))
            or (new_f < coal_f)
            or (_coal_incomplete and _result_complete)
        )
        if coalition_improved:
            self.coalition_best_solution = deepcopy(result)
            self.coalition_best_agent    = self.agent_id
            self.best_coalition_improved = True
            self._assign_next_task(result)

        self.current_solution = result
        self._reinsert_child(result)
        self.di_cycle_count += 1

        weights_to_share = None
        if self.di_cycle_count >= self.di_cycle_length or self._check_stagnation():
            weights_to_share = self._finish_di_cycle()

        return coalition_improved, weights_to_share

    def receive_coalition_best(
        self,
        solution,
        sender_id: int,
        sender_weights: Optional[list] = None,
    ) -> tuple:
        """
        Ingest a coalition-best from a peer.
        Returns (coalition_improved, prepend_adopted) where:
          coalition_improved — True if our coalition best was updated.
          prepend_adopted    — True if the prepend-first-task modification was
                               responsible for making the candidate strictly
                               better than the received solution (and should
                               therefore be re-broadcast to peers).
        """
        if solution is None:
            return False, False

        candidate = self._remove_covered(deepcopy(solution))
        if not candidate or not candidate[0]:
            return False, False

        prepend_adopted = False
        if self.is_append_first_task and self.current_task is not None:
            modified = self._prepend_current_task(candidate)
            if modified is not None and self.fitness(modified) < self.fitness(candidate):
                candidate = modified
                prepend_adopted = True

        num_uncovered = sum(1 for c in self.is_covered if not c)
        _coal_incomplete = (
            num_uncovered > 0
            and self.coalition_best_solution is not None
            and len(self.coalition_best_solution[0]) < num_uncovered
        )
        _candidate_complete = (num_uncovered == 0 or len(candidate[0]) >= num_uncovered)

        if (self.fitness(candidate) < self.fitness(self.coalition_best_solution)
                or (_coal_incomplete and _candidate_complete)):
            self.coalition_best_solution = candidate
            self.coalition_best_agent    = sender_id
            self._assign_next_task(candidate)
            if sender_weights is not None:
                self.received_weight_matrices.append(sender_weights)
            return True, prepend_adopted

        if sender_weights is not None:
            self.received_weight_matrices.append(sender_weights)
        return False, False

    def handle_task_covered(self, task_id: int) -> None:
        """React to a coverage event from any robot."""
        if self.is_covered[task_id]:
            return
        self.is_covered[task_id] = True

        if self.is_task_locked and self.locked_task == task_id:
            self._reset_task_lock_state()

        self._strip_covered_solutions()
        self._assign_next_task(self.coalition_best_solution)

    def kill_robot(self, robot_id: int) -> None:
        """
        React to robot_id failing.

        If this agent IS the failed robot: marks itself inactive so the DES
        loop stops scheduling further operator events for it.

        Otherwise: redistributes the failed robot's tasks into this agent's
        own segment across all solutions (mirrors SimpleSimulator purge_agent).
        """
        self.failed_agents[robot_id] = True
        if robot_id == self.agent_id:
            self.is_alive = False
            return

        def _purge(solution):
            if solution is None or not solution[0]:
                return solution
            order, alloc = list(solution[0]), list(solution[1])
            if robot_id >= len(alloc) or alloc[robot_id] == 0:
                return (order, alloc)
            start = sum(alloc[:robot_id])
            end   = start + alloc[robot_id]
            purged_tasks = order[start:end]
            del order[start:end]
            alloc[robot_id] = 0
            # my_start is computed AFTER zeroing alloc[robot_id] so the index
            # is correct regardless of whether agent_id < or > robot_id.
            my_start = sum(alloc[:self.agent_id])
            if my_start > len(order):
                my_start = len(order)
            order[my_start:my_start] = purged_tasks
            alloc[self.agent_id] += len(purged_tasks)
            return (order, alloc)

        self.coalition_best_solution = _purge(self.coalition_best_solution)
        self.current_solution        = _purge(self.current_solution)
        self.local_best_solution     = _purge(self.local_best_solution)
        self.di_cycle_start_solution = _purge(self.di_cycle_start_solution)
        if self.population:
            self.population = [_purge(s) for s in self.population]
        self._assign_next_task(self.coalition_best_solution)

    def revive_robot(self, robot_id: int) -> None:
        """
        React to robot_id being revived.

        The failed robot's alloc slot still exists (at 0) in surviving agents'
        solutions, so no structural change is needed for them.  If this is the
        revived agent itself, clear any stale pre-failure assignment from its
        own local solution state so it waits until optimisation assigns it new
        work after revival.
        """
        self.failed_agents[robot_id] = False
        if robot_id == self.agent_id:
            self.is_alive = True
            self._clear_self_assignment_after_revive()
        self._assign_next_task(self.coalition_best_solution)

    def reinitialise_population(self, seed_solution=None,
                                committed_positions=None,
                                committed_times=None) -> None:
        """
        Rebuild the population after a fleet change.

        If seed_solution is provided (e.g. a repaired coalition-best), the
        population is generated as perturbations of that seed so that
        optimization quality earned before the event is preserved.
        Otherwise a fresh heuristic population is generated from scratch.

        committed_positions : dict {robot_id -> (x, y)} — where each surviving
            robot will *be* when it finishes its current committed travel leg.
            Used by the greedy init instead of the mid-travel interpolated pose.
        committed_times : dict {robot_id -> float} — remaining travel time for
            each surviving robot (0.0 if already idle).

        Preserves learned weights (weight matrix / UCB bandit).
        """
        if seed_solution is not None and seed_solution[0]:
            self.population = self._generate_population_from_seed(seed_solution)
            self.coalition_best_solution = deepcopy(seed_solution)
        else:
            self.population = self._generate_population(
                committed_positions=committed_positions,
                committed_times=committed_times,
            )
            _, best = self._select_solution()
            self.coalition_best_solution = deepcopy(best)
        self.current_parent_idx, self.current_solution = self._select_solution()
        self.local_best_solution = None
        self.di_cycle_start_solution = None
        self.di_cycle_count = 0
        self.no_improvement_attempt_count = 0
        self._reset_task_lock_state()
        self.current_task = None
        self._assign_next_task(self.coalition_best_solution)

    def _generate_population_from_seed(self, base: tuple) -> list:
        """Build a population of perturbations around a given base solution."""
        population = [deepcopy(base)]
        num_tasks = len(base[0])
        for _ in range(1, self.pop_size):
            num_swaps = random.randint(1, max(1, num_tasks // 4))
            population.append(self._perturb_solution(base, num_swaps))
        return population

    def _clear_self_assignment_after_revive(self) -> None:
        recipient_id = next(
            (idx for idx, failed in enumerate(self.failed_agents)
             if idx != self.agent_id and not failed),
            None,
        )

        def _clear(solution):
            if solution is None or not solution[0]:
                return solution
            order, alloc = list(solution[0]), list(solution[1])
            if self.agent_id >= len(alloc) or alloc[self.agent_id] == 0:
                return (order, alloc)

            start = sum(alloc[:self.agent_id])
            end = start + alloc[self.agent_id]
            stale_tasks = order[start:end]
            del order[start:end]
            alloc[self.agent_id] = 0

            if recipient_id is not None and recipient_id < len(alloc):
                dest_idx = sum(alloc[:recipient_id])
                if dest_idx > len(order):
                    dest_idx = len(order)
                order[dest_idx:dest_idx] = stale_tasks
                alloc[recipient_id] += len(stale_tasks)

            return (order, alloc)

        self.coalition_best_solution = _clear(self.coalition_best_solution)
        self.current_solution        = _clear(self.current_solution)
        self.local_best_solution     = _clear(self.local_best_solution)
        self.di_cycle_start_solution = _clear(self.di_cycle_start_solution)
        if self.population:
            self.population = [_clear(s) for s in self.population]
        self._reset_task_lock_state()
        self.current_task = None

    # ------------------------------------------------------------------ #
    # Task assignment                                                      #
    # ------------------------------------------------------------------ #

    def _assign_next_task(self, solution) -> None:
        if solution is None or not isinstance(solution, (tuple, list)):
            self.current_task = None
            return

        ordered, alloc = solution

        if self.is_task_locked:
            t = self.locked_task
            if t is not None and 0 <= t < len(self.is_covered) and not self.is_covered[t]:
                self.current_task = t
                return
            self._reset_task_lock_state()

        if self.agent_id >= len(alloc) or alloc[self.agent_id] <= 0:
            candidate = None
        else:
            start = sum(alloc[:self.agent_id])
            end   = start + alloc[self.agent_id]
            candidate = next(
                (int(t) for t in ordered[start:end]
                 if 0 <= int(t) < len(self.is_covered) and not self.is_covered[int(t)]),
                None,
            )

        # ABA / None-B-None oscillation detection
        if candidate != self._last_raw_candidate:
            self._last_raw_candidate = candidate
            self._uniq_hist.append(candidate)

        if len(self._uniq_hist) == 3:
            A, B, C = self._uniq_hist
            is_aba        = (A == C) and (A != B)
            is_none_b_none = (A is None) and (C is None) and (B is not None)

            if is_aba or is_none_b_none:
                self.osc_count += 1
                lock_target = B if is_none_b_none else C
                if self.osc_count >= self.max_osc_before_lock and not self.is_task_locked:
                    self.is_task_locked = True
                    self.locked_task    = int(lock_target)
                    self.current_task   = int(lock_target)
                    return
            else:
                self.osc_count = 0

        self.current_task = candidate

    def _reset_task_lock_state(self) -> None:
        self.is_task_locked       = False
        self.locked_task          = None
        self.osc_count            = 0
        self._uniq_hist           = deque(maxlen=3)
        self._last_raw_candidate  = object()

    # ------------------------------------------------------------------ #
    # Population helpers                                                   #
    # ------------------------------------------------------------------ #

    def _voronoi_partition(self) -> list:
        tasks_per_agent = [[] for _ in range(self.num_agents)]
        alive = [a for a in range(self.num_agents)
                 if not self.failed_agents[a] and self.robot_poses[a] is not None]
        if not alive:
            return tasks_per_agent
        robot_xy = np.array([self.robot_poses[a] for a in alive], dtype=float)
        task_xy  = np.array(self.problem.task_poses, dtype=float)
        dists    = ((task_xy[:, None, :] - robot_xy[None, :, :]) ** 2).sum(axis=2)
        nearest  = np.argmin(dists, axis=1)
        for t, idx in enumerate(nearest):
            if not self.is_covered[t]:
                tasks_per_agent[alive[idx]].append(t)
        return tasks_per_agent

    def _nn_order(self, tasks: list, start_xy: tuple) -> list:
        if not tasks:
            return []
        pts       = np.array([self.problem.task_poses[t] for t in tasks], dtype=float)
        start_d   = np.sum((pts - np.array(start_xy)) ** 2, axis=1)
        remaining = list(range(len(tasks)))
        order     = [remaining.pop(int(np.argmin(start_d)))]
        cur       = order[0]
        while remaining:
            d   = np.sum((pts[remaining] - pts[cur]) ** 2, axis=1)
            k   = int(np.argmin(d))
            order.append(remaining.pop(k))
            cur = order[-1]
        return [tasks[i] for i in order]

    def _ensure_min_one(self, per_agent_routes: list) -> None:
        needers = [a for a, r in enumerate(per_agent_routes)
                   if not r and not self.failed_agents[a] and self.robot_poses[a] is not None]
        donors  = {i for i, r in enumerate(per_agent_routes) if len(r) > 1}
        for a in needers:
            if not donors:
                break
            src   = max(donors, key=lambda i: len(per_agent_routes[i]))
            route = per_agent_routes[src]
            dest  = np.array(self.robot_poses[a])
            k, _  = min(
                enumerate(route),
                key=lambda it: np.sum((np.array(self.problem.task_poses[it[1]]) - dest) ** 2),
            )
            per_agent_routes[a].append(route.pop(k))
            if len(per_agent_routes[src]) <= 1:
                donors.discard(src)

    def _generate_population(self, committed_positions=None, committed_times=None) -> list:
        """Dispatch to heuristic or random initialisation based on agent config."""
        if self.init_method == 'greedy':
            return self._generate_population_greedy(
                committed_positions=committed_positions,
                committed_times=committed_times,
            )
        if self.init_method == 'random' or not self.initialise_with_heuristic:
            return self._generate_population_random()
        return self._generate_population_voronoi()

    def _generate_population_random(self) -> list:
        """Randomly assign and order uncovered tasks — no Voronoi bias."""
        uncovered = [t for t, c in enumerate(self.is_covered) if not c]
        alive = [a for a in range(self.num_agents) if not self.failed_agents[a]]
        if not alive:
            return []
        population = []
        for _ in range(self.pop_size):
            tasks = uncovered[:]
            random.shuffle(tasks)
            assignment = [random.choice(alive) for _ in tasks]
            order, alloc = [], [0] * self.num_agents
            for agent in range(self.num_agents):
                agent_tasks = [t for t, a in zip(tasks, assignment) if a == agent]
                order.extend(agent_tasks)
                alloc[agent] = len(agent_tasks)
            population.append((order, alloc))
        return population

    def _perturb_solution(self, solution: tuple, num_swaps: int) -> tuple:
        """Apply random intra- or inter-route moves to diversify a solution."""
        order, alloc = list(solution[0]), list(solution[1])
        for _ in range(num_swaps):
            if random.random() < 0.5:
                # Intra-route swap: swap two tasks within one agent's route
                eligible = [a for a in range(self.num_agents) if alloc[a] >= 2]
                if eligible:
                    a     = random.choice(eligible)
                    start = sum(alloc[:a])
                    end   = start + alloc[a]
                    i, j  = random.sample(range(start, end), 2)
                    order[i], order[j] = order[j], order[i]
            else:
                # Inter-route move: relocate one task from a donor to a recipient
                donors = [a for a in range(self.num_agents) if alloc[a] >= 1]
                if len(donors) >= 1 and self.num_agents >= 2:
                    src      = random.choice(donors)
                    dst      = random.choice([a for a in range(self.num_agents) if a != src])
                    src_start = sum(alloc[:src])
                    src_end   = src_start + alloc[src]
                    pick_idx  = random.randrange(src_start, src_end)
                    task      = order.pop(pick_idx)
                    alloc[src] -= 1
                    dst_start = sum(alloc[:dst])
                    dst_end   = dst_start + alloc[dst]
                    order.insert(random.randint(dst_start, dst_end), task)
                    alloc[dst] += 1
        return (order, alloc)

    def _generate_population_voronoi(self) -> list:
        population = []
        # Build the base Voronoi + nearest-neighbour solution once.
        tpa    = self._voronoi_partition()
        routes = [
            self._nn_order([t for t in tpa[a] if not self.is_covered[t]],
                           self.robot_poses[a])
            for a in range(self.num_agents)
        ]
        self._ensure_min_one(routes)
        base_order = [t for r in routes for t in r]
        base_alloc = [len(r) for r in routes]
        base = (base_order, base_alloc)

        # First member is the clean heuristic solution; the rest are perturbed clones.
        population.append(base)
        num_tasks = len(base_order)
        for i in range(1, self.pop_size):
            num_swaps = random.randint(1, max(1, num_tasks // 4))
            population.append(self._perturb_solution(base, num_swaps))
        return population

    def _generate_population_greedy(self, committed_positions=None,
                                     committed_times=None) -> list:
        """
        Time-aware greedy construction: at each step assign the (robot, task) pair
        with the earliest simulated arrival time.

        Each robot tracks when it finishes its last task (time_available) and its
        position at that point.  The arrival time for a candidate pair is:
            time_available[robot] + dist(robot_pos, task_pos)
        (distance is a valid time surrogate because all robots share the same speed).

        This mirrors what the online greedy DES does — a robot that already has a
        long chain of assignments will not keep stealing nearby tasks from an idle
        robot that can reach them sooner.

        committed_positions : dict {robot_id -> (x, y)} — where each surviving
            robot will be once it finishes its current committed travel leg.
            Overrides the mid-travel interpolated pose for that robot.
        committed_times : dict {robot_id -> float} — remaining travel time before
            the robot becomes free.  Overrides the default 0.0 for that robot.
        """
        import math

        uncovered = [t for t, c in enumerate(self.is_covered) if not c]
        if not uncovered:
            return self._generate_population_random()

        alive_agents = [a for a in range(self.num_agents) if not self.failed_agents[a]]
        if not alive_agents:
            return self._generate_population_random()

        robot_positions = [list(p) if p is not None else [0.0, 0.0]
                           for p in self.robot_poses]
        time_available = [0.0] * self.num_agents

        # Override with committed travel state when provided.
        # A robot mid-travel will not be free at its current (interpolated)
        # position — it will be free at its goal once the remaining leg finishes.
        if committed_positions:
            for rid, pos in committed_positions.items():
                if 0 <= rid < len(robot_positions) and pos is not None:
                    robot_positions[rid] = list(pos)
        if committed_times:
            for rid, t in committed_times.items():
                if 0 <= rid < len(time_available):
                    time_available[rid] = float(t)
        per_agent_routes: list = [[] for _ in range(self.num_agents)]
        unassigned: set = set(uncovered)

        while unassigned:
            best_arrival = float('inf')
            best_robot   = None
            best_task    = None
            for a in alive_agents:
                pos = robot_positions[a]
                t_free = time_available[a]
                for t in unassigned:
                    tp      = self.problem.task_poses[t]
                    arrival = t_free + math.hypot(pos[0] - tp[0], pos[1] - tp[1])
                    if arrival < best_arrival:
                        best_arrival = arrival
                        best_robot   = a
                        best_task    = t
            if best_robot is None:
                break
            per_agent_routes[best_robot].append(best_task)
            robot_positions[best_robot] = list(self.problem.task_poses[best_task])
            time_available[best_robot]  = best_arrival
            unassigned.discard(best_task)

        self._ensure_min_one(per_agent_routes)
        base_order = [t for r in per_agent_routes for t in r]
        base_alloc = [len(r) for r in per_agent_routes]
        base = (base_order, base_alloc)

        population = [base]
        num_tasks = len(base_order)
        for _ in range(1, self.pop_size):
            num_swaps = random.randint(1, max(1, num_tasks // 4))
            population.append(self._perturb_solution(base, num_swaps))
        return population

    def _select_solution(self) -> Tuple[Optional[int], Optional[tuple]]:
        if not self.population:
            return None, None
        return min(enumerate(self.population), key=lambda it: self.fitness(it[1]))

    def _select_random_solution(self) -> Tuple[Optional[int], Optional[tuple]]:
        if not self.population:
            return None, None
        idx = random.randrange(len(self.population))
        return idx, self.population[idx]

    def _apply_operator(self, operator: Operator) -> Optional[tuple]:
        if not self.current_solution or not self.current_solution[0]:
            return None
        valid = [s for s in (self.population or []) if s and s[0]]
        if len(valid) <= 1:
            return None
        expected_tasks = len(self.current_solution[0])
        try:
            result = OperatorFunctions.apply_op(
                operator,
                self.current_solution,
                valid,
                self.problem.cost_matrix,
                list(self.problem.current_robot_cost_matrix),
                self.problem.initial_robot_cost_matrix,
                robot_to_depot_cost=self.problem.robot_to_depot_cost,
            )
            # Discard any result that dropped tasks (operator bug defence)
            if result is None or not result[0] or len(result[0]) < expected_tasks:
                return None
            return result
        except Exception:
            import traceback
            print(f'[WARN] agent={self.agent_id} operator={operator} raised an exception:',
                  flush=True)
            traceback.print_exc()
            return None

    def _reinsert_child(self, child: tuple) -> None:
        if child is None or not self.population or self.current_parent_idx is None:
            return
        try:
            if self.fitness(child) < self.fitness(
                self.population[self.current_parent_idx]
            ):
                self.population[self.current_parent_idx] = deepcopy(child)
        except (IndexError, Exception):
            pass

    def fitness(self, sol, robot_cost_matrix_snapshot=None) -> float:
        if sol is None or self.problem.current_robot_cost_matrix is None:
            return float('inf')
        try:
            if robot_cost_matrix_snapshot is not None:
                original = self.problem.current_robot_cost_matrix
                self.problem.current_robot_cost_matrix = robot_cost_matrix_snapshot
                try:
                    return SimpleFitness.fitness_function_robot_pose(sol, self.problem)
                finally:
                    self.problem.current_robot_cost_matrix = original
            return SimpleFitness.fitness_function_robot_pose(sol, self.problem)
        except Exception:
            return float('inf')

    def _remove_covered(self, solution: Optional[tuple]) -> Optional[tuple]:
        if solution is None:
            return None
        order, alloc = list(solution[0]), list(solution[1])
        new_order, new_alloc = [], []
        cursor = 0
        for count in alloc:
            kept = [t for t in order[cursor:cursor + count]
                    if not self.is_covered[int(t)]]
            new_order.extend(kept)
            new_alloc.append(len(kept))
            cursor += count
        return (new_order, new_alloc)

    def _strip_covered_solutions(self) -> None:
        for attr in ('current_solution', 'coalition_best_solution', 'local_best_solution',
                     'di_cycle_start_solution'):
            sol = getattr(self, attr)
            if sol is not None:
                setattr(self, attr, self._remove_covered(sol))
        if self.population:
            self.population = [self._remove_covered(s) for s in self.population]

    def _prepend_current_task(self, solution: tuple) -> Optional[tuple]:
        if self.current_task is None or solution is None:
            return None
        order, alloc = list(solution[0]), list(solution[1])
        task = self.current_task
        if task not in order:
            return None
        pos          = order.index(task)
        cursor, owner = 0, None
        for i, a in enumerate(alloc):
            if cursor <= pos < cursor + a:
                owner = i
                break
            cursor += a
        if owner is None:
            return None
        order.pop(pos)
        if owner != self.agent_id:
            alloc[owner]         -= 1
            alloc[self.agent_id] += 1
        start = sum(alloc[:self.agent_id])
        order.insert(start, task)
        return (order, alloc)

    def _op_index(self, operator: Operator) -> int:
        ops = self.intensifiers + self.diversifiers
        try:
            return ops.index(operator)
        except ValueError:
            return 0

    def _build_classical_admissibility_matrix(self) -> list:
        n_int = len(self.intensifiers)
        n_div = len(self.diversifiers)
        rows = []
        rows.append([0.0] * n_int + [1.0] * n_div)
        rows.append([1.0] * n_int + [0.0] * n_div)
        for i in range(n_int):
            row = [1.0] * n_int + [0.0] * n_div
            row[i] = 0.0
            rows.append(row)
        return rows

    def _admissible_operator_indices(self, condition: int) -> list:
        row = self.operator_admissibility[int(condition)]
        return [idx for idx, allowed in enumerate(row) if allowed > 0.0]

    def _next_condition_after_operator(self, op_idx: int) -> int:
        op_idx = int(op_idx)
        if op_idx < len(self.intensifiers):
            return 2 + op_idx
        return 1

    # ------------------------------------------------------------------ #
    # DI-cycle learning                                                    #
    # ------------------------------------------------------------------ #

    def _check_stagnation(self) -> bool:
        """True when the last two history entries are both non-improving intensifications.

        Implements the stagnation termination condition of Eq. (4):
            (op(H_{k-2}) ∈ I ∧ g(H_{k-2}) ≤ 0) ∧ (op(H_{k-1}) ∈ I ∧ g(H_{k-1}) ≤ 0)

        Note: in the code gain = new_f − cur_f, so gain >= 0 means no improvement
        (the paper defines gain as F(before) − F(after), flipping the sign).
        Requires at least 2 history entries (|H| ≥ 2); the paper notes evaluation
        is only possible from the third step onwards (|H| ≥ 3), which is equivalent
        here because we check the two most recent entries after appending the current one.
        """
        if len(self.previous_experience) < 2:
            return False
        n_int = len(self.intensifiers)
        for _, op_idx, gain in self.previous_experience[-2:]:
            if int(op_idx) >= n_int or gain < 0:
                return False
        return True

    # ------------------------------------------------------------------ #
    # Learning methods                                                     #
    # ------------------------------------------------------------------ #

    def _best_episode_cutoff(self) -> int:
        """Index up to which experience is considered (cumulative gain minimum).

        Returns the index *after* the step that achieved the best cumulative
        gain, so that step is included in the weight update (range(cutoff) is
        exclusive of cutoff, hence +1).
        """
        cumulative, total = [], 0.0
        for _, _, gain in self.previous_experience:
            total += gain
            cumulative.append(total)
        if self.best_local_improved and cumulative:
            return cumulative.index(min(cumulative)) + 1
        return len(self.previous_experience)

    def _apply_time_discount(self, reward: float, gain: float, wall_time: float) -> float:
        """
        Scale reward by a time-discount factor:
            reward * (1 + λ·t)^sign(gain)
        Improvements (gain < 0, reward > 0) are diminished for slow operators.
        Degradations (gain > 0, reward < 0) are amplified for slow operators.
        """
        if not self.time_discount or wall_time <= 0.0:
            return reward
        factor = 1.0 + self.time_discount_lambda * wall_time
        if gain < 0:
            return reward / factor
        return reward * factor

    def _update_reward_ema(self, gain: float) -> None:
        """Update the exponential moving average of improvement.

        Called on every operator application (gain=0 for null/empty results).
        improvement = -gain  (positive when the solution actually got better).
        """
        improvement = -gain
        delta = improvement - self._reward_ema_mean
        self._reward_ema_mean += self._reward_ema_alpha * delta
        self._reward_ema_var   = ((1.0 - self._reward_ema_alpha)
                                  * (self._reward_ema_var
                                     + self._reward_ema_alpha * delta ** 2))
        self._reward_ema_n    += 1

    def _relative_reward(self, gain: float, wall_time: float = 0.0) -> float:
        """Compute reward relative to the running EMA baseline.

        reward = (improvement - ema_mean) / (ema_std + ε)

        An operator that improves better than average gets a positive reward;
        one that improves less than average (or worsens) gets a negative reward.
        The signal stays centred throughout the run regardless of absolute
        improvement magnitude, so Q-values reflect relative operator quality.
        """
        improvement = -gain
        std    = max(self._reward_ema_var ** 0.5, 1e-6)
        r      = (improvement - self._reward_ema_mean) / std * self.positive_reward
        return self._apply_time_discount(r, gain, wall_time)

    def _update_div_reward_ema(self, cycle_gain: float) -> None:
        """Update the diversifier EMA with the total gain across one DI cycle.

        Called once per completed DI cycle.  Uses the same alpha as the
        per-step EMA but operates at cycle granularity, so the effective
        window is ~(1/alpha) cycles rather than steps.
        """
        improvement = -cycle_gain   # positive = cycle improved the solution
        delta = improvement - self._div_ema_mean
        self._div_ema_mean += self._reward_ema_alpha * delta
        self._div_ema_var   = ((1.0 - self._reward_ema_alpha)
                               * (self._div_ema_var
                                  + self._reward_ema_alpha * delta ** 2))
        self._div_ema_n    += 1

    def _relative_div_reward(self, cycle_gain: float) -> float:
        """Compute a diversifier reward relative to the cycle-level EMA baseline.

        reward = (cycle_improvement - div_ema_mean) / (div_ema_std + ε)

        A diversifier that enables a better-than-average cycle gets a positive
        reward; one that leads to a worse-than-average cycle gets a negative
        reward.  All diversifiers active in the same cycle share this reward,
        reflecting that the cycle outcome is a joint result.
        """
        improvement = -cycle_gain
        std = max(self._div_ema_var ** 0.5, 1e-6)
        return (improvement - self._div_ema_mean) / std * self.positive_reward

    def _learning_step(self, condition: int, op_idx: int, improved: bool,
                       gain: float = 0.0, wall_time: float = 0.0) -> None:
        """Per-step Q-learning update (used by Q_LEARNING_STEP and Q_LEARNING_SEPARATE)."""
        q        = self.weight_matrix.weights[condition][op_idx]
        next_condition = self._next_condition_after_operator(op_idx)
        max_next = max(self.weight_matrix.weights[next_condition])
        if self.is_relative_reward:
            reward = self._relative_reward(gain, wall_time)
        else:
            reward = self.positive_reward if improved else self.negative_reward
            reward = self._apply_time_discount(reward, gain, wall_time)
        q_new    = q + self.lr * (reward + self.gamma_decay * max_next - q)
        self.weight_matrix.weights[condition][op_idx] = max(q_new, 1e-6)

    def _learning_qlearning(self, diversifiers_only: bool = False,
                            improved: Optional[bool] = None,
                            cycle_reward: Optional[float] = None) -> None:
        """Cycle-end Q-learning update (Q_LEARNING and Q_LEARNING_SEPARATE diversifiers).

        improved:      override the reward signal; defaults to self.best_local_improved.
        cycle_reward:  when set and is_relative_reward=True, all operators in this
                       cycle share this pre-computed reward (used for diversifiers
                       whose reward is based on total cycle gain, not per-step gain).
        """
        min_idx = self._best_episode_cutoff()
        n_int   = len(self.intensifiers)
        seen    = set()
        _improved = improved if improved is not None else self.best_local_improved
        _fixed_reward = self.positive_reward if _improved else self.negative_reward
        for i in range(min_idx):
            cond, op_col, step_gain = self.previous_experience[i]
            op_col = int(op_col)
            if diversifiers_only and op_col < n_int:
                continue
            key = (cond, op_col)
            if key in seen:
                continue
            seen.add(key)
            q        = self.weight_matrix.weights[cond][op_col]
            next_condition = self._next_condition_after_operator(op_col)
            max_next = max(self.weight_matrix.weights[next_condition])
            if self.is_relative_reward:
                reward = cycle_reward if cycle_reward is not None else self._relative_reward(step_gain)
            else:
                reward = _fixed_reward
            q_new    = q + self.lr * (reward + self.gamma_decay * max_next - q)
            self.weight_matrix.weights[cond][op_col] = max(q_new, 1e-6)

    def _learning_ferreira(self) -> None:
        """Cycle-end Ferreira et al. incremental update."""
        if not self.best_local_improved:
            return
        min_idx = self._best_episode_cutoff()
        seen    = set()
        for i in range(min_idx):
            cond, op_col, _ = self.previous_experience[i]
            key = (cond, int(op_col))
            if key in seen:
                continue
            seen.add(key)
            increment = self.eta if self.best_coalition_improved else 1.0
            self.weight_matrix.weights[cond][int(op_col)] += increment

    def _finish_di_cycle(self) -> Optional[list]:
        """
        Weight update at end of DI cycle — dispatches to the active learning method.
        Returns the weight matrix if it should be broadcast to peers, else None.
        """
        if self.learning_method == LearningMethod.Q_LEARNING:
            self._learning_qlearning(diversifiers_only=False)
        elif self.learning_method == LearningMethod.Q_LEARNING_SEPARATE:
            cycle_total_gain = sum(g for _, _, g in self.previous_experience)
            self._update_div_reward_ema(cycle_total_gain)
            div_reward = (self._relative_div_reward(cycle_total_gain)
                          if self.is_relative_reward else None)
            self._learning_qlearning(diversifiers_only=True,
                                     improved=self.best_cycle_start_improved,
                                     cycle_reward=div_reward)
        elif self.learning_method == LearningMethod.Q_LEARNING_SEPARATE_GAIN:
            self._learning_qlearning(diversifiers_only=True)
        elif self.learning_method == LearningMethod.FERREIRA:
            self._learning_ferreira()
        # Q_LEARNING_SEPARATE, Q_LEARNING_STEP, Q_LEARNING_STEP_GAIN, Q_LEARNING_GAIN, UCB, UNIFORM: no cycle-end weight update

        _mimetism_applicable = self.learning_method not in (
            LearningMethod.UCB, LearningMethod.UNIFORM
        )
        weights_to_share = None
        if self.is_mimetism_enabled and _mimetism_applicable:
            if self.best_coalition_improved:
                weights_to_share = deepcopy(self.weight_matrix.weights)
            if self.received_weight_matrices:
                for wm in self.received_weight_matrices:
                    for i in range(len(self.weight_matrix.weights)):
                        for j in range(len(self.weight_matrix.weights[i])):
                            self.weight_matrix.weights[i][j] = (
                                (1 - self.rho) * self.weight_matrix.weights[i][j]
                                + self.rho * wm[i][j]
                            )
                self.received_weight_matrices.clear()

        if self.is_inject_best_on_cycle and self.coalition_best_solution and self.population:
            if random.random() < self.inject_best_prob:
                worst = max(range(len(self.population)),
                            key=lambda k: self.fitness(self.population[k]))
                self.population[worst] = deepcopy(self.coalition_best_solution)
                self.current_solution = deepcopy(self.coalition_best_solution)
                self.current_parent_idx = worst
                self.no_improvement_attempt_count = 0

        if self._logger is not None:
            op_list = self.intensifiers + self.diversifiers
            named_experiences = [
                [cond, op_list[int(op_idx)].name, gain]
                for cond, op_idx, gain in self.previous_experience
            ]
            self._logger.di_cycle_complete(
                sim_time=self._current_sim_time,
                agent_id=self.agent_id,
                di_cycle_num=self.di_cycle_total,
                best_local_improved=self.best_local_improved,
                best_coalition_improved=self.best_coalition_improved,
                experiences=named_experiences,
                weights=self.weight_matrix.weights,
            )

        self.di_cycle_total         += 1
        self.previous_experience     = []
        self.di_cycle_start_solution = None
        self.di_cycle_count          = 0
        self.best_local_improved        = False
        self.best_coalition_improved    = False
        self.best_cycle_start_improved  = False

        return weights_to_share

    # ------------------------------------------------------------------ #
    # Weight persistence                                                   #
    # ------------------------------------------------------------------ #

    def save_weights(self, path: str) -> None:
        """Persist learned policy state to a JSON file.

        For Q-learning methods: saves the weight matrix.
        For UCB: saves the per-operator reward history and global pull count.
        """
        import json
        import os

        data = {
            "method": self.learning_method.value,
            "num_intensifiers": len(self.intensifiers),
            "num_diversifiers": len(self.diversifiers),
        }
        if self.learning_method == LearningMethod.UCB:
            data["ucb_history"] = [list(h) for h in self.ucb_bandit._history]
            data["ucb_N"] = self.ucb_bandit.N
        else:
            data["weight_matrix"] = [list(row) for row in self.weight_matrix.weights]

        data["reward_ema_mean"] = self._reward_ema_mean
        data["reward_ema_var"]  = self._reward_ema_var
        data["reward_ema_n"]    = self._reward_ema_n
        data["div_ema_mean"]    = self._div_ema_mean
        data["div_ema_var"]     = self._div_ema_var
        data["div_ema_n"]       = self._div_ema_n

        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f)

    def load_weights(self, path: str) -> None:
        """Load and apply policy state from a JSON file produced by save_weights().

        Raises ValueError if the method or operator dimensions do not match.
        """
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        if data["method"] != self.learning_method.value:
            raise ValueError(
                f"Weight file method '{data['method']}' does not match "
                f"agent method '{self.learning_method.value}'"
            )
        n_int = data["num_intensifiers"]
        n_div = data["num_diversifiers"]
        if n_int != len(self.intensifiers) or n_div != len(self.diversifiers):
            raise ValueError(
                f"Weight file shape ({n_int} int, {n_div} div) does not match "
                f"agent shape ({len(self.intensifiers)} int, {len(self.diversifiers)} div)"
            )

        if self.learning_method == LearningMethod.UCB:
            for i, history in enumerate(data["ucb_history"]):
                self.ucb_bandit._history[i].clear()
                self.ucb_bandit._history[i].extend(history)
            self.ucb_bandit.N = data["ucb_N"]
        else:
            self.weight_matrix.weights = [list(row) for row in data["weight_matrix"]]

        if "reward_ema_mean" in data:
            self._reward_ema_mean = data["reward_ema_mean"]
            self._reward_ema_var  = data["reward_ema_var"]
            self._reward_ema_n    = data["reward_ema_n"]
        if "div_ema_mean" in data:
            self._div_ema_mean = data["div_ema_mean"]
            self._div_ema_var  = data["div_ema_var"]
            self._div_ema_n    = data["div_ema_n"]

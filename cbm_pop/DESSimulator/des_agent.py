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
    Q_LEARNING_GAIN     = 'Q-Learning-improveoncurrent'   # step reward = improvement on current solution
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
        method:                str   = 'Q-Learning',
        ucb_c:                 float = 1.414,
        ucb_window:            int   = 200,
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

        # Solution state
        self.population:              Optional[list]          = None
        self.current_solution:        Optional[tuple]         = None
        self.local_best_solution:     Optional[tuple]         = None
        self.coalition_best_solution: Optional[tuple]         = None
        self.coalition_best_agent:    Optional[int]           = None

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
        self.di_cycle_count:           int  = 0
        self.di_cycle_total:           int  = 0
        self.iteration_count:          int  = 0
        self.no_improvement_attempt_count: int = 0
        self.best_local_improved:      bool = False
        self.best_coalition_improved:  bool = False
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

        self.problem.update_robot_cost_matrix(self.robot_poses)
        self.problem.initialize_robot_initial_pose_cost_matrix(self.initial_robot_poses)

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

        condition = ConditionFunctions.perceive_condition_row(
            self.previous_experience, self.intensifiers, self.diversifiers
        )
        enabled = self.intensifiers + self.diversifiers
        if self.learning_method == LearningMethod.UCB:
            operator = enabled[self.ucb_bandit.select()]
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
            return False

        result = self._remove_covered(step.result)
        if not result or not result[0]:
            self.no_improvement_attempt_count += 1
            return False

        num_uncovered = sum(1 for c in self.is_covered if not c)
        new_f  = self._fitness(result)
        cur_f  = self._fitness(self.current_solution)
        loc_f  = self._fitness(self.local_best_solution)
        coal_f = self._fitness(self.coalition_best_solution)

        gain    = new_f - cur_f
        op_idx  = self._op_index(step.operator)
        self.previous_experience.append([step.condition, op_idx, gain])

        step_improved_local   = (loc_f == float('inf')) or (new_f < loc_f)
        step_improved_current = gain < 0

        # Step-level learning
        if self.learning_method == LearningMethod.Q_LEARNING_STEP:
            self._learning_step(step.condition, op_idx, step_improved_local)
        elif (self.learning_method == LearningMethod.Q_LEARNING_SEPARATE
              and step.operator in self.intensifiers):
            self._learning_step(step.condition, op_idx, step_improved_local)
        elif self.learning_method == LearningMethod.Q_LEARNING_GAIN:
            self._learning_step(step.condition, op_idx, step_improved_current)
        elif self.learning_method == LearningMethod.UCB:
            self.ucb_bandit.update(op_idx, -gain)
            self.ucb_bandit.N += 1
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

        if self.di_cycle_count >= self.di_cycle_length:
            self._finish_di_cycle()

        return coalition_improved

    def receive_coalition_best(
        self,
        solution,
        sender_id: int,
        sender_weights: Optional[list] = None,
    ) -> bool:
        """
        Ingest a coalition-best from a peer.
        Returns True if this improved our coalition best.
        """
        if solution is None:
            return False

        candidate = self._remove_covered(deepcopy(solution))
        if not candidate or not candidate[0]:
            return False

        if self.is_append_first_task and self.current_task is not None:
            modified = self._prepend_current_task(candidate)
            if modified is not None and self._fitness(modified) < self._fitness(candidate):
                candidate = modified

        num_uncovered = sum(1 for c in self.is_covered if not c)
        _coal_incomplete = (
            num_uncovered > 0
            and self.coalition_best_solution is not None
            and len(self.coalition_best_solution[0]) < num_uncovered
        )
        _candidate_complete = (num_uncovered == 0 or len(candidate[0]) >= num_uncovered)

        if (self._fitness(candidate) < self._fitness(self.coalition_best_solution)
                or (_coal_incomplete and _candidate_complete)):
            self.coalition_best_solution = candidate
            self.coalition_best_agent    = sender_id
            self._assign_next_task(candidate)
            if sender_weights is not None:
                self.received_weight_matrices.append(sender_weights)
            return True

        if sender_weights is not None:
            self.received_weight_matrices.append(sender_weights)
        return False

    def handle_task_covered(self, task_id: int) -> None:
        """React to a coverage event from any robot."""
        if self.is_covered[task_id]:
            return
        self.is_covered[task_id] = True

        if self.is_task_locked and self.locked_task == task_id:
            self._reset_task_lock_state()

        self._strip_covered_solutions()
        self._assign_next_task(self.coalition_best_solution)

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
        robot_xy = np.array(self.robot_poses, dtype=float)
        task_xy  = np.array(self.problem.task_poses, dtype=float)
        dists    = ((task_xy[:, None, :] - robot_xy[None, :, :]) ** 2).sum(axis=2)
        nearest  = np.argmin(dists, axis=1)
        for t, a in enumerate(nearest):
            if not self.is_covered[t]:
                tasks_per_agent[a].append(t)
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
        needers = [a for a, r in enumerate(per_agent_routes) if not r]
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

    def _generate_population(self) -> list:
        """Dispatch to heuristic or random initialisation based on agent config."""
        if self.initialise_with_heuristic:
            return self._generate_population_voronoi()
        return self._generate_population_random()

    def _generate_population_random(self) -> list:
        """Randomly assign and order uncovered tasks — no Voronoi bias."""
        uncovered = [t for t, c in enumerate(self.is_covered) if not c]
        population = []
        for _ in range(self.pop_size):
            tasks = uncovered[:]
            random.shuffle(tasks)
            assignment = [random.randrange(self.num_agents) for _ in tasks]
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

    def _select_solution(self) -> Tuple[Optional[int], Optional[tuple]]:
        if not self.population:
            return None, None
        return min(enumerate(self.population), key=lambda it: self._fitness(it[1]))

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
            if self._fitness(child) < self._fitness(
                self.population[self.current_parent_idx]
            ):
                self.population[self.current_parent_idx] = deepcopy(child)
        except (IndexError, Exception):
            pass

    def _fitness(self, sol) -> float:
        if sol is None or self.problem.current_robot_cost_matrix is None:
            return float('inf')
        try:
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
        for attr in ('current_solution', 'coalition_best_solution', 'local_best_solution'):
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

    # ------------------------------------------------------------------ #
    # DI-cycle learning                                                    #
    # ------------------------------------------------------------------ #

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

    def _learning_step(self, condition: int, op_idx: int, improved: bool) -> None:
        """Per-step Q-learning update (used by Q_LEARNING_STEP and Q_LEARNING_SEPARATE)."""
        q        = self.weight_matrix.weights[condition][op_idx]
        max_next = max(self.weight_matrix.weights[condition])
        reward   = self.positive_reward if improved else self.negative_reward
        q_new    = q + self.lr * (reward + self.gamma_decay * max_next - q)
        self.weight_matrix.weights[condition][op_idx] = max(q_new, 1e-6)

    def _learning_qlearning(self, diversifiers_only: bool = False) -> None:
        """Cycle-end Q-learning update (Q_LEARNING and Q_LEARNING_SEPARATE diversifiers)."""
        min_idx = self._best_episode_cutoff()
        n_int   = len(self.intensifiers)
        seen    = set()
        reward  = self.positive_reward if self.best_local_improved else self.negative_reward
        for i in range(min_idx):
            cond, op_col, _ = self.previous_experience[i]
            op_col = int(op_col)
            if diversifiers_only and op_col < n_int:
                continue
            key = (cond, op_col)
            if key in seen:
                continue
            seen.add(key)
            q        = self.weight_matrix.weights[cond][op_col]
            max_next = max(self.weight_matrix.weights[cond])
            q_new    = q + self.lr * (reward + self.gamma_decay * max_next - q)
            self.weight_matrix.weights[cond][op_col] = max(q_new, 1e-6)

    def _learning_ferreira(self) -> None:
        """Cycle-end Ferreira et al. incremental update."""
        min_idx = self._best_episode_cutoff()
        seen    = set()
        for i in range(min_idx):
            cond, op_col, _ = self.previous_experience[i]
            key = (cond, int(op_col))
            if key in seen:
                continue
            seen.add(key)
            increment = 1.0 if self.best_coalition_improved else self.eta
            self.weight_matrix.weights[cond][int(op_col)] += increment

    def _finish_di_cycle(self) -> Optional[list]:
        """
        Weight update at end of DI cycle — dispatches to the active learning method.
        Returns the weight matrix if it should be broadcast to peers, else None.
        """
        if self.learning_method == LearningMethod.Q_LEARNING:
            self._learning_qlearning(diversifiers_only=False)
        elif self.learning_method == LearningMethod.Q_LEARNING_SEPARATE:
            self._learning_qlearning(diversifiers_only=True)
        elif self.learning_method == LearningMethod.FERREIRA:
            self._learning_ferreira()
        # Q_LEARNING_STEP, Q_LEARNING_GAIN, UCB, UNIFORM: no cycle-end weight update

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
                            key=lambda k: self._fitness(self.population[k]))
                self.population[worst] = deepcopy(self.coalition_best_solution)

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
        self.di_cycle_count          = 0
        self.best_local_improved     = False
        self.best_coalition_improved = False

        return weights_to_share

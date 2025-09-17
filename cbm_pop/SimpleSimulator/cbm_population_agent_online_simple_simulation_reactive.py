import json
import math
import random
import sys
import traceback
from time import time
import matplotlib.pyplot as plt
import numpy as np
from builtin_interfaces.msg import Time
from matplotlib import animation, patches
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
from copy import deepcopy
from random import sample
from collections import Counter  # <-- AUDIT

from cbm_pop.Operator import Operator
from cbm_pop.Condition import ConditionFunctions
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator_Fuctions import OperatorFunctions
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.Problem import Problem
from rclpy.node import Node
from std_msgs.msg import String, Float32
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
from cbm_pop_interfaces.msg import Solution, Weights, EnvironmentalRepresentation, SimplePosition, FinishedCoverage
from enum import Enum
from math import radians, cos, sin, asin, sqrt
from std_msgs.msg import Bool

class LearningMethod(Enum):
    FERREIRA = "Ferreira_et_al."
    Q_LEARNING = "Q-Learning"


class CBMPopulationAgentOnlineSimpleSimulationReactive(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, learning_method,
                 lr=0.5,
                 gamma_decay=0.99,
                 positive_reward=1,
                 negative_reward=-0.5,
                 num_tsp_agents=10,
                 enable_chk_logs: bool = False):

        """
        Initialises the agent on startup
        """
        super().__init__(node_name)

        settings_str = f"""
                    ================= Agent Configuration =================
                    Agent ID:            {agent_id}
                    Node Name:           {node_name}
                    Learning Method:     {learning_method}

                    Population Size:     {pop_size}
                    Eta (learning rate): {eta}
                    Rho (discount):      {rho}
                    DI Cycle Length:     {di_cycle_length}
                    Epsilon:             {epsilon}
                    Iterations:          {num_iterations}
                    Solution Attempts:   {num_solution_attempts}

                    RL Parameters:
                      - LR:              {lr}
                      - Gamma Decay:     {gamma_decay}
                      - Positive Reward: {positive_reward}
                      - Negative Reward: {negative_reward}

                    Number of TSP Agents: {num_tsp_agents}
                    =======================================================
                    """

        from collections import defaultdict, Counter
        self._proposal_stats = defaultdict(lambda: {
            "count": 0,
            "removed_uncovered": Counter(),
            "added_covered": Counter(),
        })

        self.current_task = None
        self.task_poses = [(i + 0.5, j + 0.5) for i in range(15) for j in range(15)]
        self.num_tasks = len(self.task_poses)

        # Core config
        self.is_generating = False
        self.pop_size = pop_size
        self.eta = eta
        self.rho = rho
        self.di_cycle_length = di_cycle_length
        self.num_iterations = num_iterations
        self.num_tsp_agents = num_tsp_agents
        self.is_finished = False

        # Solution state
        self.agent_best_solution = None
        self.coalition_best_solution = None
        self.local_best_solution = None
        self.coalition_best_agent = None

        # Operators and weights
        self.intensifiers = [Operator.ONE_MOVE, Operator.TWO_SWAP, Operator.TWO_OPT_INTRA]
        self.diversifiers = [
            Operator.BEST_COST_ROUTE_CROSSOVER,
            Operator.INTRA_DEPOT_REMOVAL,
            Operator.INTRA_DEPOT_SWAPPING,
            Operator.SINGLE_ACTION_REROUTING
        ]
        self.weight_matrix = WeightMatrix(len(self.intensifiers), len(self.diversifiers))

        self.population = None
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_ID = agent_id
        self.true_agent_ID = self.agent_ID
        self.received_weight_matrices = []

        if isinstance(learning_method, str):
            try:
                learning_method = LearningMethod(learning_method)
            except ValueError:
                raise ValueError(
                    f"Invalid learning method '{learning_method}'. Must be one of: "
                    f"{[e.value for e in LearningMethod]}"
                )
        self.learning_method = learning_method

        self.new_robot_cost_matrix = None
        self.is_new_robot_cost_matrix = False
        self.last_env_rep_timestamps = {}

        # Iteration state
        self.iteration_count = 0
        self.di_cycle_count = 0
        self.no_improvement_attempt_count = 0
        self.best_coalition_improved = False
        self.best_local_improved = False

        # Runtime data
        self.initial_robot_poses = [None] * self.num_tsp_agents
        self.robot_poses = [None] * self.num_tsp_agents
        self.current_solution = None
        self.is_covered = [False] * self.num_tasks
        self.cost_matrix = self.calculate_cost_matrix()

        self.robot_cost_matrix = [None] * self.num_tsp_agents
        self.robot_initial_pose_cost_matrix = [None] * self.num_tsp_agents
        self.last_purge_agent_true_id = None
        self.is_agent_tobe_purged = False
        self.failed_agents = [False] * self.num_tsp_agents
        self.purged_agents = [False] * self.num_tsp_agents
        self.agent_to_revive = None
        self.am_i_failed = False
        self.ros_timer = None
        self.agent_timeouts = [False] * self.num_tsp_agents
        self.finished_robots = [False] * self.num_tsp_agents
        self.is_all_poses = False

        self.cb_group = ReentrantCallbackGroup()
        self.me_cb_group = MutuallyExclusiveCallbackGroup()

        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(Solution, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10)
        self.weight_publisher = self.create_publisher(Weights, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            Weights, 'weight_matrix', self.weight_update_callback, 10)

        # Global pose subscribers
        self.global_pose_subscribers = []
        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id}/global_pose'
            sub = self.create_subscription(
                SimplePosition,
                topic,
                lambda msg, agent=id: self.global_pose_callback(msg),
                10,
                callback_group=self.cb_group
            )
            self.global_pose_subscribers.append(sub)

        # Goal publisher
        self.goal_pose_publisher = self.create_publisher(
            SimplePosition,
            f'/central_control/uas_{agent_id}/goal_pose',
            10)

        # Kill robot subscribers
        self.kill_robot_subscribers = []
        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id + 1}/kill_robot'
            sub = self.create_subscription(
                Bool,
                topic,
                lambda msg, agent=(id + 1): self.kill_robot_callback(msg, agent),
                10
            )
            self.kill_robot_subscribers.append(sub)

        # Revive subscriber
        self.revive_robot_sub = self.create_subscription(
            Bool,
            f'/central_control/uas_{agent_id}/revive_robot',
            self.revive_robot_callback,
            10
        )

        # Finished coverage pub/sub
        self.finished_coverage_pub = self.create_publisher(
            FinishedCoverage,
            f'/central_control/finished_coverage',
            10)
        self.finished_coverage_sub = self.create_subscription(
            FinishedCoverage,
            f'/central_control/finished_coverage',
            self.finished_coverage_callback,
            10
        )

        # Timers
        self.run_goal_publisher_timer = self.create_timer(0.5, self.publish_goal_pose, callback_group=self.cb_group)
        self.run_cost_matrix_recalculation = self.create_timer(0.25, self.robot_cost_matrix_recalculation,
                                                               callback_group=self.me_cb_group)
        self.environmental_representation_subscriber = self.create_subscription(
            EnvironmentalRepresentation,
            '/environmental_representation',
            self.environmental_representation_callback,
            10,
            callback_group=self.cb_group
        )
        self.environmental_representation_publisher = self.create_publisher(
            EnvironmentalRepresentation,
            '/environmental_representation',
            10
        )
        self.environmental_representation_timer = self.create_timer(2, self.environmental_representation_timer_callback,
                                                                    callback_group=self.cb_group)
        self.solution_publisher_timer = self.create_timer(2, self.regular_solution_publish_timer,
                                                          callback_group=self.cb_group)
        self.create_timer(5, self.check_stale_agents)

        # Q-learning params
        self.lr = lr
        self.reward = 0
        self.new_reward = 0
        self.gamma_decay = gamma_decay
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

        self.run_timer = None
        self.is_loop_started = False
        self.task_covered = -1
        self.is_new_task_covered = False


        '''
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # Inside __init__
        self.fig, self.ax = plt.subplots()
        self.im = None  # placeholder for the cost matrix image
        self._plt_lock = threading.Lock()
        '''


    '''def _start_cost_plot(self):
        """Start the cost matrix animation thread."""
        self.ani = animation.FuncAnimation(self.fig, self._update_cost_plot, interval=1000)
        plt.show()
    '''

    '''def _update_cost_plot(self, _):
        """Update plot with latest robot cost matrix."""
        with self._plt_lock:
            if self.robot_cost_matrix is None or not isinstance(self.robot_cost_matrix, np.ndarray):
                return

            self.ax.clear()
            self.ax.set_title(f"Robot Cost Matrix (Agent {self.agent_ID})")
            im = self.ax.imshow(self.robot_cost_matrix, cmap='viridis', interpolation='nearest')
            self.ax.set_xlabel("Tasks")
            self.ax.set_ylabel("Robots")

            # Draw a rectangle to outline the current task cell
            if hasattr(self, 'current_task') and self.current_task is not None:
                rect = patches.Rectangle(
                    (self.current_task - 0.5, self.agent_ID - 0.5),  # (x, y) in data coords
                    1, 1, linewidth=2, edgecolor='red', facecolor='none'
                )
                self.ax.add_patch(rect)
    '''

    # ======================= AUDIT HELPERS ==========================
    def _audit_solution(self, where: str, sol):
        """
        Check invariants for a single (order, allocations) solution and log any issues.
        """
        if sol is None:
            # separate callsite for WARN
            self.get_logger().warning(f"[AUDIT:{where}] solution is None")
            return

        try:
            order, alloc = sol
        except Exception as e:
            # separate callsite for ERROR
            self.get_logger().error(f"[AUDIT:{where}] bad solution shape: {type(sol)} error={e}")
            return

        from collections import Counter
        issues = []
        msg_parts = []

        if not isinstance(order, (list, tuple)) or not isinstance(alloc, (list, tuple)):
            issues.append("bad_types")
            msg_parts.append(f"types(order={type(order)}, alloc={type(alloc)})")

        if sum(alloc) != len(order):
            issues.append("alloc_sum_mismatch")
            msg_parts.append(f"sum(alloc)={sum(alloc)} != len(order)={len(order)}")

        bad_idx = [t for t in order if not isinstance(t, int) or t < 0 or t >= self.num_tasks]
        if bad_idx:
            issues.append("out_of_range")
            msg_parts.append(f"out_of_range={bad_idx[:10]}{'...' if len(bad_idx) > 10 else ''}")

        c = Counter(order)
        dups = [t for t, cnt in c.items() if cnt > 1]
        if dups:
            issues.append("duplicates")
            msg_parts.append(f"duplicates={dups[:10]}{'...' if len(dups) > 10 else ''}")

        covered_present = [t for t in order if 0 <= t < self.num_tasks and self.is_covered[t]]
        if covered_present:
            issues.append("covered_present")
            msg_parts.append(f"covered_present={covered_present[:10]}{'...' if len(covered_present) > 10 else ''}")

        uncovered = {t for t in range(self.num_tasks) if not self.is_covered[t]}
        missing_uncovered = sorted(list(uncovered - set(order)))
        if missing_uncovered:
            issues.append("missing_uncovered")
            msg_parts.append(
                f"missing_uncovered={missing_uncovered[:10]}{'...' if len(missing_uncovered) > 10 else ''}")

        cursor = 0
        per_agent_oob = []
        for i, a in enumerate(alloc):
            if a < 0 or cursor + a > len(order):
                per_agent_oob.append((i, a, cursor))
            cursor += max(0, a)
        if per_agent_oob:
            issues.append("slice_bounds")
            msg_parts.append(f"slice_oob={per_agent_oob}")

        if issues:
            # separate callsite for WARN
            self.get_logger().warning(f"[AUDIT:{where}] issues={issues} :: {' | '.join(msg_parts)}")
        else:
            # separate callsite for INFO
            self.get_logger().info(f"[AUDIT:{where}] OK len(order)={len(order)} sum(alloc)={sum(alloc)}")

    def _diff_solutions(self, where: str, old_sol, new_sol, label_old="OLD", label_new="NEW"):
        """
        Log task-level differences between two solutions (ignoring allocations).
        Flags any uncovered tasks removed or added.
        """
        old_set = set(old_sol[0]) if old_sol else set()
        new_set = set(new_sol[0]) if new_sol else set()

        removed = sorted(list(old_set - new_set))
        added   = sorted(list(new_set - old_set))

        removed_uncovered = [t for t in removed if 0 <= t < self.num_tasks and not self.is_covered[t]]
        removed_covered   = [t for t in removed if 0 <= t < self.num_tasks and self.is_covered[t]]
        added_uncovered   = [t for t in added if 0 <= t < self.num_tasks and not self.is_covered[t]]
        added_covered     = [t for t in added if 0 <= t < self.num_tasks and self.is_covered[t]]

        if removed or added:
            self.get_logger().warning(
                f"[AUDIT:{where}] DIFF {label_old}->{label_new} | "
                f"removed_uncovered={removed_uncovered} removed_covered={removed_covered} "
                f"added_uncovered={added_uncovered} added_covered={added_covered}"
            )

    def _audit_population(self, where: str, pop):
        """Run _audit_solution on each member of a population."""
        if not pop:
            self.get_logger().warning(f"[AUDIT:{where}] population is empty or None")
            return
        for i, sol in enumerate(pop):
            self._audit_solution(f"{where}.pop[{i}]", sol)
    # ===================== END AUDIT HELPERS ========================

    def calculate_cost_matrix(self):
        """
        Returns a cost matrix representing the traversal cost from each task_pose to each other task_pose, this is
        constructed by calculating the drone_distance between all the task poses in the agents task_pose list.
        """
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    x1, y1 = self.task_poses[i][0], self.task_poses[i][1]
                    x2, y2 = self.task_poses[j][0], self.task_poses[j][1]
                    x_dist = abs(x2 - x1)
                    y_dist = abs(y2 - y1)
                    cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))
                    cost_map[i][j] = cost
        return cost_map

    def calculate_robot_cost_matrix(self):
        """
        Returns a cost map representing the traversal cost from each robot_pose to each task_pose calculated using
        drone_distance.
        """
        valid_robot_poses = [pose for pose in self.robot_poses if pose is not None]
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_robots, num_tasks))
        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                x1, y1 = robot_pose
                x2, y2 = task_pose
                x_dist = x2 - x1
                y_dist = y2 - y1
                cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))
                cost_map[i][j] = cost
        return cost_map

    def calculate_robot_inital_pose_cost_matrix(self):
        """
        Returns a cost map representing the traversal cost from each initial_robot_pose to each each task_pose calculated
        using drone_distance.
        """
        valid_robot_poses = self.initial_robot_poses
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_robots, num_tasks))
        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                x1, y1 = robot_pose
                x2, y2 = task_pose
                x_dist = x2 - x1
                y_dist = y2 - y1
                cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))
                cost_map[i][j] = cost
        self.robot_initial_pose_cost_matrix = cost_map

    def kill_robot_callback(self, msg, failed_agent_true_id):
        """
        This Fails a robot.
        """
        print(f"Kill signal received for agent: {failed_agent_true_id}")
        if failed_agent_true_id == self.true_agent_ID:
            self.am_i_failed = True
            if msg.data:
                goal_pose = SimplePosition()
                goal_pose.x_position = self.robot_poses[self.true_agent_ID][0]
                goal_pose.y_position = self.robot_poses[self.true_agent_ID][1]
                self.goal_pose_publisher.publish(goal_pose)
                self.current_task = None

    def purge_agent(self, purge_agent_true_id):
        """
        Removes a "purged agent" from all solutions.
        """
        print("Purging")
        agent_idx = purge_agent_true_id - 1 - sum(self.failed_agents[:purge_agent_true_id])
        num_agents = len(self.population[0][1]) if self.population else 0

        def update_solution(solution):
            if solution is None:
                return None
            order, allocations = solution
            if num_agents <= 1:
                return (order, allocations)

            new_order = list(order)
            new_alloc = list(allocations)

            new_owner_idx = agent_idx - 1 if agent_idx > 0 else len(new_alloc) - 1
            counter = 0
            for i in range(agent_idx):
                counter += new_alloc[i]
            start_idx = counter
            end_idx = start_idx + new_alloc[agent_idx]
            purged_tasks = new_order[start_idx:end_idx]

            print(f"alloc_before: {agent_idx}: {new_alloc}")
            del new_alloc[agent_idx]
            del new_order[start_idx:end_idx]
            new_owner_idx = agent_idx - 1 if agent_idx > 0 else len(new_alloc) - 1
            print(f"alloc_after: {agent_idx}: {new_alloc}")
            counter = 0
            print(f"{new_owner_idx + 1}")
            for i in range(new_owner_idx + 1):
                counter += new_alloc[i]

            if agent_idx == 0:
                new_order.extend(purged_tasks)
            else:
                new_order[counter:counter] = purged_tasks

            new_alloc[new_owner_idx] += len(purged_tasks)
            return (new_order, new_alloc)

        # Update all relevant solutions with DIFF/AUDIT
        for idx, solution in enumerate(self.population):
            before = solution
            self.population[idx] = update_solution(solution)
            self._diff_solutions(f"purge_agent.pop[{idx}]", before, self.population[idx])
            self._audit_solution(f"purge_agent.pop[{idx}]", self.population[idx])

        before_cur = self.current_solution
        self.current_solution = update_solution(self.current_solution)
        self._diff_solutions("purge_agent.current", before_cur, self.current_solution)
        self._audit_solution("purge_agent.current", self.current_solution)

        before_coal = self.coalition_best_solution
        self.coalition_best_solution = update_solution(self.coalition_best_solution)
        self._diff_solutions("purge_agent.coalition", before_coal, self.coalition_best_solution)
        self._audit_solution("purge_agent.coalition", self.coalition_best_solution)
        print(f"Agent {self.agent_ID}: {self.coalition_best_solution}")

    def revive_robot_callback(self, msg):
        """
        Revives the agent.
        """
        self.am_i_failed = False
        self.is_agent_tobe_purged = False
        self.last_purge_agent_true_id = None
        agent_id = self.true_agent_ID
        print(f"Revive request received for Agent {agent_id}")

        if not self.failed_agents[agent_id - 1] and agent_id != self.true_agent_ID:
            print(f"Agent {agent_id} is already active. Ignoring revive request.")
            return

        if self.run_timer is None:
            print(f"Restarting run_step for Agent {agent_id}.")
            self.run_timer = self.create_timer(0.01, self.run_step, callback_group=self.me_cb_group)

        if self.environmental_representation_timer is None:
            self.environmental_representation_timer = self.create_timer(5,
                                                                        self.environmental_representation_timer_callback,
                                                                        callback_group=self.cb_group)

        if self.run_goal_publisher_timer is None:
            self.run_goal_publisher_timer = self.create_timer(0.5, self.publish_goal_pose,
                                                              callback_group=self.cb_group)

        if self.solution_publisher_timer is None:
            self.solution_publisher_timer = self.create_timer(2, self.regular_solution_publish_timer,
                                                              callback_group=self.cb_group)

        if self.environmental_representation_subscriber is None:
            self.environmental_representation_subscriber = self.create_subscription(
                EnvironmentalRepresentation,
                '/environmental_representation',
                self.environmental_representation_callback,
                10,
                callback_group=self.cb_group
            )

        print(f"Agent {agent_id} successfully revived and all timers restarted.")

    def unpurge_agent(self, agent_id):
        """
        Reintroduces the specified agent into all the solutions, and recalculates robot cost matrix.
        """
        print(f"Reviving Agent {agent_id}...")
        self.failed_agents[agent_id - 1] = False
        self.purged_agents[agent_id - 1] = False
        self.is_agent_tobe_purged = False

        def reintegrate_agent(solution):
            if solution is None:
                return None
            order, allocations = solution
            new_order = list(order)
            new_alloc = list(allocations)
            if agent_id - 1 < len(new_alloc):
                new_alloc.insert(agent_id - 1, 0)
                print(f"Inserted empty path for Agent {agent_id}")
            return (new_order, new_alloc)

        for idx, solution in enumerate(self.population):
            before = solution
            self.population[idx] = reintegrate_agent(solution)
            self._diff_solutions(f"unpurge_agent.pop[{idx}]", before, self.population[idx])
            self._audit_solution(f"unpurge_agent.pop[{idx}]", self.population[idx])

        before_cur = self.current_solution
        self.current_solution = reintegrate_agent(self.current_solution)
        self._diff_solutions("unpurge_agent.current", before_cur, self.current_solution)
        self._audit_solution("unpurge_agent.current", self.current_solution)

        before_coal = self.coalition_best_solution
        self.coalition_best_solution = reintegrate_agent(self.coalition_best_solution)
        self._diff_solutions("unpurge_agent.coalition", before_coal, self.coalition_best_solution)
        self._audit_solution("unpurge_agent.coalition", self.coalition_best_solution)

        self.new_robot_cost_matrix = self.calculate_robot_cost_matrix()
        self.is_new_robot_cost_matrix = True

    def _active_agent_indices(self):
        """Indices of robots that are not purged/failed (in coalition space)."""
        return [i for i in range(self.num_tsp_agents) if not self.purged_agents[i]]

    def _voronoi_partition(self):
        """
        Assign each task to the nearest active robot by Euclidean distance from robot_poses.
        """
        active = self._active_agent_indices()
        tasks_per_agent = [[] for _ in range(self.num_tsp_agents)]
        robot_xy = np.array([self.robot_poses[i] for i in active], dtype=float)
        task_xy = np.array(self.task_poses, dtype=float)
        dists = ((task_xy[:, None, :] - robot_xy[None, :, :]) ** 2).sum(axis=2)
        nearest_active_idx = np.argmin(dists, axis=1)
        for t, a_local in enumerate(nearest_active_idx):
            a_true = active[a_local]
            if not self.is_covered[t]:
                tasks_per_agent[a_true].append(t)
        return tasks_per_agent

    def _nn_order(self, tasks, start_xy, jitter=0.0, rng=None):
        """
        Nearest-neighbor ordering starting from the task closest to start_xy.
        """
        if not tasks:
            return []
        rng = rng or random
        pts = np.array([self.task_poses[t] for t in tasks], dtype=float)
        if jitter > 0:
            pts = pts + rng.uniform(-jitter, jitter) * np.ones_like(pts)
        start_d = np.sum((pts - np.array(start_xy)) ** 2, axis=1)
        first_idx = int(np.argmin(start_d))
        remaining = list(range(len(tasks)))
        order_idx = [remaining.pop(first_idx)]
        cur = order_idx[0]
        while remaining:
            d = np.sum((pts[remaining] - pts[cur]) ** 2, axis=1)
            k = int(np.argmin(d))
            order_idx.append(remaining.pop(k))
            cur = order_idx[-1]
        return [tasks[i] for i in order_idx]

    def _two_opt_once(self, order):
        if len(order) < 4:
            return order, False
        pts = np.array([self.task_poses[t] for t in order], dtype=float)

        def seg_len(a, b):
            return float(np.linalg.norm(pts[b] - pts[a]))

        n = len(order)
        for i in range(n - 3):
            for k in range(i + 2, n - 1):
                a, b = i, i + 1
                c, d = k, k + 1
                old = seg_len(a, b) + seg_len(c, d)
                new = seg_len(a, c) + seg_len(b, d)
                if new + 1e-12 < old:
                    cand = order[:a + 1] + list(reversed(order[a + 1:c + 1])) + order[c + 1:]
                    return cand, True
        return order, False

    def _two_opt_polish(self, order, max_passes=2):
        """Run a couple of 2-opt passes; always pass/receive (order, flag)."""
        if isinstance(order, np.ndarray):
            cur = order.tolist()
        elif isinstance(order, tuple):
            cur = list(order)
        elif isinstance(order, list):
            cur = order
        else:
            return []
        for _ in range(max_passes):
            cur, imp = self._two_opt_once(cur)
            if not imp:
                break
        return cur

    def generate_population_voronoi(self, use_two_opt=True):
        """
        Build an initial population using Voronoi assignment + simple route heuristic per agent.
        """
        if not all(p is not None for p in self.robot_poses):
            raise RuntimeError("Robot poses are required before Voronoi initialization.")

        population = []
        rng = random.Random()

        def _dist_xy(idx_task, xy):
            tx, ty = self.task_poses[idx_task]
            return (tx - xy[0]) ** 2 + (ty - xy[1]) ** 2

        def _ensure_min_one(per_agent_routes):
            total_tasks = sum(len(r) for r in per_agent_routes)
            if total_tasks == 0:
                return
            needers = [a for a, r in enumerate(per_agent_routes) if len(r) == 0]
            if not needers:
                return
            donors = {i for i, r in enumerate(per_agent_routes) if len(r) > 1}
            for a in needers:
                if not donors:
                    break
                src = max(donors, key=lambda i: len(per_agent_routes[i]))
                src_route = per_agent_routes[src]
                if len(src_route) <= 1:
                    donors.discard(src)
                    continue
                dest_xy = self.robot_poses[a]
                k, task_to_move = min(
                    enumerate(src_route),
                    key=lambda it: _dist_xy(it[1], dest_xy)
                )
                src_route.pop(k)
                per_agent_routes[a].append(task_to_move)
                if len(per_agent_routes[src]) <= 1:
                    donors.discard(src)
            if use_two_opt:
                for i, r in enumerate(per_agent_routes):
                    if len(r) >= 4 and rng.random() < 0.5:
                        per_agent_routes[i] = self._two_opt_polish(r, max_passes=1)

        for sol_idx in range(self.pop_size):
            tasks_per_agent = self._voronoi_partition()
            per_agent_routes = []
            for a in range(self.num_tsp_agents):
                tasks = tasks_per_agent[a]
                if not tasks:
                    per_agent_routes.append([])
                    continue
                start_xy = self.robot_poses[a]
                jitter = 0.01 * rng.random()
                route = self._nn_order(tasks, start_xy, jitter=jitter, rng=rng)
                if use_two_opt and rng.random() < 0.7:
                    route = self._two_opt_polish(route, max_passes=2)
                route = [t for t in route if not self.is_covered[t]]
                per_agent_routes.append(route)

            _ensure_min_one(per_agent_routes)
            ordered_task_list = [t for r in per_agent_routes for t in r]
            allocation_counts = [len(r) for r in per_agent_routes]
            candidate = (ordered_task_list, allocation_counts)
            try:
                self._audit_solution(f"gen_voronoi.sol_{sol_idx}", candidate, level="INFO")
            except Exception:
                pass
            population.append(candidate)

        return population

    def generate_population(self):
        """
        Generates the initial solution population randomly.
        """
        population = []
        print("Generating Random Population...")

        for i in range(self.pop_size):
            print(f"Generating solution {i}")
            tasks = list(range(self.num_tasks))
            random.shuffle(tasks)
            agent_assignments = [[] for _ in range(self.num_tsp_agents)]
            for task in tasks:
                chosen_agent = random.randint(0, self.num_tsp_agents - 1)
                agent_assignments[chosen_agent].append(task)
            ordered_task_list = [task for agent_tasks in agent_assignments for task in agent_tasks]
            task_allocation_counts = [len(agent_tasks) for agent_tasks in agent_assignments]
            population.append((ordered_task_list, task_allocation_counts))

        print("Generated random population")
        print(f"{population[0]}")
        return population

    def set_coalition_best_solution(self, solution):
        """
        Sets coalition best and assigns next task.
        """
        self.coalition_best_solution = deepcopy(solution)
        self.assign_next_task(solution)

    def assign_next_task(self, solution):
        """
        Extracts and sets the current next task for the agent from a given solution.
        """
        try:
            if solution is None:
                print(f"[assign_next_task] Agent {self.agent_ID}: solution is None")
                self.current_task = None
                return

            if not isinstance(solution, (tuple, list)) or len(solution) < 2:
                print(
                    f"[assign_next_task] Agent {self.agent_ID}: solution has invalid type/length: {type(solution)}, len={len(solution) if hasattr(solution, '__len__') else 'n/a'}")
                self.current_task = None
                return

            ordered_task_list, allocation_counts = solution

            if ordered_task_list is None or allocation_counts is None:
                print(
                    f"[assign_next_task] Agent {self.agent_ID}: solution elements are None (ordered={ordered_task_list is None}, allocs={allocation_counts is None})")
                self.current_task = None
                return

            if not isinstance(allocation_counts, (list, tuple)):
                print(
                    f"[assign_next_task] Agent {self.agent_ID}: allocation_counts is not a list/tuple (got {type(allocation_counts)})")
                self.current_task = None
                return

            if self.agent_ID < 0 or self.agent_ID >= len(allocation_counts):
                raise ValueError(f"Invalid robot_id {self.agent_ID} for alloc length {len(allocation_counts)}")

            if self.is_covered is None:
                print(f"[assign_next_task] Agent {self.agent_ID}: self.is_covered is None")
                self.current_task = None
                return

            num_tasks = allocation_counts[self.agent_ID]
            if not isinstance(num_tasks, int):
                print(
                    f"[assign_next_task] Agent {self.agent_ID}: num_tasks not int (got {type(num_tasks)} -> {num_tasks})")
                self.current_task = None
                return

            if num_tasks <= 0:
                self.current_task = None
                return

            start_index = sum(allocation_counts[:self.agent_ID])
            end_index = start_index + num_tasks

            if start_index < 0 or end_index > len(ordered_task_list):
                print(
                    f"[assign_next_task] Agent {self.agent_ID}: slice OOB (start={start_index}, end={end_index}, len(ordered)={len(ordered_task_list)})")
                end_index = min(end_index, len(ordered_task_list))
                start_index = max(0, min(start_index, end_index))

            agent_tasks = ordered_task_list[start_index:end_index]

            for task in agent_tasks:
                if task is None:
                    print(
                        f"[assign_next_task] Agent {self.agent_ID}: encountered None task in slice {start_index}:{end_index}")
                    continue
                if task < 0 or task >= len(self.is_covered):
                    print(
                        f"[assign_next_task] Agent {self.agent_ID}: bad task index {task} (covered len {len(self.is_covered)})")
                    continue
                if not self.is_covered[task]:
                    self.current_task = task
                    if self.is_finished == True:
                        self.is_finished = False
                        m = FinishedCoverage()
                        m.finished = False
                        m.robot_id = self.agent_ID
                        self.finished_coverage_pub.publish(m)
                    return

            self.current_task = None

        except Exception as ex:
            try:
                dbg = {
                    "agent_id": self.agent_ID,
                    "alloc_len": len(solution[1]) if (
                                solution and isinstance(solution, (tuple, list)) and len(solution) > 1 and solution[
                            1] is not None) else "n/a",
                    "ordered_len": len(solution[0]) if (
                                solution and isinstance(solution, (tuple, list)) and len(solution) > 0 and solution[
                            0] is not None) else "n/a",
                }
            except Exception:
                dbg = {"agent_id": self.agent_ID, "alloc_len": "err", "ordered_len": "err"}
            print(f"Agent: {self.agent_ID} has {ex} while assigning tasks. Debug: {dbg}")
            self.current_task = None

    def select_solution(self):
        """
        Finds and returns the fittest solution
        """
        best_solution = min(self.population, key=lambda sol: Fitness.fitness_function(
            sol, self.cost_matrix))
        return best_solution

    def update_experience(self, condition, operator, gain):
        """
        Adds details of the current iteration to the experience memory.
        """
        op_order = self.intensifiers + self.diversifiers
        try:
            op_col = op_order.index(operator)
        except Exception:
            op_col = int(operator)
        self.previous_experience.append([condition, op_col, gain])

    def individual_learning_old(self):
        cumulative_gain = 0.0
        best = 0.0
        idx_min = -1
        for i, (_, _, gain) in enumerate(self.previous_experience):
            cumulative_gain += gain
            if cumulative_gain < best:
                best = cumulative_gain
                idx_min = i
        elements_before_best = self.previous_experience[:idx_min + 1] if idx_min != -1 else []
        pairs = {(cond, int(op_col)) for cond, op_col, _ in elements_before_best}
        for row, col in pairs:
            self.weight_matrix.weights[row][col] += self.eta
        return self.weight_matrix.weights

    def individual_learning_step(self, cur_condition, next_condition, op_col, improved: bool):
        """
        TD(0) update on the *latest* (state, action) only.
        - cur_condition, next_condition: row indices in the condition matrix.
        - op_col: column index of the operator used this iteration.
        - improved: True if this iteration improved local best (reward positive), else negative.
        """
        row = int(cur_condition)
        col = int(op_col)
        current_q = self.weight_matrix.weights[row][col]
        max_next_q = max(self.weight_matrix.weights[next_condition]) if next_condition is not None else 0.0
        reward = self.positive_reward if improved else self.negative_reward

        updated_q = current_q + self.lr * (reward + self.gamma_decay * max_next_q - current_q)
        self.weight_matrix.weights[row][col] = max(updated_q, 1e-6)

    def mimetism_learning(self, received_weights, rho):
        """
        Perform mimetism learning by updating self.weight_matrix.weights using multiple sets of received weights.
        """
        for weight_set in received_weights:
            if len(weight_set) != len(self.weight_matrix.weights) or len(weight_set[0]) != len(
                    self.weight_matrix.weights[0]):
                raise ValueError("Dimension mismatch between weight_matrix.weights and received weights.")
            for i in range(len(self.weight_matrix.weights)):
                for j in range(len(self.weight_matrix.weights[i])):
                    self.weight_matrix.weights[i][j] = (
                            (1 - rho) * self.weight_matrix.weights[i][j] + rho * weight_set[i][j]
                    )

    def stopping_criterion(self, iteration_count):
        """
        Returns true if current number of iterations exceeds the limit.
        """
        return iteration_count > self.num_iterations

    def end_of_di_cycle(self, cycle_count):
        """
        Returns true if the current cycle is equal to or over the length of a DI cycle.
        """
        if cycle_count >= self.di_cycle_length:
            return True
        return False

    def weight_update_callback(self, msg):
        """
        Stores received weight matrices.
        """
        received_weights = self.weight_matrix.unpack_weights(weights_msg=msg, agent_id=self.agent_ID)
        if received_weights is not None:
            self.received_weight_matrices.append(received_weights)

    def finished_coverage_callback(self, msg):
        self.finished_robots[int(msg.robot_id)] = bool(msg.finished)
        print(f"Finished robots:  {self.finished_robots}")
        if all(self.finished_robots):
            print("Coverage Complete")
            self.destroy_node()
            rclpy.shutdown()
            print("ROS2 system shut down.")

    # --- Helpers ---

    def _active_agent_index(self):
        """
        Returns this agent's index in the 'active' (non-purged) agent list.
        """
        active_indices = [i for i, purged in enumerate(self.purged_agents) if not purged]
        return active_indices.index(self.true_agent_ID)

    def _segment_bounds(self, allocations, agent_idx):
        """
        Given allocations and an agent_idx in active indexing, return (start, end).
        """
        start = sum(allocations[:agent_idx])
        end = start + allocations[agent_idx]
        return start, end

    def _next_task_of_agent(self, solution, agent_idx):
        """
        Returns the next (first) task of agent_idx in the given solution, or None if none.
        """
        order, allocations = solution
        start, end = self._segment_bounds(allocations, agent_idx)
        if end > start:
            return order[start]
        return None

    def _evaluate_solution(self, solution):
        """
        Wrapper to evaluate fitness with your existing matrices.
        """
        order, allocations = solution
        return Fitness.fitness_function_robot_pose(
            solution,
            self.cost_matrix,
            [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged],
            self.robot_initial_pose_cost_matrix
        )

    def try_preserve_next_task(self, received_solution):
        """
        Attempt to modify 'received_solution' so that THIS agent keeps its current 'next task'
        (from self.coalition_best_solution). If modified solution has better fitness, return it.
        """
        if self.coalition_best_solution is None:
            return received_solution

        my_idx = self._active_agent_index()
        my_current_next = self._next_task_of_agent(self.coalition_best_solution, my_idx)
        if my_current_next is None:
            return received_solution

        r_order, r_alloc = received_solution
        order = list(r_order)
        alloc = list(r_alloc)

        try:
            task_pos = order.index(my_current_next)
        except ValueError:
            return received_solution

        cumulative = 0
        owner_idx = None
        for idx, a in enumerate(alloc):
            if task_pos < cumulative + a:
                owner_idx = idx
                break
            cumulative += a
        if owner_idx is None:
            return received_solution

        my_start, my_end = self._segment_bounds(alloc, my_idx)
        if owner_idx == my_idx and task_pos == my_start:
            return received_solution

        removed_task = order.pop(task_pos)
        alloc[owner_idx] -= 1
        if task_pos < my_start:
            my_start -= 1
            my_end -= 1
        order.insert(my_start, removed_task)
        alloc[my_idx] += 1

        if any(a < 0 for a in alloc):
            return received_solution

        modified = (order, alloc)
        base_f = self._evaluate_solution(received_solution)
        mod_f = self._evaluate_solution(modified)
        return modified if mod_f < base_f else received_solution

    def solution_update_callback(self, msg):
        """
        - Observe coverage from diffs (coalition -> received). Tasks missing in the
          received plan are treated as newly covered.
        - Sanitize the received plan: strip covered tasks per segment; dedupe per segment.
        - Compare against our sanitized coalition and adopt only if fitter.
        - Never trims uncovered tasks.
        """

        # ---- helpers (local) ---------------------------------------------------
        def _dedupe_preserve_segments(sol):
            """Remove duplicates per segment, keeping the earliest occurrence."""
            if not sol:
                return sol
            order, alloc = sol
            order = list(order)
            alloc = list(alloc)

            # map task -> first position we keep
            first_pos = {}
            for pos, t in enumerate(order):
                if t not in first_pos:
                    first_pos[t] = pos

            # rebuild per segment preserving earliest occurrences
            new_order, new_alloc = [], []
            cursor = 0
            for a_count in alloc:
                start, end = cursor, cursor + a_count
                cursor = end
                seg = []
                for i in range(start, end):
                    t = order[i]
                    if first_pos.get(t, i) == i:  # keep only first occurrence
                        seg.append(t)
                new_order.extend(seg)
                new_alloc.append(len(seg))
            return (new_order, new_alloc)

        def _sanitize_strip_covered(sol):
            """Strip already-covered tasks per segment (preserves segment sizes)."""
            if not sol:
                return sol
            order, alloc = sol
            # reuse your existing per-segment stripper
            return self.remove_covered_tasks_from_solution((list(order), list(alloc)))

        def _fitness(sol):
            """Safe fitness (returns +inf if matrices not ready). Lower is better."""
            if not sol:
                return float("inf")
            try:
                return Fitness.fitness_function_robot_pose(
                    sol,
                    self.cost_matrix,
                    [self.robot_cost_matrix[i] for i, p in enumerate(self.purged_agents) if not p],
                    self.robot_initial_pose_cost_matrix
                )
            except Exception as e:
                # Log once per callsite to aid debugging, but don't crash adoption.
                try:
                    self.get_logger().error(
                        f"[AUDIT:solution_update_callback.fitness_error] {type(e).__name__}: {e}"
                    )
                except Exception:
                    pass
                return float("inf")

        def _diff_observe_coverage(cur_sol, recv_sol):
            """Infer coverage: tasks in cur but missing in recv => handle_covered_task()."""
            if not cur_sol or not recv_sol:
                return
            cur_order, _ = cur_sol
            recv_order, _ = recv_sol
            recv_set = set(recv_order)
            newly = [t for t in cur_order
                     if 0 <= t < self.num_tasks and not self.is_covered[t] and t not in recv_set]
            if newly:
                self.get_logger().warning(
                    f"[AUDIT:solution_update_callback.observe] newly_covered={newly[:10]}"
                    f"{'...' if len(newly) > 10 else ''}"
                )
            for t in newly:
                # idempotent; also updates populations/coalition/current internally
                self.handle_covered_task(t)

        # ---- parse received ----------------------------------------------------
        try:
            recv_order = list(msg.order)
            recv_alloc = list(msg.allocations)
        except Exception:
            # Fallback if the subscriber already hands us (order, alloc)
            if isinstance(msg, (tuple, list)) and len(msg) == 2:
                recv_order, recv_alloc = list(msg[0]), list(msg[1])
            else:
                self.get_logger().error("[AUDIT:solution_update_callback] bad incoming message")
                return

        received = (recv_order, recv_alloc)

        # For logging: diff old coalition -> raw received (before sanitize)
        if self.coalition_best_solution:
            self._diff_solutions("solution_update_callback", self.coalition_best_solution, received,
                                 label_old="coalition", label_new="received")

        # ---- (1) Observe coverage from missing tasks ---------------------------
        cur_snapshot = self.coalition_best_solution or self.current_solution
        _diff_observe_coverage(cur_snapshot, received)

        # ---- (2) Sanitize the received solution --------------------------------
        # strip covered tasks per segment, then dedupe per segment
        cand = _sanitize_strip_covered(received)
        cand = _dedupe_preserve_segments(cand)
        self._audit_solution("solution_update_callback.candidate_sanitized", cand)

        # Optional: log if covered tasks were present in the raw received plan
        covered_in_recv = [t for t in recv_order
                           if 0 <= t < self.num_tasks and self.is_covered[t]]
        if covered_in_recv:
            self.get_logger().warning(
                f"[AUDIT:solution_update_callback.strip_recv] stripped_covered={covered_in_recv[:10]}"
                f"{'...' if len(covered_in_recv) > 10 else ''}"
            )

        # ---- (3) Sanitize our coalition for a fair compare ---------------------
        cur = self.coalition_best_solution
        if cur:
            cur = _sanitize_strip_covered(cur)
            cur = _dedupe_preserve_segments(cur)
            self._audit_solution("solution_update_callback.coalition_sanitized", cur)

        # ---- (4) Compare and adopt if strictly better (lower fitness) ----------
        cand_f = _fitness(cand)
        cur_f = _fitness(cur) if cur else float("inf")

        # If candidate is empty after sanitize/dedupe, ignore.
        if not cand or not cand[0]:
            self.get_logger().info("[AUDIT:solution_update_callback.ignored] empty_candidate_after_sanitize")
            return

        if cand_f < cur_f:
            self.set_coalition_best_solution(cand)
            self.get_logger().info(
                f"[AUDIT:solution_update_callback.adopted] score_old={cur_f:.6f} score_new={cand_f:.6f}"
            )
            # (Optionally publish here; you already have a periodic publisher.)
        else:
            self.get_logger().info(
                f"[AUDIT:solution_update_callback.ignored] score_old={cur_f:.6f} score_new={cand_f:.6f}"
            )

    def remove_extra_tasks(self, solution, target_task_count):
        """
        Trim the *end* of each agent's segment until global task count equals target_task_count.
        """
        if solution is None:
            return None

        self._audit_solution("remove_extra_tasks.IN", solution)

        order, allocations = deepcopy(solution)
        order, allocations = self.remove_covered_tasks_from_solution((order, allocations))

        total = sum(allocations)
        if total <= target_task_count:
            out = (order, allocations)
            self._audit_solution("remove_extra_tasks.OUT_nochange", out)
            return out

        # Simple trimming: cycle agents, remove from the tail of each segment
        idx = len(allocations) - 1
        while sum(allocations) > target_task_count and any(a > 0 for a in allocations):
            if allocations[idx] > 0:
                # remove last task of this agent's segment
                end = sum(allocations[:idx + 1])
                start = end - allocations[idx]
                if end - 1 >= start:
                    removed_task = order.pop(end - 1)
                    allocations[idx] -= 1
                    self.get_logger().warning(f"[AUDIT:remove_extra_tasks] trimming task={removed_task} from agent={idx}")
            idx = (idx - 1) % len(allocations)

        out = (order, allocations)
        self._diff_solutions("remove_extra_tasks", solution, out)
        self._audit_solution("remove_extra_tasks.OUT", out)
        return out

    def global_pose_callback(self, msg):
        """
        Receives and processes a global pose from another agent.
        """
        agent = None
        try:
            agent = msg.robot_id
            if self.initial_robot_poses[agent] is None:
                self.initial_robot_poses[agent] = (msg.x_position, msg.y_position)

            task = deepcopy(self.current_task)

            if msg:
                self.robot_poses[agent] = (msg.x_position, msg.y_position)

            if all(pose is not None for pose in self.robot_poses) and self.is_loop_started is False and self.is_all_poses is False:
                self.is_all_poses = True
                self.new_robot_cost_matrix = self.calculate_robot_cost_matrix()
                self.is_new_robot_cost_matrix = True
                self.calculate_robot_inital_pose_cost_matrix()

                if self.population is None:
                    if self.is_generating is False:
                        self.is_generating = True
                        self.population = self.generate_population_voronoi()
                        self._audit_population("population_voronoi_init", self.population)
                        self.current_solution = self.select_solution()
                        self._audit_solution("select_solution.current", self.current_solution)

                self.run_timer = self.create_timer(0.1, self.run_step, callback_group=self.me_cb_group)
                self.is_loop_started = True

            if self.task_poses is not None and task is not None:
                x = msg.x_position
                y = msg.y_position
                goal_x = self.task_poses[task][0]
                goal_y = self.task_poses[task][1]
                distance = math.sqrt(((x - goal_x) ** 2) + ((y - goal_y) ** 2))

                if agent == self.true_agent_ID and distance < 0.1:
                    self.task_covered = task
                    self.is_new_task_covered = True
                    self.is_covered[task] = True
                    self.assign_next_task(self.coalition_best_solution)

            print(f"{self.coalition_best_solution[1][self.agent_ID - 1]}")
            if self.coalition_best_solution is not None and self.coalition_best_solution[1][self.agent_ID - 1] == 0:
                x = msg.x_position
                y = msg.y_position
                goal_x = self.initial_robot_poses[self.true_agent_ID][0]
                goal_y = self.initial_robot_poses[self.true_agent_ID][1]
                distance = math.sqrt(((x - goal_x) ** 2) + ((y - goal_y) ** 2))

                if agent == self.true_agent_ID and distance < 0.1:
                    print(f"[INFO] Agent {self.agent_ID} finished coverage!.")
                    self.is_finished = True
                    m = FinishedCoverage()
                    m.finished = True
                    m.robot_id = self.agent_ID
                    self.finished_coverage_pub.publish(m)

        except Exception as e:
            error_message = (
                f"[ERROR] Exception in global_pose_callback for agent {agent}:\n"
                f"    Error Type: {type(e).__name__}\n"
                f"    Error Message: {e}\n"
                f"    Stack Trace:\n{traceback.format_exc()}"
            )
            print(f"{self.coalition_best_solution}")
            print(error_message)
            self.get_logger().error(error_message)

    def handle_covered_task(self, covered_task):
        """
        Handles a task covered by this agent.
        Updates all solutions to reflect the new state and publishes the new environmental representation.
        """
        if not self.is_loop_started:
            return

        if 0 <= covered_task < len(self.is_covered) and self.is_covered[covered_task]:
            return

        self.get_logger().info(f"[AUDIT:covered] Agent {self.true_agent_ID} confirmed covered_task={covered_task}")
        self.is_covered[covered_task] = True

        rep = EnvironmentalRepresentation()
        rep.agent_id = self.true_agent_ID
        rep.is_covered = list(self.is_covered)
        self.environmental_representation_publisher.publish(rep)

        def update_solution(solution):
            if solution is None:
                return None
            order, allocations = solution
            if not order or not allocations:
                return solution

            new_order = list(order)
            new_alloc = list(allocations)

            try:
                pos = new_order.index(covered_task)
            except ValueError:
                return (new_order, new_alloc)

            cum = 0
            owner_idx = None
            for idx, a in enumerate(new_alloc):
                if pos < cum + a:
                    owner_idx = idx
                    break
                cum += a

            del new_order[pos]
            if owner_idx is not None:
                new_alloc[owner_idx] = max(0, new_alloc[owner_idx] - 1)

            if sum(new_alloc) != len(new_order):
                fixed = []
                cursor = 0
                for a in new_alloc:
                    take = min(a, max(0, len(new_order) - cursor))
                    fixed.append(take)
                    cursor += take
                new_alloc = fixed

            return (new_order, new_alloc)

        for i, sol in enumerate(self.population):
            before = sol
            self.population[i] = update_solution(sol)
            self._diff_solutions(f"handle_covered_task.population[{i}]", before, self.population[i],
                                 label_old="before", label_new="after")
            self._audit_solution(f"handle_covered_task.population[{i}]", self.population[i])

        before_cur = self.current_solution
        self.current_solution = update_solution(self.current_solution)
        self._diff_solutions("handle_covered_task.current", before_cur, self.current_solution)
        self._audit_solution("handle_covered_task.current", self.current_solution)

        before_coal = self.coalition_best_solution
        self.coalition_best_solution = update_solution(self.coalition_best_solution)
        self._diff_solutions("handle_covered_task.coalition", before_coal, self.coalition_best_solution)
        self._audit_solution("handle_covered_task.coalition", self.coalition_best_solution)

        self.assign_next_task(self.coalition_best_solution or self.current_solution)

    def environmental_representation_callback(self, msg):
        """
        On receiving an environmental representation, handle the coverage.
        """
        if not self.am_i_failed:
            agent_id = msg.agent_id
            self.last_env_rep_timestamps[agent_id] = time()

            for i in range(len(msg.is_covered)):
                if msg.is_covered[i] and not self.is_covered[i]:
                    self.handle_covered_task(i)
                    self.is_covered[i] = True
            if self.failed_agents[agent_id - 1]:
                self.agent_to_revive = agent_id

    def check_stale_agents(self):
        """
        If a message hasn't been received by an agent within a threshold period, set the agent as failed and to be purged.
        """
        if not self.am_i_failed:
            current_time = time()
            timeout_threshold = 15
            for agent_id, last_time in self.last_env_rep_timestamps.items():
                if current_time - last_time > timeout_threshold and self.failed_agents[agent_id - 1] is False:
                    self.is_agent_tobe_purged = True
                    self.last_purge_agent_true_id = agent_id
                    self.failed_agents[agent_id - 1] = True
                    if self.true_agent_ID > agent_id:
                        self.agent_ID = self.agent_ID - 1
                    self.get_logger().warning(
                        f"Agent {agent_id} has not sent an update for {current_time - last_time:.2f} seconds.")

    def environmental_representation_timer_callback(self):
        """
        Publish this agent's environmental representation.
        """
        if self.am_i_failed:
            self.environmental_representation_timer.cancel()
            self.environmental_representation_timer = None

        rep = EnvironmentalRepresentation()
        rep.agent_id = self.true_agent_ID
        rep.is_covered = self.is_covered
        self.environmental_representation_publisher.publish(rep)

    def publish_goal_pose(self):
        """
        Publish this agent's current goal pose to the flight controller.
        """
        if self.am_i_failed:
            self.run_goal_publisher_timer.cancel()
            self.run_goal_publisher_timer = None
        if self.current_task is not None and self.task_poses:
            goal_pose = SimplePosition()
            goal_pose.robot_id = self.agent_ID
            goal_pose.x_position = float(self.task_poses[self.current_task][0])
            goal_pose.y_position = float(self.task_poses[self.current_task][1])
            self.goal_pose_publisher.publish(goal_pose)
        else:
            try:
                if self.am_i_failed:
                    goal_pose = SimplePosition()
                    goal_pose.x_position = self.robot_poses[self.true_agent_ID][0]
                    goal_pose.y_position = self.robot_poses[self.true_agent_ID][1]
                    self.goal_pose_publisher.publish(goal_pose)
                    return

                if self.initial_robot_poses[self.true_agent_ID] is not None:
                    goal_pose = SimplePosition()
                    goal_pose.x_position = self.initial_robot_poses[self.true_agent_ID][0]
                    goal_pose.y_position = self.initial_robot_poses[self.true_agent_ID][1]
                    self.goal_pose_publisher.publish(goal_pose)
            except:
                print("Goal pose error")

    def select_random_solution(self):
        """
        Sample a random solution from the population.
        """
        temp_solution = sample(population=self.population, k=1)[0]
        if temp_solution != self.current_solution:
            return temp_solution

    def robot_cost_matrix_recalculation(self):
        """Recalculate and update the robot cost matrix."""
        if all(pose is not None for pose in self.robot_poses):
            self.new_robot_cost_matrix = self.calculate_robot_cost_matrix()
            self.is_new_robot_cost_matrix = True

    def regular_solution_publish_timer(self):
        """
        Publishes the coalition best solution.
        """
        if self.am_i_failed:
            self.solution_publisher_timer.cancel()
            self.solution_publisher_timer = None
        if self.coalition_best_solution is not None:
            solution = Solution()
            solution.id = self.agent_ID
            solution.order = self.coalition_best_solution[0]
            solution.allocations = self.coalition_best_solution[1]
            self.solution_publisher.publish(solution)

    def remove_covered_tasks_from_solution(self, solution):
        """
        Remove already-covered tasks while preserving per-agent boundaries.
        """
        if solution is None:
            return None

        order, allocations = deepcopy(solution)
        new_order = []
        new_allocs = []

        idx = 0
        for count in allocations:
            seg = order[idx:idx + count]
            seg_kept = [t for t in seg if not self.is_covered[t]]
            new_order.extend(seg_kept)
            new_allocs.append(len(seg_kept))
            idx += count

        audited = (new_order, new_allocs)
        self._audit_solution("remove_covered_tasks_from_solution", audited)
        return audited

    def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """
        print("RunStep")

        if self.am_i_failed:
            return

        if self.is_new_robot_cost_matrix:
            self.robot_cost_matrix = self.new_robot_cost_matrix
            self.is_new_robot_cost_matrix = False

        if self.am_i_failed:
            print(f"Im agent {self.true_agent_ID} and im still here!")

        if self.is_new_task_covered:
            self.handle_covered_task(self.task_covered)
            self.task_covered = -1
            self.is_new_task_covered = False

        if self.current_solution is not None:

            if self.current_solution:
                before = self.current_solution
                self.current_solution = self.remove_covered_tasks_from_solution(self.current_solution)
                self._diff_solutions("run_step.clean_current", before, self.current_solution,
                                     label_old="current_before", label_new="current_after")
            if self.coalition_best_solution:
                before_c = self.coalition_best_solution
                self.coalition_best_solution = self.remove_covered_tasks_from_solution(self.coalition_best_solution)
                self._diff_solutions("run_step.clean_coalition", before_c, self.coalition_best_solution,
                                     label_old="coal_before", label_new="coal_after")

            if self.is_agent_tobe_purged and not self.purged_agents[self.last_purge_agent_true_id - 1] and len(
                    self.current_solution[1]) > self.failed_agents.count(False):
                print("Purging agent")
                self.purge_agent(self.last_purge_agent_true_id)
                self.purged_agents[self.last_purge_agent_true_id - 1] = True
                self.is_agent_tobe_purged = None

            if self.agent_to_revive:
                if self.agent_ID >= self.agent_to_revive:
                    print(f"I was robot: {self.agent_ID}, now becoming robot: {self.agent_ID + 1}")
                    self.agent_ID = self.agent_ID + 1
                self.unpurge_agent(self.agent_to_revive)
                self.agent_to_revive = None

            num_false = self.failed_agents.count(False)

            if self.stopping_criterion(self.iteration_count):
                self.run_timer.cancel()
                return

            condition = ConditionFunctions.perceive_condition_row(self.previous_experience, self.intensifiers,
                                                                  self.diversifiers)

            if self.no_improvement_attempt_count >= self.no_improvement_attempts:
                self.current_solution = self.select_random_solution()
                self._audit_solution("run_step.select_random_solution", self.current_solution)
                self.no_improvement_attempt_count = 0



            operator = OperatorFunctions.choose_operator(self.weight_matrix.weights, condition,
                                                         self.intensifiers + self.diversifiers)

            # Column index for the chosen operator (matches your weight matrix layout)
            op_order = self.intensifiers + self.diversifiers
            try:
                op_col = op_order.index(operator)
            except Exception:
                op_col = int(operator)

            c_new = None
            try:
                c_new = run_with_timeout(
                    OperatorFunctions.apply_op,
                    args=(
                        operator,
                        self.current_solution,
                        self.population,
                        self.cost_matrix,
                        [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged],
                        self.robot_initial_pose_cost_matrix
                    ),
                    timeout=5.0
                )
                if c_new is None:
                    print(f"[TIMEOUT] Operator {operator} took too long. Skipping this step.")
                    self.no_improvement_attempt_count += 1
                    return
                # DIFF/AUDIT after operator
                self._diff_solutions("apply_op", self.current_solution, c_new, label_old="current", label_new="c_new")
                self._audit_solution("apply_op.c_new", c_new)

            except Exception as e:
                self.get_logger().error(f"[CHK] Exception during operator apply: {e}")
                print(f"Issue with applying operator: {e}")
                print(f"New solution came out empty:\n old solution: {self.current_solution}\n"
                      f"operator applied {operator}")

            try:
                c_pres = self.preserve_next_task_if_better(self.current_solution, c_new)
                if c_pres is not None and c_pres is not c_new:
                    self._diff_solutions("preserve_next_task_if_better", c_new, c_pres,
                                         label_old="c_new", label_new="c_pres")
                    self._audit_solution("preserve_next_task_if_better.c_pres", c_pres)
                    c_new = c_pres
            except Exception as e:
                self.get_logger().warn(f"[preserve_next_task] skipped due to: {e}")

            new_solution_fitness = Fitness.fitness_function_robot_pose(
                c_new,
                self.cost_matrix,
                [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged],
                self.robot_initial_pose_cost_matrix
            )

            current_solution_fitness = Fitness.fitness_function_robot_pose(
                self.current_solution,
                self.cost_matrix,
                [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged],
                self.robot_initial_pose_cost_matrix
            )

            if self.local_best_solution:
                local_best_solution_fitness = Fitness.fitness_function_robot_pose(
                    self.local_best_solution,
                    self.cost_matrix,
                    [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged],
                    self.robot_initial_pose_cost_matrix
                )
            if self.coalition_best_solution:
                coalition_best_solution_fitness = Fitness.fitness_function_robot_pose(
                    self.coalition_best_solution,
                    self.cost_matrix,
                    [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged],
                    self.robot_initial_pose_cost_matrix,
                    islog=False
                )

            if c_new:
                gain = new_solution_fitness - current_solution_fitness
                self.update_experience(condition, operator, gain)

                if self.local_best_solution is None or new_solution_fitness < local_best_solution_fitness:
                    self.local_best_solution = deepcopy(c_new)
                    self.best_local_improved = True
                    self.no_improvement_attempt_count = 0
                else:
                    self.no_improvement_attempt_count += 1

                if self.coalition_best_solution is None or new_solution_fitness < coalition_best_solution_fitness:
                    self.set_coalition_best_solution(c_new)
                    self.coalition_best_agent = self.agent_ID
                    self.best_coalition_improved = True
                    solution = Solution()
                    solution.id = self.agent_ID
                    solution.order = self.coalition_best_solution[0]
                    solution.allocations = self.coalition_best_solution[1]
                    self.solution_publisher.publish(solution)

                self.current_solution = c_new
                self.di_cycle_count += 1

                # --- Per-iteration learning (TD(0)) ---
                # 'condition' is the state *before* applying the operator this iteration.
                # Define 'next_condition' as the perceived state *after* the step.
                next_condition = ConditionFunctions.perceive_condition_row(
                    self.previous_experience, self.intensifiers, self.diversifiers
                )
                self.individual_learning_step(condition, next_condition, op_col, self.best_local_improved)
                # Optional: keep only short memory so it doesn't grow unbounded
                if len(self.previous_experience) > 32:
                    self.previous_experience = self.previous_experience[-32:]

                self.best_local_improved = False

                if self.best_coalition_improved:
                    self.best_coalition_improved = False
                    msg = Weights()
                    msg_dict = self.weight_matrix.pack_weights(self.agent_ID)
                    msg.id = msg_dict["id"]
                    msg.rows = msg_dict["rows"]
                    msg.cols = msg_dict["cols"]
                    msg.weights = msg_dict["weights"]
                    self.weight_publisher.publish(msg)

                row_idx = ConditionFunctions.perceive_condition_row(
                    self.previous_experience, self.intensifiers, self.diversifiers
                )
                stuck_row = 1 + len(self.intensifiers)
                if self.end_of_di_cycle(self.di_cycle_count) or row_idx == stuck_row:

                    if self.received_weight_matrices:
                        self.mimetism_learning(self.received_weight_matrices, self.rho)
                        self.received_weight_matrices = []

                    self.previous_experience = []
                    self.di_cycle_count = 0

                self.iteration_count += 1
            else:
                print("Something went wrong with applying the operator and resulted in a None.")

        # Sentinel audits (periodic)
        try:
            if self.current_solution:
                self._audit_solution("sentinel.current", self.current_solution)
            if self.coalition_best_solution:
                self._audit_solution("sentinel.coalition", self.coalition_best_solution)
        except Exception:
            pass


def generate_problem(num_tasks):
    """
    Randomly generates a problem of size `number_tasks`
    :return: Randomly generated symmetrical cost matrix representing the problem
    """
    np.random.seed(0)
    size = num_tasks
    cost_matrix = np.random.randint(1, 100, size=(size, size))
    cost_matrix = (cost_matrix + cost_matrix.T) // 2
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


# Deneme

import concurrent.futures

def run_with_timeout(func, args=(), kwargs=None, timeout=1.0):
    """
    Runs a function with a timeout. Returns the result or None on timeout.
    """
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

def main(args=None):
    rclpy.init(args=args)

    temp_node = Node("parameter_loader")
    temp_node.declare_parameter("agent_id", 1)
    temp_node.declare_parameter("runtime", -1.0)
    temp_node.declare_parameter("learning_method", "Ferreira_et_al.")

    temp_node.declare_parameter("lr", 0.5)
    temp_node.declare_parameter("gamma_decay", 0.99)
    temp_node.declare_parameter("positive_reward", 1.0)
    temp_node.declare_parameter("negative_reward", -0.5)
    temp_node.declare_parameter("num_tsp_agents", 10)

    agent_id = temp_node.get_parameter("agent_id").value
    runtime = temp_node.get_parameter("runtime").value
    learning_method = temp_node.get_parameter("learning_method").value

    lr = temp_node.get_parameter("lr").value
    gamma_decay = temp_node.get_parameter("gamma_decay").value
    positive_reward = temp_node.get_parameter("positive_reward").value
    negative_reward = temp_node.get_parameter("negative_reward").value
    num_tsp_agents = temp_node.get_parameter("num_tsp_agents").value
    temp_node.destroy_node()

    node_name = f"cbm_population_agent_{agent_id}"
    agent = CBMPopulationAgentOnlineSimpleSimulationReactive(
        pop_size=10, eta=0.1, rho=0.1, di_cycle_length=10, epsilon=0.01,
        num_iterations=9999999, num_solution_attempts=21, agent_id=agent_id,
        node_name=node_name, learning_method=learning_method, num_tsp_agents=num_tsp_agents, lr=lr,
        gamma_decay=gamma_decay, positive_reward=positive_reward, negative_reward=negative_reward
    )
    print("CBMPopulationAgentOnlineSimpleSimulation has been initialized.")

    def shutdown_callback():
        agent.get_logger().info("LLM-Interface-agent Runtime completed. Shutting down.")
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if runtime != -1:
        print("Hit shutdown_callback")
        agent.create_timer(runtime, shutdown_callback)

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        if rclpy.ok():
            executor.shutdown()
            rclpy.shutdown()
            sys.exit(0)


if __name__ == '__main__':
    main()

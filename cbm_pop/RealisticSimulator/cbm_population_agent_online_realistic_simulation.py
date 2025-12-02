import math
import random
import rclpy
import numpy as np
from time import time
from math import radians, sin, asin, cos, sqrt
from builtin_interfaces.msg import Time
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from copy import deepcopy
from cbm_pop.Operator import Operator
from cbm_pop.Condition import ConditionFunctions
from cbm_pop.RealisticSimulator.realistic_fitness import RealisticFitness
from cbm_pop.Operator_Fuctions import OperatorFunctions
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.RealisticSimulator.realistic_problem import RealisticProblem, ProblemClass
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import Bool
from std_msgs.msg import String, Float32
from rclpy.executors import MultiThreadedExecutor
from pygeodesy.geoids import GeoidPGM
from cbm_pop_interfaces.msg import (Solution,
                                    Weights,
                                    EnvironmentalRepresentation,
                                    FinishedCoverage,
                                    CurrentTask)
from geographic_msgs.msg import GeoPose, GeoPoseStamped

class LearningMethod(Enum):
    FERREIRA = "Ferreira_et_al."
    Q_LEARNING = "Q-Learning"


class CBMPopulationAgentOnlineRealisticSimulation(Node):
    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, learning_method,
                 lr=0.5,
                 gamma_decay=0.99,
                 positive_reward=1,
                 negative_reward=-0.5,
                 num_tsp_agents=10,
                 problem_size=15,
                 lock_mode: bool = False,
                 preserve_next_task=False,
                 use_ucb: bool = False,
                 ucb_c: float = 1.0,
                 inject_best_on_cycle: bool = False,
                 inject_best_prob: float = 0.90,
                 initialise_with_heuristic=True,
                 problem_class=ProblemClass.FourWhichWay,
                 problem_seed=1):

        """
        Initialises the agent on startup
        """
        super().__init__(node_name)

        # --- core flags / logging toggle ---
        self.geoid = GeoidPGM('/home/ajifoster3/Documents/Software/ros_ws/src/CBM-POP_Implementation/egm96-5.pgm')
        self.islogging = True

        if self.islogging:
            self.get_logger().info("===== Initialising CBMPopulationAgentOnlineRealisticSimulation =====")
            self.get_logger().info(
                f"Using geoid file: /home/ajifoster3/Documents/Software/ros_ws/src/CBM-POP_Implementation/egm96-5.pgm")

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
                    Lock Mode            {lock_mode}
                    Preserve_next_task   {preserve_next_task}

                    RL Parameters:
                      - LR:              {lr}
                      - Gamma Decay:     {gamma_decay}
                      - Positive Reward: {positive_reward}
                      - Negative Reward: {negative_reward}

                    UCB:
                      - Enabled:         {use_ucb}
                      - Exploration c:   {ucb_c}

                    Post-cycle injection:
                      - Enabled:         {inject_best_on_cycle}
                      - Probability:     {inject_best_prob}

                    Number of TSP Agents: {num_tsp_agents}
                    Problem Size:         {problem_size}
                    Problem Class:        {problem_class}
                    Problem Seed:         {problem_seed}
                    =======================================================
                    """
        if self.islogging:
            self.get_logger().info(settings_str)

        from collections import defaultdict, Counter
        self._proposal_stats = defaultdict(lambda: {
            "count": 0,
            "removed_uncovered": Counter(),
            "added_covered": Counter(),
        })

        self.is_task_locked = False
        self.max_switches_before_lock = 2  # Track maximum allowed switches
        self.switch_count = 0  # Counter for task switching
        self.last_task = None  # Store the last assigned task

        self.current_task = None
        self.current_tasks = [-1] * num_tsp_agents

        self.current_parent_idx = None

        self.initialise_with_heuristic = initialise_with_heuristic

        if self.islogging:
            self.get_logger().info("Constructing RealisticProblem...")

        self.problem = RealisticProblem(ProblemClass(problem_class))

        if self.islogging:
            self.get_logger().info(f"RealisticProblem created with {self.problem.num_tasks} tasks.")

        self.lock_mode = lock_mode
        self.preserve_next_task = preserve_next_task

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
        self.intensifiers = [Operator.ONE_MOVE, Operator.TWO_SWAP, Operator.TWO_OPT_INTRA,
                             Operator.NEAREST_K_RELOCATION]
        self.diversifiers = [
            Operator.BEST_COST_ROUTE_CROSSOVER,
            Operator.INTRA_DEPOT_REMOVAL,
            Operator.INTRA_DEPOT_SWAPPING,
            Operator.SINGLE_ACTION_REROUTING
        ]
        self.weight_matrix = WeightMatrix(len(self.intensifiers), len(self.diversifiers))

        if self.islogging:
            self.get_logger().info(f"Operators initialised: "
                                   f"{len(self.intensifiers)} intensifiers, {len(self.diversifiers)} diversifiers.")
            self.get_logger().info(
                f"Weight matrix shape: {len(self.weight_matrix.weights)} x {len(self.weight_matrix.weights[0])}")

        self.population = None
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_ID = agent_id
        self.agent_ID = self.agent_ID
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

        if self.islogging:
            self.get_logger().info(f"Learning method resolved to enum: {self.learning_method}")

        self.inject_best_on_cycle = inject_best_on_cycle
        self.inject_best_prob = inject_best_prob

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
        self.is_covered = [False] * self.problem.num_tasks

        self.last_purge_agent_true_id = None
        self.failed_agents = [False] * self.num_tsp_agents
        self.agents_to_revive = [False] * self.num_tsp_agents
        self.purged_agents = [False] * self.num_tsp_agents
        self.agent_to_revive = None
        self.am_i_failed = False
        self.ros_timer = None
        self.agent_timeouts = [False] * self.num_tsp_agents
        self.finished_robots = [False] * self.num_tsp_agents
        self.is_all_poses = False

        self.cb_group = ReentrantCallbackGroup()
        self.me_cb_group = MutuallyExclusiveCallbackGroup()

        if self.islogging:
            self.get_logger().info("Creating ROS publishers and subscribers...")

        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(Solution, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.__solution_update_callback, 10)
        self.weight_publisher = self.create_publisher(Weights, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            Weights, 'weight_matrix', self.__weight_update_callback, 10)
        self.current_task_publisher = self.create_publisher(CurrentTask, 'current_task', 10)
        if self.lock_mode:
            self.current_task_subscriber = self.create_subscription(
                CurrentTask, 'current_task', self.current_task_update_callback, 10)
            self.current_task_publisher_timer = self.create_timer(
                1, self.regular_current_task_publish_timer, callback_group=self.cb_group
            )

        self.kill_robot_subscribers = []
        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id}/kill_robot'
            sub = self.create_subscription(
                Bool,
                topic,
                lambda msg, agent=id: self.kill_robot_callback(msg, agent),
                10
            )
            self.kill_robot_subscribers.append(sub)

        self.revive_robot_sub = self.create_subscription(
            Bool,
            f'/central_control/uas_{self.agent_ID}/revive_robot',
            self.revive_robot_callback,
            10
        )

        # Global pose subscribers
        self.global_pose_subscribers = []
        for i in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{i}/global_pose'
            sub = self.create_subscription(
                GeoPoseStamped,
                topic,
                lambda msg, agent=i: self.__global_pose_callback(msg, agent),
                10,
                callback_group=self.cb_group
            )
            self.global_pose_subscribers.append(sub)

        # Goal publisher
        self.goal_pose_publisher = self.create_publisher(
            GeoPoseStamped,
            f'/central_control/uas_{self.agent_ID}/goal_pose',
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
            self.__finished_coverage_callback,
            10
        )

        # Timers
        self.run_goal_publisher_timer = self.create_timer(
            0.5, self.__publish_goal_pose, callback_group=self.cb_group
        )

        self.environmental_representation_subscriber = self.create_subscription(
            EnvironmentalRepresentation,
            '/environmental_representation',
            self.__environmental_representation_callback,
            40,
            callback_group=self.cb_group
        )
        self.environmental_representation_publisher = self.create_publisher(
            EnvironmentalRepresentation,
            '/environmental_representation',
            10
        )
        self.environmental_representation_timer = self.create_timer(
            2, self.__environmental_representation_timer_callback,
            callback_group=self.cb_group
        )
        self.solution_publisher_timer = self.create_timer(
            2, self.__regular_solution_publish_timer,
            callback_group=self.cb_group
        )

        if self.islogging:
            self.get_logger().info("ROS publishers, subscribers, and timers created.")

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

        # UCB settings/state
        self.use_ucb = use_ucb
        self.ucb_c = ucb_c
        self.operator_usage_counts = {}

        # Timer to monitor stale agents
        self.create_timer(5, self.check_stale_agents)

        if self.islogging:
            self.get_logger().info("Initialisation complete. Waiting for robot poses and environmental updates.")

    def __diff_solutions(self, where: str, old_sol, new_sol, label_old="OLD", label_new="NEW"):
        """
        Log task-level differences between two solutions (ignoring allocations).
        Flags any uncovered tasks removed or added.
        """
        old_set = set(old_sol[0]) if old_sol else set()
        new_set = set(new_sol[0]) if new_sol else set()

        removed = sorted(list(old_set - new_set))
        added   = sorted(list(new_set - old_set))

        removed_uncovered = [t for t in removed if 0 <= t < self.problem.num_tasks and not self.is_covered[t]]
        removed_covered   = [t for t in removed if 0 <= t < self.problem.num_tasks and self.is_covered[t]]
        added_uncovered   = [t for t in added if 0 <= t < self.problem.num_tasks and not self.is_covered[t]]
        added_covered     = [t for t in added if 0 <= t < self.problem.num_tasks and self.is_covered[t]]

        if removed or added:
            self.get_logger().warning(
                f"[AUDIT:{where}] DIFF {label_old}->{label_new} | "
                f"removed_uncovered={removed_uncovered} removed_covered={removed_covered} "
                f"added_uncovered={added_uncovered} added_covered={added_covered}"
            )

    # ===================== END AUDIT HELPERS ========================

    def _active_agent_indices(self):
        """Indices of robots that are not purged/failed (in coalition space)."""
        return [i for i in range(self.num_tsp_agents) if not self.purged_agents[i]]

    def __voronoi_partition(self):
        """
        Assign tasks to the nearest active robot using 3D haversine distance
        (lat, lon + altitude).
        """
        active = self._active_agent_indices()
        tasks_per_agent = [[] for _ in range(self.num_tsp_agents)]

        if not all(self.robot_poses[i] is not None for i in active):
            raise RuntimeError("Robot poses required before Voronoi initialisation.")

        # Preload robot positions
        robot_geo = []
        for i in active:
            p = self.robot_poses[i].position
            robot_geo.append((p.latitude, p.longitude, p.altitude))
        robot_geo = np.array(robot_geo, dtype=float)  # shape (N_active, 3)

        # Preload task positions
        task_geo = []
        for t in self.problem.task_poses:
            task_geo.append((t["latitude"], t["longitude"], t["altitude"]))
        task_geo = np.array(task_geo, dtype=float)  # shape (N_tasks, 3)

        # Compute distances [N_tasks, N_active]
        dists = np.zeros((task_geo.shape[0], robot_geo.shape[0]), dtype=float)

        for i_task, (tlat, tlon, talt) in enumerate(task_geo):
            for i_robot, (rlat, rlon, ralt) in enumerate(robot_geo):
                dists[i_task, i_robot] = self.haversine(
                    tlat, tlon, talt,
                    rlat, rlon, ralt
                )

        nearest = np.argmin(dists, axis=1)

        # Assign each task to nearest active robot
        for t_index, a_local_idx in enumerate(nearest):
            a_true = active[a_local_idx]
            if not self.is_covered[t_index]:
                tasks_per_agent[a_true].append(t_index)

        return tasks_per_agent

    def __nn_order(self, tasks, start_geo_pose, jitter=0.0, rng=None):
        """
        3D nearest-neighbour ordering using haversine(lat,lon,alt)
        instead of Euclidean XY.

        tasks: list of task indices
        start_geo_pose: GeoPose (full pose)
        """
        if not tasks:
            return []

        rng = rng or random

        # Extract robot start in lat/lon/alt
        sp = start_geo_pose.position
        start_lat = sp.latitude
        start_lon = sp.longitude
        start_alt = sp.altitude

        # Preload task geos + jitter
        task_geo = []
        for t in tasks:
            p = self.problem.task_poses[t]
            lat, lon, alt = p["latitude"], p["longitude"], p["altitude"]

            if jitter > 0:
                lat += rng.uniform(-jitter * 1e-6, jitter * 1e-6)  # tiny jitter in deg
                lon += rng.uniform(-jitter * 1e-6, jitter * 1e-6)
                alt += rng.uniform(-jitter, jitter)

            task_geo.append((lat, lon, alt))

        # Convert to numpy arrays for easy handling
        task_geo = np.array(task_geo, dtype=float)

        # Find nearest starting task
        d0 = [
            self.haversine(start_lat, start_lon, start_alt, lat, lon, alt)
            for lat, lon, alt in task_geo
        ]
        first = int(np.argmin(d0))

        remaining = list(range(len(tasks)))
        order_idx = [remaining.pop(first)]
        cur = order_idx[0]

        # Greedy NN loop
        while remaining:
            (clat, clon, calt) = task_geo[cur]
            d = [
                self.haversine(clat, clon, calt,
                               task_geo[j][0],
                               task_geo[j][1],
                               task_geo[j][2])
                for j in remaining
            ]
            k = int(np.argmin(d))
            order_idx.append(remaining.pop(k))
            cur = order_idx[-1]

        # Convert local indices to actual task IDs
        return [tasks[i] for i in order_idx]

    def __two_opt_once(self, order):
        """
        Single 2-opt move using 3D haversine distance between task waypoints.
        `order` is a list of task indices.
        """
        n = len(order)
        if n < 4:
            return order, False

        def seg_len(idx_a, idx_b):
            """
            Distance between order[idx_a] and order[idx_b] using 3D haversine.
            """
            t_a = self.problem.task_poses[order[idx_a]]
            t_b = self.problem.task_poses[order[idx_b]]

            return self.haversine(
                t_a["latitude"], t_a["longitude"], t_a["altitude"],
                t_b["latitude"], t_b["longitude"], t_b["altitude"],
            )

        # Standard 2-opt: try swapping edges (a,b) and (c,d)
        for i in range(n - 3):
            for k in range(i + 2, n - 1):
                a, b = i, i + 1
                c, d = k, k + 1

                old = seg_len(a, b) + seg_len(c, d)
                new = seg_len(a, c) + seg_len(b, d)

                if new + 1e-12 < old:
                    # Build improved tour: reverse segment (b..c)
                    new_order = (
                            order[:a + 1]
                            + list(reversed(order[a + 1:c + 1]))
                            + order[c + 1:]
                    )
                    return new_order, True

        return order, False

    def __generate_population_voronoi(self, use_two_opt=True):
        """
        Build an initial population using Voronoi assignment + simple route heuristic per agent.
        """
        if not all(p is not None for p in self.robot_poses):
            raise RuntimeError("Robot poses are required before Voronoi initialization.")

        population = []
        rng = random.Random()

        def _dist_geo(task_idx, agent_true_idx: int):
            """
            Squared 3D haversine distance between a task and an agent's pose.
            Used only for comparing which agent is closer (no need for sqrt).
            """
            t = self.problem.task_poses[task_idx]
            tp_lat = t["latitude"]
            tp_lon = t["longitude"]
            tp_alt = t["altitude"]

            rp = self.robot_poses[agent_true_idx].position
            rp_lat = rp.latitude
            rp_lon = rp.longitude
            rp_alt = rp.altitude

            d = self.haversine(tp_lat, tp_lon, tp_alt,
                               rp_lat, rp_lon, rp_alt)
            return d * d

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

                # a is the agent index we’re giving a task TO
                k, task_to_move = min(
                    enumerate(src_route),
                    key=lambda it: _dist_geo(it[1], a)  # pass agent index, not GeoPose
                )
                src_route.pop(k)
                per_agent_routes[a].append(task_to_move)
                if len(per_agent_routes[src]) <= 1:
                    donors.discard(src)

        for sol_idx in range(self.pop_size):
            tasks_per_agent = self.__voronoi_partition()
            per_agent_routes = []
            for a in range(self.num_tsp_agents):
                tasks = tasks_per_agent[a]
                if not tasks:
                    per_agent_routes.append([])
                    continue
                start_pose = self.robot_poses[a]
                jitter = 0
                route = self.__nn_order(tasks, start_pose, jitter=jitter, rng=rng)
                route = [t for t in route if not self.is_covered[t]]
                per_agent_routes.append(route)

            _ensure_min_one(per_agent_routes)
            ordered_task_list = [t for r in per_agent_routes for t in r]
            allocation_counts = [len(r) for r in per_agent_routes]
            candidate = (ordered_task_list, allocation_counts)
            population.append(candidate)

        return population

    def __generate_population(self):
        """
        Generates the initial solution population randomly.
        """
        population = []
        print("Generating Random Population...")

        for i in range(self.pop_size):
            print(f"Generating solution {i}")
            tasks = list(range(self.problem.num_tasks))
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

    def _set_coalition_best_solution(self, solution):
        """
        Sets coalition best and assigns next task.
        """
        self.coalition_best_solution = deepcopy(solution)
        if not self.lock_mode or self.current_task is None:
            self.__assign_next_task(solution)

    def __assign_next_task(self, solution):
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

                old_task = self.current_task
                if not self.is_covered[task]:
                    # If already locked, stay on the locked task
                    if self.is_task_locked:
                        print(f"Still locked on task {task}")
                        return

                    # Track switching behaviour
                    if self.last_task == task and task != self.current_task:
                        self.switch_count += 1
                    else:
                        self.switch_count = 0  # Reset if task changed

                    # Check for locking condition
                    if self.switch_count >= self.max_switches_before_lock:
                        print(f"Locking on task {task} after {self.switch_count} switches")
                        self.current_task = task
                        self.is_task_locked = True
                    else:
                        self.current_task = task

                    if self.current_task != old_task:
                        self.last_task = old_task
                    if self.islogging:
                        print(f"Current Task is {self.current_task}")
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

    def __select_solution(self):
        # Reseed if needed
        if not self.population:
            self.get_logger().warning("[SELECT] population was empty; reseeding…")
            try:
                if self.initialise_with_heuristic:
                    self.population = self.__generate_population_voronoi()
                else:
                    self.population = self.__generate_population()
            except Exception as e:
                self.get_logger().error(f"[SELECT] reseed failed: {e}")
                self.population = []

        if not self.population:
            self.get_logger().info("[SELECT] no candidates available (likely all tasks covered).")
            return None, None

        # Choose best candidate
        idx, sol = min(
            enumerate(self.population),
            key=lambda it: self._fitness(it[1])
        )
        return idx, sol

    def __update_experience(self, condition, operator, gain):
        """
        Adds details of the current iteration to the experience memory.
        """
        op_order = self.intensifiers + self.diversifiers
        try:
            op_col = op_order.index(operator)
        except Exception:
            op_col = int(operator)
        self.previous_experience.append([condition, op_col, gain])

    def __individual_learning_old(self):
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
            self.weight_matrix.weights[row][col] += 1 if self.best_coalition_improved else 1*self.eta
        if self.islogging:
            print(f"weights: {self.weight_matrix.weights}")
        return self.weight_matrix.weights

    def __individual_learning(self):
        """
        Learning with Bellman Equation.
        First-visit per (condition, op): only one update per unique pair
        in the prefix up to min_fitness_index.
        """
        cumulative_gains = []
        total_gain = 0
        for _, _, gain in self.previous_experience:
            total_gain += gain
            cumulative_gains.append(total_gain)

        if self.best_local_improved and cumulative_gains:
            min_fitness_index = cumulative_gains.index(min(cumulative_gains))
        else:
            min_fitness_index = len(self.previous_experience)

        # NEW: de-duplicate updates per unique (condition, op) in the prefix
        seen_pairs = set()

        for i in range(min_fitness_index):
            condition, op_col, gain = self.previous_experience[i]
            row = condition
            col = int(op_col)

            key = (row, col)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            current_q = self.weight_matrix.weights[row][col]
            if i + 1 < len(self.previous_experience):
                next_condition = self.previous_experience[i + 1][0]
                max_next_q = max(self.weight_matrix.weights[next_condition])
            else:
                max_next_q = 0

            # keep your original reward logic
            self.reward = self.positive_reward if self.best_local_improved else self.negative_reward

            updated_q = current_q + self.lr * (self.reward + self.gamma_decay * max_next_q - current_q)
            updated_q = max(updated_q, 1e-6)
            self.weight_matrix.weights[row][col] = updated_q

        return self.weight_matrix.weights

    def __mimetism_learning(self, received_weights, rho):
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

    def __stopping_criterion(self, iteration_count):
        """
        Returns true if current number of iterations exceeds the limit.
        """
        return iteration_count > self.num_iterations

    def __end_of_di_cycle(self, cycle_count):
        """
        Returns true if the current cycle is equal to or over the length of a DI cycle.
        """
        if cycle_count >= self.di_cycle_length:
            return True
        return False

    def __weight_update_callback(self, msg):
        """
        Stores received weight matrices.
        """
        received_weights = self.weight_matrix.unpack_weights(weights_msg=msg, agent_id=self.agent_ID)
        if received_weights is not None:
            self.received_weight_matrices.append(received_weights)

    def current_task_update_callback(self, msg):
        try:
            # Record peer’s current head
            if self.current_tasks[msg.agent_id] != msg.current_task:
                self.current_tasks[msg.agent_id] = msg.current_task
            if self.islogging:
                print(f"current tasks: {self.current_tasks}")
            # Optional: resolve head clash by yielding if we’re worse
            if msg.agent_id != self.agent_ID and msg.current_task == self.current_task and self.current_task is not None:
                print(f"Resolving current task conflict: Agent ID: {self.agent_ID}, My current task: {self.current_task}, current tasks: {self.current_tasks}")
                my = self.problem.current_robot_cost_matrix[self.agent_ID][self.current_task]
                theirs = self.problem.current_robot_cost_matrix[msg.agent_id][self.current_task]
                if theirs < my:
                    self.__assign_next_task(self.coalition_best_solution)
        except Exception as e:
            error_message = (
                f"[ERROR] Exception in current task update callback:\n"
                f"    Error Type: {type(e).__name__}\n"
                f"    Error Message: {e}\n"
                f"    Stack Trace:\n{traceback.format_exc()}"
            )
            print(error_message)

    def handle_allocated_task(self, current_task):
        """
        Handles a task allocated to this agent.
        Updating all it's solutions to reflect the new state, and publishing it's new environmental representation.
        *** TODO: This shouldn't always publish! ***
        """
        if self.is_loop_started:

            # Function to update a given solution
            def update_solution(solution, current_task):
                if solution is None:
                    return None

                order, allocations = solution
                counter = 0
                for agent_idx, task_count in enumerate(allocations):
                    if current_task in order[counter:counter + task_count]:
                        order.remove(current_task)  # Remove task from order
                        allocations[agent_idx] -= 1  # Decrease allocation count
                        break
                    counter += task_count
                return order, allocations

            for idx, solution in enumerate(self.population):
                self.population[idx] = update_solution(solution, current_task)

            # Update current_solution and coalition_best_solution if they exist
            self.current_solution = update_solution(self.current_solution, current_task)
            self.coalition_best_solution = update_solution(self.coalition_best_solution, current_task)

    def __finished_coverage_callback(self, msg):
        self.finished_robots[int(msg.robot_id)] = bool(msg.finished)
        if all(self.finished_robots) and not getattr(self, "_shutdown_timer_set", False):
            self._shutdown_timer_set = True
            self.get_logger().info("Coverage Complete — shutting down in 5 seconds...")
            # One-shot timer to trigger shutdown (not destroy) after 5s
            self.create_timer(5.0, lambda: rclpy.shutdown())


    # --- Helpers ---

    def __active_agent_index(self):
        """
        Returns this agent's index in the 'active' (non-purged) agent list.
        """
        active_indices = [i for i, purged in enumerate(self.purged_agents) if not purged]
        return active_indices.index(self.agent_ID)

    def __segment_bounds(self, allocations, agent_idx):
        """
        Given allocations and an agent_idx in active indexing, return (start, end).
        """
        start = sum(allocations[:agent_idx])
        end = start + allocations[agent_idx]
        return start, end

    def __next_task_of_agent(self, solution, agent_idx):
        """
        Returns the next (first) task of agent_idx in the given solution, or None if none.
        """
        order, allocations = solution
        start, end = self.__segment_bounds(allocations, agent_idx)
        if end > start:
            return order[start]
        return None

    def __evaluate_solution(self, solution):
        """
        Wrapper to evaluate fitness with your existing matrices.
        """
        order, allocations = solution
        return RealisticFitness.fitness_function_robot_pose(
            solution,
            self.problem
        )

    def __solution_update_callback(self, msg):
        """
        - Observe coverage from diffs (coalition -> received). Tasks missing in the
          received plan are treated as newly covered.
        - Sanitize the received plan: strip covered tasks per segment; dedupe per segment.
        - Compare against our sanitized coalition and adopt only if fitter.
        - Never trims uncovered tasks.
        """
        if self.am_i_failed:
            return

        def __sanitize_strip_covered(sol):
            """Strip already-covered tasks per segment (preserves segment sizes)."""
            if not sol:
                return sol
            order, alloc = sol
            return self.__remove_covered_tasks_from_solution((list(order), list(alloc)))

        try:
            recv_order = list(msg.order)
            recv_alloc = list(msg.allocations)
        except Exception:
            if isinstance(msg, (tuple, list)) and len(msg) == 2:
                recv_order, recv_alloc = list(msg[0]), list(msg[1])
            else:
                return

        active_agents = self._active_agent_indices()
        expected_alloc_len = len(active_agents)
        if len(recv_alloc) != expected_alloc_len:
            self.get_logger().info(
                f"[solution_update_callback] ignoring solution: alloc len={len(recv_alloc)} "
                f"!= active agents={expected_alloc_len}"
            )
            return

        received = (recv_order, recv_alloc)

        cand = __sanitize_strip_covered(received)

        cur = self.coalition_best_solution
        if cur:
            cur = __sanitize_strip_covered(cur)

        cand_f = self._fitness(
            cand
        )
        cur_f = self._fitness(cur,
        ) if cur else float("inf")

        if not cand or not cand[0]:
            if self.islogging:
                self.get_logger().info("[AUDIT:solution_update_callback.ignored] empty_candidate_after_sanitize")
            return

        if cand_f < cur_f:
            self._set_coalition_best_solution(cand)

    def __remove_extra_tasks(self, solution, target_task_count):
        """
        Trim the *end* of each agent's segment until global task count equals target_task_count.
        """
        if solution is None:
            return None

        order, allocations = deepcopy(solution)
        order, allocations = self.__remove_covered_tasks_from_solution((order, allocations))

        total = sum(allocations)
        if total <= target_task_count:
            out = (order, allocations)
            return out

        idx = len(allocations) - 1
        while sum(allocations) > target_task_count and any(a > 0 for a in allocations):
            if allocations[idx] > 0:
                end = sum(allocations[:idx + 1])
                start = end - allocations[idx]
                if end - 1 >= start:
                    removed_task = order.pop(end - 1)
                    allocations[idx] -= 1
            idx = (idx - 1) % len(allocations)

        out = (order, allocations)
        return out

    def __global_pose_callback(self, msg, agent):
        """
        Receives and processes a global GeoPose from another agent.

        - Stores initial and current poses (GeoPose) for each agent.
        - Once all agents have reported a pose, computes cost matrices, generates the population,
          and starts the optimisation loop.
        - If this agent reaches its current task goal (in geo space), marks the task as covered
          and assigns the next one.
        - If this agent has no remaining tasks and has returned to its initial pose, publishes
          a FinishedCoverage message.
        """
        try:
            # Store the first received pose for each agent as its initial pose
            if self.initial_robot_poses[agent] is None:
                self.initial_robot_poses[agent] = deepcopy(msg.pose)

            task = deepcopy(self.current_task)

            # Always update the latest pose for this agent
            if msg.pose:
                self.robot_poses[agent] = deepcopy(msg.pose)

            if all(pose is not None for pose in self.robot_poses) \
                    and self.is_loop_started is False \
                    and self.is_all_poses is False:

                self.is_all_poses = True

                # These should now work with arrays of GeoPoses instead of (x, y)
                self.problem.update_robot_cost_matrix(self.robot_poses)
                self.problem.initialize_robot_initial_pose_cost_matrix(self.initial_robot_poses)

                if self.population is None:
                    if self.is_generating is False:
                        self.is_generating = True
                        if self.initialise_with_heuristic:
                            self.population = self.__generate_population_voronoi()
                        else:
                            self.population = self.__generate_population()
                        self.current_parent_idx, self.current_solution = self.__select_solution()

                self.run_timer = self.create_timer(0.1, self.run_step, callback_group=self.me_cb_group)
                self.is_loop_started = True

            if self.problem.task_poses is not None and task is not None and msg.pose:

                global_lat = msg.pose.position.latitude
                global_lon = msg.pose.position.longitude
                global_alt = msg.pose.position.altitude
                geoid_height = self.geoid.height(global_lat, global_lon)
                global_alt = global_alt - geoid_height

                # Goal (task) position – assumed dict with lat/lon/alt
                goal_lat = self.problem.task_poses[task]["latitude"]
                goal_lon = self.problem.task_poses[task]["longitude"]
                goal_alt = self.problem.task_poses[task]["altitude"]

                distance = self.haversine(global_lat, global_lon, global_alt,
                                          goal_lat, goal_lon, goal_alt)

                # Only act when *this* agent reaches its goal
                if agent == self.agent_ID and distance < 0.4:
                    if self.islogging:
                        self.get_logger().info(f"Covered task: {task}")
                    self.__handle_covered_task(task)

                    if self.is_task_locked:
                        self.get_logger().info(
                            f"Unlocking task {self.current_task} — coverage complete"
                        )
                        self.is_task_locked = False
                        self.switch_count = 0
                        self.__assign_next_task(self.coalition_best_solution)

                    # Assign next task regardless of lock state
                    self.__assign_next_task(self.coalition_best_solution)

            if (
                    self.coalition_best_solution is not None
                    and self.coalition_best_solution[1][self.agent_ID] == 0  # keep your current indexing
                    and msg.pose
                    and self.initial_robot_poses[self.agent_ID] is not None
            ):

                # Current position
                global_lat = msg.pose.position.latitude
                global_lon = msg.pose.position.longitude
                global_alt = msg.pose.position.altitude
                geoid_height = self.geoid.height(global_lat, global_lon)
                global_alt = global_alt - geoid_height

                # Initial (home) position
                init_pose = self.initial_robot_poses[self.agent_ID]
                goal_lat = init_pose.position.latitude
                goal_lon = init_pose.position.longitude
                goal_alt = init_pose.position.altitude

                distance_home = self.haversine(global_lat, global_lon, global_alt,
                                               goal_lat, goal_lon, goal_alt)

                if agent == self.agent_ID and distance_home < 0.4 and all(self.is_covered):
                    self.get_logger().info(f"[INFO] Agent {self.agent_ID} finished coverage!.")
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

    def __handle_covered_task(self, covered_task: int | None = None):
        """
        Handles a task that has been covered.
        If `covered_by` is provided, we first try to remove the task from that agent's
        segment; if it's not found there (plans drift), we remove it wherever it is.
        Only the agent that actually covered the task (covered_by == self.agent_ID)
        will advance to a new next task.
        """

        # Exit if coverage hasn't started
        if not self.is_loop_started:
            return

        # Exit if the task is already covered
        if 0 <= covered_task < len(self.is_covered) and self.is_covered[covered_task]:
            return

        # Set the task as Covered
        self.is_covered[covered_task] = True

        # Publish the Coverage List with the newly covered task
        if not self.am_i_failed:
            rep = EnvironmentalRepresentation()
            rep.agent_id = self.agent_ID
            rep.is_covered = list(self.is_covered)
            self.environmental_representation_publisher.publish(rep)

        # Returns the index of the first and last task for the given robot
        def _seg_bounds(allocs, idx):
            s = sum(allocs[:idx])
            e = s + allocs[idx]
            return s, e

        def update_solution(solution):
            # Exit if the solutions is none
            if solution is None:
                return None

            # Split solution into components
            order, allocations = solution

            # Exit if a component is null
            if not order or not allocations:
                return solution

            # Copy the solution components
            new_order = list(order)
            new_alloc = list(allocations)

            # If the covering robot is known
            removed = False
            #if covered_by is not None and 0 <= covered_by < len(new_alloc) and new_alloc[covered_by] > 0:
            #    s, e = _seg_bounds(new_alloc, covered_by)
            #    try:
            #        # get index of covered task
            #        rel = new_order[s:e].index(covered_task)
            #        pos = s + rel
            #        # delete covered task
            #        del new_order[pos]
            #        # decrease allocation for the covering robot
            #        new_alloc[covered_by] = max(0, new_alloc[covered_by] - 1)
            #        removed = True
            #    except ValueError:
            #        pass

            # If we don't know which robot covered the task we need to find which robots the task is
            if not removed and covered_task in new_order:
                pos = new_order.index(covered_task)
                cum = 0
                owner_idx = None
                for idx, a in enumerate(new_alloc):
                    if pos < cum + a:
                        owner_idx = idx
                        break
                    cum += a
                del new_order[pos]
                if owner_idx is not None and new_alloc[owner_idx] > 0:
                    new_alloc[owner_idx] -= 1

            if sum(new_alloc) != len(new_order):
                fixed = []
                cursor = 0
                for a in new_alloc:
                    take = min(a, max(0, len(new_order) - cursor))
                    fixed.append(take)
                    cursor += take
                new_alloc = fixed

            self.task_covered = -1
            self.is_new_task_covered = False
            return (new_order, new_alloc)

        for i, sol in enumerate(self.population):
            before = sol
            self.population[i] = update_solution(sol)

        before_cur = self.current_solution
        self.current_solution = update_solution(self.current_solution)

        before_coal = self.coalition_best_solution
        self.coalition_best_solution = update_solution(self.coalition_best_solution)

    def __environmental_representation_callback(self, msg: EnvironmentalRepresentation):
        """
        On receiving another agent's environmental representation, merge coverage.
        """
        try:
            if self.am_i_failed and all(msg.is_covered):
                print(f"[INFO] Purged Agent {self.agent_ID} reporting finished coverage!.")
                self.is_finished = True
                m = FinishedCoverage()
                m.finished = True
                m.robot_id = self.agent_ID
                self.finished_coverage_pub.publish(m)
            if self.am_i_failed:
                return
            agent_id = msg.agent_id
            self.last_env_rep_timestamps[agent_id] = time()

            for i in range(len(msg.is_covered)):
                if msg.is_covered[i] and not self.is_covered[i]:
                    self.__handle_covered_task(i)

            if self.failed_agents[agent_id]:
                self.agents_to_revive[agent_id] = True

        except Exception as e:
            error_message = (
                f"[ERROR] Exception in environmental_representation_callback:\n"
                f"    Error Type: {type(e).__name__}\n"
                f"    Error Message: {e}\n"
                f"    Stack Trace:\n{traceback.format_exc()}"
            )
            self.get_logger().error(error_message)

    def __environmental_representation_timer_callback(self):
        """
        Publish this agent's environmental representation.
        """
        if self.am_i_failed:
            print("Cancelling environmental representation timer")
            self.environmental_representation_timer.cancel()
            self.environmental_representation_timer = None
            return

        rep = EnvironmentalRepresentation()
        rep.agent_id = self.agent_ID
        rep.is_covered = list(self.is_covered)
        self.environmental_representation_publisher.publish(rep)

    def __publish_goal_pose(self):
        """
        Publish this agent's current goal pose to the flight controller.
        Logs *exactly* which goal is used and why.
        """
        # ======= Failed agent: freeze goal at current/initial pose =======
        if self.am_i_failed:
            if self.run_goal_publisher_timer is not None:
                self.run_goal_publisher_timer.cancel()
                self.run_goal_publisher_timer = None

            if self.islogging:
                self.get_logger().warning(
                    f"[GOAL_PUB] Agent {self.agent_ID} is FAILED — publishing frozen goal pose."
                )

            goal_pose = GeoPoseStamped()
            goal_pose.header.stamp = Time()
            goal_pose.header.frame_id = "world"

            # Use last known pose if available, otherwise initial
            src_pose = self.robot_poses[self.agent_ID] or self.initial_robot_poses[self.agent_ID]
            if src_pose is None:
                if self.islogging:
                    self.get_logger().error(
                        f"[GOAL_PUB] Agent {self.agent_ID} FAILED but has no pose to freeze."
                    )
                return

            goal_pose.pose = deepcopy(src_pose)

            if self.islogging:
                p = goal_pose.pose.position
                self.get_logger().info(
                    f"[GOAL_PUB] Frozen goal for FAILED agent {self.agent_ID}: "
                    f"lat={p.latitude:.7f}, lon={p.longitude:.7f}, alt(ellipsoid)={p.altitude:.3f}"
                )

            self.goal_pose_publisher.publish(goal_pose)
            return

        # ======= Active agent: task goal =======
        if self.current_task is not None and self.problem.task_poses:
            tp = self.problem.task_poses[self.current_task]

            goal_pose = GeoPoseStamped()
            goal_pose.header.stamp = Time()
            goal_pose.header.frame_id = "world"

            goal_pose.pose.position.latitude = tp["latitude"]
            goal_pose.pose.position.longitude = tp["longitude"]
            goal_pose.pose.position.altitude = tp["altitude"]

            goal_pose.pose.orientation.x = float(tp["orientation_x"])
            goal_pose.pose.orientation.y = float(tp["orientation_y"])
            goal_pose.pose.orientation.z = float(tp["orientation_z"])
            goal_pose.pose.orientation.w = float(tp["orientation_w"])

            if self.islogging:
                self.get_logger().info(
                    f"[GOAL_PUB] Agent {self.agent_ID} publishing TASK goal: "
                    f"task={self.current_task}, "
                    f"lat={tp['latitude']:.7f}, lon={tp['longitude']:.7f}, alt={tp['altitude']:.3f}"
                )

            self.goal_pose_publisher.publish(goal_pose)
            return

        # ======= No current task: send home pose (if known) =======
        try:
            if self.initial_robot_poses[self.agent_ID] is not None:
                init_pose = self.initial_robot_poses[self.agent_ID]

                goal_pose = GeoPoseStamped()
                goal_pose.header.stamp = Time()
                goal_pose.header.frame_id = "world"

                goal_pose.pose.position.latitude = init_pose.position.latitude
                goal_pose.pose.position.longitude = init_pose.position.longitude

                geoid_height = self.geoid.height(
                    goal_pose.pose.position.latitude,
                    goal_pose.pose.position.longitude
                )
                # Convert stored ellipsoidal to orthometric
                goal_pose.pose.position.altitude = init_pose.position.altitude - geoid_height

                goal_pose.pose.orientation.x = float(init_pose.orientation.x)
                goal_pose.pose.orientation.y = float(init_pose.orientation.y)
                goal_pose.pose.orientation.z = float(init_pose.orientation.z)
                goal_pose.pose.orientation.w = float(init_pose.orientation.w)

                if self.islogging:
                    self.get_logger().info(
                        f"[GOAL_PUB] Agent {self.agent_ID} has no current task — "
                        f"publishing HOME goal: "
                        f"lat={goal_pose.pose.position.latitude:.7f}, "
                        f"lon={goal_pose.pose.position.longitude:.7f}, "
                        f"alt(ortho)={goal_pose.pose.position.altitude:.3f}"
                    )

                self.goal_pose_publisher.publish(goal_pose)
            else:
                if self.islogging:
                    self.get_logger().warning(
                        f"[GOAL_PUB] Agent {self.agent_ID} has no current task and no initial pose; "
                        f"no goal published."
                    )
        except Exception as e:
            if self.islogging:
                self.get_logger().error(f"[GOAL_PUB] Goal pose error: {e}")
            print("Goal pose error")

    def __select_random_solution(self):
        if not self.population:
            return None, None
        idx = random.randrange(len(self.population))
        return idx, self.population[idx]

    def regular_current_task_publish_timer(self):
        """
        Publishes the coalition best solution.
        """
        if self.current_task is not None:
            current_task = CurrentTask()
            current_task.agent_id = self.agent_ID
            current_task.current_task = self.current_task
            self.current_task_publisher.publish(current_task)
            if self.islogging:
                print(f"Publishing current task: {self.current_task}")

    def __regular_solution_publish_timer(self):
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

    def __remove_covered_tasks_from_solution(self, solution):
        """
        Remove already-covered tasks while preserving per-agent boundaries.
        Fail fast if any invalid task IDs or slice issues are detected.
        """
        if solution is None:
            return None

        from copy import deepcopy
        order, allocations = deepcopy(solution)

        n_tasks = len(self.is_covered)
        new_order, new_allocs = [], []

        cursor = 0
        total_alloc = sum(allocations)
        if total_alloc > len(order):
            msg = (f"[FATAL] sum(allocations)={total_alloc} > len(order)={len(order)}; "
                   f"allocations inconsistent with order.")
            print(msg)
            raise IndexError(msg)

        for seg_idx, count in enumerate(allocations):
            seg_end = cursor + int(count)
            if seg_end > len(order):
                msg = (f"[FATAL] seg_idx={seg_idx} slice out of bounds: "
                       f"cursor={cursor}, count={count}, len(order)={len(order)}")
                print(msg)
                raise IndexError(msg)

            seg = order[cursor:seg_end]
            seg_kept = []

            for t in seg:
                try:
                    ti = int(t)
                except Exception:
                    msg = f"[FATAL] Non-integer task id in seg={seg_idx}: t={t!r}"
                    print(msg)
                    raise IndexError(msg)

                if ti < 0 or ti >= n_tasks:
                    msg = (f"[FATAL] Out-of-range task id in seg={seg_idx}: "
                           f"t={ti}, n_tasks={n_tasks}")
                    print(msg)
                    raise IndexError(msg)

                if not self.is_covered[ti]:
                    seg_kept.append(ti)

            new_order.extend(seg_kept)
            new_allocs.append(len(seg_kept))
            cursor = seg_end

        if sum(new_allocs) != len(new_order):
            msg = (f"[FATAL] new_allocs sum mismatch: "
                   f"sum(new_allocs)={sum(new_allocs)} len(new_order)={len(new_order)}")
            print(msg)
            raise IndexError(msg)

        return (new_order, new_allocs)

    def __preserve_next_task_if_better(self, reference_solution, target_solution, base_f=None):
        """
        Try to modify target_solution so that THIS agent keeps the same 'next task'
        as in self.current_task (preferred) or, if unavailable, from reference_solution.
        """
        if not target_solution:
            return target_solution

        my_idx = self.__active_agent_index()
        task_to_preserve = getattr(self, "current_task", None)
        if task_to_preserve is None and reference_solution:
            task_to_preserve = self.__next_task_of_agent(reference_solution, my_idx)
        if task_to_preserve is None:
            return target_solution

        order = list(target_solution[0])
        alloc = list(target_solution[1])

        try:
            pos = order.index(task_to_preserve)
        except ValueError:
            return target_solution

        cum, owner_idx = 0, None
        for idx, a in enumerate(alloc):
            if pos < cum + a:
                owner_idx = idx
                break
            cum += a
        if owner_idx is None:
            return target_solution

        my_start, my_end = self.__segment_bounds(alloc, my_idx)
        if owner_idx == my_idx and pos == my_start:
            return target_solution

        removed = order.pop(pos)
        alloc[owner_idx] -= 1
        if pos < my_start:
            my_start -= 1
            my_end -= 1
        order.insert(my_start, removed)
        alloc[my_idx] += 1
        if any(a < 0 for a in alloc):
            return target_solution
        modified = (order, alloc)
        if base_f is None:
            base_f = self.__evaluate_solution(target_solution)
        mod_f = self.__evaluate_solution(modified)
        return modified if mod_f < base_f else target_solution

    def __update_robot_cost_matrix(self):
        self.problem.update_robot_cost_matrix(self.robot_poses)

    def __apply_operator(self, operator):
        """
        Call an operator only when there is at least one other valid candidate in the population.
        Log errors using RcutilsLogger.error with traceback text.
        """
        import traceback

        if not self.current_solution or not isinstance(self.current_solution, (list, tuple)) \
                or len(self.current_solution) < 2 or not self.current_solution[0]:
            return None

        def _valid(sol):
            if not sol or not isinstance(sol, (list, tuple)) or len(sol) < 2:
                return False
            order, alloc = sol
            return isinstance(order, (list, tuple)) and len(order) > 0 and isinstance(alloc, (list, tuple))

        pop = self.population or []
        valid = [s for s in pop if _valid(s)]
        if self.islogging:
            print(len(valid))

        if len(valid) <= 1:
            self.get_logger().debug("[OP] Skipping operator: not enough valid candidates in population.")
            return None

        try:
            return run_with_timeout(
                OperatorFunctions.apply_op,
                args=(
                    operator,
                    self.current_solution,
                    valid,
                    self.problem.cost_matrix,
                    [self.problem.current_robot_cost_matrix[i]
                     for i, purged in enumerate(self.purged_agents) if not purged],
                    self.problem.initial_robot_cost_matrix
                ),
                timeout=5.0
            )
        except Exception as e:
            tb = traceback.format_exc()
            self.get_logger().error(f"[OP] apply_op raised: {type(e).__name__}: {e}\n{tb}")
            return None

    def __calculate_solution_fitnesses(self, c_new):
        new_solution_fitness = RealisticFitness.fitness_function_robot_pose(
            c_new,
            self.problem
        )
        current_solution_fitness = RealisticFitness.fitness_function_robot_pose(
            self.current_solution,
            self.problem
        )
        local_best_solution_fitness = None
        if self.local_best_solution:
            local_best_solution_fitness = RealisticFitness.fitness_function_robot_pose(
                self.local_best_solution,
                self.problem
            )
        coalition_best_solution_fitness = None
        if self.coalition_best_solution:
            coalition_best_solution_fitness = RealisticFitness.fitness_function_robot_pose(
                self.coalition_best_solution,
                self.problem
            )
        return coalition_best_solution_fitness, current_solution_fitness, local_best_solution_fitness, new_solution_fitness

    def __calculate_locking_solution_fitnesses(self, c_new):
        new_solution_fitness = self._fitness(
            c_new
        )
        current_solution_fitness = self._fitness(
            self.current_solution
        )
        local_best_solution_fitness = None
        if self.local_best_solution:
            local_best_solution_fitness = self._fitness(
                self.local_best_solution
            )
        coalition_best_solution_fitness = None
        if self.coalition_best_solution:
            coalition_best_solution_fitness = self._fitness(
                self.coalition_best_solution
            )
        return coalition_best_solution_fitness, current_solution_fitness, local_best_solution_fitness, new_solution_fitness

    def __is_stuck(self):
        if len(self.previous_experience) >= 2:
            n_int = len(self.intensifiers)

            _, op_prev, gain_prev = self.previous_experience[-2]
            _, op_last, gain_last = self.previous_experience[-1]

            op_prev = int(op_prev)
            op_last = int(op_last)

            both_intensifiers = (op_prev < n_int) and (op_last < n_int)
            no_or_positive_both = (gain_prev >= 0.0) and (gain_last >= 0.0)

            if both_intensifiers and no_or_positive_both:
                if self.islogging:
                    print("Am Stuck")
                return True

        return False

    def __update_coalition_best(self, c_new):
        self._set_coalition_best_solution(c_new)
        self.coalition_best_agent = self.agent_ID
        self.best_coalition_improved = True
        solution = Solution()
        solution.id = self.agent_ID
        solution.order = self.coalition_best_solution[0]
        solution.allocations = self.coalition_best_solution[1]
        self.solution_publisher.publish(solution)

    def __update_local_best(self, c_new):
        self.local_best_solution = deepcopy(c_new)
        self.best_local_improved = True

    def __try_preserve_next_task(self, c_new):
        try:
            c_pres = self.__preserve_next_task_if_better(self.current_solution, c_new)
            if c_pres is not None and c_pres is not c_new:
                c_new = c_pres
        except Exception as e:
            self.get_logger().warning(f"[preserve_next_task] skipped due to: {e}")
        return c_new

    def __finish_di_cycle(self):
        learning_method_switch = {
            LearningMethod.FERREIRA: self.__individual_learning_old,
            LearningMethod.Q_LEARNING: self.__individual_learning
        }
        learning_function = learning_method_switch.get(self.learning_method)
        if self.islogging:
            print(f"{self.previous_experience}")
        if learning_function:
            self.weight_matrix.weights = learning_function()
        else:
            self.get_logger().error(f"[CHK] Unknown learning method: {self.learning_method}")
        self.best_local_improved = False
        if self.best_coalition_improved:
            self.best_coalition_improved = False
            # Use Weights for traditional approaches
            msg = Weights()
            msg_dict = self.weight_matrix.pack_weights(self.agent_ID)
            msg.id = msg_dict["id"]
            msg.rows = msg_dict["rows"]
            msg.cols = msg_dict["cols"]
            msg.weights = msg_dict["weights"]
            self.weight_publisher.publish(msg)
        if self.received_weight_matrices:
            self.__mimetism_learning(self.received_weight_matrices, self.rho)
            self.received_weight_matrices = []

        try:
            if (self.inject_best_on_cycle
                    and self.coalition_best_solution
                    and self.population
                    and random.random() < self.inject_best_prob):
                worst_idx, _ = max(
                    enumerate(self.population),
                    key=lambda it: self._fitness(it[1])
                )
                if self.population[worst_idx] != self.coalition_best_agent:
                    self.population[worst_idx] = deepcopy(self.coalition_best_solution)

                # Make it the current solution and align the parent slot
                self.current_solution = deepcopy(self.coalition_best_solution)
                self.current_parent_idx = worst_idx
                self.no_improvement_attempt_count = 0
                if self.islogging:
                    self.get_logger().info(
                        f"[INJECT] Replaced worst idx={worst_idx} with coalition best; "
                        f"set as current_solution."
                    )
        except Exception as e:
            self.get_logger().warning(f"[INJECT] failed to inject coalition best: {e}")

        self.previous_experience = []
        self.di_cycle_count = 0

    def _fitness(self, sol):
        """Uniform fitness wrapper that respects lock_mode."""
        if sol is None or self.problem.current_robot_cost_matrix is None or len(self.problem.current_robot_cost_matrix) < self.num_tsp_agents:
            return float("inf")
        if self.lock_mode:
            return RealisticFitness.fitness_function_locked_tasks(sol, self.problem, self.current_tasks)
        return RealisticFitness.fitness_function_robot_pose(sol, self.problem)

    def _reinsert_child(
            self,
            child_solution,
            replace_policy: str = "always",  # "better" | "non_worse" | "always"
            fallback: str = "worst"  # "worst" | "none"
    ):
        """
        Put `child_solution` back into the pool, ideally replacing the parent at `self.current_parent_idx`.

        replace_policy:
          - "better":   replace parent only if child strictly improves fitness
          - "non_worse":replace if child <= parent
          - "always":   always replace parent slot

        fallback:
          - "worst": if parent index unavailable, replace the worst pool member if child is better
          - "none":  do nothing if parent index unavailable
        """
        if child_solution is None or not self.population:
            return

        def _ok_idx(i: int) -> bool:
            return isinstance(i, int) and 0 <= i < len(self.population)

        # Try to replace the parent slot
        if _ok_idx(getattr(self, "current_parent_idx", None)):
            try:
                parent_fit = self._fitness(self.population[self.current_parent_idx])
                child_fit = self._fitness(child_solution)

                do_replace = (
                        (replace_policy == "always") or
                        (replace_policy == "non_worse" and child_fit <= parent_fit) or
                        (replace_policy == "better" and child_fit < parent_fit)
                )

                if do_replace:
                    self.population[self.current_parent_idx] = deepcopy(child_solution)
            except Exception as e:
                self.get_logger().warning(
                    f"[REINSERT] failed to replace parent at {self.current_parent_idx}: {e}"
                )
            return

        # Fallback strategy
        if fallback == "worst":
            try:
                worst_idx, worst_sol = max(
                    enumerate(self.population), key=lambda it: self._fitness(it[1])
                )
                child_fit = self._fitness(child_solution)
                if child_fit < self._fitness(worst_sol):
                    self.population[worst_idx] = deepcopy(child_solution)
            except Exception as e:
                self.get_logger().warning(f"[REINSERT] fallback failed: {e}")

    def kill_robot_callback(self, msg, failed_agent_id):
        """
        This fails a robot and freezes its goal at its current (or initial) GeoPose.
        """
        print(f"Kill signal received for agent: {failed_agent_id}")
        if failed_agent_id == self.agent_ID and not self.am_i_failed:
            self.am_i_failed = True
            print("I'm failed")
            self.failed_agents[self.agent_ID] = True

            if msg.data:
                src = self.robot_poses[self.agent_ID] or self.initial_robot_poses[self.agent_ID]
                if src is None:
                    self.current_task = None
                    return

                goal_pose = GeoPoseStamped()
                goal_pose.header.stamp = Time()
                goal_pose.header.frame_id = "world"
                goal_pose.pose = deepcopy(src)
                self.goal_pose_publisher.publish(goal_pose)

                self.current_task = None

    def revive_robot_callback(self, msg):
        """
        Revives the agent.
        If a revive message is received for the agent, it will set itself as unfailed, to be unpurged, and sets the
        last_purge_agent_true_id to none(is this right?).
        Then all necessary timers and subscribers are started.
        """
        print(f"Revive request received for Agent {self.agent_ID}")
        self.am_i_failed = False
        if not self.failed_agents[
            self.agent_ID]:  # If this agent was never failed, ignore the request
            print(f"Agent {self.agent_ID} is already active. Ignoring revive request.")
            return
        self.failed_agents[self.agent_ID] = False
        self.purged_agents[self.agent_ID] = False

        # Ensure `run_timer` is restarted if not already running
        if self.run_timer is None:
            print(f"Restarting run_step for Agent {self.agent_ID}.")
            self.run_timer = self.create_timer(0.1, self.run_step, callback_group=self.me_cb_group)

        # Restart other necessary timers
        if self.environmental_representation_timer is None:
            self.environmental_representation_timer = self.create_timer(5,
                                                                        self.__environmental_representation_timer_callback,
                                                                        callback_group=self.cb_group)

        if self.run_goal_publisher_timer is None:
            self.run_goal_publisher_timer = self.create_timer(0.5, self.__publish_goal_pose,
                                                              callback_group=self.cb_group)

        if self.solution_publisher_timer is None:
            self.solution_publisher_timer = self.create_timer(2, self.__regular_solution_publish_timer,
                                                              callback_group=self.cb_group)

        if self.environmental_representation_subscriber is None:
            self.environmental_representation_subscriber = self.create_subscription(
                EnvironmentalRepresentation,
                '/environmental_representation',
                self.__environmental_representation_callback,
                40,
                callback_group=self.cb_group
            )

        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.__solution_update_callback, 10)

        self.__assign_next_task(self.coalition_best_solution)
        print(self.current_task)

        print(f"Agent {self.agent_ID} successfully revived and all timers restarted.")

    def purge_agent(self, purge_agent_true_id):
        """
        Purge the LAST active agent (highest true id not yet purged).
        Assumes purges happen strictly from the end in descending true_id order.
        """

        def update_solution(sol):
            if sol is None:
                return None
            order, allocations = sol
            if not allocations:
                return (order, allocations)  # nothing to purge

            # The last active agent is the last segment
            idx_in_alloc = len(allocations) - 1

            # Slice bounds of that last segment
            start_idx = sum(allocations[:-1])
            count = allocations[-1]
            end_idx = start_idx + count

            # Bounds guard (paranoia)
            if start_idx < 0 or end_idx > len(order):
                self.get_logger().warning(
                    f"[purge_last_agent] OOB slice start={start_idx} end={end_idx} len(order)={len(order)} "
                    f"allocs={allocations}"
                )
                return (order, allocations)

            # Extract tasks to reassign
            purged_tasks = order[start_idx:end_idx]

            # Remove last segment
            del order[start_idx:end_idx]
            allocations.pop()  # drop last

            if not allocations:
                # Edge case: that was the only agent; no one to reassign to
                return (order, allocations)

            # Decide the recipient (choose yourself, or nearest, or previous agent, etc.)
            new_owner_true = self.agent_ID

            # Map recipient true id to current index after pop.
            # Because we purge from the end in descending order, the active order is 0..len(alloc)-1.
            new_owner_idx = min(new_owner_true, len(allocations) - 1)

            # Insert at the END of the recipient's segment
            insert_pos = sum(allocations[:new_owner_idx]) + allocations[new_owner_idx]
            order[insert_pos:insert_pos] = purged_tasks
            allocations[new_owner_idx] += len(purged_tasks)

            # Invariant check
            if sum(allocations) != len(order):
                self.get_logger().warning(
                    f"[purge_last_agent] fixing alloc/order mismatch; sum={sum(allocations)} len(order)={len(order)}"
                )
                # Minimal repair (re-sum per segment cap)
                fixed, cursor = [], 0
                for a in allocations:
                    take = min(a, max(0, len(order) - cursor))
                    fixed.append(take);
                    cursor += take
                allocations = fixed

            return (order, allocations)

        # Apply to all holders
        if self.population:
            for i, sol in enumerate(self.population):
                self.population[i] = update_solution(sol)
        self.current_solution = update_solution(self.current_solution)
        self.local_best_solution = update_solution(self.local_best_solution)
        self.coalition_best_solution = update_solution(self.coalition_best_solution)

        # Re-pick next task after topology change
        self.__assign_next_task(self.current_solution)

    def unpurge_agent(self, agent_id):
        print(f"Reviving Agent {agent_id}...")
        # Mark agent as revived
        self.failed_agents[agent_id ] = False
        self.purged_agents[agent_id] = False
        self.agents_to_revive[agent_id] = False
        self.finished_robots[agent_id] = False
        # Function to insert an empty path at the right position
        def reintegrate_agent(solution):
            if solution is None:
                return None

            order, allocations = solution
            if agent_id < len(allocations):  # Ensure valid index
                # Insert an empty task allocation for the revived agent
                allocations.insert(agent_id, 0)  # No tasks assigned
                print(f"Inserted empty path for Agent {agent_id}")
            return (order, allocations)

        # Reintegrate the agent into all solutions with an empty path
        for idx, solution in enumerate(self.population):
            self.population[idx] = reintegrate_agent(solution)
        self.current_solution = reintegrate_agent(self.current_solution)
        self.coalition_best_solution = reintegrate_agent(self.coalition_best_solution)
        # Update cost matrix to include revived agent

    def check_stale_agents(self):
        """
        If a message hasn't been received by an agent within a threshold period, set the agent as failed and to be
        purged.
        """
        if not self.am_i_failed:
            current_time = time()
            timeout_threshold = 3  # Define a threshold (e.g., 10 seconds)

            for agent_id, last_time in list(self.last_env_rep_timestamps.items()):
                if self.islogging:
                    print(f"{agent_id}: time {current_time - last_time}")
                if current_time - last_time > timeout_threshold and self.failed_agents[agent_id] is False:
                    self.failed_agents[agent_id] = True
                    self.get_logger().warning(
                        f"(agent_{self.agent_ID}) Agent {agent_id} has not sent an update for {current_time - last_time:.2f} seconds.")

    @staticmethod
    def haversine(lat1, lon1, alt1, lat2, lon2, alt2):
        """
        Calculates Haversine distance
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ asin( √a )
        d = R ⋅ c
        φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6378160)
        """
        R = 6378160  # Earth radius in miles (for km use 6372.8)

        # Convert lat/lon from degrees to radians
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)

        # Haversine formula for surface distance
        a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
        c = 2 * asin(sqrt(a))
        haversine_distance = R * c  # Great-circle distance in miles

        # Compute altitude difference in miles
        alt_diff = alt2 - alt1  # Altitude difference in miles

        # Compute 3D distance using Pythagorean theorem
        distance_3D = sqrt(haversine_distance ** 2 + alt_diff ** 2)

        return distance_3D


    def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """

        if self.am_i_failed:
            return

        # Update the cost matrix with any new robot positions.
        self.__update_robot_cost_matrix()

        if self.current_solution is not None and len(self.current_solution[0]) > 0:

            if self.current_solution:
                self.current_solution = self.__remove_covered_tasks_from_solution(self.current_solution)
            if self.coalition_best_solution:
                self.coalition_best_solution = self.__remove_covered_tasks_from_solution(self.coalition_best_solution)


            if self.failed_agents != self.purged_agents:
                print("Purging agent")
                for i in range(len(self.purged_agents)):
                    if self.purged_agents[i] != self.failed_agents[i]:
                        self.purged_agents[i] = True
                        self.purge_agent(i)

            if any(self.agents_to_revive):
                print("Reviving agent")
                for i in range(len(self.agents_to_revive)):
                    if self.agents_to_revive[i]:
                        self.unpurge_agent(i)

            # if self.agent_to_revive:
            #     if self.agent_ID >= self.agent_to_revive:
            #         print(f"I was robot: {self.agent_ID}, now becoming robot: {self.agent_ID + 1}")
            #         self.agent_ID = self.agent_ID + 1
            #     self.unpurge_agent(self.agent_to_revive)
            #     self.agent_to_revive = None


            if self.__stopping_criterion(self.iteration_count):
                self.run_timer.cancel()
                return

            condition = ConditionFunctions.perceive_condition_row(self.previous_experience, self.intensifiers,
                                                                  self.diversifiers)

            if self.no_improvement_attempt_count >= self.no_improvement_attempts:
                self.current_parent_idx, self.current_solution = self.__select_random_solution()
                self.no_improvement_attempt_count = 0

            enabled_ops = self.intensifiers + self.diversifiers

            operator = OperatorFunctions.choose_operator(self.weight_matrix.weights, condition, enabled_ops)

            c_new = self.__apply_operator(operator)
            if c_new is None:
                print(f"[TIMEOUT] Operator {operator} took too long. Skipping this step.")
                self.no_improvement_attempt_count += 1
                return

            if self.preserve_next_task:
                c_new = self.__try_preserve_next_task(c_new)

            if self.lock_mode:
                (coalition_best_solution_fitness, current_solution_fitness,
                 local_best_solution_fitness, new_solution_fitness) = self.__calculate_locking_solution_fitnesses(c_new)
            else:
                (coalition_best_solution_fitness, current_solution_fitness,
                 local_best_solution_fitness, new_solution_fitness) = self.__calculate_solution_fitnesses(c_new)

            if c_new:
                gain = new_solution_fitness - current_solution_fitness
                self.__update_experience(condition, operator, gain)

                if self.local_best_solution is None or new_solution_fitness < local_best_solution_fitness:
                    self.__update_local_best(c_new)
                    self.no_improvement_attempt_count = 0
                else:
                    self.no_improvement_attempt_count += 1

                if self.coalition_best_solution is None or new_solution_fitness < coalition_best_solution_fitness:
                    self.__update_coalition_best(c_new)

                self.current_solution = c_new
                self._reinsert_child(c_new, replace_policy="better", fallback="worst")
                self.di_cycle_count += 1

                if self.__end_of_di_cycle(self.di_cycle_count) or self.__is_stuck():
                    self.__finish_di_cycle()

                self.iteration_count += 1
            else:
                print("Something went wrong with applying the operator and resulted in a None.")




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

import sys, traceback, signal
import rclpy
from rclpy.node import Node

import sys, os, signal, traceback, threading, asyncio, faulthandler


def main(args=None):
    # ---------- Extra debugging hooks ----------
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1)

    os.environ.setdefault("PYTHONASYNCIODEBUG", "1")
    try:
        loop = asyncio.get_event_loop()
        def _asyncio_exc_handler(loop, context):
            msg = context.get("message", "asyncio exception")
            exc = context.get("exception")
            print(f"[ASYNCIO-EXC] {msg}", file=sys.stderr)
            if exc:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
        loop.set_exception_handler(_asyncio_exc_handler)
    except Exception:
        pass

    def _thread_excepthook(args: threading.ExceptHookArgs):
        print(f"[THREAD-EXC:{args.thread.name}] {args.exc_type.__name__}: {args.exc_value}", file=sys.stderr)
        traceback.print_tb(args.exc_traceback)
    threading.excepthook = _thread_excepthook

    PKG_HINTS = ("cbm_pop", "RealisticSimulator")
    def print_root_cause(e: BaseException):
        frames = list(traceback.walk_tb(e.__traceback__))
        chosen = None
        for frame, lineno in reversed(frames):
            if any(h in frame.f_code.co_filename for h in PKG_HINTS):
                chosen = (frame, lineno)
                break
        if not chosen and frames:
            chosen = frames[-1]
        print(f"\n[ROOT-CAUSE] {type(e).__name__}: {e}", file=sys.stderr)
        if chosen:
            frame, lineno = chosen
            print(f"  at {frame.f_code.co_filename}:{lineno} in {frame.f_code.co_name}()", file=sys.stderr)
        print("[TRACEBACK]", file=sys.stderr)
        traceback.print_exception(type(e), e, e.__traceback__)

    sys.excepthook = lambda exctype, value, tb: print_root_cause(value)

    # ---------- ROS 2 setup ----------
    rclpy.init(args=args)
    temp_node = Node("parameter_loader")
    params = {
        "agent_id": 0,
        "runtime": -1.0,
        "learning_method": "Q-Learning",
        "lr": 0.5,
        "gamma_decay": 0.99,
        "positive_reward": 1.0,
        "negative_reward": -0.5,
        "num_tsp_agents": 10,
        "problem_size": 15,
        "lock_mode": False,
        "preserve_next_task": False,
        # NEW PARAMETERS
        "eta": 0.1,
        "rho": 0.1,
        # UCB toggle/strength
        "use_ucb": False,
        "ucb_c": 1.0,
        "inject_best_on_cycle": True,  # or False by default
        "inject_best_prob": 0.90,
        "initialise_with_heuristic": True,
        "problem_class": "FourWhichWay",
        "problem_seed": 1
    }
    for k, v in params.items():
        temp_node.declare_parameter(k, v)

    agent_id        = temp_node.get_parameter("agent_id").value
    runtime         = temp_node.get_parameter("runtime").value
    learning_method = temp_node.get_parameter("learning_method").value
    lr              = temp_node.get_parameter("lr").value
    gamma_decay     = temp_node.get_parameter("gamma_decay").value
    positive_reward = temp_node.get_parameter("positive_reward").value
    negative_reward = temp_node.get_parameter("negative_reward").value
    num_tsp_agents  = temp_node.get_parameter("num_tsp_agents").value
    problem_size    = temp_node.get_parameter("problem_size").value
    lock_mode       = temp_node.get_parameter("lock_mode").value
    preserve_next_task = temp_node.get_parameter("preserve_next_task").value
    eta             = temp_node.get_parameter("eta").value
    rho             = temp_node.get_parameter("rho").value
    use_ucb         = temp_node.get_parameter("use_ucb").value
    ucb_c           = temp_node.get_parameter("ucb_c").value
    inject_best_on_cycle = temp_node.get_parameter("inject_best_on_cycle").value
    inject_best_prob = temp_node.get_parameter("inject_best_prob").value
    initialise_with_heuristic = temp_node.get_parameter("initialise_with_heuristic").value

    problem_class = temp_node.get_parameter("problem_class").value
    problem_seed = temp_node.get_parameter("problem_seed").value

    temp_node.destroy_node()

    node_name = f"cbm_population_agent_{agent_id}"
    agent = CBMPopulationAgentOnlineRealisticSimulation(
        pop_size=10, eta=eta, rho=rho, di_cycle_length=10, epsilon=0.01,
        num_iterations=9999999, num_solution_attempts=21,
        agent_id=agent_id, node_name=node_name, learning_method=learning_method,
        num_tsp_agents=num_tsp_agents, lr=lr,
        gamma_decay=gamma_decay, positive_reward=positive_reward,
        negative_reward=negative_reward, lock_mode=lock_mode, preserve_next_task=preserve_next_task,
        use_ucb=use_ucb, ucb_c=ucb_c, problem_size=problem_size,
        inject_best_on_cycle=inject_best_on_cycle,
        inject_best_prob=inject_best_prob,
        initialise_with_heuristic=initialise_with_heuristic,
        problem_class=problem_class,
        problem_seed=problem_seed
    )
    print("CBMPopulationAgentOnlineRealisticSimulation has been initialized.")

    shutdown_reason = {"value": None}
    def begin_shutdown(reason: str):
        if shutdown_reason["value"] is None:
            shutdown_reason["value"] = reason
            try:
                agent.get_logger().info(f"[SHUTDOWN] reason={reason}")
            except Exception:
                print(f"[SHUTDOWN] reason={reason}")
            try:
                agent.destroy_node()
            except Exception:
                pass
            if rclpy.ok():
                try:
                    executor.shutdown()
                except Exception:
                    pass
                try:
                    rclpy.shutdown()
                except Exception:
                    pass

    signal.signal(signal.SIGINT,  lambda *_: begin_shutdown("SIGINT"))
    signal.signal(signal.SIGTERM, lambda *_: begin_shutdown("SIGTERM"))

    if runtime != -1:
        agent.create_timer(runtime, lambda: begin_shutdown("runtime_timer"))

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        begin_shutdown("KeyboardInterrupt")
    except Exception as e:
        print_root_cause(e)
        begin_shutdown(f"exception:{type(e).__name__}:{e}")
    finally:
        if shutdown_reason["value"] is None:
            begin_shutdown("spin_returned")
        print(f"[MAIN] shutdown complete; reason={shutdown_reason['value']}")
        sys.exit(0)

if __name__ == "__main__":
    main()

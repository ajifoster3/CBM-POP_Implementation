import argparse
import csv
import sys
import math
from datetime import datetime
import os
from time import time
import json
from collections import Counter

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from cbm_pop.RealisticSimulator.realistic_fitness import RealisticFitness
from cbm_pop.RealisticSimulator.realistic_problem import ProblemClass
from cbm_pop_interfaces.msg import (
    Solution,
    EnvironmentalRepresentation,
    FinishedCoverage,
    CumulativeReward,
    CurrentTask,
)
from std_msgs.msg import Bool
from geographic_msgs.msg import GeoPoseStamped


class ProblemAdapter:
    """
    Minimal adapter exposing:
      - cost_matrix                (N x N)
      - current_robot_cost_matrix  (R x N)
      - initial_robot_cost_matrix  (R x N)
    """
    def __init__(self):
        self.cost_matrix = None
        self.current_robot_cost_matrix = None
        self.initial_robot_cost_matrix = None


class RealisticFitnessLogger(Node):
    def __init__(self, args, problem_class: ProblemClass = ProblemClass.FourWhichWay):
        super().__init__('fitness_logger')
        print("Starting Logger")
        self.logging_start_time = time()

        # kill-robot trigger config from arguments
        self.is_kill_enabled = bool(args.enable_kill)
        self.kill_threshold = float(args.kill_threshold)
        self.number_to_kill = int(args.num_to_kill)
        self.is_kill_published = False
        self.kill_pubs = []  # set in start_listeners if enabled

        # revive-robot trigger config from arguments
        self.is_revive_enabled = bool(args.enable_revive)
        self.revive_threshold = float(args.revive_threshold)
        self.number_to_revive = self.number_to_kill
        self.is_revive_published = False
        self.revive_pubs = []

        # Timeouts
        self.timeout = 10.0
        self.last_env_update_time = None
        self.termination_timeout = 20.0

        self.declare_parameter('timeout', 0.0)
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        # Allow overriding agents if you want (otherwise auto-detect from first Solution)
        self.declare_parameter('num_tsp_agents', 10)
        self.declare_parameter('problem_size', 15)
        self.num_tsp_agents = self.get_parameter('num_tsp_agents').get_parameter_value().integer_value
        self.problem_size = self.get_parameter('problem_size').get_parameter_value().integer_value

        if self.problem_size < 1:
            raise ValueError("problem_size must be >= 1")

        # Output root
        self.declare_parameter('parent_log_dir', 'resources/run_logs')
        parent_log_dir = self.get_parameter('parent_log_dir').get_parameter_value().string_value

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join(parent_log_dir, timestamp)
        os.makedirs(self.run_folder, exist_ok=True)

        # Auto-detected agent count used for latching matrices
        self.expected_agents = None  # set from len(msg.allocations) on first Solution

        self.current_tasks = [-1] * self.num_tsp_agents
        self.is_all_poses_set = False

        # Output files
        self.position_log_file = os.path.join(self.run_folder, "robot_positions.csv")
        self.cumulative_log_file = os.path.join(self.run_folder, "cumulative_reward.csv")
        self.fitness_log_file = os.path.join(self.run_folder, "fitness_log_file.csv")
        self.best_solution_file = os.path.join(self.run_folder, "current_best_solution.csv")
        self.environmental_log_file = os.path.join(self.run_folder, "environmental_representation.csv")
        self.debug_log_file = os.path.join(self.run_folder, "debug_log.txt")

        # CSV headers
        with open(self.position_log_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["ROS Time (seconds)", "Agent ID", "Latitude", "Longitude", "Altitude"])
        with open(self.fitness_log_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["ROS Time (seconds)", "Agent ID", "Fitness", "Paths per Robot"])
        with open(self.best_solution_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["ROS Time (seconds)", "Agent ID", "Fitness", "Order", "Allocations"])
        with open(self.environmental_log_file, mode="w", newline="") as f:
            csv.writer(f).writerow(["ROS Time (seconds)", "Agent ID", "Is Covered"])
        with open(self.debug_log_file, mode="w") as f:
            f.write("Debug log for realistic_fitness_logger\n")

        # -------- Load task poses from JSON (lat / lon / alt) --------
        if problem_class == ProblemClass.FourWhichWay:
            filename = "/home/ajifoster3/Downloads/Poses/4Whichway_geoposes.json"
        else:
            raise NotImplementedError(f"Unsupported problem class: {problem_class}")

        with open(filename, "r") as file:
            print(f"Using task pose file: {filename}")
            data = json.load(file)

        # Each entry is a dict with lat / lon / alt + orientation
        self.task_poses = [
            {
                "latitude": item["GeoPose"]["position"]["latitude"],
                "longitude": item["GeoPose"]["position"]["longitude"],
                "altitude": item["GeoPose"]["position"]["altitude"],
                "orientation_x": item["GeoPose"]["orientation"]["x"],
                "orientation_y": item["GeoPose"]["orientation"]["y"],
                "orientation_z": item["GeoPose"]["orientation"]["z"],
                "orientation_w": item["GeoPose"]["orientation"]["w"]
            }
            for item in data
        ]

        # Problem adapter + matrices
        self.problem = ProblemAdapter()
        self.problem.cost_matrix = self.calculate_task_task_cost_matrix()

        # Robot poses & matrices (lat, lon, alt tuples)
        self.initial_robot_poses = [None] * self.num_tsp_agents
        self.robot_poses = [None] * self.num_tsp_agents
        self.problem.initial_robot_cost_matrix = None
        self.problem.current_robot_cost_matrix = None

        # State
        self.best_fitness = None
        self.best_solution = None
        self.is_covered = [False] * len(self.task_poses)
        self.finished_robots = [False] * self.num_tsp_agents

        # Diagnostics
        self.received_solutions = 0
        self.scored_solutions = 0
        self.skipped_solutions = 0
        self.skip_reasons = Counter()

        # Subscriptions
        cb_group = ReentrantCallbackGroup()
        self.global_pose_subscribers = []
        for agent_id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{agent_id}/global_pose'
            sub = self.create_subscription(
                GeoPoseStamped,
                topic,
                lambda msg, agent=agent_id: self.global_pose_callback(msg, agent),
                10,
                callback_group=cb_group
            )
            self.global_pose_subscribers.append(sub)

        # Subscribe to best_solution immediately (don’t gate on poses)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10
        )

        self.finished_coverage_sub = self.create_subscription(
            FinishedCoverage, '/central_control/finished_coverage', self.finished_coverage_callback, 10
        )
        self.cumulative_reward_subscriber = self.create_subscription(
            CumulativeReward, '/cumulative_reward', self.cumulative_reward_callback, 10
        )
        self.current_task_subscriber = self.create_subscription(
            CurrentTask, 'current_task', self.current_task_update_callback, 10
        )
        self.environmental_subscriber = self.create_subscription(
            EnvironmentalRepresentation, '/environmental_representation', self.environmental_representation_callback, 10
        )
        self.stop_subscriber = self.create_subscription(
            Bool, 'stop_plotting', self.stop_callback, 10
        )


        if self.is_revive_enabled:
            for i in range(self.number_to_revive):
                self.revive_pubs.append(self.create_publisher(
                    Bool,
                    f"/central_control/uas_{self.num_tsp_agents - 1 - i}/revive_robot",
                    10
                ))

        # Timers
        self.create_timer(1.0, self.check_timeout)
        self.create_timer(5.0, self.debug_heartbeat)

        self.debug(f"Logger started. Output dir: {self.run_folder}")

    # ---------------- Utilities ----------------

    def debug(self, msg):
        stamp = time() - self.logging_start_time
        line = f"[{stamp:8.3f}] {msg}\n"
        # Console and file
        self.get_logger().info(msg)
        with open(self.debug_log_file, "a") as f:
            f.write(line)

    def calculate_task_task_cost_matrix(self):
        """
        Task–task cost: 3D haversine between task GeoPoses.
        """
        n = len(self.task_poses)
        M = np.zeros((n, n))

        for i, ti in enumerate(self.task_poses):
            lat1 = ti["latitude"]
            lon1 = ti["longitude"]
            alt1 = ti["altitude"]
            for j, tj in enumerate(self.task_poses):
                if i == j:
                    continue
                lat2 = tj["latitude"]
                lon2 = tj["longitude"]
                alt2 = tj["altitude"]
                M[i, j] = self.haversine(lat1, lon1, alt1, lat2, lon2, alt2)

        return M

    def calculate_robot_to_task_cost_matrix(self, robot_positions):
        """
        Robot–task cost: 3D haversine from robot (lat, lon, alt) to task (lat, lon, alt).
        """
        n = len(self.task_poses)
        r = len(robot_positions)
        M = np.zeros((r, n))

        for i, pos in enumerate(robot_positions):
            if pos is None:
                continue

            lat1, lon1, alt1 = pos

            for j, tj in enumerate(self.task_poses):
                lat2 = tj["latitude"]
                lon2 = tj["longitude"]
                alt2 = tj["altitude"]
                M[i, j] = self.haversine(lat1, lon1, alt1, lat2, lon2, alt2)

        return M

    def matrices_ready(self):
        if self.problem is None or self.problem.cost_matrix is None:
            return False
        if self.expected_agents is None:
            return False
        irm = self.problem.initial_robot_cost_matrix
        crm = self.problem.current_robot_cost_matrix
        if irm is None or crm is None:
            return False
        return (
            irm.shape == (self.expected_agents, len(self.task_poses)) and
            crm.shape == (self.expected_agents, len(self.task_poses)) and
            np.isfinite(irm).all() and np.isfinite(crm).all()
        )

    def validate_matrices(self):
        reasons = []
        # cost_matrix
        if self.problem.cost_matrix is None:
            reasons.append("cost_matrix=None")
        else:
            if self.problem.cost_matrix.shape[0] != self.problem.cost_matrix.shape[1]:
                reasons.append(f"cost_matrix not square {self.problem.cost_matrix.shape}")
            if not np.isfinite(self.problem.cost_matrix).all():
                reasons.append("cost_matrix has non-finite")
        # initial
        if self.problem.initial_robot_cost_matrix is None:
            reasons.append("initial_robot_cost_matrix=None")
        else:
            if self.expected_agents is not None and \
               self.problem.initial_robot_cost_matrix.shape[0] != self.expected_agents:
                reasons.append(
                    f"initial_robot_cost_matrix rows={self.problem.initial_robot_cost_matrix.shape[0]} != expected_agents={self.expected_agents}"
                )
            if not np.isfinite(self.problem.initial_robot_cost_matrix).all():
                reasons.append("initial_robot_cost_matrix has non-finite")
        # current
        if self.problem.current_robot_cost_matrix is None:
            reasons.append("current_robot_cost_matrix=None")
        else:
            if self.expected_agents is not None and \
               self.problem.current_robot_cost_matrix.shape[0] != self.expected_agents:
                reasons.append(
                    f"current_robot_cost_matrix rows={self.problem.current_robot_cost_matrix.shape[0]} != expected_agents={self.expected_agents}"
                )
            if not np.isfinite(self.problem.current_robot_cost_matrix).all():
                reasons.append("current_robot_cost_matrix has non-finite")
        return reasons

    def validate_solution(self, msg: Solution):
        order = list(msg.order)
        alloc = list(msg.allocations)
        reasons = []
        if sum(alloc) != len(order):
            reasons.append(f"alloc_sum({sum(alloc)}) != len(order)({len(order)})")
        if any(a < 0 for a in alloc):
            reasons.append("negative allocation(s)")
        if any(i < 0 or i >= len(self.task_poses) for i in order):
            reasons.append("order has out-of-range task index")
        return reasons

    # ---------------- Timers & Callbacks ----------------

    def debug_heartbeat(self):
        missing = []
        if self.expected_agents is not None:
            missing = [i for i in range(self.expected_agents) if self.robot_poses[i] is None]
        self.debug(
            f"HB: solutions recv={self.received_solutions}, scored={self.scored_solutions}, "
            f"skipped={self.skipped_solutions} {dict(self.skip_reasons)} | "
            f"mat_ready={self.matrices_ready()} poses_set={self.is_all_poses_set} "
            f"expected_agents={self.expected_agents} waiting_for={missing} "
            f"init_rows={None if self.problem.initial_robot_cost_matrix is None else self.problem.initial_robot_cost_matrix.shape} "
            f"cur_rows={None if self.problem.current_robot_cost_matrix is None else self.problem.current_robot_cost_matrix.shape}"
        )

    def check_timeout(self):
        if self.last_env_update_time is None:
            self.last_env_update_time = time()
            return
        elapsed = time() - self.last_env_update_time
        if elapsed > self.termination_timeout:
            self.get_logger().warn(
                f"No EnvironmentalRepresentation received for {elapsed:.2f}s. Shutting down...")
            self.destroy_node()
            rclpy.shutdown()

    def finished_coverage_callback(self, msg: FinishedCoverage):
        self.finished_robots[int(msg.robot_id)] = bool(msg.finished)
        if all(self.finished_robots[: (self.expected_agents or self.num_tsp_agents)]):
            self.create_timer(5.0, self.__shutdown_ros2)

    def __shutdown_ros2(self):
        print("Shutting down ROS2 system now.")
        self.destroy_node()
        rclpy.shutdown()

    def cumulative_reward_callback(self, msg: CumulativeReward):
        with open(self.cumulative_log_file, mode='a', newline='') as file:
            timestamp = time() - self.logging_start_time
            csv.writer(file).writerow([msg.agent_id, msg.cumulative_reward, timestamp])

    def global_pose_callback(self, msg: GeoPoseStamped, agent: int):
        # Extract robot position from GeoPoseStamped
        lat = msg.pose.position.latitude
        lon = msg.pose.position.longitude
        alt = msg.pose.position.altitude

        # Ensure arrays are large enough
        if agent >= len(self.initial_robot_poses):
            extend = agent + 1 - len(self.initial_robot_poses)
            self.initial_robot_poses.extend([None] * extend)
            self.robot_poses.extend([None] * extend)

        # Store initial pose (first time only)
        if self.initial_robot_poses[agent] is None:
            self.initial_robot_poses[agent] = (lat, lon, alt)

        # Always update current pose
        self.robot_poses[agent] = (lat, lon, alt)

        # Latch matrices when all poses known
        if self.expected_agents is not None and not self.is_all_poses_set:
            have = sum(p is not None for p in self.robot_poses[:self.expected_agents])
            if have == self.expected_agents:
                self.is_all_poses_set = True
                self.problem.initial_robot_cost_matrix = self.calculate_robot_to_task_cost_matrix(
                    self.initial_robot_poses[:self.expected_agents]
                )
                self.problem.current_robot_cost_matrix = self.calculate_robot_to_task_cost_matrix(
                    self.robot_poses[:self.expected_agents]
                )
                self.debug(f"Matrices latched for {self.expected_agents} agents")

        # Continuously refresh robot-to-task costs
        if self.is_all_poses_set and self.expected_agents is not None:
            self.problem.current_robot_cost_matrix = self.calculate_robot_to_task_cost_matrix(
                self.robot_poses[:self.expected_agents]
            )

        # Log position to CSV
        timestamp = time() - self.logging_start_time
        with open(self.position_log_file, mode="a", newline="") as f:
            csv.writer(f).writerow([timestamp, agent, lat, lon, alt])

    def current_task_update_callback(self, msg: CurrentTask):
        # Lazily ensure current_tasks is long enough
        if msg.agent_id >= len(self.current_tasks):
            self.current_tasks.extend([-1] * (msg.agent_id + 1 - len(self.current_tasks)))
        self.current_tasks[msg.agent_id] = msg.current_task

    def solution_update_callback(self, msg: Solution):
        self.received_solutions += 1
        timestamp = time() - self.logging_start_time
        self.debug(f"Solution recv id={msg.id} len(order)={len(msg.order)} alloc_sum={sum(msg.allocations)}")

        # Detect agent count on first solution
        if self.expected_agents is None:
            self.expected_agents = len(msg.allocations)
            self.debug(f"expected_agents set to {self.expected_agents} from first Solution")
            # If we already have all those poses, latch now
            have = sum(p is not None for p in self.robot_poses[:self.expected_agents])
            if have == self.expected_agents:
                self.is_all_poses_set = True
                self.problem.initial_robot_cost_matrix = self.calculate_robot_to_task_cost_matrix(
                    self.initial_robot_poses[:self.expected_agents]
                )
                self.problem.current_robot_cost_matrix = self.calculate_robot_to_task_cost_matrix(
                    self.robot_poses[:self.expected_agents]
                )
                self.debug(f"Matrices latched immediately on first Solution for {self.expected_agents} agents")
            else:
                missing = [i for i in range(self.expected_agents) if self.robot_poses[i] is None]
                self.debug(f"Waiting for poses from agents {missing}")

        # Validate Solution structure
        sol_issues = self.validate_solution(msg)
        if sol_issues:
            self.skipped_solutions += 1
            for r in sol_issues:
                self.skip_reasons[f"solution:{r}"] += 1
            self.debug(f"SKIP solution id={msg.id}: {'; '.join(sol_issues)}")
            return

        # Ensure matrices are ready
        if not self.matrices_ready():
            self.skipped_solutions += 1
            self.skip_reasons["matrices:not_ready"] += 1
            if self.expected_agents is not None:
                missing = [i for i in range(self.expected_agents) if self.robot_poses[i] is None]
                self.debug(f"SKIP solution id={msg.id}: matrices not ready; missing poses from {missing}")
            else:
                self.debug(f"SKIP solution id={msg.id}: matrices not ready; expected_agents unknown")
            return

        mat_issues = self.validate_matrices()
        if mat_issues:
            self.skipped_solutions += 1
            for r in mat_issues:
                self.skip_reasons[f"matrices:{r}"] += 1
            self.debug(f"SKIP solution id={msg.id}: {'; '.join(mat_issues)}")
            return

        # Build paths per agent (list of lists of task indices)
        task_order = list(msg.order)
        allocations = list(msg.allocations)
        cursor = 0
        paths = []
        for count in allocations:
            paths.append(task_order[cursor:cursor + count])
            cursor += count
        paths_json = json.dumps(paths)

        # Score
        solution = (task_order, allocations)
        try:
            fitness = RealisticFitness.fitness_function_robot_pose(
                solution, self.problem, alpha=0.5, islog=False
            )
        except Exception as e:
            self.skipped_solutions += 1
            self.skip_reasons["fitness:exception"] += 1
            self.debug(f"SKIP solution id={msg.id}: fitness exception {type(e).__name__}: {e}")
            return

        # Log fitness + paths row
        with open(self.fitness_log_file, mode='a', newline='') as f:
            csv.writer(f).writerow([timestamp, msg.id, fitness, paths_json])

        self.scored_solutions += 1
        self.debug(f"SCORED solution id={msg.id} -> fitness={fitness:.3f}")

        # Track best
        is_better = (self.best_fitness is None) or (fitness < self.best_fitness)
        if is_better:
            self.best_fitness = fitness
            self.best_solution = (solution[0][:], solution[1][:])
            with open(self.best_solution_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ROS Time (seconds)", "Agent ID", "Fitness", "Order", "Allocations"])
                writer.writerow([
                    timestamp, msg.id, fitness,
                    json.dumps(solution[0]), json.dumps(solution[1])
                ])
            self.debug(f"NEW BEST id={msg.id} fitness={fitness:.3f}")

    def environmental_representation_callback(self, msg: EnvironmentalRepresentation):
        self.last_env_update_time = time()
        timestamp = time() - self.logging_start_time
        with open(self.environmental_log_file, mode="a", newline="") as f:
            csv.writer(f).writerow([timestamp, msg.agent_id, json.dumps(list(msg.is_covered))])

        for i in range(len(msg.is_covered)):
            if msg.is_covered[i]:
                if i < len(self.is_covered):
                    self.is_covered[i] = True

        # kill logic — only if enabled
        if self.is_kill_enabled and not self.is_kill_published:
            total_tasks = len(self.is_covered)
            if total_tasks > 0:
                covered = sum(self.is_covered)
                ratio = covered / total_tasks
                if ratio >= self.kill_threshold:
                    # pick random agents to kill (once)
                    if not self.kill_pubs:
                        import random
                        n = self.expected_agents or self.num_tsp_agents
                        k = min(self.number_to_kill, n)
                        ids = random.sample(range(n), k)
                        self.kill_pubs = [
                            self.create_publisher(Bool, f"/central_control/uas_{rid}/kill_robot", 10)
                            for rid in ids
                        ]

                    # publish kill
                    for pub in self.kill_pubs:
                        for _ in range(4):
                            kill_msg = Bool()
                            kill_msg.data = True
                            pub.publish(kill_msg)
                    self.is_kill_published = True

        if self.is_revive_enabled and self.is_kill_published and not self.is_revive_published:
            total_tasks = len(self.is_covered)
            if total_tasks > 0:
                covered = sum(self.is_covered)
                ratio = covered / total_tasks
                if ratio >= self.revive_threshold:
                    if self.revive_pubs is not None:
                        for i in self.revive_pubs:
                            for _ in range(4):
                                revive_msg = Bool()
                                revive_msg.data = True
                                i.publish(revive_msg)
                        self.is_revive_published = True


    def stop_callback(self, msg: Bool):
        if msg.data:
            self.debug("Stop signal received. Terminating node.")
            self.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    @staticmethod
    def haversine(lat1, lon1, alt1, lat2, lon2, alt2):
        """
        3D Haversine distance in meters using altitude in meters.
        """
        R = 6378137.0  # Earth radius in meters

        dLat = math.radians(lat2 - lat1)
        dLon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)

        a = (math.sin(dLat / 2) ** 2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        surface = R * c

        dz = alt2 - alt1
        return math.sqrt(surface * surface + dz * dz)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run realistic fitness logger")
    parser.add_argument('--enable_kill', action='store_true',
                        help='Enable publishing a kill message once coverage reaches threshold')
    parser.add_argument('--kill_threshold', type=float, default=0.20,
                        help='Coverage fraction (0-1) at which to publish kill (default 0.20)')
    parser.add_argument('--num_to_kill', type=int, default=1,
                        help='Number of robots to kill if kill enabled')
    parser.add_argument('--enable_revive', action='store_true',
                        help='Enable publishing a revival message once coverage reaches a threshold')
    parser.add_argument('--revive_threshold', type=float, default=0.80,
                        help='Coverage fraction (0-1) at which to publish revive (default 0.80)')

    # IMPORTANT: don't let argparse choke on ROS2 args
    if argv is None:
        argv = sys.argv[1:]
    parsed_args, ros_args = parser.parse_known_args(argv)

    # give ROS2 ONLY the leftover args (including --ros-args -p ...)
    rclpy.init(args=ros_args)

    fitness_logger = RealisticFitnessLogger(args=parsed_args)
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(fitness_logger)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        fitness_logger.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

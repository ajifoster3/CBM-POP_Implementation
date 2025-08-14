import csv
import math
from datetime import datetime
import os
from time import time
import json  # <-- added

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup

from cbm_pop.Fitness import Fitness
from cbm_pop_interfaces.msg import (
    Solution,
    EnvironmentalRepresentation,
    SimplePosition,
    FinishedCoverage,
    CumulativeReward,
    CurrentTask,
)
from std_msgs.msg import Bool


class SimpleFitnessLogger(Node):
    def __init__(self):
        super().__init__('fitness_logger')
        self.logging_start_time = time()

        self.timeout = 10
        self.last_env_update_time = None
        self.termination_timeout = 20.0

        self.declare_parameter('timeout', 0.0)
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        # Get directory name from parameter
        self.declare_parameter('parent_log_dir', 'resources/run_logs')
        parent_log_dir = self.get_parameter('parent_log_dir').get_parameter_value().string_value

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_folder = os.path.join(parent_log_dir, timestamp)
        os.makedirs(self.run_folder, exist_ok=True)

        self.num_tsp_agents = 10
        self.current_tasks = [-1] * self.num_tsp_agents
        self.is_all_poses_set = False

        self.position_log_file = os.path.join(self.run_folder, "robot_positions.csv")
        self.cumulative_log_file = os.path.join(self.run_folder, "cumulative_reward.csv")
        self.fitness_log_file = os.path.join(self.run_folder, "fitness_log_file.csv")

        # new: file that always stores ONLY the current best solution
        self.best_solution_file = os.path.join(self.run_folder, "current_best_solution.csv")

        self.task_poses = [(i + 0.5, j + 0.5) for i in range(10) for j in range(10)]
        self.cost_matrix = self.calculate_cost_matrix()
        self.robot_inital_pose_cost_matrix = [None] * self.num_tsp_agents

        # Positions CSV header
        with open(self.position_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Agent ID", "Latitude", "Longitude", "Altitude"])

        # Fitness CSV header (timestamp, agent id, fitness)
        with open(self.fitness_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "ROS Time (seconds)",
                "Agent ID",
                "Fitness"
            ])

        # Best-solution CSV header (single-row file that gets overwritten on improvements)
        with open(self.best_solution_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "ROS Time (seconds)",
                "Agent ID",
                "Fitness",
                "Order",
                "Allocations"
            ])

        self.clock_time = None
        self.is_covered = None
        self.persistent_environmental_representation = None
        self.environmental_representation = None
        self.cumulative_reward_subscriber = None

        self.stop_subscriber = self.create_subscription(Bool, 'stop_plotting', self.stop_callback, 10)
        self.best_fitness = None
        self.best_solution = None  # <-- track best solution tuple (order, allocations)
        self.stop_flag = False

        self.initial_robot_poses = [None] * self.num_tsp_agents
        self.robot_poses = [None] * self.num_tsp_agents
        self.finished_robots = [False] * self.num_tsp_agents

        cb_group = ReentrantCallbackGroup()
        self.global_pose_subscribers = []

        for agent_id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{agent_id}/global_pose'
            sub = self.create_subscription(
                SimplePosition,
                topic,
                lambda msg, agent=agent_id: self.global_pose_callback(msg, agent),
                10,
                callback_group=cb_group
            )
            self.global_pose_subscribers.append(sub)

        self.finished_coverage_sub = self.create_subscription(
            FinishedCoverage,
            f'/central_control/finished_coverage',
            self.finished_coverage_callback,
            10
        )

        self.cumulative_reward_subscriber = self.create_subscription(
            CumulativeReward,
            '/cumulative_reward',
            self.cumulative_reward_callback,
            10
        )

        self.current_task_subscriber = self.create_subscription(
            CurrentTask,
            'current_task',
            self.current_task_update_callback,
            10
        )

    def calculate_cost_matrix(self):
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    x1, y1 = self.task_poses[i]
                    x2, y2 = self.task_poses[j]
                    cost = math.hypot(x2 - x1, y2 - y1)
                    cost_map[i][j] = cost
        return cost_map

    def calculate_robot_inital_pose_cost_matrix(self):
        valid_robot_poses = self.initial_robot_poses
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_robots, num_tasks))

        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                x1, y1 = robot_pose
                x2, y2 = task_pose
                cost_map[i][j] = math.hypot(x2 - x1, y2 - y1)

        self.robot_inital_pose_cost_matrix = cost_map

    def check_timeout(self):
        if self.last_env_update_time is None:
            self.last_env_update_time = time()

        elapsed_time = time() - self.last_env_update_time
        if elapsed_time > self.termination_timeout:
            self.get_logger().warn(
                f"No EnvironmentalRepresentation received for {elapsed_time:.2f} seconds. Shutting down...")
            self.destroy_node()
            rclpy.shutdown()

    def finished_coverage_callback(self, msg):
        self.finished_robots[int(msg.robot_id)] = bool(msg.finished)
        if all(self.finished_robots):
            print("Coverage Complete")
            self.destroy_node()
            rclpy.shutdown()
            print("ROS2 system shut down.")

    def cumulative_reward_callback(self, msg):
        self.log_cumulative_reward((msg.agent_id, msg.cumulative_reward))

    def log_cumulative_reward(self, cumulative_reward):
        with open(self.cumulative_log_file, mode='a', newline='') as file:
            timestamp = time() - self.logging_start_time
            writer = csv.writer(file)
            # Columns: agent_id, cumulative_reward, timestamp
            writer.writerow([cumulative_reward[0], cumulative_reward[1], timestamp])

    def global_pose_callback(self, msg, agent):
        if self.initial_robot_poses[agent] is None:
            self.initial_robot_poses[agent] = (msg.x_position, msg.y_position)

        self.robot_poses[agent] = (msg.x_position, msg.y_position)

        if all(p is not None for p in self.robot_poses) and not self.is_all_poses_set:
            self.is_all_poses_set = True
            # Build initial pose cost matrix
            self.calculate_robot_inital_pose_cost_matrix()
            self.solution_subscriber = self.create_subscription(
                Solution, 'best_solution', self.solution_update_callback, 10)

        timestamp = time() - self.logging_start_time
        with open(self.position_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, agent,
                msg.x_position,
                msg.y_position,
                ""  # altitude not provided by SimplePosition
            ])

    def current_task_update_callback(self, msg):
        self.current_tasks[msg.agent_id] = msg.current_task

    def solution_update_callback(self, msg):
        if msg and all(x is not None for x in self.current_tasks):
            timestamp = time() - self.logging_start_time
            solution = (msg.order, msg.allocations)
            fitness = Fitness.fitness_function_robot_pose(
                solution,
                self.cost_matrix,
                self.robot_inital_pose_cost_matrix,
                self.robot_inital_pose_cost_matrix
            )
            # Log: time, agent id (from Solution.id), fitness
            with open(self.fitness_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, msg.id, fitness])

            # --- Track and persist ONLY the current best solution (overwrite file on improvement) ---
            is_better = (self.best_fitness is None) or (fitness < self.best_fitness)  # assume minimization
            if is_better:
                self.best_fitness = fitness
                self.best_solution = (list(msg.order), list(msg.allocations))
                with open(self.best_solution_file, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["ROS Time (seconds)", "Agent ID", "Fitness", "Order", "Allocations"])
                    writer.writerow([
                        timestamp,
                        msg.id,
                        fitness,
                        json.dumps(list(msg.order)),
                        json.dumps(list(msg.allocations))
                    ])
        else:
            self.get_logger().warning("Received empty message or incomplete task data")

    def stop_callback(self, msg):
        if msg.data:
            self.get_logger().info("Stop signal received. Terminating node.")
            self.destroy_node()
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    fitness_logger = SimpleFitnessLogger()
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

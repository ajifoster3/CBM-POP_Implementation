import csv
import math
from datetime import datetime
from time import time
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.timer import Timer
from cbm_pop.Fitness import Fitness
from cbm_pop.fitness_logger import FitnessLogger
from cbm_pop_interfaces.msg import Solution
from std_msgs.msg import Bool
from cbm_pop_interfaces.msg import EnvironmentalRepresentation, SimplePosition, FinishedCoverage, CumulativeReward, CurrentTask
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from time import time
import numpy as np
import os  # Add at top if not already imported

class SimpleFitnessLogger(Node):
    def __init__(self):
        super().__init__('fitness_logger')
        self.logging_start_time = time()

        self.timeout = 10
        self.last_env_update_time = None  # Track last received EnvironmentalRepresentation time
        self.termination_timeout = 20.0

        self.declare_parameter('timeout', 0.0)
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        self.num_tsp_agents = 10
        self.current_tasks = [-1] * self.num_tsp_agents
        self.is_all_poses_set = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.run_folder = f"resources/run_logs/{timestamp}"
        os.makedirs(self.run_folder, exist_ok=True)

        self.position_log_file = os.path.join(self.run_folder, "robot_positions.csv")
        self.cumulative_log_file = os.path.join(self.run_folder, "cumulative_reward.csv")
        self.fitness_log_file = os.path.join(self.run_folder, "fitness_log_file.csv")

        self.task_poses = [(i + 0.5, j + 0.5) for i in range(10) for j in range(10)]
        self.cost_matrix = self.calculate_cost_matrix()
        self.robot_inital_pose_cost_matrix = [None] * self.num_tsp_agents

        with open(self.position_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Agent ID", "Latitude", "Longitude", "Altitude"])

        self.clock_time = None
        self.is_covered = None
        self.persistent_environmental_representation = None
        self.environmental_representation = None
        self.cumulative_reward_subscriber = None

        self.stop_subscriber = self.create_subscription(Bool, 'stop_plotting', self.stop_callback, 10)
        self.best_fitness = None
        self.stop_flag = False

        # Runtime data
        self.initial_robot_poses = [None] * (self.num_tsp_agents)
        self.robot_poses = [None] * (self.num_tsp_agents)
        self.finished_robots = [False] * self.num_tsp_agents

        cb_group = ReentrantCallbackGroup()

        self.global_pose_subscribers = []
        for agent_id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{agent_id}/global_pose'

            # Wrap the callback to include agent_id
            sub = self.create_subscription(
                SimplePosition,
                topic,
                lambda msg, agent=agent_id: self.global_pose_callback(msg, agent),
                10,
                callback_group=cb_group
            )

            self.global_pose_subscribers.append(sub)  # Store subscription to prevent garbage collection

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
            CurrentTask, 'current_task', self.current_task_update_callback, 10)

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
                    # Extract positions
                    x1, y1 = self.task_poses[i][0], self.task_poses[i][1]
                    x2, y2 = self.task_poses[j][0], self.task_poses[j][1]

                    # Compute horizontal and vertical distances
                    x_dist = abs(x2 - x1)
                    y_dist = abs(y2 - y1)
                    # Compute cost using trajectory generation function
                    cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))

                    cost_map[i][j] = cost
        return cost_map

    def calculate_robot_inital_pose_cost_matrix(self):
        """
        Returns a cost map representing the traversal cost from each initial_robot_pose to each each task_pose calculated
        using drone_distance.
        """

        # Filter out None values from robot_poses

        valid_robot_poses = self.initial_robot_poses
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        # Initialize cost matrix
        cost_map = np.zeros((num_robots, num_tasks))

        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                # Extract positions
                x1, y1 = robot_pose
                x2, y2 = task_pose

                # Compute horizontal and vertical distances
                x_dist = x2 - x1
                y_dist = y2 - y1
                # Compute cost using trajectory generation function
                cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))

                cost_map[i][j] = cost
        self.robot_inital_pose_cost_matrix = cost_map

    def check_timeout(self):
        """Check if 10 seconds have passed without receiving an EnvironmentalRepresentation message."""
        if self.last_env_update_time is None:
            self.last_env_update_time = time()  # Initialize on first check

        elapsed_time = time() - self.last_env_update_time

        if elapsed_time > self.termination_timeout:
            self.get_logger().warn(
                f"No EnvironmentalRepresentation received for {elapsed_time:.2f} seconds. Shutting down...")
            self.destroy_node()
            rclpy.shutdown()

    def finished_coverage_callback(self, msg):
        print(f"finished robot ID: {msg.robot_id}")
        self.finished_robots[int(msg.robot_id)] = bool(msg.finished)
        if all(self.finished_robots):
            print("Coverage Complete")
            self.destroy_node()  # Stop this node

            # Shutdown ROS2 system
            rclpy.shutdown()  # This ensures that ROS2 itself is properly shut down
            print("ROS2 system shut down.")

    def cumulative_reward_callback(self, msg):
        cumulative_reward = msg.agent_id, msg.cumulative_reward
        self.log_cumulative_reward(cumulative_reward)

    def log_cumulative_reward(self, cumulative_reward):
        with open(self.cumulative_log_file, mode='a', newline='') as file:
            timestamp = time() - self.logging_start_time
            writer = csv.writer(file)
            writer.writerow([cumulative_reward[0], cumulative_reward[1], timestamp])

    def global_pose_callback(self, msg, agent):
        # Store the first received pose for each agent
        if self.initial_robot_poses[agent] is None:
            self.initial_robot_poses[agent] = (msg.x_position, msg.y_position)

        if all(pose is not None for pose in self.robot_poses) and self.is_all_poses_set is False:
            self.is_all_poses_set = True
            self.initial_robot_pose_cost_matrix = self.calculate_robot_inital_pose_cost_matrix()
            self.solution_subscriber = self.create_subscription(Solution, 'best_solution',
                                                                self.solution_update_callback,
                                                                10)

        self.robot_poses[agent] = (msg.x_position, msg.y_position)

        timestamp = time() - self.logging_start_time
        with open(self.position_log_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                timestamp, agent,
                msg.x_position,
                msg.y_position
            ])

    def clock_callback(self, msg):
        self.clock_time = msg.clock.sec + msg.clock.nanosec / 1e9


    def current_task_update_callback(self, msg):
        self.current_tasks[msg.agent_id] = msg.current_task

    def solution_update_callback(self, msg):
        if msg and all(x is not None for x in self.current_tasks) is not None:
            timestamp = time() - self.logging_start_time
            solution = (msg.order, msg.allocations)
            fitness = Fitness.fitness_function_locked_tasks(solution, self.cost_matrix, self.current_tasks, self.robot_inital_pose_cost_matrix)
            with open(self.fitness_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, fitness])
        else:
            self.get_logger().warning("Received empty message or no clock time available")

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

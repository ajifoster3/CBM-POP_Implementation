import csv
import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from time import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.timer import Timer
import matplotlib.pyplot as plt
from cbm_pop.Fitness import Fitness
from cbm_pop.Problem import Problem
from cbm_pop.path_cost_calculator import calculate_drone_distance
from cbm_pop_interfaces.msg import Solution
from std_msgs.msg import Bool
from rosgraph_msgs.msg import Clock
from cbm_pop_interfaces.msg import EnvironmentalRepresentation
from math import radians, cos, sin, asin, sqrt
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geographic_msgs.msg import GeoPoseStamped
from pygeodesy.geoids import GeoidPGM


class FitnessLogger(Node):
    def __init__(self):
        super().__init__('fitness_logger')

        self.timeout = 10
        self.last_env_update_time = None  # Track last received EnvironmentalRepresentation time
        self.termination_timeout = 20.0

        self.geoid = GeoidPGM('/home/ajifoster3/Documents/Software/ros2_ws/src/CBM-POP_Implementation/egm96-5.pgm')
        self.declare_parameter('timeout', 0.0)
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        self.num_tsp_agents = 5
        filename = "/home/ajifoster3/Downloads/Poses/random120/3.json"
        with open(filename, "r") as file:
            print(f"using {filename}")
            data = json.load(file)
        self.task_poses = [
            {
                "latitude": item["GeoPose"]["position"]["latitude"],
                "longitude": item["GeoPose"]["position"]["longitude"],
                "altitude": item["GeoPose"]["position"]["altitude"]
            }
            for item in data
        ]

        num_tasks = len(self.task_poses)
        self.cost_matrix = np.zeros((num_tasks, num_tasks))
        self.robot_cost_matrix = [None] * self.num_tsp_agents
        self.robot_inital_pose_cost_matrix = [None] * self.num_tsp_agents
        self.is_all_poses_set = False
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    lat1, lon1, alt1 = self.task_poses[i].values()
                    lat2, lon2, alt2 = self.task_poses[j].values()

                    total_horizontal_distance = self.haversine(lat1, lon1, 0, lat2, lon2, 0)
                    total_vertical_distance = alt2 - alt1
                    cost = calculate_drone_distance(total_horizontal_distance, total_vertical_distance, 3, 1.45, 11)
                    self.cost_matrix[i][j] = cost

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.position_log_file = f"resources/run_logs/robot_positions_{timestamp}.csv"
        self.log_file = f"resources/run_logs/fitness_logs_{timestamp}.csv"
        self.env_log_file = f"resources/run_logs/environmental_logs_{timestamp}.csv"

        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Fitness_Value"])
        with open(self.env_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Environmental_Representation"])
        with open(self.position_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Agent ID", "Latitude", "Longitude", "Altitude"])



        self.clock_time = None
        self.is_covered = None
        self.persistent_environmental_representation = None
        self.environmental_representation = None
        self.solution_subscriber = None
        self.create_subscription(Clock, '/clock', self.clock_callback, 10)
        self.create_subscription(EnvironmentalRepresentation, '/environmental_representation',
                                 self.environmental_representation_callback, 10)

        self.stop_subscriber = self.create_subscription(Bool, 'stop_plotting', self.stop_callback, 10)
        self.best_fitness = None
        self.stop_flag = False

        killed_agent = random.randint(1, 5)
        self.topic = f'/central_control/uas_{killed_agent}/kill_robot'

        # Create a publisher for the kill command
        self.publisher = self.create_publisher(Bool, self.topic, 10)


        # Runtime data
        self.initial_robot_poses = [None] * (self.num_tsp_agents)
        self.robot_poses = [None] * (self.num_tsp_agents)

        cb_group = ReentrantCallbackGroup()

        self.run_cost_matrix_recalculation = self.create_timer(15, self.robot_cost_matrix_recalculation,
                                                               callback_group=cb_group)

        self.timeout_checker = self.create_timer(1.0, self.check_timeout, callback_group=cb_group)  # Runs every second

        self.global_pose_subscribers = []
        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id + 1}/global_pose'

            # Wrap the callback to include agent_id
            sub = self.create_subscription(
                GeoPoseStamped,
                topic,
                lambda msg, agent=(id + 1): self.global_pose_callback(msg, agent),
                10,
                callback_group=cb_group
            )

            self.global_pose_subscribers.append(sub)  # Store subscription to prevent garbage collection

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

    def global_pose_callback(self, msg, agent):
        # Store the first received pose for each agent
        if self.initial_robot_poses[agent - 1] is None:
            self.initial_robot_poses[agent - 1] = deepcopy(msg.pose)

        if all(pose is not None for pose in self.robot_poses) and self.is_all_poses_set is False:
            self.is_all_poses_set = True
            self.robot_cost_matrix_recalculation()
            self.solution_subscriber = self.create_subscription(Solution, 'best_solution',
                                                                self.solution_update_callback,
                                                                10)

        self.robot_poses[agent - 1] = deepcopy(msg.pose)

        # Log positions to the uniquely named file
        if self.clock_time is not None:
            with open(self.position_log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    self.clock_time, agent,
                    msg.pose.position.latitude,
                    msg.pose.position.longitude,
                    msg.pose.position.altitude
                ])

    def robot_cost_matrix_recalculation(self):
        if all(pose is not None for pose in self.robot_poses):
            if self.solution_subscriber is None:
                self.solution_subscriber = self.create_subscription(Solution, 'best_solution',
                                                                    self.solution_update_callback,
                                                                    10)
            self.calculate_robot_cost_matrix()
            self.calculate_robot_inital_pose_cost_matrix()

    def calculate_robot_cost_matrix(self):
        # Filter out None values from robot_poses

        valid_robot_poses = [pose for pose in self.robot_poses if pose is not None]
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        # Initialize cost matrix
        cost_map = np.zeros((num_robots, num_tasks))

        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                # Extract positions
                geoid_height = self.geoid.height(robot_pose.position.latitude, robot_pose.position.longitude)
                lat1, lon1, alt1 = robot_pose.position.latitude, robot_pose.position.longitude, robot_pose.position.altitude - geoid_height
                lat2, lon2, alt2 = task_pose["latitude"], task_pose["longitude"], task_pose["altitude"]

                # Compute horizontal and vertical distances
                total_horizontal_distance = self.haversine(lat1, lon1, 0, lat2, lon2, 0)
                total_vertical_distance = alt2 - alt1

                # Compute cost using trajectory generation function
                cost = calculate_drone_distance(
                    total_horizontal_distance, total_vertical_distance, 3, 1.45, 11)

                cost_map[i][j] = cost
        self.robot_cost_matrix = cost_map

    def calculate_robot_inital_pose_cost_matrix(self):
        # Filter out None values from robot_poses

        valid_robot_poses = self.initial_robot_poses
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        # Initialize cost matrix
        cost_map = np.zeros((num_robots, num_tasks))

        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                # Extract positions
                lat1, lon1, alt1 = robot_pose.position.latitude, robot_pose.position.longitude, robot_pose.position.altitude
                lat2, lon2, alt2 = task_pose["latitude"], task_pose["longitude"], task_pose["altitude"]

                # Compute horizontal and vertical distances
                total_horizontal_distance = self.haversine(lat2, lon2, 0, lat1, lon1, 0)
                total_vertical_distance = alt1 - alt2

                # Compute cost using trajectory generation function
                cost = calculate_drone_distance(
                    total_horizontal_distance, total_vertical_distance, 3, 1.45, 11)

                cost_map[i][j] = cost
        self.robot_inital_pose_cost_matrix = cost_map

    def publish_kill_signal(self):
        msg = Bool()
        msg.data = True
        self.publisher.publish(msg)
        print(f'Published kill signal to {self.topic}')

        # Destroy the timer after execution
        self.timer.cancel()

    def clock_callback(self, msg):
        self.clock_time = msg.clock.sec + msg.clock.nanosec / 1e9

    def environmental_representation_callback(self, msg):
        self.last_env_update_time = time()  # Update last received time

        new_representation = msg.is_covered
        if self.persistent_environmental_representation is None:
            self.persistent_environmental_representation = new_representation
            changed = True
        else:
            updated_representation = [prev or current for prev, current in
                                      zip(self.persistent_environmental_representation, new_representation)]
            changed = updated_representation != self.persistent_environmental_representation
            self.persistent_environmental_representation = updated_representation

        if changed and self.clock_time is not None:
            with open(self.env_log_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.clock_time, str(self.persistent_environmental_representation)])

    def solution_update_callback(self, msg):
        if msg and self.clock_time is not None:
            solution = (msg.order, msg.allocations)
            fitness = Fitness.fitness_function_robot_pose(solution, self.cost_matrix, self.robot_cost_matrix,
                                                          self.robot_inital_pose_cost_matrix)
            if self.best_fitness is None or fitness < self.best_fitness:
                self.best_fitness = fitness
                self.get_logger().info(f"ROS Time: {self.clock_time:.2f}s, Fitness: {fitness}")
                self.log_fitness(self.clock_time, fitness)
        else:
            self.get_logger().warning("Received empty message or no clock time available")

    def log_fitness(self, ros_time, fitness):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ros_time, fitness])

    def stop_callback(self, msg):
        if msg.data:
            self.get_logger().info("Stop signal received. Terminating node.")
            self.destroy_node()
            rclpy.shutdown()

    def haversine(self, lat1, lon1, alt1, lat2, lon2, alt2):
        R = 6378160
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)
        a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
        c = 2 * asin(sqrt(a))
        haversine_distance = R * c
        alt_diff = alt2 - alt1
        return sqrt(haversine_distance ** 2 + alt_diff ** 2)


def main(args=None):
    rclpy.init(args=args)

    fitness_logger = FitnessLogger()

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

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


class FitnessLoggerOffline(Node):
    def __init__(self):
        super().__init__('fitness_logger')

        self.start_system_time = time()

        self.timeout = 10
        self.termination_timeout = 60.0

        self.geoid = GeoidPGM('/home/ajifoster3/Documents/Software/ros_ws/src/CBM-POP_Implementation/egm96-5.pgm')
        self.declare_parameter('timeout', 0.0)
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        self.num_tsp_agents = 5
        filename = "/home/ajifoster3/Downloads/Poses/generated_geoposes_less.json"
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


        filename = "/home/ajifoster3/Downloads/Poses/RobotPoses/random_geoposes_4.json"
        with open(filename, "r") as file:
            print(f"using {filename}")
            robot_data = json.load(file)
        self.robot_poses = [
            {
                "latitude": item["GeoPose"]["position"]["latitude"],
                "longitude": item["GeoPose"]["position"]["longitude"],
                "altitude": item["GeoPose"]["position"]["altitude"],
                "orientation_x": item["GeoPose"]["orientation"]["x"],
                "orientation_y": item["GeoPose"]["orientation"]["y"],
                "orientation_z": item["GeoPose"]["orientation"]["z"],
                "orientation_w": item["GeoPose"]["orientation"]["w"]
            }
            for item in robot_data
        ]
        self.initial_robot_poses = self.robot_poses

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

        self.solution_subscriber = None
        self.robot_cost_matrix_recalculation()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"resources/run_logs/fitness_logs_{timestamp}.csv"

        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Fitness_Value"])



        self.clock_time = None
        self.is_covered = None
        self.persistent_environmental_representation = None
        self.environmental_representation = None
        self.create_subscription(Clock, '/clock', self.clock_callback, 10)

        self.stop_subscriber = self.create_subscription(Bool, 'stop_plotting', self.stop_callback, 10)
        self.best_fitness = None
        self.stop_flag = False

        cb_group = ReentrantCallbackGroup()

        self.timeout_checker = self.create_timer(1.0, self.check_timeout, callback_group=cb_group)  # Runs every second

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
                geoid_height = self.geoid.height(robot_pose["latitude"], robot_pose["longitude"])
                lat1, lon1, alt1 = robot_pose["latitude"], robot_pose["longitude"], robot_pose["altitude"] - geoid_height
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
                lat1, lon1, alt1 = robot_pose["latitude"], robot_pose["longitude"], robot_pose["altitude"]
                lat2, lon2, alt2 = task_pose["latitude"], task_pose["longitude"], task_pose["altitude"]

                # Compute horizontal and vertical distances
                total_horizontal_distance = self.haversine(lat2, lon2, 0, lat1, lon1, 0)
                total_vertical_distance = alt1 - alt2

                # Compute cost using trajectory generation function
                cost = calculate_drone_distance(
                    total_horizontal_distance, total_vertical_distance, 3, 1.45, 11)

                cost_map[i][j] = cost
        self.robot_inital_pose_cost_matrix = cost_map

    def clock_callback(self, msg):
        self.clock_time = msg.clock.sec + msg.clock.nanosec / 1e9


    def solution_update_callback(self, msg):
        # Get the current system time
        elapsed_system_time = time() - self.start_system_time  # This gets the time in seconds since the epoch

        # You no longer need to check 'self.clock_time is not None'
        # and 'msg' should always be a valid message object if the callback is triggered.
        solution = (msg.order, msg.allocations)
        fitness = Fitness.fitness_function_robot_pose(
            solution,
            self.cost_matrix,
            self.robot_cost_matrix,
            self.robot_inital_pose_cost_matrix  # Ensure this parameter is correctly handled in your Fitness class
        )

        if self.best_fitness is None or fitness < self.best_fitness:
            self.best_fitness = fitness
            # Log system time instead of ROS time
            self.get_logger().info(f"System Time: {elapsed_system_time:.2f}s, Fitness: {fitness}")
            self.log_fitness(elapsed_system_time, fitness)

    def log_fitness(self, ros_time, fitness):
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ros_time, fitness])

    def stop_callback(self, msg):
        if msg.data:
            self.get_logger().info("Stop signal received. Terminating node.")
            self.destroy_node()
            rclpy.shutdown()

    def check_timeout(self):
        if time() > self.start_system_time + self.timeout+5:
            self.get_logger().warn(
                f"Shutting down logger...")
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
    print("Launching offline fitness logger")
    fitness_logger = FitnessLoggerOffline()

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

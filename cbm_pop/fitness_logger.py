import csv
import json
import sys
from datetime import datetime

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


class FitnessLogger(Node):
    def __init__(self):
        super().__init__('fitness_logger')

        self.declare_parameter('timeout', 0.0)
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        num_tasks = 150

        with open("/home/ajifoster3/Downloads/all_geoposes_wind_turbine.json", "r") as file:
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
        self.log_file = f"resources/run_logs/fitness_logs_{timestamp}.csv"
        self.env_log_file = f"resources/run_logs/environmental_logs_{timestamp}.csv"
        self.coverage_log_file = f"resources/run_logs/coverage_logs_{timestamp}.csv"

        with open(self.log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Fitness_Value"])
        with open(self.env_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Environmental_Representation"])
        with open(self.coverage_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Is_Covered"])

        self.clock_time = None
        self.is_covered = None
        self.persistent_environmental_representation = None
        self.environmental_representation = None

        self.create_subscription(Clock, '/clock', self.clock_callback, 10)
        self.create_subscription(EnvironmentalRepresentation, '/environmental_representation',
                                 self.environmental_representation_callback, 10)
        self.solution_subscriber = self.create_subscription(Solution, 'best_solution', self.solution_update_callback,
                                                            10)
        self.stop_subscriber = self.create_subscription(Bool, 'stop_plotting', self.stop_callback, 10)
        self.best_fitness = None
        self.stop_flag = False

        self.topic = f'/central_control/uas_2/kill_robot'

        # Create a publisher for the kill command
        self.publisher = self.create_publisher(Bool, self.topic, 10)

        # Create a timer that will trigger the kill command after 50 seconds
        # self.timer = self.create_timer(20.0, self.publish_kill_signal)

        self.initial_robot_poses = [None] * (5)
        self.robot_poses = [None] * (5)

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
            fitness = Fitness.fitness_function(solution, self.cost_matrix)
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
    rclpy.spin(fitness_logger)
    fitness_logger.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

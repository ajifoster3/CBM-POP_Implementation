import csv
import sys
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.timer import Timer
import matplotlib.pyplot as plt
from cbm_pop.Fitness import Fitness
from cbm_pop.Problem import Problem
from cbm_pop_interfaces.msg import Solution
from std_msgs.msg import Bool


class FitnessLogger(Node):
    def __init__(self):
        super().__init__('fitness_logger')

        self.declare_parameter('timeout', 0.0)  # Default is 0.0, meaning no timeout
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        num_tasks = 150
        problem = Problem()
        problem.load_cost_matrix("resources/150_Task_Problem.csv", "csv")
        self.cost_matrix = problem.cost_matrix

        # Store logs with timestamps
        self.iteration_logs = []
        self.start_time = self.get_clock().now()  # Record start time

        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10)

        self.best_fitness = None

        self.stop_subscriber = self.create_subscription(
            Bool, 'stop_plotting', self.stop_callback, 10)

        self.stop_flag = False  # Flag to track when to stop

        if self.timeout > 0:
            self.get_logger().info(f"Timeout set to {self.timeout} seconds. Node will stop automatically.")
            self.timer = self.create_timer(self.timeout, self.stop_due_to_timeout)

    def solution_update_callback(self, msg):
        # Callback to process incoming solution updates
        if msg is not None:
            solution = (msg.order, msg.allocations)
            fitness = Fitness.fitness_function(solution, self.cost_matrix)
            if self.best_fitness is None or fitness < self.best_fitness:
                self.best_fitness = fitness
                elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds / 1e9  # Time in seconds
                self.get_logger().info(f"Time: {elapsed_time:.2f}s, Fitness: {fitness}")
                self.iteration_logs.append((elapsed_time, fitness))
        else:
            self.get_logger().warning("Received empty message")

    def stop_callback(self, msg):
        if msg.data:
            self.stop_flag = True
            self.get_logger().info("Stop signal received. Plotting fitness values...")
            self.plot_fitness()
            self.destroy_node()
            rclpy.shutdown()

    def stop_due_to_timeout(self):
        if not self.stop_flag:
            self.get_logger().info("Timeout reached. Plotting fitness values...")
            self.plot_fitness()
            self.destroy_node()
            rclpy.shutdown()

    def plot_fitness(self):
        # Plot fitness values over time
        if len(self.iteration_logs) > 0:
            times = [log[0] for log in self.iteration_logs]
            fitness_values = [log[1] for log in self.iteration_logs]

            # Plot the fitness values
            plt.figure()
            plt.plot(times, fitness_values, marker='o', linestyle='-')
            plt.xlabel("Time (seconds)")
            plt.ylabel("Fitness Value")
            plt.title("Fitness Values Over Time")
            plt.grid(True)
            plt.show()

            # Generate a unique filename with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = "resources/run_logs/"
            file_name = f"{directory}fitness_logs_{timestamp}.csv"
            try:
                with open(file_name, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Write the header
                    writer.writerow(["Time(seconds)", "Fitness_Value"])
                    # Write the data
                    writer.writerows(self.iteration_logs)
                self.get_logger().info(f"Fitness values logged to {file_name}")
            except Exception as e:
                self.get_logger().error(f"Failed to log fitness values: {e}")
        else:
            self.get_logger().info("No fitness values to plot or log.")

def main(args=None):
    try:
        rclpy.init(args=args)
    except Exception as e:
        print(f"Failed to initialize ROS 2: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        fitness_logger = FitnessLogger()
    except Exception as e:
        print(f"Failed to initialize FitnessLogger: {e}", file=sys.stderr)
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)

    try:
        rclpy.spin(fitness_logger)
    except KeyboardInterrupt:
        pass
    finally:
        fitness_logger.destroy_node()
        if rclpy.ok():  # Prevent double shutdown
            rclpy.shutdown()


if __name__ == '__main__':
    main()

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
        problem.load_cost_matrix("150_Task_Problem.csv", "csv")
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

            plt.figure()
            plt.plot(times, fitness_values, marker='o', linestyle='-')
            plt.xlabel("Time (seconds)")
            plt.ylabel("Fitness Value")
            plt.title("Fitness Values Over Time")
            plt.grid(True)
            plt.show()
        else:
            self.get_logger().info("No fitness values to plot.")


def main(args=None):
    rclpy.init(args=args)
    fitness_logger = FitnessLogger()

    try:
        rclpy.spin(fitness_logger)
    except KeyboardInterrupt:
        pass
    finally:
        fitness_logger.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

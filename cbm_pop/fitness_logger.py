import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re

from cbm_pop.Fitness import Fitness
from cbm_pop.Problem import Problem
from cbm_pop_interfaces.msg import Solution
import datetime

class FitnessLogger(Node):
    def __init__(self):
        super().__init__('fitness_logger')

        num_tasks = 150
        problem = Problem()
        problem.load_cost_matrix("src/cbm_pop/150_Task_Problem.csv", "csv")
        self.cost_matrix = problem.cost_matrix


        # Fitness değerlerini depolayacağız
        self.iteration_logs = []

        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10)

    def log_handler(self, msg):
        # Log mesajını alıyoruz
        if 'Current best solution fitness_QL' in msg:
            # Fitness değerini regex ile çekiyoruz
            match = re.search(r"fitness_QL = (\d+\.\d+)", msg)
            if match:
                fitness_value = float(match.group(1))
                # Fitness değerini kaydet
                self.iteration_logs.append((len(self.iteration_logs) + 1, fitness_value))

    def solution_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        if msg is not None:
            solution = (msg.order, msg.allocations)
            fitness = Fitness.fitness_function(solution, self.cost_matrix)
            print(fitness)
            self.iteration_logs.append((len(self.iteration_logs) + 1, fitness))
        else:
            print("Received empty message")

    def update_plot(self, frame):
        # Verileri güncelle
        if len(self.iteration_logs) > 0:
            iterations = [log[0] for log in self.iteration_logs]
            fitness_values = [log[1] for log in self.iteration_logs]
            self.line.set_data(iterations, fitness_values)
            self.ax.relim()
            self.ax.autoscale_view()
        return self.line,


def main(args=None):
    rclpy.init(args=args)
    fitness_logger = FitnessLogger()

    # ROS 2 spin (logları dinlemek için)
    rclpy.spin(fitness_logger)

    fitness_logger.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

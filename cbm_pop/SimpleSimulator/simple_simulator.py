import sys
import time
import threading
import argparse
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation import LearningMethod
from cbm_pop.SimpleSimulator.simulator_robot import SimulatorRobot as robot
from cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation import CBMPopulationAgentOnlineSimpleSimulation as agent
from cbm_pop.SimpleSimulator.simple_fitness_logger import SimpleFitnessLogger
from cbm_pop_interfaces.msg import EnvironmentalRepresentation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


class SimpleSimulator:
    def __init__(self, tasks, environmental_bounds, robot_starting_positions, obstacles, args):
        self.simulation_done = False
        self.environmental_bounds = environmental_bounds
        self.tasks = tasks
        self.robot_starting_positions = robot_starting_positions
        self.obstacles = obstacles
        self.robots = []
        self.lock = threading.Lock()
        self.finished_robots = [False] * args.num_robots
        self.is_covered = [False] * len(self.tasks)
        self.task_scat = None

        for i in range(len(robot_starting_positions)):
            self.robots.append(robot(i, robot_starting_positions[i], args.speed, args.num_robots))

    def start_simulation_thread(self, on_complete=None):
        def simulation_loop():
            while not all(robot.is_finished for robot in self.robots):
                with self.lock:
                    for robot in self.robots:
                        robot.move_robot()
                time.sleep(0.05)
            print("Simulation complete.")
            if on_complete:
                on_complete()

        threading.Thread(target=simulation_loop, daemon=True).start()

    def start_animation(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(self.environmental_bounds[0], self.environmental_bounds[1])
        self.ax.set_ylim(self.environmental_bounds[2], self.environmental_bounds[3])
        self.ax.set_title("Live Robot Positions")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        task_xs, task_ys = zip(*self.tasks)
        self.task_scat = self.ax.scatter(task_xs, task_ys, s=10, label='Tasks')
        self.robot_scatter = self.ax.scatter([], [], c='#80B3FF', label='Robots')
        self.robot_labels = [self.ax.text(0, 0, str(i), fontsize=8, color='black', ha='center')
                             for i in range(len(self.robots))]

        def update(frame):
            with self.lock:
                positions = []
                for i, robot in enumerate(self.robots):
                    try:
                        pos = robot.get_robot_position()[-1]
                        positions.append(pos)
                        self.robot_labels[i].set_position((pos[0], pos[1] + 0.2))  # offset label above robot
                    except IndexError:
                        continue

                if positions:
                    self.robot_scatter.set_offsets(np.array(positions))

                task_colors = ['blue' if covered else 'red' for covered in self.is_covered]
                self.task_scat.set_facecolor(task_colors)


            return self.robot_scatter, self.task_scat, *self.robot_labels

        self.ani = FuncAnimation(self.fig, update, interval=5)

        # Custom legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Uncovered Task', markerfacecolor='red', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Covered Task', markerfacecolor='blue', markersize=6),
            Line2D([0], [0], marker='o', color='w', label='Robot', markerfacecolor='#80B3FF', markersize=6),
        ]
        self.ax.legend(handles=legend_elements)

    def start_coverage_listener(self, node):
        self.coverage_subscriber = node.create_subscription(
            EnvironmentalRepresentation,
            '/environmental_representation',
            self.environmental_representation_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )

    def environmental_representation_callback(self, msg):
        for i in range(len(msg.is_covered)):
            if msg.is_covered[i]:
                self.is_covered[i] = True


def main():
    print("Initializing ROS...")
    parser = argparse.ArgumentParser(description="Run CBM-POP Simulation")
    parser.add_argument('--num_robots', type=int, default=2, help='Number of robots')
    parser.add_argument('--speed', type=float, default=0.05, help='Robot speed')
    parser.add_argument('--env_size', type=int, default=5, help='Environment size (square)')
    args = parser.parse_args()

    rclpy.init(args=sys.argv)

    tasks = [(i + 0.5, j + 0.5) for i in range(10) for j in range(10)]
    env_bounds = [0, 10, 0, 10]
    xmin, xmax, ymin, ymax = env_bounds

    starts = [[(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))] for _ in range(args.num_robots)]

    simulator = SimpleSimulator(tasks, env_bounds, starts, obstacles=1, args=args)
    print("Simulator created.")

    executor = MultiThreadedExecutor()
    for robot in simulator.robots:
        executor.add_node(robot)

    # Add ROS node to listen for task coverage updates
    coverage_listener_node = rclpy.create_node('coverage_listener')
    simulator.start_coverage_listener(coverage_listener_node)
    executor.add_node(coverage_listener_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def shutdown():
        print("All robots finished.")

        # Compute path lengths
        path_lengths = [np.sum(
            np.linalg.norm(np.diff(np.array(robot.get_robot_position()), axis=0), axis=1)
        ) for robot in simulator.robots]

        longest_path = max(path_lengths)
        average_path = np.mean(path_lengths)

        print(f"Longest robot path length: {longest_path:.2f}")
        print(f"Average robot path length: {average_path:.2f}")

        # Shutdown
        plt.close('all')
        executor.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    simulator.start_simulation_thread(on_complete=shutdown)
    simulator.start_animation()

    try:
        while rclpy.ok():
            plt.pause(0.1)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted. Manual shutdown.")
        shutdown()


if __name__ == '__main__':
    main()

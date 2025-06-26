import sys
import time
import threading
import argparse
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation import LearningMethod
from cbm_pop.SimpleSimulator.simulator_robot import SimulatorRobot as robot
from cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation \
    import CBMPopulationAgentOnlineSimpleSimulation as agent
from cbm_pop.SimpleSimulator.simple_fitness_logger import SimpleFitnessLogger
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SimpleSimulator:
    def __init__(self, tasks, environmental_bounds, robot_starting_positions,
                 obstacles, args):
        self.simulation_done = False
        self.environmental_bounds = environmental_bounds
        self.tasks = tasks
        self.robot_starting_positions = robot_starting_positions
        self.obstacles = obstacles
        self.robots = []
        self.lock = threading.Lock()
        self.finished_robots = [False] * args.num_robots

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
        self.ax.scatter(task_xs, task_ys, c='black', s=10, label='Tasks')

        self.robot_scatter = self.ax.scatter([], [], c='blue', label='Robots')

        def update(frame):
            with self.lock:
                positions = []
                for robot in self.robots:
                    try:
                        pos = robot.get_robot_position()[-1]
                        positions.append(pos)
                    except IndexError:
                        continue
                if positions:
                    self.robot_scatter.set_offsets(np.array(positions))
            return self.robot_scatter,

        self.ani = FuncAnimation(self.fig, update, interval=50)
        self.ax.legend()


def main():
    print("Initializing ROS...")
    parser = argparse.ArgumentParser(description="Run CBM-POP Simulation")

    # Environment & robot arguments
    parser.add_argument('--num_robots', type=int, default=2, help='Number of robots')
    parser.add_argument('--speed', type=float, default=0.05, help='Robot speed')
    parser.add_argument('--env_size', type=int, default=5, help='Environment size (square)')

    args = parser.parse_args()
    rclpy.init(args=sys.argv)

    tasks = [(i + 0.5, j + 0.5) for i in range(10) for j in range(10)]
    env_bounds = [0, 10, 0, 10]
    xmin, xmax, ymin, ymax = env_bounds

    starts = [[(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))] for _ in range(args.num_robots)]

    simulator = SimpleSimulator(
        tasks, env_bounds, starts, obstacles=1,
        args=args
    )
    print("Simulator created.")

    executor = MultiThreadedExecutor()
    for robot in simulator.robots:
        executor.add_node(robot)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def shutdown():
        print("All robots finished. Shutting down nodes...")
        plt.close('all')  # force close the plot window
        executor.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    simulator.start_simulation_thread(on_complete=shutdown)
    simulator.start_animation()

    try:
        while rclpy.ok():
            plt.pause(0.1)  # keep GUI responsive
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted. Manual shutdown.")
        shutdown()


if __name__ == '__main__':
    main()

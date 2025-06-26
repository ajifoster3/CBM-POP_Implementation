import sys
from random import random

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation import LearningMethod
from cbm_pop.SimpleSimulator.simulator_robot import SimulatorRobot as robot
from cbm_pop.SimpleSimulator.cbm_population_agent_online_simple_simulation \
import CBMPopulationAgentOnlineSimpleSimulation as agent
import threading
import time
from cbm_pop.SimpleSimulator.simple_fitness_logger import SimpleFitnessLogger
from cbm_pop_interfaces.msg import FinishedCoverage
import argparse


class SimpleSimulator:
    def __init__(self, tasks, environmental_bounds, robot_starting_positions,
                 obstacles, args):

        self.environmental_bounds = environmental_bounds
        self.tasks = tasks
        self.robot_starting_positions = robot_starting_positions
        self.obstacles = obstacles
        self.t = 0
        self.robot_positions = self.robot_starting_positions
        self.robots = []
        self.agents = []
        self.lock = threading.Lock()
        self.robot_trajectories = [[] for _ in range(len(robot_starting_positions))]
        self.finished_robots = [False] * args.num_robots

        for i in range(len(robot_starting_positions)):
            self.robots.append(robot(i, robot_starting_positions[i], args.speed, args.num_robots))
            self.agents.append(agent(
                pop_size=args.pop_size,
                eta=args.eta,
                rho=args.rho,
                di_cycle_length=args.di_cycle_length,
                epsilon=args.epsilon,
                num_iterations=args.num_iterations,
                num_solution_attempts=args.num_solution_attempts,
                agent_id=i,
                node_name=f"cbm_population_agent_{i}",
                learning_method=args.learning_method,
                lr=args.lr,
                gamma_decay=args.gamma_decay,
                positive_reward=args.positive_reward,
                negative_reward=args.negative_reward,
                num_tsp_agents=args.num_robots
            ))

    def start_simulation_thread(self, on_complete=None):
        def simulation_loop():
            while not all(robot.is_finished for robot in self.robots):
                with self.lock:
                    for i, robot in enumerate(self.robots):
                        robot.move_robot()
                time.sleep(0.01)
            print("Simulation complete.")
            if on_complete:
                on_complete()

        threading.Thread(target=simulation_loop, daemon=True).start()



def main():

    print("Initializing ROS...")
    parser = argparse.ArgumentParser(description="Run CBM-POP Simulation")
    # Environment & robot arguments
    parser.add_argument('--num_robots', type=int, default=2, help='Number of robots')
    parser.add_argument('--speed', type=float, default=0.05, help='Robot speed')
    parser.add_argument('--env_size', type=int, default=5, help='Environment size (square)')

    # Agent hyperparameters
    parser.add_argument('--pop_size', type=int, default=10)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--di_cycle_length', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--num_iterations', type=int, default=9999999)
    parser.add_argument('--num_solution_attempts', type=int, default=21)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma_decay', type=float, default=0.99)
    parser.add_argument('--positive_reward', type=float, default=1.0)
    parser.add_argument('--negative_reward', type=float, default=-0.5)
    parser.add_argument('--learning_method', type=str, choices=['Q-Learning', 'Ferreira_et_al.'], default='Q-Learning')

    args = parser.parse_args()

    rclpy.init(args=sys.argv)  # Pass full argv to rclpy

    tasks = [(i + 0.5, j + 0.5) for i in range(15) for j in range(15)]
    env_bounds = [0, 25, 0, 25]
    xmin, xmax, ymin, ymax = env_bounds

    starts = [[(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))] for _ in range(args.num_robots)]

    simulator = SimpleSimulator(
        tasks, env_bounds, starts, obstacles=1,
        args=args  # pass entire args namespace
    )
    print("Simulator created.")

    executor = MultiThreadedExecutor()
    for robot in simulator.robots:
        executor.add_node(robot)
    for agent in simulator.agents:
        executor.add_node(agent)
    logger = SimpleFitnessLogger()
    executor.add_node(logger)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()


    # === Shutdown callback ===
    def shutdown():
        print("All robots finished. Shutting down nodes...")

        # Destroy all nodes BEFORE shutting down ROS
        for robot in simulator.robots:
            if robot.context.ok():
                robot.destroy_node()
        for agent in simulator.agents:
            if agent.context.ok():
                agent.destroy_node()
        if logger.context.ok():
            logger.destroy_node()

        rclpy.shutdown()

    simulator.start_simulation_thread(on_complete=shutdown)

    try:
        while rclpy.ok():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted. Manual shutdown.")
        shutdown()


if __name__ == '__main__':
    main()

import csv
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
from cbm_pop_interfaces.msg import EnvironmentalRepresentation, SimplePosition, FinishedCoverage
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from time import time

class SimpleFitnessLogger(Node):
    def __init__(self):
        super().__init__('fitness_logger')
        self.logging_start_time = time()

        self.timeout = 10
        self.last_env_update_time = None  # Track last received EnvironmentalRepresentation time
        self.termination_timeout = 20.0

        self.declare_parameter('timeout', 0.0)
        self.timeout = self.get_parameter('timeout').get_parameter_value().double_value

        self.num_tsp_agents = 5
        self.is_all_poses_set = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.position_log_file = f"resources/run_logs/robot_positions_{timestamp}.csv"


        with open(self.position_log_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ROS Time (seconds)", "Agent ID", "Latitude", "Longitude", "Altitude"])

        self.clock_time = None
        self.is_covered = None
        self.persistent_environmental_representation = None
        self.environmental_representation = None
        self.solution_subscriber = None

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
        self.finished_robots[int(msg.robot_id)] = bool(msg.finished)
        if all(self.finished_robots):
            print("Coverage Complete")
            self.destroy_node()


    def global_pose_callback(self, msg, agent):
        # Store the first received pose for each agent
        if self.initial_robot_poses[agent] is None:
            self.initial_robot_poses[agent] = (msg.x_position, msg.y_position)

        if all(pose is not None for pose in self.robot_poses) and self.is_all_poses_set is False:
            self.is_all_poses_set = True

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

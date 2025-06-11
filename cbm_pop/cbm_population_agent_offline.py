import json
import random
import sys
import traceback
from time import time

from builtin_interfaces.msg import Time
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
from copy import deepcopy
from random import sample
from cbm_pop.Condition import ConditionFunctions, Condition
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator_Fuctions import OperatorFunctions
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.Problem import Problem
from rclpy.node import Node
from std_msgs.msg import String, Float32
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
from cbm_pop.path_cost_calculator import calculate_drone_distance
from cbm_pop_interfaces.msg import Solution, Weights, EnvironmentalRepresentation
from enum import Enum
from math import radians, cos, sin, asin, sqrt
from pygeodesy.geoids import GeoidPGM
from std_msgs.msg import Bool
from geographic_msgs.msg import GeoPose, GeoPoseStamped

class LearningMethod(Enum):
    FERREIRA = "Ferreira_et_al."
    Q_LEARNING = "Q-Learning"


class CBMPopulationAgentOffline(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_tasks, num_tsp_agents, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, cost_matrix, learning_method):
        """
        Initialises the agent on startup
        """
        super().__init__(node_name)
        self.geoid = GeoidPGM('/home/ajifoster3/Documents/Software/ros_ws/src/CBM-POP_Implementation/egm96-5.pgm')

        self.current_task = None
        filename = "/home/ajifoster3/Downloads/Poses/generated_geoposes_less.json"
        with open(filename, "r") as file:
            print(f"using {filename}")
            data = json.load(file)
        self.task_poses = [
            {
                "latitude": item["GeoPose"]["position"]["latitude"],
                "longitude": item["GeoPose"]["position"]["longitude"],
                "altitude": item["GeoPose"]["position"]["altitude"],
                "orientation_x": item["GeoPose"]["orientation"]["x"],
                "orientation_y": item["GeoPose"]["orientation"]["y"],
                "orientation_z": item["GeoPose"]["orientation"]["z"],
                "orientation_w": item["GeoPose"]["orientation"]["w"]
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

        self.num_tasks = len(self.task_poses)
        self.is_generating = False
        self.pop_size = pop_size
        self.eta = eta
        self.rho = rho
        self.di_cycle_length = di_cycle_length
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.num_tsp_agents = 5
        self.agent_best_solution = None
        self.coalition_best_solution = None
        self.local_best_solution = None
        self.coalition_best_agent = None
        self.num_intensifiers = 2
        self.num_diversifiers = 4

        self.cost_matrix = self.calculate_cost_matrix()
        self.robot_cost_matrix = self.calculate_robot_cost_matrix()
        self.population = self.generate_population()
        self.current_solution = self.select_solution()
        self.weight_matrix = WeightMatrix(self.num_intensifiers, self.num_diversifiers, True)
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_ID = agent_id  # This will change as teamsize changes
        self.true_agent_ID = self.agent_ID  # This is permanent
        self.received_weight_matrices = []
        self.learning_method = learning_method
        self.last_env_rep_timestamps = {}

        # Iteration state
        self.iteration_count = 0
        self.di_cycle_count = 0
        self.no_improvement_attempt_count = 0
        self.best_coalition_improved = False
        self.best_local_improved = False

        # Runtime data

        self.cb_group = ReentrantCallbackGroup()
        self.me_cb_group = MutuallyExclusiveCallbackGroup()

        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(Solution, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10, callback_group=self.me_cb_group)
        self.weight_publisher = self.create_publisher(Weights, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            Weights, 'weight_matrix', self.weight_update_callback, 10)
        # Subscriber to the global position
        # List to store subscriptions

        self.solution_publisher_timer = self.create_timer(2, self.regular_solution_publish_timer,
                                                          callback_group=self.cb_group)

        # Q_learning
        # Q_learning parameter
        self.lr = 0.1  # RL learning Rate
        self.reward = 0  # RL reward initial 0
        self.new_reward = 0  # RL tarafından secilecek. initial 0
        self.gamma_decay = 0.99  # it can be change interms of iteration

        self.run_timer = self.create_timer(0.1, self.run_step)

    def calculate_robot_cost_matrix(self):
        """
        Returns a cost map representing the traversal cost from each robot_pose to each task_pose calculated using
        drone_distance.
        """
        # Filter out None values from robot_poses
        valid_robot_poses = [pose for pose in self.robot_poses if pose is not None]
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
                total_horizontal_distance = self.haversine(lat1, lon1, 0, lat2, lon2, 0)
                total_vertical_distance = alt2 - alt1

                # Compute cost using trajectory generation function
                cost = calculate_drone_distance(
                    total_horizontal_distance, total_vertical_distance, 3, 1.45, 11)

                cost_map[i][j] = cost

        return cost_map

    def calculate_cost_matrix(self):
        """
        Returns a cost matrix representing the traversal cost from each task_pose to each other task_pose, this is
        constructed by calculating the drone_distance between all the task poses in the agents task_pose list.
        """
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    # Extract positions
                    lat1, lon1, alt1 = self.task_poses[i]["latitude"], self.task_poses[i]["longitude"], \
                        self.task_poses[i]["altitude"]
                    lat2, lon2, alt2 = self.task_poses[j]["latitude"], self.task_poses[j]["longitude"], \
                        self.task_poses[j]["altitude"]

                    # Compute horizontal and vertical distances
                    total_horizontal_distance = self.haversine(lat1, lon1, 0, lat2, lon2, 0)
                    total_vertical_distance = alt2 - alt1

                    # Compute cost using trajectory generation function
                    cost = calculate_drone_distance(
                        total_horizontal_distance, total_vertical_distance, 3, 1.45, 11)

                    cost_map[i][j] = cost
        return cost_map

    def calculate_robot_inital_pose_cost_matrix(self):
        """
        Returns a cost map representing the traversal cost from each initial_robot_pose to each each task_pose calculated
        using drone_distance.
        """

        # Filter out None values from robot_poses

        valid_robot_poses = self.robot_poses
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

    def generate_population(self):
        """
        Generates the initial solution population with random task ordering.
        """
        population = []
        print("Generating Population (Random Order)...")

        for i in range(self.pop_size):
            print(f"Generating solution {i}")

            # Start with all tasks
            all_tasks_to_assign = list(range(self.num_tasks))
            random.shuffle(all_tasks_to_assign)  # Shuffle tasks initially for a random starting point

            agent_assignments = [[] for _ in range(self.num_tsp_agents)]
            task_allocation_counts = [0] * self.num_tsp_agents

            current_tasks = list(range(self.num_tasks))
            random.shuffle(current_tasks)  # Randomize the order in which tasks are processed

            for task in current_tasks:
                # Randomly pick an agent for this task
                chosen_agent = random.randint(0, self.num_tsp_agents - 1)
                agent_assignments[chosen_agent].append(task)
                task_allocation_counts[chosen_agent] += 1

            # To generate a random *overall* task order and then *derive* allocations:
            ordered_task_list = list(range(self.num_tasks))
            random.shuffle(ordered_task_list)

            # Re-initialize for this new approach
            agent_assignments = [[] for _ in range(self.num_tsp_agents)]
            task_allocation_counts = [0] * self.num_tsp_agents

            for idx, task_id in enumerate(ordered_task_list):
                chosen_agent = random.randint(0, self.num_tsp_agents - 1)
                agent_assignments[chosen_agent].append(task_id)
                task_allocation_counts[chosen_agent] += 1

            # The `ordered_task_list` is already the randomly ordered list.
            # The `task_allocation_counts` are derived from how tasks were assigned.

            population.append((ordered_task_list, task_allocation_counts))

        print("Generated random population")
        print(f"{population[0]}")
        return population


    def set_coalition_best_solution(self, solution):
        """
        Sets the given solution as the best coalition solution and the gets the robots next task from the solution.
        """
        self.coalition_best_solution = solution

    def select_solution(self):
        """
        Finds and returns the fittest solution
        :return: The fittest solution
        """
        # Select the best solution from the population based on fitness score
        best_solution = min(self.population, key=lambda sol: Fitness.fitness_function(
            sol, self.cost_matrix))  # Assuming lower score is better
        return best_solution

    def update_experience(self, condition, operator, gain):
        """
        Adds details of the current iteration to the experience memory.
        :param condition: The previous condition
        :param operator: The operator applied
        :param gain: The resulting change in the current solution's fitness
        :return: None
        """
        self.previous_experience.append([condition, operator, gain])
        pass

    def individual_learning_old(self):
        # Update weight matrix (if needed) based on learning (not fully implemented in this example)
        abs_gain = 0
        index_best_fitness = -1
        for i in range(len(self.previous_experience)):
            current_gain = abs_gain + self.previous_experience[i][2]
            if current_gain < abs_gain:
                index_best_fitness = i
            abs_gain += current_gain

            # Get elements before index_best_fitness
        elements_before_best = self.previous_experience[:index_best_fitness + 1] if index_best_fitness != -1 else []
        condition_operator_pairs = [(item[0], item[1]) for item in elements_before_best]
        condition_operator_pairs = list(set(condition_operator_pairs))
        for pair in condition_operator_pairs:
            self.weight_matrix.weights[pair[0].value][pair[1].value - 1] += self.eta

        return self.weight_matrix.weights

    # ----------------------------------------------------------------------------
    def individual_learning(self):
        """
        Learning with Bellman Equation.
        Updates the weight matrix (Q(s, a)) using the Q-learning formula.
        Only updates the weights for experiences before the lowest relative fitness solution.
        :return: Updated weight matrix
        """
        # Compute cumulative gains
        cumulative_gains = []
        total_gain = 0
        for experience in self.previous_experience:
            _, _, gain = experience
            total_gain += gain
            cumulative_gains.append(total_gain)

        # Find the index of the lowest relative fitness solution (min cumulative gain)
        if self.best_local_improved:
            min_fitness_index = cumulative_gains.index(min(cumulative_gains))
        else:
            min_fitness_index = len(self.previous_experience)

        # Update weights only for experiences before this index
        for i in range(min_fitness_index):
            condition, operator, gain = self.previous_experience[i]

            # Current Q value
            current_q = self.weight_matrix.weights[condition.value][operator.value - 1]

            # Determine next condition
            if i + 1 < len(self.previous_experience):
                next_condition = self.previous_experience[i + 1][
                    0]  # Next condition is the first element of the next experience
                max_next_q = max(self.weight_matrix.weights[next_condition.value])  # Best future Q-value
            else:
                max_next_q = 0  # No future state, assume no future reward

            # Estimate future rewards (single-step Q-learning)
            max_next_q = max(self.weight_matrix.weights[condition.value])

            if self.best_local_improved:
                self.reward = 1
            else:
                self.reward = -0.5

            updated_q = current_q + self.lr * (self.reward + self.gamma_decay * max_next_q - current_q)
            updated_q = max(updated_q, 1e-6)
            # Update weight matrix
            self.weight_matrix.weights[condition.value][operator.value - 1] = updated_q

        return self.weight_matrix.weights

    # ------------------------------------------------------------------------------------------------

    def mimetism_learning(self, received_weights, rho):
        # mimetisim learnng will stay same
        """
        Perform mimetism learning by updating self.weight_matrix.weights using multiple sets of received weights.
        For each weight in each received set:
            w_a = (1 - rho) * w_a + rho * w_b
        :param received_weights: A 3D list (or array) of received weights. Each "slice" is a 2D matrix of weights.
        :param rho: The learning rate, a value between 0 and 1.
        :return: Updated weight matrix.
        """
        # Iterate over each set of received weights
        for weight_set in received_weights:
            # Check dimensions match
            if len(weight_set) != len(self.weight_matrix.weights) or len(weight_set[0]) != len(
                    self.weight_matrix.weights[0]):
                raise ValueError("Dimension mismatch between weight_matrix.weights and received weights.")
                # Update self.weight_matrix.weights for each element
            for i in range(len(self.weight_matrix.weights)):  # Rows
                for j in range(len(self.weight_matrix.weights[i])):  # Columns
                    self.weight_matrix.weights[i][j] = (
                            (1 - rho) * self.weight_matrix.weights[i][j] + rho * weight_set[i][j]
                    )

    def stopping_criterion(self, iteration_count):
        """
        Returns true is the current number of iterations is over a set iteration limit.
        """
        # Define a stopping criterion (e.g., a fixed number of iterations)
        return iteration_count > self.num_iterations

    def end_of_di_cycle(self, cycle_count):
        """
        Returns true if the current cycle is equal to or over the length of a DI cycle.
        """
        if cycle_count >= self.di_cycle_length:
            return True
        return False  # Placeholder; replace with actual condition

    def weight_update_callback(self, msg):
        """
        Stores received weight matrices.
        """
        # Callback to process incoming weight matrix updates
        print("Received weights")
        received_weights = self.weight_matrix.unpack_weights(weights_msg=msg, agent_id=self.agent_ID)
        if received_weights is not None:
            self.received_weight_matrices.append(received_weights)

    def solution_update_callback(self, msg):
        """
        Receives and processes a proposed best coalition solution from another agent.
        """
        # Callback to process incoming solution updates
        if msg is not None and self.coalition_best_solution is not None:
            received_solution = (msg.order, msg.allocations)

            # Check if the received solution has fewer tasks than the current best solution
            received_task_count = sum(msg.allocations)
            current_task_count = sum(self.coalition_best_solution[1])

            # Now check fitness as before
            their_solution_fitness = Fitness.fitness_function_robot_pose(received_solution, self.cost_matrix,
                                                                         self.robot_cost_matrix, self.robot_cost_matrix)

            our_solution_fitness = Fitness.fitness_function_robot_pose(self.coalition_best_solution, self.cost_matrix,
                                                                       self.robot_cost_matrix, self.robot_cost_matrix)

            # If the received solution is better, update the coalition best solution
            if their_solution_fitness < our_solution_fitness:
                self.set_coalition_best_solution(received_solution)
                self.coalition_best_agent = msg.id
        elif msg is not None and self.coalition_best_solution is None:
            received_solution = (msg.order, msg.allocations)
            self.set_coalition_best_solution(received_solution)
            self.coalition_best_agent = msg.id


    @staticmethod
    def haversine(lat1, lon1, alt1, lat2, lon2, alt2):
        """
        Calculates Haversine distance
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ asin( √a )
        d = R ⋅ c
        φ is latitude, λ is longitude, R is earth’s radius (mean radius = 6378160)
        """
        R = 6378160  # Earth radius in miles (for km use 6372.8)

        # Convert lat/lon from degrees to radians
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)

        # Haversine formula for surface distance
        a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
        c = 2 * asin(sqrt(a))
        haversine_distance = R * c  # Great-circle distance in miles

        # Compute altitude difference in miles
        alt_diff = alt2 - alt1  # Altitude difference in miles

        # Compute 3D distance using Pythagorean theorem
        distance_3D = sqrt(haversine_distance ** 2 + alt_diff ** 2)

        return distance_3D

    def select_random_solution(self):
        """
        Sample a random solution from the population.
        """
        temp_solution = sample(population=self.population, k=1)[0]
        if temp_solution != self.current_solution:
            return temp_solution

    def regular_solution_publish_timer(self):
        """
        Publishes the coalition best solution.
        """
        if self.coalition_best_solution is not None:
            solution = Solution()
            solution.id = self.agent_ID
            solution.order = self.coalition_best_solution[0]
            solution.allocations = self.coalition_best_solution[1]
            self.solution_publisher.publish(solution)

    def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """
        if self.current_solution is not None:
            # Ensure the current solution and coalition best solution do not contain covered tasks
            if self.stopping_criterion(self.iteration_count):
                self.get_logger().info("Stopping criterion met. Shutting down.")
                self.run_timer.cancel()
                return

            condition = ConditionFunctions.perceive_condition(self.previous_experience)
            if self.no_improvement_attempt_count >= self.no_improvement_attempts:
                self.current_solution = self.select_random_solution()
                self.no_improvement_attempt_count = 0

            operator = OperatorFunctions.choose_operator(self.weight_matrix.weights, condition)
            c_new = None
            try:
                c_new = OperatorFunctions.apply_op(
                    operator,
                    self.current_solution,
                    self.population,
                    self.cost_matrix,
                    self.robot_cost_matrix,
                    self.robot_cost_matrix
                )
            except Exception as e:
                print(f"Issue with applying operator: {e}")
            if c_new:
                gain = Fitness.fitness_function_robot_pose(c_new, self.cost_matrix,
                                                           self.robot_cost_matrix, self.robot_cost_matrix) - \
                       Fitness.fitness_function_robot_pose(self.current_solution, self.cost_matrix,
                                                           self.robot_cost_matrix, self.robot_cost_matrix)
                self.update_experience(condition, operator, gain)

                if self.local_best_solution is None or \
                        Fitness.fitness_function_robot_pose(c_new, self.cost_matrix,
                                                            self.robot_cost_matrix, self.robot_cost_matrix) < Fitness.fitness_function_robot_pose(
                    self.local_best_solution, self.cost_matrix,
                    self.robot_cost_matrix, self.robot_cost_matrix):
                    self.local_best_solution = deepcopy(c_new)
                    self.best_local_improved = True
                    self.no_improvement_attempt_count = 0
                else:
                    self.no_improvement_attempt_count += 1

                if self.coalition_best_solution is None or \
                        Fitness.fitness_function_robot_pose(c_new, self.cost_matrix,
                                                            self.robot_cost_matrix, self.robot_cost_matrix) < Fitness.fitness_function_robot_pose(
                    self.coalition_best_solution, self.cost_matrix,
                    self.robot_cost_matrix, self.robot_cost_matrix):
                    self.set_coalition_best_solution(deepcopy(c_new))
                    self.coalition_best_agent = self.agent_ID
                    self.best_coalition_improved = True
                    print(f"publishing best coalition: {Fitness.fitness_function_robot_pose(self.coalition_best_solution, self.cost_matrix, self.robot_cost_matrix, self.robot_cost_matrix)}")
                    solution = Solution()
                    solution.id = self.agent_ID
                    solution.order = self.coalition_best_solution[0]
                    solution.allocations = self.coalition_best_solution[1]
                    self.solution_publisher.publish(solution)

                self.current_solution = c_new
                self.di_cycle_count += 1

                condition = ConditionFunctions.perceive_condition(self.previous_experience)
                if condition == Condition.C_4:
                    print("*********************** Im Condition C4 ***********************")
                if self.end_of_di_cycle(self.di_cycle_count) or condition == Condition.C_4:

                    learning_method_switch = {
                        LearningMethod.FERREIRA: self.individual_learning_old,
                        LearningMethod.Q_LEARNING: self.individual_learning
                    }

                    condition = ConditionFunctions.perceive_condition(self.previous_experience)

                    # Call the appropriate function based on the current learning method
                    learning_function = learning_method_switch.get(LearningMethod(self.learning_method))
                    if learning_function:
                        self.weight_matrix.weights = learning_function()
                    else:
                        self.get_logger().error(f"Unknown learning method: {self.learning_method}")

                    self.best_local_improved = False
                    if self.best_coalition_improved:
                        self.best_coalition_improved = False
                        msg = Weights()
                        msg_dict = self.weight_matrix.pack_weights(self.agent_ID)
                        msg.id = msg_dict["id"]
                        msg.rows = msg_dict["rows"]
                        msg.cols = msg_dict["cols"]
                        msg.weights = msg_dict["weights"]

                        self.weight_publisher.publish(msg)

                    if self.received_weight_matrices:
                        self.mimetism_learning(self.received_weight_matrices, self.rho)
                        self.received_weight_matrices = []

                    self.previous_experience = []
                    self.di_cycle_count = 0

                self.iteration_count += 1
            else:
                print("Something went wrong with applying the operator and resulted in a None.")


def generate_problem(num_tasks):
    """
    Randomly generates a problem of size `number_tasks`
    :return: Randomly generated symmetrical cost matrix representing the problem
    """
    # Set the random seed for reproducibility
    np.random.seed(0)
    # Generate a random symmetrical 20x20 cost matrix
    size = num_tasks
    cost_matrix = np.random.randint(1, 100, size=(size, size))
    # Make the matrix symmetrical
    cost_matrix = (cost_matrix + cost_matrix.T) // 2
    # Set the diagonal to zero (no cost for staying at the same location)
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


# Deneme


def main(args=None):
    rclpy.init(args=args)

    temp_node = Node("parameter_loader")
    temp_node.declare_parameter("agent_id", 1)
    temp_node.declare_parameter("problem_filename", "resources/150_Task_Problem.csv")
    temp_node.declare_parameter("runtime", 60.0)
    temp_node.declare_parameter("learning_method", "Ferreira et al.")

    agent_id = temp_node.get_parameter("agent_id").value
    problem_filename = temp_node.get_parameter("problem_filename").value
    runtime = temp_node.get_parameter("runtime").value
    learning_method = temp_node.get_parameter("learning_method").value
    temp_node.destroy_node()

    problem = Problem()
    problem.load_cost_matrix(problem_filename, "csv")
    num_tasks = len(problem.cost_matrix)

    node_name = f"cbm_population_agent_{agent_id}"
    start_system_time = time()
    agent = CBMPopulationAgentOffline(
        pop_size=10, eta=0.1, rho=0.25, di_cycle_length=5, epsilon=0.01,
        num_tasks=num_tasks, num_tsp_agents=10, num_iterations=9999999,
        num_solution_attempts=21, agent_id=agent_id, node_name=node_name,
        cost_matrix=problem.cost_matrix, learning_method=learning_method
    )
    print("CBMPopulationAgentOnline has been initialized.")

    def shutdown_callback():
        agent.get_logger().info("LLM-Interface-agent Runtime completed. Shutting down.")
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if runtime != -1:
        # Create a timer for shutdown
        a_time = runtime-(time()-start_system_time-3)
        print(f"Hit shutdown_callback with runtime: {a_time}")
        agent.create_timer(a_time, shutdown_callback, callback_group=agent.cb_group)

    # asyncio.run(agent.generate_population())

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

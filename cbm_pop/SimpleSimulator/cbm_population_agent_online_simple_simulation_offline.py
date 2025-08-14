import json
import math
import random
import sys
import traceback
from time import time
import matplotlib.pyplot as plt
import numpy as np
from builtin_interfaces.msg import Time
from matplotlib import animation, patches
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
from cbm_pop_interfaces.msg import Solution, Weights, EnvironmentalRepresentation, SimplePosition, FinishedCoverage
from enum import Enum
from math import radians, cos, sin, asin, sqrt
from std_msgs.msg import Bool

class LearningMethod(Enum):
    FERREIRA = "Ferreira_et_al."
    Q_LEARNING = "Q-Learning"


class CBMPopulationAgentOnlineSimpleSimulationOffline(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, learning_method,
                 lr = 0.5,
                 gamma_decay = 0.99,
                 positive_reward = 1,
                 negative_reward = -0.5,
                 num_tsp_agents = 10):

        """
        Initialises the agent on startup
        """
        super().__init__(node_name)
        print("Initialising Agent")
        self.current_task = None
        self.task_poses = [(i + 0.5, j + 0.5) for i in range(10) for j in range(10)]

        self.num_tasks = len(self.task_poses)
        self.is_generating = False
        self.pop_size = pop_size
        self.eta = eta
        self.rho = rho
        self.di_cycle_length = di_cycle_length
        self.num_iterations = num_iterations
        self.num_tsp_agents = num_tsp_agents
        self.agent_best_solution = None
        self.coalition_best_solution = None
        self.received_coalition_best_solution = None # This is a pending best solution received from another agent.
        self.local_best_solution = None
        self.coalition_best_agent = None
        self.num_intensifiers = 2
        self.num_diversifiers = 4
        self.population = None
        self.weight_matrix = WeightMatrix(self.num_intensifiers, self.num_diversifiers, True)
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_ID = agent_id  # This will change as teamsize changes
        self.true_agent_ID = self.agent_ID  # This is permanent
        self.received_weight_matrices = []

        # temperature Annealing settings
        self.step_count = 0
        self.anneal_horizon = 2000
        self.T0 = 2.0
        self.T1 = 0.4

        if isinstance(learning_method, str):
            try:
                learning_method = LearningMethod(learning_method)
            except ValueError:
                raise ValueError(
                    f"Invalid learning method '{learning_method}'. Must be one of: "
                    f"{[e.value for e in LearningMethod]}"
                )
        self.learning_method = learning_method

        self.new_robot_cost_matrix = None
        self.is_new_robot_cost_matrix = False
        self.last_env_rep_timestamps = {}

        # Iteration state
        self.iteration_count = 0
        self.di_cycle_count = 0
        self.no_improvement_attempt_count = 0
        self.best_coalition_improved = False
        self.best_local_improved = False

        # Runtime data
        self.initial_robot_poses = [None] * self.num_tsp_agents
        self.robot_poses = [None] * self.num_tsp_agents
        self.current_solution = None
        self.is_covered = [False] * self.num_tasks
        self.cost_matrix = self.calculate_cost_matrix()
        self.robot_initial_pose_cost_matrix = [None] * self.num_tsp_agents
        self.agent_to_revive = None
        self.ros_timer = None
        self.agent_timeouts = [False] * self.num_tsp_agents
        self.finished_robots = [False] * self.num_tsp_agents

        self.cb_group = ReentrantCallbackGroup()
        self.me_cb_group = MutuallyExclusiveCallbackGroup()

        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(Solution, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10)
        self.weight_publisher = self.create_publisher(Weights, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            Weights, 'weight_matrix', self.weight_update_callback, 10)
        # Subscriber to the global position
        # List to store subscriptions
        self.global_pose_subscribers = []

        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id}/global_pose'

            # Wrap the callback to include agent_id
            sub = self.create_subscription(
                SimplePosition,
                topic,
                lambda msg, agent=id: self.global_pose_callback(msg),
                10,
                callback_group=self.cb_group
            )

            self.global_pose_subscribers.append(sub)  # Store subscription to prevent garbage collection

        # Publisher for goal position
        self.goal_pose_publisher = self.create_publisher(
            SimplePosition,
            f'/central_control/uas_{agent_id}/goal_pose',
            10)

        self.solution_publisher_timer = self.create_timer(2, self.regular_solution_publish_timer,
                                                          callback_group=self.cb_group)

        # Q_learning
        # Q_learning parameter
        self.lr = lr  # RL learning Rate
        self.reward = 0  # RL reward initial 0
        self.new_reward = 0  # RL tarafından secilecek. initial 0
        self.gamma_decay = gamma_decay  # it can be change interms of iteration
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

        self.run_timer = None
        self.is_loop_started = False
        print("Agent initialised.")

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
                    x1, y1 = self.task_poses[i][0], self.task_poses[i][1]
                    x2, y2 = self.task_poses[j][0], self.task_poses[j][1]

                    # Compute horizontal and vertical distances
                    x_dist = abs(x2 - x1)
                    y_dist = abs(y2 - y1)
                    # Compute cost using trajectory generation function
                    cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))

                    cost_map[i][j] = cost
        return cost_map

    def calculate_robot_inital_pose_cost_matrix(self):
        """
        Returns a cost map representing the traversal cost from each initial_robot_pose to each each task_pose calculated
        using drone_distance.
        """

        # Filter out None values from robot_poses

        valid_robot_poses = self.initial_robot_poses
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        # Initialize cost matrix
        cost_map = np.zeros((num_robots, num_tasks))

        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                # Extract positions
                x1, y1 = robot_pose
                x2, y2 = task_pose

                # Compute horizontal and vertical distances
                x_dist = x2 - x1
                y_dist = y2 - y1
                # Compute cost using trajectory generation function
                cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))

                cost_map[i][j] = cost
        self.robot_initial_pose_cost_matrix = cost_map

    def generate_population(self):
        """
        Generates the initial solution population randomly.
        Each task is randomly assigned to one of the agents.
        """
        population = []
        print("Generating Random Population...")

        for i in range(self.pop_size):
            #print(f"Generating solution {i}")

            # Randomly shuffle tasks
            tasks = list(range(self.num_tasks))
            random.shuffle(tasks)

            # Randomly assign each task to an agent
            agent_assignments = [[] for _ in range(self.num_tsp_agents)]
            for task in tasks:
                chosen_agent = random.randint(0, self.num_tsp_agents - 1)
                agent_assignments[chosen_agent].append(task)

            # Flatten the ordered task list
            ordered_task_list = [
                task
                for agent_tasks in agent_assignments
                for task in agent_tasks
            ]

            # Record how many tasks are assigned to each agent
            task_allocation_counts = [len(agent_tasks) for agent_tasks in agent_assignments]

            # Append the random solution
            population.append((ordered_task_list, task_allocation_counts))

        #print("Generated random population")
        #print(f"{population[0]}")
        return population

    def set_coalition_best_solution(self, solution):
        """
        Sets the given solution as the best coalition solution and the gets the robots next task from the solution.
        """
        self.coalition_best_solution = deepcopy(solution)

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
                self.reward = self.positive_reward
            else:
                self.reward = self.negative_reward

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
        received_weights = self.weight_matrix.unpack_weights(weights_msg=msg, agent_id=self.agent_ID)
        if received_weights is not None:
            self.received_weight_matrices.append(received_weights)

    def solution_update_callback(self, msg):
        """
        Receives a processes a proposed best coalition solution from another agent.
        """
        if msg is not None and self.coalition_best_solution is not None:
                received_solution = (msg.order, msg.allocations)

                # Check if the received solution has fewer tasks than the current best solution
                received_task_count = sum(msg.allocations)
                current_task_count = sum(self.coalition_best_solution[1])

                # Now check fitness as before
                their_solution_fitness = Fitness.fitness_function_robot_pose(received_solution, self.cost_matrix,
                                                                             self.robot_initial_pose_cost_matrix,
                                                                             self.robot_initial_pose_cost_matrix)

                our_solution_fitness = Fitness.fitness_function_robot_pose(self.coalition_best_solution,
                                                                           self.cost_matrix,
                                                                           self.robot_initial_pose_cost_matrix,
                                                                           self.robot_initial_pose_cost_matrix)

                # If the received solution is better, update the coalition best solution
                if their_solution_fitness < our_solution_fitness:
                    #print(f"received a better solution: their fitness: {their_solution_fitness}, out fitness: {our_solution_fitness}")
                    self.received_coalition_best_solution = received_solution
                    self.coalition_best_agent = msg.id

    def global_pose_callback(self, msg):
        """
        Receives and processes a global pose from another agent.
        Updating the internal representation of the state.
        If coverage has yet to start and each agent has a pose, calculate the robot cost matrix and start coverage.
        If this agent has reached its goal, complete that task.
        If an agent has finished its tasks, register its completed coverage.
        """
        try:
            agent = msg.robot_id #todo: carry agent id
            # Store the first received pose for each agent
            if self.initial_robot_poses[agent] is None:
                self.initial_robot_poses[agent] = (msg.x_position, msg.y_position)

            task = deepcopy(self.current_task)

            if msg:
                self.robot_poses[agent] = (msg.x_position, msg.y_position)
            if all(pose is not None for pose in
                   self.robot_poses) and self.is_loop_started is False:
                self.is_new_robot_cost_matrix = True
                self.calculate_robot_inital_pose_cost_matrix()
                if self.population is None:
                    if self.is_generating is False:
                        self.is_generating = True
                        self.population = self.generate_population()
                        self.current_solution = self.select_solution()
                self.run_timer = self.create_timer(0.1, self.run_step, callback_group=self.me_cb_group)
                self.is_loop_started = True

            if self.task_poses is not None and task is not None:
                x = msg.x_position
                y = msg.y_position

                goal_x = self.task_poses[task][0]
                goal_y = self.task_poses[task][1]

                distance = math.sqrt(((x - goal_x) ** 2)+((y - goal_y) ** 2))

        except Exception as e:
            error_message = (
                f"[ERROR] Exception in global_pose_callback for agent {agent}:\n"
                f"    Error Type: {type(e).__name__}\n"
                f"    Error Message: {e}\n"
                f"    Stack Trace:\n{traceback.format_exc()}"
            )
            print(error_message)
            # print(error_message)

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

    def apply_operator(self, operator):
        c_new = None
        try:
            c_new = run_with_timeout(
                OperatorFunctions.apply_op,
                args=(
                    operator,
                    self.current_solution,
                    self.population,
                    self.cost_matrix,
                    self.robot_initial_pose_cost_matrix,
                    self.robot_initial_pose_cost_matrix
                ),
                timeout=5.0  # seconds, adjust as needed
            )
            if c_new is None:
                print(f"[TIMEOUT] Operator {operator} took too long. Skipping this step.")
                self.no_improvement_attempt_count += 1
                return None
            else:
                return c_new
        except Exception as e:
            print(f"Issue with applying operator: {e}")
            print(f"New solution came out empty:\n old solution: {self.current_solution}\n"
                  f"operator applied {operator}")

    def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """

        if self.current_solution is not None:

            # 1. Perceive state
            condition = ConditionFunctions.perceive_condition(self.previous_experience)

            # 2. If the solution is stagnant, use a new solution
            if self.no_improvement_attempt_count >= self.no_improvement_attempts:
                self.current_solution = self.select_random_solution()
                self.no_improvement_attempt_count = 0

            # 3. Select an Operator
            # where you currently choose the operator
            operator, probs, T = OperatorFunctions.choose_operator_annealed(
                self.weight_matrix.weights,
                condition,
                step=self.step_count,
                anneal_horizon=self.anneal_horizon,
                T0=self.T0, T1=self.T1,
                # mask=self.classic_masks[condition.value],  # <— uncomment to keep classic gating
                return_debug=True
            )
            self.step_count += 1

            # 4. Apply Said Operator
            c_new = self.apply_operator(operator)
            if c_new is None:
                return

            # 5. Handle any received coalition best solution
            if self.received_coalition_best_solution:
                self.handle_received_coalition_best_solution()

            # 6. Assess solution Fitnesses
            new_solution_fitness = Fitness.fitness_function_robot_pose(c_new, self.cost_matrix,
                                                                       self.robot_initial_pose_cost_matrix,
                                                                       self.robot_initial_pose_cost_matrix)

            current_solution_fitness = Fitness.fitness_function_robot_pose(self.current_solution, self.cost_matrix,
                                                                           self.robot_initial_pose_cost_matrix,
                                                                           self.robot_initial_pose_cost_matrix)


            if self.local_best_solution:
                local_best_solution_fitness = Fitness.fitness_function_robot_pose(self.local_best_solution,
                                                                                  self.cost_matrix,
                                                                                  self.robot_initial_pose_cost_matrix,
                                                                                  self.robot_initial_pose_cost_matrix)

            if self.coalition_best_solution:
                coalition_best_solution_fitness = Fitness.fitness_function_robot_pose(self.coalition_best_solution,
                                                                                      self.cost_matrix,
                                                                                      self.robot_initial_pose_cost_matrix,
                                                                                      self.robot_initial_pose_cost_matrix,
                                                                                      islog=False)

            # 7. Credit Assignment
            gain = new_solution_fitness - current_solution_fitness

            self.update_experience(condition, operator, gain)

            # 8. Update Local best solution

            if self.local_best_solution is None or \
                    new_solution_fitness < local_best_solution_fitness:
                self.local_best_solution = deepcopy(c_new)
                self.best_local_improved = True
                self.no_improvement_attempt_count = 0
            else:
                self.no_improvement_attempt_count += 1

            # 9. Update Coalition best solution

            if self.coalition_best_solution is None or new_solution_fitness < coalition_best_solution_fitness:
                self.set_coalition_best_solution(c_new)
                self.coalition_best_agent = self.agent_ID
                self.best_coalition_improved = True
                solution = Solution()
                solution.id = self.agent_ID
                solution.order = self.coalition_best_solution[0]
                solution.allocations = self.coalition_best_solution[1]
                self.solution_publisher.publish(solution)

            # 10. Update Current Solution

            self.current_solution = c_new
            self.di_cycle_count += 1

            # 11. Handle end of DI cycle
            condition = ConditionFunctions.perceive_condition(self.previous_experience)
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


    def handle_received_coalition_best_solution(self):
        if not self.coalition_best_solution:
            self.set_coalition_best_solution(self.received_coalition_best_solution)
            self.received_coalition_best_solution = None
        else:
            received_solution_fitness = Fitness.fitness_function_robot_pose(self.received_coalition_best_solution,
                                                                            self.cost_matrix,
                                                                            self.robot_initial_pose_cost_matrix,
                                                                            self.robot_initial_pose_cost_matrix)
            coalition_solution_fitness = Fitness.fitness_function_robot_pose(self.coalition_best_solution,
                                                                             self.cost_matrix,
                                                                             self.robot_initial_pose_cost_matrix,
                                                                             self.robot_initial_pose_cost_matrix)
            if received_solution_fitness < coalition_solution_fitness:
                self.set_coalition_best_solution(self.received_coalition_best_solution)

            self.received_coalition_best_solution = None


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

import concurrent.futures

def run_with_timeout(func, args=(), kwargs=None, timeout=1.0):
    """
    Runs a function with a timeout. Returns the result or None on timeout.
    """
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

def main(args=None):
    rclpy.init(args=args)

    print("Initialising CBM agent")
    temp_node = Node("parameter_loader")
    temp_node.declare_parameter("agent_id", 1)
    temp_node.declare_parameter("runtime", -1.0)
    temp_node.declare_parameter("learning_method", "Ferreira_et_al.")
    temp_node.declare_parameter("num_tsp_agents", 5)

    agent_id = temp_node.get_parameter("agent_id").value
    runtime = temp_node.get_parameter("runtime").value
    learning_method = temp_node.get_parameter("learning_method").value
    num_tsp_agents = temp_node.get_parameter("num_tsp_agents").value
    temp_node.destroy_node()

    node_name = f"cbm_population_agent_{agent_id}"
    agent = CBMPopulationAgentOnlineSimpleSimulationOffline(
        pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
        num_iterations=9999999, num_solution_attempts=21, agent_id=agent_id,
        node_name=node_name, learning_method=learning_method, num_tsp_agents=num_tsp_agents
    )
    print("CBMPopulationAgentOnlineSimpleSimulationOffline has been initialized.")

    def shutdown_callback():
        agent.get_logger().info("LLM-Interface-agent Runtime completed. Shutting down.")
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if runtime != -1:
        # Create a timer for shutdown
        print("Hit shutdown_callback")
        agent.create_timer(runtime, shutdown_callback)

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
            executor.shutdown()
            rclpy.shutdown()
            sys.exit(0)


if __name__ == '__main__':
    main()

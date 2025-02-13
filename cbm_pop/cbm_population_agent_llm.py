import random
import sys
import traceback
from audioop import findfactor
from collections import defaultdict

import numpy as np
from copy import deepcopy
from random import sample
from cbm_pop.Condition import ConditionFunctions
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator_Functions_LLM import OperatorFunctionsLLM, Operator
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.Problem import Problem
from rclpy.node import Node
import rclpy
from cbm_pop_interfaces.msg import Solution, Weights, FailedOperatorRequest, FailedOperatorResponse, \
    GeneratedPopulation, OECrossoverRequest, OECrossoverResponse
from enum import Enum


class LearningMethod(Enum):
    FERREIRA = "Ferreira_et_al."
    Q_LEARNING = "Q-Learning"


class CBMPopulationAgentLLM(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_tasks, num_tsp_agents, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, cost_matrix, learning_method):
        super().__init__(node_name)
        self.run_timer = None
        self.pop_size = pop_size
        self.eta = eta
        self.rho = rho
        self.di_cycle_length = di_cycle_length
        self.oe_cycle_length = 10
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.num_tasks = num_tasks
        self.num_tsp_agents = num_tsp_agents
        self.agent_best_solution = None
        self.coalition_best_solution = None
        self.local_best_solution = None
        self.coalition_best_agent = None
        self.num_intensifiers = 2
        self.num_diversifiers = 4
        self.population = self.generate_population()
        self.cost_matrix = cost_matrix
        self.current_solution = self.select_solution()
        self.weight_matrix = WeightMatrix(self.num_intensifiers, self.num_diversifiers)
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_id = agent_id
        self.received_weight_matrices = []
        self.learning_method = learning_method
        self.operator_functions = None
        # Initialize the dictionary
        self.failed_operator_dict = {}
        self.di_cycle_log = []

        # Populate the dictionary with Operator enum elements as keys and empty lists as values
        for operator in Operator:
            self.failed_operator_dict[operator] = []

        # Iteration state
        self.iteration_count = 0
        self.di_cycle_count = 0
        self.oe_cycle_count = 0
        self.no_improvement_attempt_count = 0
        self.best_coalition_improved = False
        self.best_local_improved = False
        self.cycle_operator_indexes = {}
        for operator in Operator:
            self.cycle_operator_indexes[operator] = -1

        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(Solution, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10)
        self.weight_publisher = self.create_publisher(Weights, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            Weights, 'weight_matrix', self.weight_update_callback, 10)
        self.failed_operator_publisher = self.create_publisher(FailedOperatorRequest,
                                                               'failed_operator_request' + str(self.agent_id),
                                                               10)
        self.failed_operator_subscriber = self.create_subscription(FailedOperatorResponse,
                                                                   'failed_operator_response' + str(self.agent_id),
                                                                   self.failed_operator_response_callback,
                                                                   10)
        self.oe_crossover_publisher = self.create_publisher(OECrossoverRequest,
                                                            'oe_crossover_request' + str(self.agent_id),
                                                            10)
        self.oe_crossover_subscriber = self.create_subscription(OECrossoverResponse,
                                                                'oe_crossover_response' + str(self.agent_id),
                                                                self.oe_crossover_response_callback,
                                                                10)

        # Q_learning
        # Q_learning parameter
        self.lr = 0.1  # RL learning Rate
        self.reward = 0  # RL reward initial 0
        self.new_reward = 0  # RL tarafından secilecek. initial 0
        self.gamma_decay = 0.99  # it can be change interms of iteration

        self.operator_functions = OperatorFunctionsLLM()
        self.run_timer = self.create_timer(0.1, self.run_step)


        # Add subscription to 'generated_population' topic
        #self.generated_population_sub = self.create_subscription(
        #    GeneratedPopulation,  # Adjust the message type as needed
        #    'generated_population_' + str(self.agent_id),
        #    self.generated_population_callback,
        #    10
        #)

    def run_step(self):
        """Main execution step for the optimization algorithm."""
        if self.check_stopping_criteria():
            return

        condition = self.determine_condition()
        self.handle_no_improvement()

        operator = self.operator_functions.choose_operator(
            self.weight_matrix.weights, condition
        )

        c_new, operator_index = self.apply_operator_and_validate(operator)
        gain = self.calculate_gain(c_new)
        self.update_experience_and_counters(condition, operator, gain, operator_index)

        self.update_best_solutions(c_new)
        self.handle_cycle_updates()

        self.iteration_count += 1

    def check_stopping_criteria(self):
        """Check if we should stop execution."""
        if self.stopping_criterion(self.iteration_count):
            self.get_logger().info("Stopping criterion met. Shutting down.")
            self.run_timer.cancel()
            return True
        return False

    def determine_condition(self):
        """Determine current optimization condition."""
        return ConditionFunctions.perceive_condition(self.previous_experience)

    def handle_no_improvement(self):
        """Handle case where no improvement has been made recently."""
        if self.no_improvement_attempt_count >= self.no_improvement_attempts:
            self.current_solution = self.select_random_solution()
            self.no_improvement_attempt_count = 0

    def apply_operator_and_validate(self, operator):
        """Apply selected operator and validate resulting solution."""
        while True:
            c_new, index, *error = self.operator_functions.apply_op(
                operator,
                self.current_solution,
                self.population,
                self.cost_matrix,
                self.failed_operator_dict,
                self.cycle_operator_indexes[operator]
            )
            if c_new is None:
                request = FailedOperatorRequest()
                request.operator_name = str(operator)
                request.failed_function = str(self.operator_functions.operator_function_code[operator][index])
                request.failed_function_index = int(index)
                request.error = str(error)
                self.failed_operator_publisher.publish(request)
            self.cycle_operator_indexes[operator] = index
            try:
                if c_new is None or not self.is_solutions_valid(c_new):
                    self.handle_invalid_solution(operator, index)
                    # return self.current_solution
                else:
                    return c_new, index
            except Exception as e:
                self.handle_operator_error(operator, index, e)
                # return self.current_solution

    def handle_invalid_solution(self, operator, index):
        """Handle invalid solution generated by operator."""
        if not any(existing[0] == index for existing in self.failed_operator_dict[operator]):
            self.failed_operator_dict[operator].append((index, "Invalid solution generated by operator"))
            request = FailedOperatorRequest()
            request.operator_name = str(operator)
            request.failed_function = str(self.operator_functions.operator_function_code[operator][index])
            request.failed_function_index = int(index)
            request.error = str("Invalid solution generated by operator")
            self.failed_operator_publisher.publish(request)

    def handle_operator_error(self, operator, index, error):
        """Handle errors during operator application."""
        print(f"Operator error {operator}: {error}")
        if not any(existing[0] == index for existing in self.failed_operator_dict[operator]):
            self.failed_operator_dict[operator].append((index, error))
            request = FailedOperatorRequest()
            request.operator_name = str(operator)
            request.failed_function = str(self.operator_functions.operator_function_code[operator][index])
            request.failed_function_index = int(index)
            request.error = str(error)
            self.failed_operator_publisher.publish(request)

    def calculate_gain(self, new_solution):
        """Calculate fitness gain from new solution."""
        return Fitness.fitness_function(new_solution, self.cost_matrix) - \
            Fitness.fitness_function(self.current_solution, self.cost_matrix)

    def update_experience_and_counters(self, condition, operator, gain, operator_index):
        """Update experience memory and attempt counters."""
        self.update_experience(condition, operator, gain, operator_index)
        if gain >= 0:  # Only count attempts if no improvement
            self.no_improvement_attempt_count += 1

    def update_best_solutions(self, new_solution):
        """Update local and coalition best solutions."""
        self.update_local_best(new_solution)
        self.update_coalition_best(new_solution)
        self.current_solution = new_solution

    def update_local_best(self, new_solution):
        """Update agent's local best solution."""
        current_fitness = Fitness.fitness_function(new_solution, self.cost_matrix)
        if (self.local_best_solution is None or
                current_fitness < Fitness.fitness_function(self.local_best_solution, self.cost_matrix)):
            self.local_best_solution = deepcopy(new_solution)
            self.best_local_improved = True
            self.no_improvement_attempt_count = 0

    def update_coalition_best(self, new_solution):
        """Update coalition best solution and publish if improved."""
        current_fitness = Fitness.fitness_function(new_solution, self.cost_matrix)
        if (self.coalition_best_solution is None or
                current_fitness < Fitness.fitness_function(self.coalition_best_solution, self.cost_matrix)):
            self.coalition_best_solution = deepcopy(new_solution)
            self.coalition_best_agent = self.agent_id
            self.best_coalition_improved = True
            self.publish_best_solution()

    def publish_best_solution(self):
        """Publish current best solution to the coalition."""
        solution = Solution()
        solution.id = self.agent_id
        solution.order = self.coalition_best_solution[0]
        solution.allocations = self.coalition_best_solution[1]
        self.solution_publisher.publish(solution)

    def handle_cycle_updates(self):
        """Handle DI and OE cycle updates and learning."""
        self.di_cycle_count += 1
        self.oe_cycle_count += 1

        if self.end_of_di_cycle(self.di_cycle_count):
            self.handle_di_cycle_operations()
        if self.end_of_oe_cycle(self.oe_cycle_count):
            self.handle_oe_cycle_operations()

    def perform_individual_learning(self):
        """Execute appropriate individual learning method."""
        learning_method_switch = {
            LearningMethod.FERREIRA: self.individual_learning_old,
            LearningMethod.Q_LEARNING: self.individual_learning
        }
        learning_function = learning_method_switch.get(LearningMethod(self.learning_method))
        if learning_function:
            self.weight_matrix.weights = learning_function()
        else:
            self.get_logger().error(f"Unknown learning method: {self.learning_method}")

    def publish_weight_matrix(self):
        """Publish current weight matrix to the coalition."""
        msg = Weights()
        msg_dict = self.weight_matrix.pack_weights(self.agent_id)
        msg.id = msg_dict["id"]
        msg.rows = msg_dict["rows"]
        msg.cols = msg_dict["cols"]
        msg.weights = msg_dict["weights"]
        self.weight_publisher.publish(msg)

    def generate_population(self):
        """
        Randomly generates a population of size `pop_size`
        :return: Population of solutions of size `pop_size`
        """
        # Generate initial population where each solution is a list of task allocations to agents
        population = []

        for _ in range(self.pop_size):
            # Create a list of task indexes and shuffle it for a random allocation
            allocation = list(range(self.num_tasks))
            random.shuffle(allocation)

            # Generate non-zero task counts for each agent that sum to number_tasks
            # Start with each agent assigned at least 1 task
            counts = [1] * self.num_tsp_agents
            for _ in range(self.num_tasks - self.num_tsp_agents):
                counts[random.randint(0, self.num_tsp_agents - 1)] += 1

            # Add both allocation and counts to the population
            population.append((allocation, counts))

        return population

    def select_solution(self):
        """
        Finds and returns the fittest solution
        :return: The fittest solution
        """
        # Select the best solution from the population based on fitness score
        best_solution = min(self.population, key=lambda sol: Fitness.fitness_function(
            sol, self.cost_matrix))  # Assuming lower score is better
        return best_solution

    def update_experience(self, condition, operator, gain, operator_index):
        """
        Adds details of the current iteration to the experience memory.
        :param condition: The previous condition
        :param operator: The operator applied
        :param gain: The resulting change in the current solution's fitness
        :return: None
        """
        self.previous_experience.append([condition, operator, gain, operator_index])
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

    def individual_learning(self):
        """
        Updates the weight matrix (Q(s, a)) using the Q-learning formula.
        :return: Updated weight matrix
        """
        for experience in self.previous_experience:
            condition, operator, gain, _ = experience

            # Current Q value
            current_q = self.weight_matrix.weights[condition.value][operator.value - 1]

            # Estimate future rewards (no explicit next_state, assume single-step Q-learning)
            max_next_q = max(
                self.weight_matrix.weights[condition.value])  # Max Q for current state (proxy for next state)

            # Q-learning update formula
            if gain > 0:
                self.reward = -0.5
            elif gain == 0:
                self.reward = 0
            else:
                self.reward = 1

            updated_q = current_q + self.lr * (self.reward + self.gamma_decay * max_next_q - current_q)

            # Update weight matrix
            self.weight_matrix.weights[condition.value][operator.value - 1] = updated_q

        return self.weight_matrix.weights

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
        # Define a stopping criterion (e.g., a fixed number of iterations)
        return iteration_count > self.num_iterations

    def end_of_di_cycle(self, cycle_count):
        if cycle_count >= self.di_cycle_length:
            return True
        return False

    def handle_di_cycle_operations(self):
        """Handle end of DI cycle operations."""
        if self.best_local_improved:
            self.perform_individual_learning()
            self.best_local_improved = False

        if self.best_coalition_improved:
            self.publish_weight_matrix()
            self.best_coalition_improved = False

        if self.received_weight_matrices:
            self.mimetism_learning(self.received_weight_matrices, self.rho)
            self.received_weight_matrices = []

        operator_info = [(sublist[1], sublist[3]) for sublist in
                         self.previous_experience]  # List of tuples (operator, index)
        gain = sum(sublist[2] for sublist in self.previous_experience)  # Sum of gains

        self.di_cycle_log.append([operator_info, gain])

        for operator in Operator:
            self.cycle_operator_indexes[operator] = -1
        self.previous_experience = []
        self.di_cycle_count = 0

    def end_of_oe_cycle(self, cycle_count):
        if cycle_count >= self.oe_cycle_length:
            return True
        return False

    def handle_oe_cycle_operations(self):
        """Handle end of OE cycle operations."""

        cycle_log = self.di_cycle_log

        # Extract the sequences and their corresponding gain values
        most_positive_entry = max(self.di_cycle_log, key=lambda x: x[1])  # Entry with the highest gain
        most_negative_entry = min(self.di_cycle_log, key=lambda x: x[1])  # Entry with the lowest gain

        # Extract sequences
        most_positive_sequence = most_positive_entry[0]  # Operator sequence for highest gain
        most_negative_sequence = most_negative_entry[0]  # Operator sequence for lowest gain
        pairs = self.find_comparable_pairs(most_positive_sequence, most_negative_sequence)
        if pairs:
            sampled_pair = random.sample(pairs, 1)[0]
            sampled_operator = sampled_pair[0]
            sampled_better_index = sampled_pair[1]
            sampled_worse_index = sampled_pair[2]
            request = OECrossoverRequest()
            request.better_function_code = str(self.operator_functions.operator_function_code[sampled_pair[0]][
                                                   sampled_pair[1]])
            request.worse_function_code = str(self.operator_functions.operator_function_code[sampled_pair[0]][
                                                  sampled_pair[2]])
            request.operator_name = str(sampled_pair[0])
            request.worse_function_index = int(sampled_pair[2])
            self.oe_crossover_publisher.publish(request)
        self.oe_cycle_count = 0

    def find_comparable_pairs(self, pos_sequence, neg_sequence):
        pos_dict = defaultdict(list)
        neg_dict = defaultdict(list)

        # Group indexes by operator type for positive sequence
        for operator, index in pos_sequence:
            pos_dict[operator].append(index)

        # Group indexes by operator type for negative sequence
        for operator, index in neg_sequence:
            neg_dict[operator].append(index)

        # Find operators appearing in both sequences with different indexes
        comparable_pairs = []
        for operator in pos_dict.keys() & neg_dict.keys():  # Intersection of operators
            for pos_index in set(pos_dict[operator]):
                for neg_index in set(neg_dict[operator]):
                    if pos_index != neg_index:  # Ensure different indexes
                        comparable_pairs.append((operator, neg_index, pos_index))

        return comparable_pairs  # Returns all valid pairs

    def select_random_solution(self):
        temp_solution = sample(population=self.population, k=1)[0]
        if temp_solution != self.current_solution:
            return temp_solution

    def is_solutions_valid(self, solution):
        """
        Validates that all solutions in the population meet the required constraints:
        - Each solution must have exactly two unpackable elements.
        - Each task is assigned exactly once across all agents.
        - The number of tasks assigned to each agent matches the specified distribution.
        - Tasks per agent should sum to the total number of tasks, and each element should be greater than 1.

        Parameters:
            population (list): List of solutions to validate. Each solution is a tuple of
                               (order_of_tasks, tasks_per_agent).

        Returns:
            bool: True if all solutions are valid, False otherwise.
        """
        # Ensure the solution has exactly two unpackable elements
        if not isinstance(solution, tuple) or len(solution) != 2:
            print(f"Invalid solution structure: {solution}")
            return False

        order_of_tasks, tasks_per_agent = solution

        # Ensure all elements in order_of_tasks are integers
        if not all(isinstance(task, int) for task in order_of_tasks):
            print(f"Invalid tasks (non-integer values) in solution: {order_of_tasks}")
            return False

        # Ensure all tasks are unique and within range
        if sorted(order_of_tasks) != list(range(self.num_tasks)):
            print(f"Invalid tasks in solution: {order_of_tasks}")
            return False

        # Validate task distribution among agents
        if not all(isinstance(task_count, int) for task_count in tasks_per_agent):
            print(f"Invalid tasks_per_agent (non-integer values): {tasks_per_agent}")
            return False

        if sum(tasks_per_agent) != self.num_tasks:
            print(f"Tasks per agent do not sum to the total number of tasks: {tasks_per_agent}")
            return False

        if any(task_count < 1 for task_count in tasks_per_agent):
            print(f"Each agent must have at least than 1 task: {tasks_per_agent}")
            return False

        start_idx = 0
        for task_count in tasks_per_agent:
            if not isinstance(start_idx, int) or not isinstance(task_count, int):
                print(f"Invalid slicing indices: start_idx={start_idx}, task_count={task_count}")
                return False

            assigned_tasks = order_of_tasks[start_idx:start_idx + task_count]

            # Check if assigned tasks match the required number
            if len(assigned_tasks) != task_count:
                print(f"Invalid task count for an agent: {assigned_tasks}")
                return False

            start_idx += task_count

        # Ensure no extra tasks remain unassigned
        if start_idx != len(order_of_tasks):
            print(f"Extra unassigned tasks detected: {order_of_tasks[start_idx:]}")
            return False

        return True

    # Callbacks

    def weight_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        received_weights = self.weight_matrix.unpack_weights(weights_msg=msg, agent_id=self.agent_id)
        if received_weights is not None:
            self.received_weight_matrices.append(received_weights)

    def solution_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        if msg is not None and self.coalition_best_solution is not None:
            solution = (msg.order, msg.allocations)
            if Fitness.fitness_function(solution, self.cost_matrix) < Fitness.fitness_function(
                    self.coalition_best_solution, self.cost_matrix):
                self.coalition_best_solution = solution
                self.coalition_best_agent = msg.id

    def generated_population_callback(self, msg):
        if msg is not None:
            # Extract data from the message into a dictionary of lists
            operator_population_data = {
                Operator.TWO_SWAP: list(msg.two_swap),
                Operator.ONE_MOVE: list(msg.one_move),
                Operator.BEST_COST_ROUTE_CROSSOVER: list(msg.best_cost_route_crossover),
                Operator.INTRA_DEPOT_REMOVAL: list(msg.intra_depot_removal),
                Operator.INTRA_DEPOT_SWAPPING: list(msg.intra_depot_swapping),
                Operator.SINGLE_ACTION_REROUTING: list(msg.single_action_rerouting)
            }
            self.operator_functions = OperatorFunctionsLLM(operator_population_data)
            for operator, failed_indexes in self.operator_functions.initial_failed_functions.items():
                print(f"Operator: {operator}")  # Print the key

                # Check if there are any failed indexes
                if failed_indexes:
                    for i in failed_indexes:
                        print(str(i))
                        request = FailedOperatorRequest()
                        request.operator_name = str(operator)
                        request.failed_function = str(self.operator_functions.operator_function_code[operator][i[0]])
                        request.failed_function_index = int(i[0])
                        request.error = str(i[1])
                        self.failed_operator_publisher.publish(request)
            # Timer for periodic execution of the run loop
            self.run_timer = self.create_timer(0.1, self.run_step)

    def failed_operator_response_callback(self, msg):
        operator_name = msg.operator_name
        enum_name = operator_name.split(".")[1]
        operator_enum = getattr(Operator, enum_name, None)
        failed_function_index = msg.failed_function_index
        new_function = msg.fixed_function
        success = self.operator_functions.load_new_function(operator_enum, new_function, failed_function_index)
        if success:
            self.failed_operator_dict[operator_enum] = [
                (i, e) for i, e in self.failed_operator_dict[operator_enum] if i != failed_function_index
            ]
        else:
            request = FailedOperatorRequest()
            request.operator_name = str(operator_enum)
            request.failed_function = str(self.operator_functions.operator_function_code[operator_enum][failed_function_index])
            request.failed_function_index = int(failed_function_index)
            request.error = str("Failed to compile.")
            self.failed_operator_publisher.publish(request)

    def oe_crossover_response_callback(self, msg):
        new_function_code = msg.crossover_function_code
        operator_name = msg.operator_name
        enum_name = operator_name.split(".")[1]
        operator_enum = getattr(Operator, enum_name, None)
        worse_function_index = msg.crossover_function_index
        success = self.operator_functions.load_new_function(operator_enum, new_function_code, worse_function_index)
        if success:
            self.failed_operator_dict[operator_enum] = [
                (i, e) for i, e in self.failed_operator_dict[operator_enum] if i != worse_function_index
            ]
        else:
            request = FailedOperatorRequest()
            request.operator_name = str(operator_enum)
            request.failed_function = str(self.operator_functions.operator_function_code[operator_enum][worse_function_index])
            request.failed_function_index = int(worse_function_index)
            request.error = str("Failed to compile.")
            self.failed_operator_publisher.publish(request)


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
    np.fill_diagonal(cost_matrix, 0)
    return cost_matrix


def main(args=None):
    try:
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
        agent = CBMPopulationAgentLLM(
            pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
            num_tasks=num_tasks, num_tsp_agents=20, num_iterations=1000,
            num_solution_attempts=20, agent_id=agent_id, node_name=node_name,
            cost_matrix=problem.cost_matrix, learning_method=learning_method
        )

        def shutdown_callback():
            agent.get_logger().info("CBM-POP agent Runtime completed. Shutting down.")
            agent.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

        agent.create_timer(runtime, shutdown_callback)

        try:
            rclpy.spin(agent)
        except KeyboardInterrupt:
            pass
        finally:
            if rclpy.ok():
                agent.destroy_node()
                rclpy.shutdown()

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))

        print(f"An error occurred:\n{traceback_details}", file=sys.stderr)

        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()
# Deneme

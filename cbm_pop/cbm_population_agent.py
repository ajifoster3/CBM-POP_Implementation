import random
import numpy as np
from copy import deepcopy
from random import sample
from cbm_pop.Condition import ConditionFunctions
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator import OperatorFunctions
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.Problem import Problem
from rclpy.node import Node
from std_msgs.msg import String, Float32
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
from cbm_pop_interfaces.msg import Solution, Weights

class CBMPopulationAgent(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_tasks, num_tsp_agents, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, cost_matrix):
        super().__init__(node_name)
        self.pop_size = pop_size
        self.eta = eta
        self.rho = rho
        self.di_cycle_length = di_cycle_length
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
        self.agent_ID = agent_id
        self.received_weight_matrices = []

        # Iteration state
        self.iteration_count = 0
        self.di_cycle_count = 0
        self.no_improvement_attempt_count = 0
        self.best_coalition_improved = False
        self.best_local_improved = False

        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(Solution, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10)
        self.weight_publisher = self.create_publisher(Weights, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            Weights, 'weight_matrix', self.weight_update_callback, 10)

        # Timer for periodic execution of the run loop
        self.run_timer = self.create_timer(0.1, self.run_step)


        #Q_learning
        # Q_learning parameter
        self.lr=0.1 # RL learning Rate
        self.reward=0 # RL reward initial 0
        self.new_reward=0 # RL tarafından secilecek. initial 0
        self.gamma_decay=0.99 # it can be change interms of iteration


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



    #----------------------------------------------------------------------------
    def individual_learning(self):
        """
        Updates the weight matrix (Q(s, a)) using the Q-learning formula.
        :return: Updated weight matrix
        """
        for experience in self.previous_experience:
            condition, operator, gain = experience

            # Current Q value
            current_q = self.weight_matrix.weights[condition.value][operator.value - 1]

            # Estimate future rewards (no explicit next_state, assume single-step Q-learning)
            max_next_q = max(
                self.weight_matrix.weights[condition.value])  # Max Q for current state (proxy for next state)

            # Q-learning update formula
            if gain>0:
                self.reward=2
            elif gain==0:
                self.reward=0
            else:
                self.reward=1

            updated_q = current_q + self.lr * (self.reward + self.gamma_decay * max_next_q - current_q)

            # Update weight matrix
            self.weight_matrix.weights[condition.value][operator.value - 1] = updated_q

        return self.weight_matrix.weights
    #------------------------------------------------------------------------------------------------


    def mimetism_learning(self, received_weights, rho):
        #mimetisim learnng will stay same
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
        return False  # Placeholder; replace with actual condition

    def weight_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        received_weights = self.weight_matrix.unpack_weights(weights_msg=msg, agent_id=self.agent_ID)
        if received_weights is not None:
            self.received_weight_matrices.append(received_weights)

    def solution_update_callback(self, msg):
        # Callback to process incoming weight matrix updates
        if msg is not None and self.coalition_best_solution is not None:
            solution = (msg.order, msg.allocations)
            if Fitness.fitness_function(solution, self.cost_matrix) > Fitness.fitness_function(
                        self.coalition_best_solution, self.cost_matrix):
                self.coalition_best_solution = solution
                self.coalition_best_agent = msg.id
        else:
            print("Received empty message")

    def select_random_solution(self):
        temp_solution = sample(population=self.population, k=1)[0]
        if temp_solution != self.current_solution:
            return temp_solution

    def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """
        if self.stopping_criterion(self.iteration_count):
            self.get_logger().info("Stopping criterion met. Shutting down.")
            self.run_timer.cancel()
            return

        condition = ConditionFunctions.perceive_condition(self.previous_experience)

        if self.no_improvement_attempt_count >= self.no_improvement_attempts:
            self.current_solution = self.select_random_solution()
            self.no_improvement_attempt_count = 0

        operator = OperatorFunctions.choose_operator(self.weight_matrix.weights, condition)
        c_new = OperatorFunctions.apply_op(
            operator,
            self.current_solution,
            self.population,
            self.cost_matrix
        )

        gain = Fitness.fitness_function(c_new, self.cost_matrix) - \
               Fitness.fitness_function(self.current_solution, self.cost_matrix)
        self.update_experience(condition, operator, gain)

        if self.local_best_solution is None or \
                Fitness.fitness_function(c_new, self.cost_matrix) < Fitness.fitness_function(
                    self.local_best_solution, self.cost_matrix):
            self.local_best_solution = deepcopy(c_new)
            self.best_local_improved = True
            self.no_improvement_attempt_count = 0
        else:
            self.no_improvement_attempt_count += 1

        if self.coalition_best_solution is None or \
                Fitness.fitness_function(c_new, self.cost_matrix) < Fitness.fitness_function(
                    self.coalition_best_solution, self.cost_matrix):
            self.coalition_best_solution = deepcopy(c_new)
            self.coalition_best_agent = self.agent_ID
            self.best_coalition_improved = True
            solution = Solution()
            solution.id = self.agent_ID
            solution.order = self.coalition_best_solution[0]
            solution.allocations = self.coalition_best_solution[1]
            self.solution_publisher.publish(solution)

        self.current_solution = c_new
        self.di_cycle_count += 1

        if self.end_of_di_cycle(self.di_cycle_count):
            if self.best_local_improved:
                self.weight_matrix.weights = self.individual_learning()
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
        self.get_logger().info(f"Iteration {self.iteration_count}: Current best solution fitness_QL = "
                               f"{Fitness.fitness_function(self.local_best_solution, self.cost_matrix)}")

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

def main(args=None):
    rclpy.init(args=args)

    node = Node("parameter_loader")
    node.declare_parameter("agent_id", 1)  # Default agent_id
    node.declare_parameter("problem_filename", "150_Task_Problem.csv")  # Default problem_filename

    agent_id = node.get_parameter("agent_id").value
    problem_filename = node.get_parameter("problem_filename").value
    node.destroy_node()  # Clean up the temporary node

    num_tasks = 150
    problem = Problem()
    problem.load_cost_matrix(problem_filename, "csv")
    num_tasks = len(problem.cost_matrix)

    # Create and run the agent node
    node_name = f"cbm_population_agent_{agent_id}"
    agent = CBMPopulationAgent(
        pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
        num_tasks=num_tasks, num_tsp_agents=5, num_iterations=1000,
        num_solution_attempts=20, agent_id=agent_id, node_name=node_name,
        cost_matrix=problem.cost_matrix
    )

    try:
        rclpy.spin(agent)  # Run the ROS2 executor
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
# Deneme
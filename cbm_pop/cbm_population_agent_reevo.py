import asyncio
import random
import numpy as np
from copy import deepcopy
from random import sample


from cbm_pop.Condition import ConditionFunctions
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator import OperatorFunctions
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.Problem import Problem
from cbm_pop_interfaces.msg import Solution, Weights
from rclpy.node import Node
from std_msgs.msg import String, Float32
import rclpy
from cbm_pop.reevo.PopulationGenerator import PopulationGenerator

class CBMPopulationAgentReevo(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_tasks, num_tsp_agents, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, cost_matrix):
        super().__init__(node_name)
        self.pop_size = pop_size
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.num_tasks = num_tasks
        self.num_tsp_agents = num_tsp_agents
        self.population_gen = PopulationGenerator()
        self.population = asyncio.run(self.population_gen.generate_population(self.pop_size))
        self.cost_matrix = cost_matrix
        self.agent_ID = agent_id


        # ROS publishers and subscribers


        # Timer for periodic execution of the run loop
        self.run_timer = self.create_timer(0.1, self.run_step)


    def generate_heuristic_population(self, population_size):
        """
        Use the Population Generator to generate a population of heuristics
        :return: Population of solutions of size `pop_size`
        """

        return self.population_gen.generate_population(population_size)


    def select_solution(self):
        """
        Finds and returns the fittest solution
        :return: The fittest solution
        """

    def update_experience(self, condition, operator, gain):
        """
        Adds details of the current iteration to the experience memory.
        :param condition: The previous condition
        :param operator: The operator applied
        :param gain: The resulting change in the current solution's fitness
        :return: None
        """

    def individual_learning(self):
        # Update weight matrix (if needed) based on learning (not fully implemented in this example)
        return None

    def mimetism_learning(self, received_weights, rho):
        """
        Perform mimetism learning by updating self.weight_matrix.weights using multiple sets of received weights.
        For each weight in each received set:
            w_a = (1 - rho) * w_a + rho * w_b
        :param received_weights: A 3D list (or array) of received weights. Each "slice" is a 2D matrix of weights.
        :param rho: The learning rate, a value between 0 and 1.
        :return: Updated weight matrix.
        """
        # Iterate over each set of received weights

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


    async def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """

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

async def ros_loop(node):
    """
    Async ROS spin loop.
    """
    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0)
        await asyncio.sleep(0.01)  # Small sleep to allow cooperative multitasking

def main(args=None):
    rclpy.init(args=args)

    node = Node("parameter_loader")
    node.declare_parameter("agent_id", 1)  # Default agent_id
    node.declare_parameter("problem_filename", "src/cbm_pop/150_Task_Problem.csv")  # Default problem_filename

    agent_id = node.get_parameter("agent_id").value
    problem_filename = node.get_parameter("problem_filename").value
    node.destroy_node()  # Clean up the temporary node

    num_tasks = 150
    problem = Problem()
    problem.load_cost_matrix(problem_filename, "csv")
    num_tasks = len(problem.cost_matrix)

    # Create and run the agent node
    node_name = f"cbm_population_agent_{agent_id}"
    agent = CBMPopulationAgentReevo(
        pop_size=3, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
        num_tasks=num_tasks, num_tsp_agents=5, num_iterations=1000,
        num_solution_attempts=20, agent_id=agent_id, node_name=node_name,
        cost_matrix=problem.cost_matrix
    )

    for sol in agent.population["ga_crossover"]:
        print(sol)

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        # Run the ROS loop alongside other async tasks
        loop.run_until_complete(ros_loop(agent))
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
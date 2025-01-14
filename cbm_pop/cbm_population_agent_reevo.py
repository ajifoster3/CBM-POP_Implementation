from future.standard_library import exclude_local_folder_imports
from rclpy.node import Node
from std_msgs.msg import String, Float32
import rclpy
import asyncio
import numpy as np
from random import sample
from cbm_pop.Fitness import Fitness
from cbm_pop.Problem import Problem
from cbm_pop.reevo.ShortTermReflector import ShortTermReflector
from cbm_pop.reevo.LongTermReflector import LongTermReflector
from cbm_pop.reevo.PopulationGenerator import PopulationGenerator
from cbm_pop.reevo.ElitistMutation import ElitistMutation
from cbm_pop.reevo.GeneticAlgorithm import GeneticAlgorithm
from cbm_pop.reevo.Crossover import Crossover
from cbm_pop.reevo import reevo_config


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
        self.longterm_reflector = LongTermReflector()

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

    def get_solution_fitnesses(self):
        solution_fitnesses = []
        for solution_ID in range(self.pop_size):
            genetic_algorithm = GeneticAlgorithm(20,
                                                 len(self.cost_matrix),
                                                 5,
                                                 self.population["ga_crossover"][solution_ID],
                                                 self.population["ga_mutation"][solution_ID],
                                                 self.cost_matrix)
            solution_fitnesses.append(genetic_algorithm.run_genetic_algorithm(100))
        return solution_fitnesses

    async def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """

    def perform_short_term_reflection(self, population, fitnesses):
        valid_indexes = [index for index, fitness in enumerate(fitnesses) if fitness != -1]

        sample_index_pair_list = []

        # Ensure there are at least two valid fitnesses to sample from
        if len(valid_indexes) >= 2:
            for i in range(5):  # Repeat the sampling process 5 times
                # Randomly sample two unique indexes
                sampled_indexes = sample(valid_indexes, 2)
                sample_index_pair_list.append(sampled_indexes)
        else:
            print("Not enough valid fitnesses to sample two unique indexes.")

        reflection_list = []

        for sample_index_pair in sample_index_pair_list:
            # Sort the indexes based on their fitness values in ascending order (lower is better)
            better_idx, worse_idx = sorted(sample_index_pair, key=lambda idx: fitnesses[idx])

            # Assign the corresponding population values
            better_code_1 = population[reevo_config.function_name["ga_crossover"]][better_idx]  # Lower fitness (better)
            worse_code_1 = population[reevo_config.function_name["ga_crossover"]][worse_idx]  # Higher fitness (worse)
            better_code_2 = population[reevo_config.function_name["ga_mutation"]][better_idx]  # Lower fitness (better)
            worse_code_2 = population[reevo_config.function_name["ga_mutation"]][worse_idx]  # Higher fitness (worse)

            short_term_reflector = ShortTermReflector()
            reflection = short_term_reflector.fetch_reflection(
                function_name_1=reevo_config.function_name["ga_crossover"],
                function_name_2=reevo_config.function_name["ga_mutation"],
                problem_description=reevo_config.problem_description["task_allocation"],
                function_description_1=reevo_config.function_description["ga_crossover"],
                function_description_2=reevo_config.function_description["ga_mutation"],
                worse_code_1=worse_code_1,
                better_code_1=better_code_1,
                worse_code_2=worse_code_2,
                better_code_2=better_code_2
            )
            reflection_list.append({"reflection": reflection, "better_id": better_idx, "worse_id": worse_idx})
        return reflection_list

    def perform_crossover(self, population, reflections):
        offspring_population = {"ga_crossover": [], "ga_mutation": []}
        for reflection in reflections:
            crossover = Crossover()
            offspring_crossover_operator = crossover.perform_crossover(
                function_name=reevo_config.function_name["ga_crossover"],
                task_description=reevo_config.problem_description["task_allocation"],
                function_signature0="heuristics_v0",
                worse_code=population["ga_crossover"][reflection["worse_id"]],
                function_signature1="heuristics_v1",
                better_code=population["ga_crossover"][reflection["better_id"]],
                shortterm_reflection=reflection["reflection"])
            offspring_population["ga_crossover"].append(offspring_crossover_operator)
            offspring_mutation_operator = crossover.perform_crossover(
                function_name=reevo_config.function_name["ga_mutation"],
                task_description=reevo_config.problem_description["task_allocation"],
                function_signature0="heuristics_v0",
                worse_code=population["ga_mutation"][reflection["worse_id"]],
                function_signature1="heuristics_v1",
                better_code=population["ga_mutation"][reflection["better_id"]],
                shortterm_reflection=reflection["reflection"])
            offspring_population["ga_mutation"].append(offspring_mutation_operator)
        return offspring_population

    def perform_longterm_reflection(self, shortterm_reflections):
        st_reflections = ''.join([entry['reflection'] for entry in shortterm_reflections])
        lt_reflector = LongTermReflector()
        lt_reflector_operator = lt_reflector.perform_longterm_reflection(st_reflections,
                                                                         reevo_config.problem_description[
                                                                             "task_allocation"])
        return lt_reflector_operator

    def perform_elitism_mutation(self, fitnesses, longterm_reflections):
        index_min = min(range(len(fitnesses)), key=fitnesses.__getitem__)
        best_fitness_crossover = self.population[reevo_config.function_name["ga_crossover"]][index_min]
        best_fitness_mutation = self.population[reevo_config.function_name["ga_mutation"]][index_min]
        elitism_mutation = ElitistMutation()

        crossover_offspring = []
        mutation_offspring = []

        for i in range(5):
            crossover_offspring.append(elitism_mutation.perform_elitism_mutation(reevo_config.function_name["ga_crossover"],
                                                             reevo_config.problem_description["task_allocation"],
                                                             longterm_reflections,
                                                             "ga_crossover_v1",
                                                             best_fitness_crossover))
            mutation_offspring.append(elitism_mutation.perform_elitism_mutation(reevo_config.function_name["ga_mutation"],
                                                             reevo_config.problem_description["task_allocation"],
                                                             longterm_reflections,
                                                             "ga_mutation_v1",
                                                             best_fitness_mutation))
        return {"ga_crossover": crossover_offspring, "ga_mutation": mutation_offspring}


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
        pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
        num_tasks=num_tasks, num_tsp_agents=5, num_iterations=1000,
        num_solution_attempts=20, agent_id=agent_id, node_name=node_name,
        cost_matrix=problem.cost_matrix
    )

    fitnesses = agent.get_solution_fitnesses()

    reflections = agent.perform_short_term_reflection(agent.population, fitnesses)

    # Current Choice: Evaluate both operators at once, crossover individually.
    crossover_offspring = agent.perform_crossover(agent.population, reflections)



    longterm_reflections = agent.perform_longterm_reflection(reflections)

    elitism_mutation_offspring = agent.perform_elitism_mutation(fitnesses, longterm_reflections)

    combined_offspring = {
        "ga_crossover": crossover_offspring["ga_crossover"] + elitism_mutation_offspring["ga_crossover"],
        "ga_mutation": crossover_offspring["ga_mutation"] + elitism_mutation_offspring["ga_mutation"]
    }

    for operator in combined_offspring:
        for offspring in combined_offspring[operator]:
            print(offspring)
            print()

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

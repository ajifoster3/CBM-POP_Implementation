import random
from cbm_pop.Problem import Problem
from cbm_pop.Fitness import Fitness
import signal


class TimeoutException(Exception):
    """Custom exception for signaling a timeout."""
    pass


def timeout_handler(signum, frame):
    """Handler function for the alarm signal."""
    raise TimeoutException("Function call timed out.")


class GeneticAlgorithm:
    def __init__(self,
                 population_size,
                 number_of_tasks,
                 number_of_agents,
                 combined_function_code,
                 cost_matrix):
        self.population_size = population_size
        self.number_of_tasks = number_of_tasks
        self.number_of_agents = number_of_agents

        combined_function_code = combined_function_code.replace("''' python\n", "").strip("'''")
        self.combined_function = self._load_combined_function(combined_function_code)
        self.initial_population = self.generate_initial_population()
        self.cost_matrix = cost_matrix
        self.is_operators_invalid = False

    def run_genetic_algorithm(self, number_of_iterations):
        population = self.initial_population
        initial_fittest = min(
            Fitness.fitness_function(solution, cost_matrix=self.cost_matrix) for solution in population)
        for i in range(number_of_iterations):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(2)
            try:
                population = self.ga_iteration(population)
                signal.alarm(0)  # Cancel the alarm if the function finishes
            except TimeoutException:
                print(f"Iteration {i} of ga_iteration timed out after {2} seconds.")
                return -1

            if self.is_operators_invalid:
                return -1
        return initial_fittest - min(
            Fitness.fitness_function(solution, cost_matrix=self.cost_matrix) for solution in population)

    def ga_iteration(self, population):
        if self.is_operators_invalid:
            return -1
        population.sort(key=lambda solution: Fitness.fitness_function(solution, cost_matrix=self.cost_matrix))
        elitism_selected_solutions = population[:int(self.population_size * 0.2)]
        crossover_selected_solutions = population[:int(self.population_size * 0.5)]
        generated_solutions = []
        for i in range(len(population) - len(elitism_selected_solutions)):
            sampled_solutions = random.sample(crossover_selected_solutions, 2)
            try:
                child_solution = self.perform_combined(sampled_solutions[0], sampled_solutions[1], self.cost_matrix)
            except Exception as e:
                self.is_operators_invalid = True
                break
            generated_solutions.append(child_solution)
        generated_solutions.extend(elitism_selected_solutions)
        population = generated_solutions
        try:
            population.sort(key=lambda solution: Fitness.fitness_function(solution, cost_matrix=self.cost_matrix))
        except:
            self.is_operators_invalid = True
        return population

    def _load_combined_function(self, code):
        """
        Dynamically loads Python functions from a code string.
        """
        try:
            local_namespace = {}
            exec(code, globals(), local_namespace)
            # Ensure all required functions are loaded
            combined_function = local_namespace.get('ga_combined_v2')

            if not combined_function:
                raise ValueError("Failed to load one or more required functions.")

            return combined_function
        except Exception as e:
            print(f"Failed to load functions: {e}")
            self.is_operators_invalid = True
            return -1, -1, -1

    def generate_initial_population(self):
        population = []
        for _ in range(self.population_size):
            population.append(self.generate_solution())
        return population

    def generate_solution(self):
        order_of_tasks = list(range(self.number_of_tasks))
        random.shuffle(order_of_tasks)
        tasks_per_agent = self.generate_non_zero_integers_with_sum()
        return order_of_tasks, tasks_per_agent

    def generate_non_zero_integers_with_sum(self):
        """
        Generates `x` non-zero integers whose sum is `y`.
        Parameters:
            self.number_of_tasks (int): Number of integers.
            self.number_of_agents (int): Desired sum of the integers.

        Returns:
            list: A list of `x` non-zero integers whose sum is `y`.
        """
        if self.number_of_agents <= 0:
            raise ValueError("number_of_tasks must be greater than 0.")
        if self.number_of_tasks < self.number_of_agents:
            raise ValueError("number_of_agents must be less than the number_of_tasks to ensure non-zero integers.")

        # Start with x ones to ensure all numbers are non-zero
        nums = [1] * self.number_of_agents
        remaining = self.number_of_tasks - self.number_of_agents  # Remaining sum to distribute

        # Distribute the remaining sum randomly
        for i in range(remaining):
            import random
            nums[random.randint(0, self.number_of_agents - 1)] += 1

        return nums

    def perform_combined(self, parent1, parent2, cost_matrix):
        try:
            return self.combined_function(parent1, parent2, cost_matrix)
        except ValueError:
            print(ValueError)
            self.is_operators_invalid = True

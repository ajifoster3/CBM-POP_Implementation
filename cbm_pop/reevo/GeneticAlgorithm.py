import random
from cbm_pop.Problem import Problem

class GeneticAlgorithm:
    def __init__(self, population_size, number_of_tasks, number_of_agents, crossover_function_code, mutation_function_code):
        self.population_size = population_size
        self.number_of_tasks = number_of_tasks
        self.number_of_agents = number_of_agents
        self.crossover_function = self._load_crossover_function(crossover_function_code)
        self.mutation_function = self._load_mutation_function(mutation_function_code)

    def _load_crossover_function(self, code):
        """
        Dynamically loads a Python function from a code string.
        """
        local_namespace = {}
        exec(code, globals(), local_namespace)
        return local_namespace['ga_crossover_v2']

    def _load_mutation_function(self, code):
        """
        Dynamically loads a Python function from a code string.
        """
        local_namespace = {}
        exec(code, globals(), local_namespace)
        return local_namespace['ga_mutation_v2']

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

    def perform_crossover(self, parent1, parent2, cost_matrix):
        return self.crossover_function(parent1, parent2, cost_matrix)

    def perform_mutation(self, parent, cost_matrix):
        return self.mutation_function(parent, cost_matrix)

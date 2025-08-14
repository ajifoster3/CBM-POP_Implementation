import random
from copy import deepcopy
from enum import Enum
import numpy as np

from cbm_pop.Fitness import Fitness
from cbm_pop.Operator import Operator


class OperatorFunctions:
    # Define operator function map with static method references
    operator_function_map = {
        Operator.TWO_SWAP: lambda current_solution, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix: OperatorFunctions.two_swap(
            current_solution,
            cost_matrix,
            robot_cost_matrix, inital_robot_cost_matrix),
        Operator.ONE_MOVE: lambda current_solution, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix: OperatorFunctions.one_move(
            current_solution,
            cost_matrix,
            robot_cost_matrix, inital_robot_cost_matrix),
        Operator.BEST_COST_ROUTE_CROSSOVER: lambda current_solution,
                                                   population,
                                                   cost_matrix,
                                                   robot_cost_matrix, inital_robot_cost_matrix: OperatorFunctions.best_cost_route_crossover(
            current_solution, population, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix),
        Operator.INTRA_DEPOT_REMOVAL: lambda current_solution: OperatorFunctions.intra_depot_removal(current_solution),
        Operator.INTRA_DEPOT_SWAPPING: lambda current_solution: OperatorFunctions.intra_depot_swapping(
            current_solution),
        # Operator.INTER_DEPOT_SWAPPING: lambda current_solution: OperatorFunctions.inter_depot_swapping(
        #    current_solution),
        Operator.SINGLE_ACTION_REROUTING: lambda current_solution, cost_matrix,
                                                 robot_cost_matrix, inital_robot_cost_matrix: OperatorFunctions.single_action_rerouting(
            current_solution, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix),
    }

    @staticmethod
    def choose_operator(weights, condition):
        """
        Stochastically choose an operator for a condition using the weights.
        :param weights: Weights
        :param condition: Current condition
        :return: The chosen operator
        """
        # Choose an operator (e.g., mutation, crossover) based on weight matrix W and current state
        # For simplicity, we only apply a mutation operator in this example
        row = weights[condition.value]
        operators = list(Operator)

        # Randomly select an operator based on weights in `row`
        chosen_operator = random.choices(operators, weights=row, k=1)[0]
        return chosen_operator

    @staticmethod
    def choose_operator_annealed(
            weights,
            condition,
            *,
            step: int,
            anneal_horizon: int = 200,
            T0: float = 2.0,
            T1: float = 0.4,
            mask=None,  # optional multiplicative mask (e.g., classic gating)
            epsilon: float = 1e-6,  # tiny floor so ops never die completely
            rng: "np.random.Generator | None" = None,
            return_debug: bool = False
    ):
        """
        Pick an operator with softmax over weights at temperature T(step).

        T(step) = T0 + (T1 - T0) * min(1, step / anneal_horizon).
        """

        def _to_index(x) -> int:
            # Handle Enum instances and classes, numpy scalar ints, and plain ints.
            if hasattr(x, "value"):  # Enum instance
                x = x.value
            # If someone passed the Enum *class* by accident, fail loudly.
            if isinstance(x, type) and hasattr(x, "__members__"):
                raise TypeError("condition should be an instance (e.g., Condition.FOO), not the Enum class")
            # NumPy scalar to Python int (also handles np.int64, etc.)
            if isinstance(x, np.generic):
                return int(x)
            if isinstance(x, bool):
                return int(x)
            if isinstance(x, (int, np.integer)):
                return int(x)
            raise TypeError(f"condition index must be int-like, got {type(x).__name__}: {x!r}")

        cond_idx = _to_index(condition)

        row = np.asarray(weights[cond_idx], dtype=float)
        operators = list(Operator)

        if row.shape[0] != len(operators):
            raise ValueError(
                f"Weight row has length {row.shape[0]} but there are {len(operators)} operators. "
                "Ensure your weight matrix columns match Operator enum size/order."
            )

        # Optional gating mask
        if mask is not None:
            m = np.asarray(mask, dtype=float)
            if m.shape != row.shape:
                raise ValueError("mask length must match number of operators")
            row = row * m

        # Replace non-finite with 0, then apply tiny floor so nothing is exactly zero
        row = np.where(np.isfinite(row), row, 0.0)
        row = np.maximum(row, epsilon)

        # Temperature schedule
        progress = min(1.0, float(step) / float(max(1, anneal_horizon)))
        T = T0 + (T1 - T0) * progress
        T = max(T, 1e-6)

        # Stable softmax on row/T
        logits = row / T
        logits -= logits.max()
        exp = np.exp(logits)
        Z = exp.sum()
        # Guard against degenerate cases (shouldn't happen after epsilon + stability, but cheap)
        if not np.isfinite(Z) or Z <= 0.0:
            probs = np.full_like(exp, 1.0 / exp.size)
        else:
            probs = exp / Z

        # Sample
        if rng is None:
            rng = np.random.default_rng()
        idx = int(rng.choice(len(probs), p=probs))
        op = operators[idx]

        if return_debug:
            return op, probs.tolist(), T
        return op

    @staticmethod
    def apply_op(operator, current_solution, population, cost_matrix=None, robot_cost_matrix=None,
                 inital_robot_cost_matrix=None):
        """
        Apply the operator to the current solution and return the newly generated child solution.
        :param cost_matrix: Cost matrix to calculate the cost
        :param operator: The operator to be applied
        :param current_solution: The current solution
        :param population: The population
        :return: A child solution
        """
        # Get the function based on the operator
        current_copy = deepcopy(current_solution)
        if operator in OperatorFunctions.operator_function_map:
            # Call the function and pass arguments as needed
            if operator == Operator.BEST_COST_ROUTE_CROSSOVER:
                return OperatorFunctions.operator_function_map[operator](current_copy, population, cost_matrix,
                                                                         robot_cost_matrix, inital_robot_cost_matrix)
            elif (operator == Operator.SINGLE_ACTION_REROUTING
                  or operator == Operator.TWO_SWAP
                  or operator == Operator.ONE_MOVE):
                return OperatorFunctions.operator_function_map[operator](current_copy, cost_matrix,
                                                                         robot_cost_matrix, inital_robot_cost_matrix)
            else:
                return OperatorFunctions.operator_function_map[operator](current_copy)

        # Raise an exception if the operator is not recognized
        raise Exception("Something went wrong! The selected operation doesn't exist.")

    # Diversifiers
    @staticmethod
    def best_cost_route_crossover(current_solution, population, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix):
        """
        "For two parent chromosomes, select a route to be removed
        from each. The removed nodes are inserted into the
        other parent solution at the best insertion cost."
        At the moment it selects the best solution in P
        other than the current solution.
        :param cost_matrix: The cost matrix
        :param population: Population
        :param current_solution: The current solution as a parent
        :return: A child solution
        """
        # Find the fittest solution in P that is not current_solution
        fittest_non_current_solution = min(
            (sol for sol in population if sol != current_solution),
            key=lambda sol: Fitness.fitness_function_robot_pose(sol, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix)
        )

        # Randomly select a path (route) from fittest_non_current_solution
        task_order, agent_task_counts = fittest_non_current_solution
        selected_agent = random.randint(0, len(agent_task_counts) - 1)

        # Identify the start and end indices for the selected agent's path
        start_index = sum(agent_task_counts[:selected_agent])
        end_index = start_index + agent_task_counts[selected_agent]

        # Extract the path for the selected agent
        selected_path = task_order[start_index:end_index]

        # Create a copy of current_solution to modify
        new_solution_task_order, new_solution_task_counts = deepcopy(current_solution)

        temp_count_counter = 0
        temp_task_counter = 0
        # For each agent
        for i in new_solution_task_counts:
            # For each task for that agent
            for j in range(i):
                if new_solution_task_order[temp_task_counter] in selected_path:
                    new_solution_task_order.remove(new_solution_task_order[temp_task_counter])
                    new_solution_task_counts[temp_count_counter] -= 1
                    temp_task_counter -= 1
                temp_task_counter += 1
            temp_count_counter += 1

        for task in selected_path[:]:  # Use a copy of selected_path to iterate safely
            OperatorFunctions.find_best_task_position(cost_matrix, new_solution_task_counts, new_solution_task_order,
                                                      task, robot_cost_matrix, inital_robot_cost_matrix)

        # Return the modified solution as the child solution
        return new_solution_task_order, new_solution_task_counts  # TODO: new_solution_task_counts came out as [6,6] for a 10 task problem

    @staticmethod
    def find_best_task_position(cost_matrix, new_solution_task_counts, new_solution_task_order, task,
                                robot_cost_matrix, inital_robot_cost_matrix):
        best_fitness = float('inf')
        best_position = 0
        best_agent = 0
        # Try inserting the task at each position in the task order
        for agent_index, count in enumerate(new_solution_task_counts):
            # Calculate the insertion range for this agent
            agent_start_index = sum(new_solution_task_counts[:agent_index])
            agent_end_index = agent_start_index + count

            # Try inserting within this agent's range
            for pos in range(agent_start_index, agent_end_index + 1):
                temp_order = new_solution_task_order[:]
                temp_order.insert(pos, task)

                # Update task counts temporarily for fitness calculation
                temp_counts = new_solution_task_counts[:]
                temp_counts[agent_index] += 1

                # Calculate fitness with this temporary insertion
                temp_fitness = Fitness.fitness_function_robot_pose((temp_order, temp_counts), cost_matrix,
                                                                   robot_cost_matrix, inital_robot_cost_matrix)

                # If the new fitness is better, update the best fitness, position, and agent
                if temp_fitness < best_fitness:
                    best_fitness = temp_fitness
                    best_position = pos
                    best_agent = agent_index
        # Insert the task at the best position found and update the task count for that agent
        new_solution_task_order.insert(best_position, task)
        new_solution_task_counts[best_agent] += 1

    @staticmethod
    def intra_depot_removal(current_solution):
        """
        "Two cut-points in the chromosome associated with the
        robot initial position are selected and the genetic
        material between these two cut-points is reversed."
        :param current_solution: The current solution to be mutated
        :return: A child solution
        """
        # Extract task order and agent task counts from current solution
        task_order, agent_task_counts = deepcopy(current_solution)

        # Randomly select an agent
        selected_agent = random.randint(0, len(agent_task_counts) - 1)

        # Calculate the start and end index for this agent's route
        start_index = sum(agent_task_counts[:selected_agent])
        end_index = start_index + agent_task_counts[selected_agent]

        # Perform the reversal mutation if the agent has enough tasks
        if end_index - start_index > 1:  # Ensure there are enough tasks to reverse a section
            # Randomly choose two cut points within this range
            cut1, cut2 = sorted(random.sample(range(start_index, end_index), 2))

            # Reverse the section between the two cut points
            task_order[cut1:cut2 + 1] = reversed(task_order[cut1:cut2 + 1])
        return task_order, agent_task_counts

    @staticmethod
    def intra_depot_swapping(current_solution):
        """
        Perform an intra-depot mutation by selecting two random routes
        from the solution and moving a randomly selected task from
        one route to another.
        :param current_solution: The current solution to be mutated
        :return: A mutated child solution
        """
        # Extract task order and agent task counts from the current solution
        task_order, agent_task_counts = deepcopy(current_solution)

        # Randomly select two distinct agents (routes) to swap between
        agent1, agent2 = random.sample(range(len(agent_task_counts)), 2)

        # Determine the task range for each agent
        start_index1 = sum(agent_task_counts[:agent1]) + 1
        end_index1 = start_index1 + agent_task_counts[agent1] - 1

        start_index2 = sum(agent_task_counts[:agent2])
        end_index2 = start_index2 + agent_task_counts[agent2]

        # Ensure the selected agent has tasks to swap
        if end_index1 > start_index1:
            # Randomly select a task from agent1's route
            task_index = random.randint(start_index1, end_index1 - 1)
            task = task_order.pop(task_index)
            agent_task_counts[agent1] -= 1

            # Insert the task into a random position in agent2's route
            if end_index2 > start_index2:
                insert_position = random.randint(start_index2, end_index2)
            else:
                insert_position = start_index2

            task_order.insert(insert_position, task)
            agent_task_counts[agent2] += 1

        # Return the modified solution
        return task_order, agent_task_counts

    @staticmethod
    def inter_depot_swapping(current_solution):
        """
        "Mutation of swapping nodes in the routes of different initial
        positions. Candidates for this mutation are nodes that are in
        similar proximity to more than one initial position."

        Only applicable with Depots. Not necessary yet, Ignore

        :param current_solution: The current solution to be mutated
        :return: A child solution
        """

        return current_solution

    @staticmethod
    def single_action_rerouting(current_solution, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix):
        """
        "Re-routing involves randomly selecting one action and removing
        it from the existing route. The action is then inserted at the
        best feasible insertion point within the entire chromosome."

        This selects a random task and inserts it in the best position

        :param cost_matrix: The cost matrix
        :param current_solution: The current solution to be mutated
        :return: A modified solution with improved fitness
        """
        # Deep copy to avoid modifying the original solution
        task_order, agent_task_counts = deepcopy(current_solution)

        # Select a random action (task) from the entire task order
        if not task_order:
            return current_solution  # Return unchanged if task_order is empty

        # Randomly select an action to remove
        task_index = random.randint(0, len(task_order) - 1)
        task = task_order.pop(task_index)

        # Adjust task count for the agent that lost the task
        agent_index = next(i for i, count in enumerate(agent_task_counts) if
                           sum(agent_task_counts[:i]) <= task_index < sum(agent_task_counts[:i + 1]))
        agent_task_counts[agent_index] -= 1

        OperatorFunctions.find_best_task_position(cost_matrix, agent_task_counts, task_order,
                                                  task, robot_cost_matrix, inital_robot_cost_matrix)
        return task_order, agent_task_counts

    # Intensifiers

    # I can't tell the difference between ^ single_action_rerouting and one_move
    # Single action rerouting: randomly select and action and insert at best place in chromosome
    # One move: Remove a node and insert at position that maximises fitness
    # Maybe we don't need single action rerouting either.

    @staticmethod
    def one_move(current_solution, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix):
        """
        "Removal of a node from the solution and insertion at the point
        that maximizes solution fitness"

        Move a task to a new position such that the move has the greatest increase in fitness
        Requires calculating the fitness of moving all tasks to all positions

        :param cost_matrix: The cost matrix
        :param current_solution: The current solution to be optimised
        :return: A child solution
        """
        # Deep copy to avoid modifying the original solution
        # Deep copy to avoid modifying the original solution
        task_order, agent_task_counts = deepcopy(current_solution)

        # Initialize variables to track the best task movement
        best_fitness = float('inf')
        best_task_index = None
        best_position = None
        best_agent = None
        original_agent = None

        # Loop through each task in the task order to consider moving it
        for task_index, task in enumerate(task_order):
            # Copy the current task order and remove the task from the current position
            temp_order = task_order[:]
            removed_task = temp_order.pop(task_index)

            # Identify the agent from which the task is removed
            agent_index = next(i for i, count in enumerate(agent_task_counts) if
                               sum(agent_task_counts[:i]) <= task_index < sum(agent_task_counts[:i + 1]))
            temp_counts = agent_task_counts[:]
            temp_counts[agent_index] -= 1  # Temporarily reduce task count for the removal

            # Try inserting the removed task at every possible position for each agent
            for i, count in enumerate(temp_counts):
                start_index = sum(temp_counts[:i])
                end_index = start_index + count

                # Test inserting the task in each possible position within this agent's range
                for pos in range(start_index, end_index + 1):
                    # Make a temporary copy of the order and insert the task
                    temp_order_with_insertion = temp_order[:]
                    temp_order_with_insertion.insert(pos, removed_task)

                    # Update counts for fitness calculation
                    temp_counts_with_insertion = temp_counts[:]
                    temp_counts_with_insertion[i] += 1

                    # Calculate fitness
                    temp_fitness = Fitness.fitness_function_robot_pose(
                        (temp_order_with_insertion, temp_counts_with_insertion),
                        cost_matrix, robot_cost_matrix, inital_robot_cost_matrix)

                    # Check if this move yields a better fitness
                    if temp_fitness < best_fitness:
                        best_fitness = temp_fitness
                        best_task_index = task_index
                        best_position = pos
                        best_agent = i
                        original_agent = agent_index  # Track the original agent

        # Apply the best move identified
        if best_task_index is not None:
            # Remove the task from its original position
            task_to_move = task_order.pop(best_task_index)

            # Update task counts: decrement for original agent, increment for new agent
            agent_task_counts[original_agent] -= 1
            agent_task_counts[best_agent] += 1

            # Insert the task at the new best position
            task_order.insert(best_position, task_to_move)

        return task_order, agent_task_counts

    @staticmethod
    def two_swap(current_solution, cost_matrix, robot_cost_matrix, inital_robot_cost_matrix):
        """
        "Swapping two pairs of subsequent tasks (each pair as a unit) from two different agents
        to improve solution fitness by minimizing traversal cost."

        :param current_solution: The current solution to be optimized
        :param cost_matrix: The matrix used to calculate traversal costs
        :return: A child solution with improved fitness
        """
        # Deep copy to avoid modifying the original solution
        task_order, agent_task_counts = deepcopy(current_solution)

        # Initialize variables to track the best pair swap
        best_fitness = float('inf')
        best_swap = None  # Tuple of (first_agent, first_pair_start, second_agent, second_pair_start)

        # Identify the start and end indices of tasks for each agent
        start_index = 0
        agent_task_ranges = []
        for count in agent_task_counts:
            agent_task_ranges.append((start_index, start_index + count - 1))
            start_index += count

        # Loop through each pair of agents to consider swapping task pairs
        for agent1, (start1, end1) in enumerate(agent_task_ranges):
            # Skip if agent1 has fewer than 2 tasks
            if end1 <= start1:
                continue

            for agent2, (start2, end2) in enumerate(agent_task_ranges):
                # Skip if agent2 has fewer than 2 tasks or if it's the same agent
                if agent1 >= agent2 or end2 <= start2:
                    continue

                # Generate all possible pairs of adjacent tasks for each agent
                for i in range(start1, end1):
                    if i + 1 > end1:
                        continue  # Ensure we have a valid pair in agent1
                    for j in range(start2, end2):
                        if j + 1 > end2:
                            continue  # Ensure we have a valid pair in agent2

                        # Make a temporary copy of task_order to apply the swap
                        temp_order = task_order[:]

                        # Swap the pairs: (task[i], task[i+1]) with (task[j], task[j+1])
                        temp_order[i], temp_order[i + 1], temp_order[j], temp_order[j + 1] = (
                            temp_order[j], temp_order[j + 1], temp_order[i], temp_order[i + 1]
                        )

                        # Calculate the fitness after the swap
                        temp_fitness = Fitness.fitness_function_robot_pose((temp_order, agent_task_counts), cost_matrix,
                                                                           robot_cost_matrix, inital_robot_cost_matrix)

                        # Track the best swap if it improves fitness
                        if temp_fitness < best_fitness:
                            best_fitness = temp_fitness
                            best_swap = (i, i + 1, j, j + 1)

        # Apply the best swap identified
        if best_swap is not None:
            i, i_next, j, j_next = best_swap
            task_order[i], task_order[i_next], task_order[j], task_order[j_next] = (
                task_order[j], task_order[j_next], task_order[i], task_order[i_next]
            )

        return task_order, agent_task_counts

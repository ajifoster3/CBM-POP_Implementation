import math
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
        #Operator.ONE_MOVE_GRASP: lambda current_solution, cost_matrix, robot_cost_matrix,
        #                          inital_robot_cost_matrix, grasp_alpha: OperatorFunctions.one_move_grasp(
        #    current_solution,
        #    cost_matrix,
        #    robot_cost_matrix, inital_robot_cost_matrix,
        #    alpha=grasp_alpha),
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
        # Choose an operator (e.g., mutation, crossover) based on weight matrix W and current state
        # For simplicity, we only apply a mutation operator in this example
        row = weights[condition.value]
        operators = list(Operator)
        # Randomly select an operator based on weights in `row`
        chosen_operator = random.choices(operators, weights=row, k=1)[0]
        return chosen_operator

    @staticmethod
    def choose_operator_ucb(weights, condition, operator_usage_counts, exploration_constant: float = 1.0):
        # Resolve the row index from the condition; support Enum or numeric types
        cond_idx = condition.value if hasattr(condition, 'value') else int(condition)
        row = weights[cond_idx]
        operators = list(Operator)

        # Initialise counts for unseen operators
        for op in operators:
            operator_usage_counts.setdefault(op, 0)

        total_usage = sum(operator_usage_counts.values())

        # If no operator has been used yet, fall back to uniform random choice
        if total_usage == 0:
            return random.choice(operators)

        ucb_scores = []
        for idx, op in enumerate(operators):
            avg_reward = row[idx]
            count = operator_usage_counts[op]

            if avg_reward == 0.0:
                # Force score to zero if weight is zero
                score = 0
            else:
                # Normal UCB calculation
                if count == 0:
                    bonus = float('inf')
                else:
                    bonus = exploration_constant * math.sqrt(math.log(total_usage) / count)
                score = avg_reward + bonus

            ucb_scores.append(score)

        best_index = int(np.argmax(ucb_scores))
        return operators[best_index]

    @staticmethod
    def apply_op(operator, current_solution, population, cost_matrix=None, robot_cost_matrix=None,
                 inital_robot_cost_matrix=None, grasp_alpha=None):
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
            #elif operator == Operator.ONE_MOVE_GRASP:
            #    return OperatorFunctions.one_move_grasp(
            #        current_copy,
            #        cost_matrix,
            #        robot_cost_matrix,
            #        inital_robot_cost_matrix,
            #        alpha=grasp_alpha  # <<< inject adaptive alpha
            #    )
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
        print(f"new solutions: {new_solution_task_order}")
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
    def one_move_grasp(current_solution,
                       cost_matrix,
                       robot_cost_matrix,
                       inital_robot_cost_matrix,
                       alpha: float = 0,
                       rng=None,
                       task_sample_ratio: float = 1.0,
                       position_sample_ratio: float = 1.0,
                       allow_equal: bool = False,
                       allow_non_improving: bool = False):
        """
        GRASP-style one-move relocation:
          1) Generate candidate single-task relocations across all agents/positions.
          2) Keep 'near-best' improving moves in an RCL (alpha controls greediness).
          3) Pick one at random from the RCL and apply it.

        Args:
            alpha (0..1): RCL greediness. 0→purely best; 1→broad (more random).
            rng: random generator with .random(), .shuffle(), .choice(); defaults to Python's random.
            task_sample_ratio (0..1]: randomly subsample tasks for speed/randomness.
            position_sample_ratio (0..1]: randomly subsample insertion positions per agent.
            allow_equal: treat Δ=0 as acceptable improvement.
            allow_non_improving: if no improving moves exist, build RCL from all moves (may worsen).

        Returns:
            (task_order, agent_task_counts) possibly modified; unchanged if no move selected.
        """
        from copy import deepcopy
        import random
        print("Applying one move grasp")
        if rng is None:
            rng = random

        task_order, agent_task_counts = deepcopy(current_solution)

        # Current fitness baseline
        current_fitness = Fitness.fitness_function_robot_pose(
            (task_order, agent_task_counts),
            cost_matrix, robot_cost_matrix, inital_robot_cost_matrix
        )

        N = len(task_order)

        # Cumulative boundaries per agent on the *current* flattened order
        cum = [0]
        for c in agent_task_counts:
            cum.append(cum[-1] + c)

        # Optional task subsampling
        task_idxs = list(range(N))
        rng.shuffle(task_idxs)
        if 0 < task_sample_ratio < 1.0:
            keep = max(1, int(round(task_sample_ratio * N)))
            task_idxs = task_idxs[:keep]

        candidates = []  # entries: (delta, task_index, insert_pos, new_agent, original_agent)

        for t_idx in task_idxs:
            temp_order = task_order[:]
            removed_task = temp_order.pop(t_idx)

            # Find original agent by boundaries on the original order
            orig_agent = next(i for i in range(len(agent_task_counts))
                              if cum[i] <= t_idx < cum[i + 1])

            temp_counts = agent_task_counts[:]
            temp_counts[orig_agent] -= 1

            # New boundaries after removal
            temp_cum = [0]
            for c in temp_counts:
                temp_cum.append(temp_cum[-1] + c)

            for ag, cnt in enumerate(temp_counts):
                start = temp_cum[ag]
                end = start + cnt  # insertion allowed in [start, end]

                positions = list(range(start, end + 1))
                rng.shuffle(positions)
                if 0 < position_sample_ratio < 1.0:
                    keep_p = max(1, int(round(position_sample_ratio * len(positions))))
                    positions = positions[:keep_p]

                for pos in positions:
                    order_ins = temp_order[:]
                    order_ins.insert(pos, removed_task)

                    counts_ins = temp_counts[:]
                    counts_ins[ag] += 1

                    new_fit = Fitness.fitness_function_robot_pose(
                        (order_ins, counts_ins),
                        cost_matrix, robot_cost_matrix, inital_robot_cost_matrix
                    )
                    delta = new_fit - current_fitness  # <0 improves

                    # Keep all candidates; filtering to improving happens below
                    candidates.append((delta, t_idx, pos, ag, orig_agent))

        if not candidates:
            return task_order, agent_task_counts

        # Improving filter
        if allow_equal:
            improving = [c for c in candidates if c[0] <= 0]
        else:
            improving = [c for c in candidates if c[0] < 0]

        pool = improving if improving else (candidates if allow_non_improving else [])
        if not pool:
            # No acceptable move; keep solution
            return task_order, agent_task_counts

        # Build RCL: moves with delta <= best + alpha*(worst - best)  (minimization)
        best_delta = min(pool, key=lambda x: x[0])[0]
        worst_delta = max(pool, key=lambda x: x[0])[0]
        threshold = best_delta + alpha * (worst_delta - best_delta)
        rcl = [c for c in pool if c[0] <= threshold]

        # Random choice from RCL
        delta, t_idx, pos, new_ag, orig_ag = rng.choice(rcl)

        # Apply move: pop at t_idx, then insert at pos on the shortened list (same indexing logic as eval)
        moved_task = task_order.pop(t_idx)
        agent_task_counts[orig_ag] -= 1
        agent_task_counts[new_ag] += 1
        task_order.insert(pos, moved_task)

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

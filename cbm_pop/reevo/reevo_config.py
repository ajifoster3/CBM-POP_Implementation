prompts = dict(
    generator_system_prompt =
    '''
    You are an expert in the domain of heuristics. Your task is to design heuristics that can 
    effectively solve optimization problems.
    Your response outputs Python code and nothing else.
    Format your code as a Python code string : \"\''' python ... \'''\".
    ''',
    reflector_system_prompt =
    '''
    You are an expert in the domain of optimization heuristics. Your task is to give hints to design better heuristics.
    ''',
    task_description =
    '''
    Write a {function_name} function for {problem_description}
    {function_description}
    ''',
    user_prompt_population_initialisation =
    '''
    {task_description}
    
    {seed_function}
    
    Refer to the format of a trivial design above. Be very creative and give '{function_name}_o{operator_index}'.
    Output code only and enclose your code with Python code block: \''' python ... \'''.
    
    {initial_longterm_reflection}
    ''',
    user_prompt_shortterm_reflection =
    '''
    Below are two {function_name} functions for {problem_description}
    {function_name}:
    {function_description}
    
    Your are provided with two code versions below, where the second version performs better than the first one.
    
    [Worse {function_name} code]
    {worse_code}
    
    [Better {function_name} code]
    {better_code}
    
    You respond with some hints for designing better heuristics, based on the two code versions and using less than
    40 words.
    ''',
    user_prompt_shortterm_reflection_on_blackbox_COPs =
    '''
    Below are two {function_name} functions for {problem_description}
    {function_description}
    
    Your are provided with two code versions below, where the second version performs better than the first one.
    
    [Worse code]
    {worse_code}
    
    [Better code]
    {better_code}
    
    Please infer the problem settings by comparing the two code versions and giving hints for designing better
    heuristics. You may give hints about how edge and node attributes correlate with the black-box objective value.
    Use less than 50 words.
    ''',
    user_prompt_crossover =
    '''
    {task_description}
    
    [Worse code]
    {function_signature0}
    {worse_code}
    
    [Better code]
    {function_signature1}
    {better_code}
    
    [Reflection]
    {shortterm_reflection}
    
    [improved code]
    Please write an improved function '{function_name}_v2', according to the reflection. Output code only and 
    enclose your code with Python code block: \''' python ... \'''.
    ''',
    user_prompt_longterm_reflection =
    '''
    Below is your prior long-term reflection on designing heuristics for {problem_description}
    {prior_longterm_reflection}
    
    Below are some newly gained insights.
    {new_shortterm_reflection}
    
    Write constructive hints for designing better heuristics, based on prior reflections and new insights and using 
    less than 50 words
    ''',
    user_prompt_elitist_mutation =
    '''
    {task_description}
    
    [prior reflection]
    {longterm_reflection}
    
    [Code]
    {function_signature1}
    {elitist_code}
    
    [Improved code]
    Please write a mutation function '{function_name}_v2', according to the reflection. Output code only and enclose 
    your code with Python code block: \''' python ... \'''.
    '''
)
function_name = dict(
    two_swap="two_swap",
    one_move="one_move",
    best_cost_route_crossover="best_cost_route_crossover",
    intra_depot_removal="intra_depot_removal",
    intra_depot_swapping="intra_depot_swapping",
    single_action_rerouting="single_action_rerouting",

)
function_description = dict(
    ga_crossover =
    """
    "
    Performs a Crossover operation on the Chromosome.
    This must ensure the chromosome is valid.
    A valid chromosome is made up of two components: the order of tasks, and the number of tasks per agent.
    All tasks should appear in the order of tasks once exactly. The number of tasks per agent equal the size of the 
    order of tasks.
    You're also given access to the cost_matrix, where element [i,j] is the cost of edge i->j.
    "
    """,
    ga_mutation =
    """
    "
    Performs a Mutation operation on the Chromosome.
    This must ensure the chromosome is valid."
    A valid chromosome is made up of two components: the order of tasks, and the number of tasks per agent.
    All tasks should appear in the order of tasks once exactly. The number of tasks per agent equal the size of the 
    order of tasks.
    You're also given access to the cost_matrix, where element [i,j] is the cost of edge i->j.
    "
    """,
    ga_combined =
    """
    GA combined: 
    Controls crossover and mutation.
    GA crossover:
    Performs a Crossover operation on the Chromosome.
    This must ensure the chromosome is valid.
    A valid chromosome is made up of two components: the order of tasks, and the number of tasks per agent.
    All tasks should appear in the order of tasks once exactly. The number of tasks per agent equal the size of the 
    order of tasks.
    You're also given access to the cost_matrix, where element [i,j] is the cost of edge i->j.
    GA mutation:
    Performs a Mutation operation on the Chromosome.
    This must ensure the chromosome is valid."
    A valid chromosome is made up of two components: the order of tasks, and the number of tasks per agent.
    All tasks should appear in the order of tasks once exactly. The number of tasks per agent equal the size of the 
    order of tasks.
    You're also given access to the cost_matrix, where element [i,j] is the cost of edge i->j.
    """,
    two_swap=
    """
    "Swapping two pairs of subsequent tasks (each pair as a unit) from two different agents
        to improve solution fitness by minimizing traversal cost."

        :param current_solution: The current solution to be optimized
        :param cost_matrix: The matrix used to calculate traversal costs
        :return: A child solution with improved fitness
    """,
    one_move=
    """
    "Removal of a node from the solution and insertion at the point
        that maximizes solution fitness"

        Move a task to a new position such that the move has the greatest increase in fitness
        Requires calculating the fitness of moving all tasks to all positions

        :param cost_matrix: The cost matrix
        :param current_solution: The current solution to be optimised
        :return: A child solution
    """,
    best_cost_route_crossover=
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
    """,
    intra_depot_removal=
    """
    "Two cut-points in the chromosome associated with the
        robot initial position are selected and the genetic
        material between these two cut-points is reversed."
        :param current_solution: The current solution to be mutated
        :return: A child solution
    """,
    intra_depot_swapping=
    """
    Perform an intra-depot mutation by selecting two random routes
        from the solution and moving a randomly selected task from
        one route to another.
        :param current_solution: The current solution to be mutated
        :return: A mutated child solution
    """,
    single_action_rerouting=
    """
    "Re-routing involves randomly selecting one action and removing
        it from the existing route. The action is then inserted at the
        best feasible insertion point within the entire chromosome."

        This selects a random task and inserts it in the best position

        :param cost_matrix: The cost matrix
        :param current_solution: The current solution to be mutated
        :return: A modified solution with improved fitness
    """,
)
seed_function = dict(
    ga_crossover =
    """
    def ga_crossover(parent1, parent2, cost_matrix):
        # Unpack parents
        order1, tasks1 = parent1
        order2, tasks2 = parent2
    
        # Partially Mapped Crossover (PMX) for Order of Tasks
        size = len(order1)
        start, end = sorted(random.sample(range(size), 2))
        child_order = [None] * size
    
        # Copy the segment from parent1
        child_order[start:end] = order1[start:end]
    
        # Fill remaining tasks from parent2
        mapping = {order1[i]: order2[i] for i in range(start, end)}
        for i in range(size):
            if child_order[i] is None:
                task = order2[i]
                while task in mapping:
                    task = mapping[task]
                child_order[i] = task
    
        # Single-Point Crossover for Tasks per Robot
        split_point = random.randint(1, len(tasks1) - 1)
        child_tasks = tasks1[:split_point] + tasks2[split_point:]
    
        # Ensure total task count is correct
        total_tasks = sum(tasks1)
        adjustment = total_tasks - sum(child_tasks)
        if adjustment != 0:
            for i in range(len(child_tasks)):
                if child_tasks[i] + adjustment >= 1:  # Ensure at least 1 task per robot
                    child_tasks[i] += adjustment
                    break
    
        # Return the offspring
        return (child_order, child_tasks)
    """,
    ga_mutation =
    """
    def ga_mutation(chromosome, cost_matrix):
        # Unpack the chromosome
        order_of_tasks, tasks_per_robot = chromosome
    
        # Swap Mutation for Order of Tasks
        idx1, idx2 = random.sample(range(len(order_of_tasks)), 2)
        order_of_tasks[idx1], order_of_tasks[idx2] = order_of_tasks[idx2], order_of_tasks[idx1]
    
        # Adjust Tasks per Robot
        num_robots = len(tasks_per_robot)
        robot_idx = random.randint(0, num_robots - 1)
    
        if random.random() < 0.5:  # Increase tasks for the selected robot
            if tasks_per_robot[robot_idx] < sum(tasks_per_robot) - num_robots + 1:
                tasks_per_robot[robot_idx] += 1
                # Ensure the change is balanced
                another_idx = random.choice([i for i in range(num_robots) if i != robot_idx and tasks_per_robot[i] > 1])
                tasks_per_robot[another_idx] -= 1
        else:  # Decrease tasks for the selected robot
            if tasks_per_robot[robot_idx] > 1:
                tasks_per_robot[robot_idx] -= 1
                # Ensure the change is balanced
                another_idx = random.choice([i for i in range(num_robots) if i != robot_idx])
                tasks_per_robot[another_idx] += 1
    
        # Return the mutated chromosome
        return (order_of_tasks, tasks_per_robot)
    """,
    ga_combined =
    """
        
    def ga_combined(parent1, parent2, cost_matrix):
        def ga_mutation(chromosome, cost_matrix):
            # Unpack the chromosome
            order_of_tasks, tasks_per_robot = chromosome
        
            # Swap Mutation for Order of Tasks
            idx1, idx2 = random.sample(range(len(order_of_tasks)), 2)
            order_of_tasks[idx1], order_of_tasks[idx2] = order_of_tasks[idx2], order_of_tasks[idx1]
        
            # Adjust Tasks per Robot
            num_robots = len(tasks_per_robot)
            robot_idx = random.randint(0, num_robots - 1)
        
            if random.random() < 0.5:  # Increase tasks for the selected robot
                if tasks_per_robot[robot_idx] < sum(tasks_per_robot) - num_robots + 1:
                    tasks_per_robot[robot_idx] += 1
                    # Ensure the change is balanced
                    another_idx = random.choice([i for i in range(num_robots) if i != robot_idx and tasks_per_robot[i] > 1])
                    tasks_per_robot[another_idx] -= 1
            else:  # Decrease tasks for the selected robot
                if tasks_per_robot[robot_idx] > 1:
                    tasks_per_robot[robot_idx] -= 1
                    # Ensure the change is balanced
                    another_idx = random.choice([i for i in range(num_robots) if i != robot_idx])
                    tasks_per_robot[another_idx] += 1
        
            # Return the mutated chromosome
            return (order_of_tasks, tasks_per_robot)
        
        def ga_crossover(parent1, parent2, cost_matrix):
            # Unpack parents
            order1, tasks1 = parent1
            order2, tasks2 = parent2
        
            # Partially Mapped Crossover (PMX) for Order of Tasks
            size = len(order1)
            start, end = sorted(random.sample(range(size), 2))
            child_order = [None] * size
        
            # Copy the segment from parent1
            child_order[start:end] = order1[start:end]
        
            # Fill remaining tasks from parent2
            mapping = {order1[i]: order2[i] for i in range(start, end)}
            for i in range(size):
                if child_order[i] is None:
                    task = order2[i]
                    while task in mapping:
                        task = mapping[task]
                    child_order[i] = task
        
            # Single-Point Crossover for Tasks per Robot
            split_point = random.randint(1, len(tasks1) - 1)
            child_tasks = tasks1[:split_point] + tasks2[split_point:]
        
            # Ensure total task count is correct
            total_tasks = sum(tasks1)
            adjustment = total_tasks - sum(child_tasks)
            if adjustment != 0:
                for i in range(len(child_tasks)):
                    if child_tasks[i] + adjustment >= 1:  # Ensure at least 1 task per robot
                        child_tasks[i] += adjustment
                        break
        
            # Return the offspring
            return (child_order, child_tasks)
            
        # Perform crossover
        offspring = ga_crossover(parent1, parent2, cost_matrix)
    
        # Perform mutation
        offspring = ga_mutation(offspring, cost_matrix)
    
        return offspring
    """,
    two_swap=
    """
    def two_swap(current_solution, cost_matrix):
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
                        temp_fitness = Fitness.fitness_function((temp_order, agent_task_counts), cost_matrix)

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
    """,
    one_move=
    """
    def one_move(current_solution, cost_matrix):
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
                    temp_fitness = Fitness.fitness_function((temp_order_with_insertion, temp_counts_with_insertion),
                                                            cost_matrix)

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
    """,
    best_cost_route_crossover=
    """
    def best_cost_route_crossover(current_solution, population, cost_matrix):
        def find_best_task_position(cost_matrix, new_solution_task_counts, new_solution_task_order, task):
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
                    temp_fitness = Fitness.fitness_function((temp_order, temp_counts), cost_matrix)

                    # If the new fitness is better, update the best fitness, position, and agent
                    if temp_fitness < best_fitness:
                        best_fitness = temp_fitness
                        best_position = pos
                        best_agent = agent_index
            # Insert the task at the best position found and update the task count for that agent
            new_solution_task_order.insert(best_position, task)
            new_solution_task_counts[best_agent] += 1
        # Find the fittest solution in P that is not current_solution
        fittest_non_current_solution = min(
            (sol for sol in population if sol != current_solution),
            key=lambda sol: Fitness.fitness_function(sol, cost_matrix)
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
            find_best_task_position(cost_matrix, new_solution_task_counts, new_solution_task_order,
                                                      task)

        # Return the modified solution as the child solution
        return new_solution_task_order, new_solution_task_counts # TODO: new_solution_task_counts came out as [6,6] for a 10 task problem

    """,
    intra_depot_removal=
    """
    def intra_depot_removal(current_solution):
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
    """,
    intra_depot_swapping=
    """
    def intra_depot_swapping(current_solution):
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
    """,
    single_action_rerouting=
    """
    def single_action_rerouting(current_solution, cost_matrix):
        def find_best_task_position(cost_matrix, new_solution_task_counts, new_solution_task_order, task):
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
                    temp_fitness = Fitness.fitness_function((temp_order, temp_counts), cost_matrix)

                    # If the new fitness is better, update the best fitness, position, and agent
                    if temp_fitness < best_fitness:
                        best_fitness = temp_fitness
                        best_position = pos
                        best_agent = agent_index
            # Insert the task at the best position found and update the task count for that agent
            new_solution_task_order.insert(best_position, task)
            new_solution_task_counts[best_agent] += 1
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

        find_best_task_position(cost_matrix, agent_task_counts, task_order,
                                                  task)
        return task_order, agent_task_counts
    """,
)
problem_description = dict(
    task_allocation =
    """
    A team of agents must visit all tasks in a set. Traversing between tasks has a cost. 
    This cost is to be minimised. The costs are stored in a cost matrix.
    """
)


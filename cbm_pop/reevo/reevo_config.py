prompts = dict(
    generator_system_prompt =
    '''
    You are an expert in the domain of heuristics. Your task is to design heuristics that can 
    effectively solve optimization problems.
    Your response outputs Python code and nothing else.
    Format your code as a Python code string : \"\''' ... \'''\".
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
    
    Refer to the format of a trivial design above. Be very creative and give '{function_name}_v2'.
    Output code only and enclose your code with Python code block: \''' ... \'''.
    
    {initial_longterm_reflection}
    ''',
    user_prompt_shortterm_reflection =
    '''
    Below are two {function_name} functions for {problem_description}
    {function_description}
    
    Your are provided with two code versions below, where the second version performs better than the first one.
    
    [Worse code]
    {worse_code}
    
    [Better code]
    {better_code}
    
    You respond with some hints for designing better heuristics, based on the two code versions and using less than
    20 words.
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
    enclose your code with Python code block: \''' ... \'''.
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
    your code twith Python code block: \''' ... \'''.
    '''
)
function_name = dict(
    ga_crossover = "ga_crossover",
    ga_mutation = "ga_mutation"
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
    """
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
    """
)
problem_description = dict(
    task_allocation =
    """
    A team of agents must visit all tasks in a set. Traversing between tasks has a cost. 
    This cost is to be minimised. The costs are stored in a cost matrix.
    """
)


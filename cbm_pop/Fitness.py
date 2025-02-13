
class Fitness:

    @staticmethod
    def fitness_function(solution, cost_matrix):
        """
        Returns the maximum path cost of any agent in the solution using the cost matrix.
        :param solution: Solution to be calculated
        :param cost_matrix: Cost matrix to calculate with
        :return: The maximum path cost among all agents
        """
        # Extract the task order and task count per agent
        task_order, agent_task_counts = solution

        # Track the maximum cost of any agent's path
        max_cost = 0

        counter = 0
        for j in agent_task_counts:
            agent_cost = 0  # Cost of the current agent's path
            for i in range(counter, counter + j - 1):
                task_i = task_order[i]
                task_j = task_order[i + 1]
                agent_cost += cost_matrix[task_i][task_j]

            # Update the max cost encountered
            max_cost = max(max_cost, agent_cost)

            counter += j

        return max_cost

    @staticmethod
    def fitness_function_robot_pose(solution, cost_matrix, robot_cost_matrix):
        """
        Returns the maximum path cost of any agent in the solution using the cost matrix.
        :param solution: Solution to be calculated
        :param cost_matrix: Cost matrix to calculate with
        :param robot_cost_matrix: Cost matrix from robot positions to tasks
        :return: The maximum path cost among all agents
        """
        # Extract the task order from the solution
        task_order, agent_task_counts = solution

        # Track the maximum cost of any agent's path
        max_cost = 0

        # Sum up the costs for the assigned tasks in the task order
        counter = 0
        for agent_idx, task_count in enumerate(agent_task_counts):
            if task_count > 0:
                agent_cost = robot_cost_matrix[agent_idx][task_order[counter]]  # Cost from robot to first task

                for i in range(counter, counter + task_count - 1):
                    task_i = task_order[i]
                    task_j = task_order[i + 1]
                    agent_cost += cost_matrix[task_i][task_j]

                # Update the max cost encountered
                max_cost = max(max_cost, agent_cost)

            counter += task_count

        return max_cost

import sys
import traceback


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
    def fitness_function_robot_pose(solution, cost_matrix, robot_cost_matrix, initial_robot_cost_matrix, alpha=0.5):
        """
        Computes a weighted sum of the maximum path cost and the average path cost for agents in the solution.
        :param solution: Solution to be calculated (task_order, agent_task_counts)
        :param cost_matrix: Cost matrix to calculate task-to-task costs
        :param robot_cost_matrix: Cost matrix from robot positions to tasks
        :param initial_robot_cost_matrix: Cost matrix from tasks to initial robot positions
        :param alpha: Weighting factor between max cost and average cost (0 <= alpha <= 1)
        :return: Weighted sum of max cost and average cost
        """
        try:
            task_order, agent_task_counts = solution
            max_cost = 0
            total_cost = 0
            num_agents_with_tasks = 0

            counter = 0
            for agent_idx, task_count in enumerate(agent_task_counts):
                if task_count > 0:

                    agent_cost = robot_cost_matrix[agent_idx][task_order[counter]]  # Cost from robot to first task

                    for i in range(counter, counter + task_count - 1):
                        task_i = task_order[i]
                        task_j = task_order[i + 1]
                        agent_cost += cost_matrix[task_i][task_j]

                    agent_cost += initial_robot_cost_matrix[agent_idx][task_order[counter + task_count - 1]]

                    max_cost = max(max_cost, agent_cost)
                    total_cost += agent_cost
                    num_agents_with_tasks += 1

                counter += task_count

            avg_cost = total_cost / num_agents_with_tasks if num_agents_with_tasks > 0 else 0
            weighted_cost = alpha * max_cost + (1 - alpha) * avg_cost

            return weighted_cost
        except Exception as e:
            # error_message = (
            #     f"[ERROR] Exception in fitness_function_robot_pose:\n"
            #     f"    Error Type: {type(e).__name__}\n"
            #     f"    Error Message: {e}\n"
            #     f"    Exception Traceback:\n{traceback.format_exc()}\n"
            #     f"    Full Stack Trace:\n{''.join(traceback.format_stack())}")
            #
            # print(error_message)
            # print(f"Solution {solution}")
            # print(f"robot_cost_matrix {robot_cost_matrix}")
            return 99999999
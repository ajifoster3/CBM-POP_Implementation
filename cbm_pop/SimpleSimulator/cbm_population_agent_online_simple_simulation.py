import json
import math
import random
import sys
import traceback
from time import time
import matplotlib.pyplot as plt
import numpy as np
from builtin_interfaces.msg import Time
from matplotlib import animation, patches
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
from copy import deepcopy
from random import sample
from cbm_pop.Condition import ConditionFunctions, Condition
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator_Fuctions import OperatorFunctions
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.Problem import Problem
from rclpy.node import Node
from std_msgs.msg import String, Float32
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
from cbm_pop_interfaces.msg import Solution, Weights, EnvironmentalRepresentation, SimplePosition, FinishedCoverage
from enum import Enum
from math import radians, cos, sin, asin, sqrt
from std_msgs.msg import Bool

class LearningMethod(Enum):
    FERREIRA = "Ferreira_et_al."
    Q_LEARNING = "Q-Learning"


class CBMPopulationAgentOnlineSimpleSimulation(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, learning_method,
                 lr = None,
                 gamma_decay = None,
                 positive_reward = None,
                 negative_reward = None,
                 num_tsp_agents = 5):

        """
        Initialises the agent on startup
        """
        super().__init__(node_name)
        self.current_task = None
        self.task_poses = [(i + 0.5, j + 0.5) for i in range(15) for j in range(15)]

        self.num_tasks = len(self.task_poses)
        self.is_generating = False
        self.pop_size = pop_size
        self.eta = eta
        self.rho = rho
        self.di_cycle_length = di_cycle_length
        self.num_iterations = num_iterations
        self.num_tsp_agents = num_tsp_agents
        self.agent_best_solution = None
        self.coalition_best_solution = None
        self.local_best_solution = None
        self.coalition_best_agent = None
        self.num_intensifiers = 2
        self.num_diversifiers = 4
        self.population = None
        self.weight_matrix = WeightMatrix(self.num_intensifiers, self.num_diversifiers)
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_ID = agent_id  # This will change as teamsize changes
        self.true_agent_ID = self.agent_ID  # This is permanent
        self.received_weight_matrices = []

        if isinstance(learning_method, str):
            try:
                learning_method = LearningMethod(learning_method)
            except ValueError:
                raise ValueError(
                    f"Invalid learning method '{learning_method}'. Must be one of: "
                    f"{[e.value for e in LearningMethod]}"
                )
        self.learning_method = learning_method

        self.new_robot_cost_matrix = None
        self.is_new_robot_cost_matrix = False
        self.last_env_rep_timestamps = {}

        # Iteration state
        self.iteration_count = 0
        self.di_cycle_count = 0
        self.no_improvement_attempt_count = 0
        self.best_coalition_improved = False
        self.best_local_improved = False

        # Runtime data
        self.initial_robot_poses = [None] * self.num_tsp_agents
        self.robot_poses = [None] * self.num_tsp_agents
        self.current_solution = None
        self.is_covered = [False] * self.num_tasks
        self.cost_matrix = self.calculate_cost_matrix()
        self.robot_cost_matrix = [None] * self.num_tsp_agents
        self.robot_initial_pose_cost_matrix = [None] * self.num_tsp_agents
        self.last_purge_agent_true_id = None
        self.is_agent_tobe_purged = False
        self.failed_agents = [False] * self.num_tsp_agents
        self.purged_agents = [False] * self.num_tsp_agents
        self.agent_to_revive = None
        self.am_i_failed = False
        self.ros_timer = None
        self.agent_timeouts = [False] * self.num_tsp_agents
        self.finished_robots = [False] * self.num_tsp_agents

        self.cb_group = ReentrantCallbackGroup()
        self.me_cb_group = MutuallyExclusiveCallbackGroup()

        # ROS publishers and subscribers
        self.solution_publisher = self.create_publisher(Solution, 'best_solution', 10)
        self.solution_subscriber = self.create_subscription(
            Solution, 'best_solution', self.solution_update_callback, 10)
        self.weight_publisher = self.create_publisher(Weights, 'weight_matrix', 10)
        self.weight_subscriber = self.create_subscription(
            Weights, 'weight_matrix', self.weight_update_callback, 10)
        # Subscriber to the global position
        # List to store subscriptions
        self.global_pose_subscribers = []

        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id}/global_pose'

            # Wrap the callback to include agent_id
            sub = self.create_subscription(
                SimplePosition,
                topic,
                lambda msg, agent=id: self.global_pose_callback(msg),
                10,
                callback_group=self.cb_group
            )

            self.global_pose_subscribers.append(sub)  # Store subscription to prevent garbage collection

        # Publisher for goal position
        self.goal_pose_publisher = self.create_publisher(
            SimplePosition,
            f'/central_control/uas_{agent_id}/goal_pose',
            10)

        self.kill_robot_subscribers = []

        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id + 1}/kill_robot'

            # Wrap the callback to include agent_id
            sub = self.create_subscription(
                Bool,
                topic,
                lambda msg, agent=(id + 1): self.kill_robot_callback(msg, agent),
                10
            )

            self.kill_robot_subscribers.append(sub)  # Store subscription to prevent garbage collection

        # Wrap the callback to include agent_id
        self.revive_robot_sub = self.create_subscription(
            Bool,
            f'/central_control/uas_{agent_id}/revive_robot',
            self.revive_robot_callback,
            10
        )

        self.finished_coverage_pub = self.create_publisher(
            FinishedCoverage,
            f'/central_control/finished_coverage',
            10)

        self.finished_coverage_sub = self.create_subscription(
            FinishedCoverage,
            f'/central_control/finished_coverage',
            self.finished_coverage_callback,
            10
        )

        # Timer for periodic execution of the run loop
        self.run_goal_publisher_timer = self.create_timer(0.5, self.publish_goal_pose, callback_group=self.cb_group)
        self.run_cost_matrix_recalculation = self.create_timer(0.25, self.robot_cost_matrix_recalculation,
                                                               callback_group=self.me_cb_group)

        self.environmental_representation_subscriber = self.create_subscription(
            EnvironmentalRepresentation,
            '/environmental_representation',
            self.environmental_representation_callback,
            10,
            callback_group=self.cb_group
        )

        self.environmental_representation_publisher = self.create_publisher(
            EnvironmentalRepresentation,
            '/environmental_representation',
            10
        )

        self.environmental_representation_timer = self.create_timer(2, self.environmental_representation_timer_callback,
                                                                    callback_group=self.cb_group)

        self.solution_publisher_timer = self.create_timer(2, self.regular_solution_publish_timer,
                                                          callback_group=self.cb_group)

        self.create_timer(5, self.check_stale_agents)

        # Q_learning
        # Q_learning parameter
        self.lr = lr  # RL learning Rate
        self.reward = 0  # RL reward initial 0
        self.new_reward = 0  # RL tarafından secilecek. initial 0
        self.gamma_decay = gamma_decay  # it can be change interms of iteration
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

        self.run_timer = None
        self.is_loop_started = False
        self.task_covered = -1
        self.is_new_task_covered = False
        '''
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        # Inside __init__
        self.fig, self.ax = plt.subplots()
        self.im = None  # placeholder for the cost matrix image
        self._plt_lock = threading.Lock()
        '''


    '''def _start_cost_plot(self):
        """Start the cost matrix animation thread."""
        self.ani = animation.FuncAnimation(self.fig, self._update_cost_plot, interval=1000)
        plt.show()
    '''

    '''def _update_cost_plot(self, _):
        """Update plot with latest robot cost matrix."""
        with self._plt_lock:
            if self.robot_cost_matrix is None or not isinstance(self.robot_cost_matrix, np.ndarray):
                return

            self.ax.clear()
            self.ax.set_title(f"Robot Cost Matrix (Agent {self.agent_ID})")
            im = self.ax.imshow(self.robot_cost_matrix, cmap='viridis', interpolation='nearest')
            self.ax.set_xlabel("Tasks")
            self.ax.set_ylabel("Robots")

            # Draw a rectangle to outline the current task cell
            if hasattr(self, 'current_task') and self.current_task is not None:
                rect = patches.Rectangle(
                    (self.current_task - 0.5, self.agent_ID - 0.5),  # (x, y) in data coords
                    1, 1, linewidth=2, edgecolor='red', facecolor='none'
                )
                self.ax.add_patch(rect)
    '''

    def calculate_cost_matrix(self):
        """
        Returns a cost matrix representing the traversal cost from each task_pose to each other task_pose, this is
        constructed by calculating the drone_distance between all the task poses in the agents task_pose list.
        """
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    # Extract positions
                    x1, y1 = self.task_poses[i][0], self.task_poses[i][1]
                    x2, y2 = self.task_poses[j][0], self.task_poses[j][1]

                    # Compute horizontal and vertical distances
                    x_dist = abs(x2 - x1)
                    y_dist = abs(y2 - y1)
                    # Compute cost using trajectory generation function
                    cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))

                    cost_map[i][j] = cost
        return cost_map

    def calculate_robot_cost_matrix(self):
        """
        Returns a cost map representing the traversal cost from each robot_pose to each task_pose calculated using
        drone_distance.
        """
        # Filter out None values from robot_poses
        valid_robot_poses = [pose for pose in self.robot_poses if pose is not None]
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        # Initialize cost matrix
        cost_map = np.zeros((num_robots, num_tasks))

        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                # Extract positions
                x1, y1 = robot_pose
                x2, y2 = task_pose

                # Compute horizontal and vertical distances
                x_dist = x2 - x1
                y_dist = y2 - y1
                # Compute cost using trajectory generation function
                cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))

                cost_map[i][j] = cost

        return cost_map

    def calculate_robot_inital_pose_cost_matrix(self):
        """
        Returns a cost map representing the traversal cost from each initial_robot_pose to each each task_pose calculated
        using drone_distance.
        """

        # Filter out None values from robot_poses

        valid_robot_poses = self.initial_robot_poses
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        # Initialize cost matrix
        cost_map = np.zeros((num_robots, num_tasks))

        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                # Extract positions
                x1, y1 = robot_pose
                x2, y2 = task_pose

                # Compute horizontal and vertical distances
                x_dist = x2 - x1
                y_dist = y2 - y1
                # Compute cost using trajectory generation function
                cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))

                cost_map[i][j] = cost
        self.robot_initial_pose_cost_matrix = cost_map

    def kill_robot_callback(self, msg, failed_agent_true_id):
        """
        This Fails a robot.
        Upon receiving a kill_robot_callback message, the agent checks if it's the agent to be killed.
        If not: it does nothing.
        If it is: it sets itself as "failed", sets its goal pose as its current position, and clears its current task.
        """
        print(f"Kill signal received for agent: {failed_agent_true_id}")
        if failed_agent_true_id == self.true_agent_ID:
            self.am_i_failed = True
            if msg.data:
                goal_pose = SimplePosition()


                # Set position (Latitude, Longitude, Altitude)
                goal_pose.x_position = self.robot_poses[self.true_agent_ID][0]
                goal_pose.y_position = self.robot_poses[self.true_agent_ID][1]
                self.goal_pose_publisher.publish(goal_pose)

                self.current_task = None
        # else:
        #   self.is_agent_tobe_purged = True
        #   self.last_purge_agent_true_id = failed_agent_true_id
        #   self.failed_agents[failed_agent_true_id - 1] = True
        #   if self.true_agent_ID > failed_agent_true_id:
        #       self.agent_ID = self.agent_ID - 1

    def purge_agent(self, purge_agent_true_id):
        """
        Removes a "purged agent" from the all solutions. Reassigning the agents tasks to the agent with a lower index
        cyclical).
        """
        print("Purging")
        agent_idx = purge_agent_true_id - 1 - sum(self.failed_agents[:purge_agent_true_id])
        num_agents = len(self.population[0][1]) if self.population else 0

        # Function to update a given solution
        def update_solution(solution):
            if solution is None:
                return None

            order, allocations = solution

            if num_agents <= 1:  # Edge case: If only one agent, nothing to purge
                return (order, allocations)

            # Determine the agent receiving the tasks
            new_owner_idx = agent_idx - 1 if agent_idx > 0 else len(allocations) - 1

            # Compute the start and end index of tasks for the purged agent
            counter = 0
            for i in range(agent_idx):
                counter += allocations[i]
            start_idx = counter
            end_idx = start_idx + allocations[agent_idx]

            # Extract the tasks of the purged agent
            purged_tasks = order[start_idx:end_idx]

            print(f"alloc_before: {agent_idx}: {allocations}")
            # Remove the purged agent's tasks and allocation
            del allocations[agent_idx]
            del order[start_idx:end_idx]
            # Determine the agent receiving the tasks
            new_owner_idx = agent_idx - 1 if agent_idx > 0 else len(allocations) - 1
            print(f"alloc_after: {agent_idx}: {allocations}")
            # Append the tasks to the receiving agent
            counter = 0
            print(f"{new_owner_idx + 1}")
            for i in range(new_owner_idx + 1):  # Find insertion point for new owner
                counter += allocations[i]

            if agent_idx == 0:  # Special case: Agent 1 is purged, move to last agent
                order.extend(purged_tasks)  # Append tasks at the end
            else:
                order[counter:counter] = purged_tasks  # Insert tasks at correct position

            # Update allocation count for the receiving agent
            allocations[new_owner_idx] += len(purged_tasks)

            return (order, allocations)

        # Update all relevant solutions
        for idx, solution in enumerate(self.population):
            self.population[idx] = update_solution(solution)

        self.current_solution = update_solution(self.current_solution)
        self.coalition_best_solution = update_solution(self.coalition_best_solution)
        print(f"Agent {self.agent_ID}: {self.coalition_best_solution}")

    def revive_robot_callback(self, msg):
        """
        Revives the agent.
        If a revive message is received for the agent, it will set itself as unfailed, to be unpurged, and sets the
        last_purge_agent_true_id to none(is this right?).
        Then all necessary timers and subscribers are started.
        """
        self.am_i_failed = False
        self.is_agent_tobe_purged = False
        self.last_purge_agent_true_id = None
        agent_id = self.true_agent_ID  # Ensure this is the correct agent
        print(f"Revive request received for Agent {agent_id}")

        if not self.failed_agents[agent_id - 1] and agent_id != self.true_agent_ID:  # If this agent was never failed, ignore the request
            print(f"Agent {agent_id} is already active. Ignoring revive request.")
            return

        # Ensure `run_timer` is restarted if not already running
        if self.run_timer is None:
            print(f"Restarting run_step for Agent {agent_id}.")
            self.run_timer = self.create_timer(0.01, self.run_step, callback_group=self.me_cb_group)

        # Restart other necessary timers
        if self.environmental_representation_timer is None:
            self.environmental_representation_timer = self.create_timer(5,
                                                                        self.environmental_representation_timer_callback,
                                                                        callback_group=self.cb_group)

        if self.run_goal_publisher_timer is None:
            self.run_goal_publisher_timer = self.create_timer(0.5, self.publish_goal_pose,
                                                              callback_group=self.cb_group)

        if self.solution_publisher_timer is None:
            self.solution_publisher_timer = self.create_timer(2, self.regular_solution_publish_timer,
                                                              callback_group=self.cb_group)

        if self.environmental_representation_subscriber is None:
            self.environmental_representation_subscriber = self.create_subscription(
                EnvironmentalRepresentation,
                '/environmental_representation',
                self.environmental_representation_callback,
                10,
                callback_group=self.cb_group
            )

        print(f"Agent {agent_id} successfully revived and all timers restarted.")

    def unpurge_agent(self, agent_id):
        """
        Reintroduces the specified agent into the all the solutions, and recalculates the robot cost matrix with the
        reintroduced agent.
        """
        print(f"Reviving Agent {agent_id}...")
        # Mark agent as revived
        self.failed_agents[agent_id - 1] = False
        self.purged_agents[agent_id - 1] = False
        self.is_agent_tobe_purged = False

        # Function to insert an empty path at the right position
        def reintegrate_agent(solution):
            if solution is None:
                return None

            order, allocations = solution
            if agent_id - 1 < len(allocations):  # Ensure valid index
                # Insert an empty task allocation for the revived agent
                allocations.insert(agent_id - 1, 0)  # No tasks assigned
                print(f"Inserted empty path for Agent {agent_id}")
            return (order, allocations)

        # Reintegrate the agent into all solutions with an empty path
        for idx, solution in enumerate(self.population):
            self.population[idx] = reintegrate_agent(solution)
        self.current_solution = reintegrate_agent(self.current_solution)
        self.coalition_best_solution = reintegrate_agent(self.coalition_best_solution)
        # Update cost matrix to include revived agent
        self.new_robot_cost_matrix = self.calculate_robot_cost_matrix()
        self.is_new_robot_cost_matrix = True

    def generate_population(self):
        """
        Generates the initial solution population randomly.
        Each task is randomly assigned to one of the agents.
        """
        population = []
        print("Generating Random Population...")

        for i in range(self.pop_size):
            print(f"Generating solution {i}")

            # Randomly shuffle tasks
            tasks = list(range(self.num_tasks))
            random.shuffle(tasks)

            # Randomly assign each task to an agent
            agent_assignments = [[] for _ in range(self.num_tsp_agents)]
            for task in tasks:
                chosen_agent = random.randint(0, self.num_tsp_agents - 1)
                agent_assignments[chosen_agent].append(task)

            # Flatten the ordered task list
            ordered_task_list = [
                task
                for agent_tasks in agent_assignments
                for task in agent_tasks
            ]

            # Record how many tasks are assigned to each agent
            task_allocation_counts = [len(agent_tasks) for agent_tasks in agent_assignments]

            # Append the random solution
            population.append((ordered_task_list, task_allocation_counts))

        print("Generated random population")
        print(f"{population[0]}")
        return population

    def set_coalition_best_solution(self, solution):
        """
        Sets the given solution as the best coalition solution and the gets the robots next task from the solution.
        """
        self.coalition_best_solution = deepcopy(solution)

        self.assign_next_task(solution)

    def assign_next_task(self, solution):
        """
        Extracts and sets the current next task for the agent from a given solution
        """
        try:
            if self.agent_ID >= len(solution[1]) or self.agent_ID < 0:
                raise ValueError("Invalid robot_id")

            if solution[1][self.agent_ID] != 0:
                # Compute the start index for the given robot
                start_index = sum(solution[1][:self.agent_ID])

                # Get the number of tasks assigned to the robot
                num_tasks = solution[1][self.agent_ID]

                # Extract the tasks assigned to the robot
                agent_tasks = solution[0][start_index:start_index + num_tasks]

                # Find the first uncovered task
                for task in agent_tasks:
                    if not self.is_covered[task]:  # Check if the task is not covered
                        self.current_task = task  # Assign the first uncovered task
                        return

                # If no uncovered task is found, mark the robot as finished
                self.current_task = None
            else:
                self.current_task = None
        except Exception:
            print(f"Agent: {self.agent_ID} has {Exception} while assigning tasks.")

    def select_solution(self):
        """
        Finds and returns the fittest solution
        :return: The fittest solution
        """
        # Select the best solution from the population based on fitness score
        best_solution = min(self.population, key=lambda sol: Fitness.fitness_function(
            sol, self.cost_matrix))  # Assuming lower score is better
        return best_solution

    def update_experience(self, condition, operator, gain):
        """
        Adds details of the current iteration to the experience memory.
        :param condition: The previous condition
        :param operator: The operator applied
        :param gain: The resulting change in the current solution's fitness
        :return: None
        """
        self.previous_experience.append([condition, operator, gain])
        pass

    def individual_learning_old(self):
        # Update weight matrix (if needed) based on learning (not fully implemented in this example)
        abs_gain = 0
        index_best_fitness = -1
        for i in range(len(self.previous_experience)):
            current_gain = abs_gain + self.previous_experience[i][2]
            if current_gain < abs_gain:
                index_best_fitness = i
            abs_gain += current_gain

            # Get elements before index_best_fitness
        elements_before_best = self.previous_experience[:index_best_fitness + 1] if index_best_fitness != -1 else []
        condition_operator_pairs = [(item[0], item[1]) for item in elements_before_best]
        condition_operator_pairs = list(set(condition_operator_pairs))
        for pair in condition_operator_pairs:
            self.weight_matrix.weights[pair[0].value][pair[1].value - 1] += self.eta

        return self.weight_matrix.weights

    # ----------------------------------------------------------------------------
    def individual_learning(self):
        """
        Learning with Bellman Equation.
        Updates the weight matrix (Q(s, a)) using the Q-learning formula.
        Only updates the weights for experiences before the lowest relative fitness solution.
        :return: Updated weight matrix
        """
        # Compute cumulative gains
        cumulative_gains = []
        total_gain = 0
        for experience in self.previous_experience:
            _, _, gain = experience
            total_gain += gain
            cumulative_gains.append(total_gain)

        # Find the index of the lowest relative fitness solution (min cumulative gain)
        if self.best_local_improved:
            min_fitness_index = cumulative_gains.index(min(cumulative_gains))
        else:
            min_fitness_index = len(self.previous_experience)

        # Update weights only for experiences before this index
        for i in range(min_fitness_index):
            condition, operator, gain = self.previous_experience[i]

            # Current Q value
            current_q = self.weight_matrix.weights[condition.value][operator.value - 1]

            # Determine next condition
            if i + 1 < len(self.previous_experience):
                next_condition = self.previous_experience[i + 1][
                    0]  # Next condition is the first element of the next experience
                max_next_q = max(self.weight_matrix.weights[next_condition.value])  # Best future Q-value
            else:
                max_next_q = 0  # No future state, assume no future reward

            # Estimate future rewards (single-step Q-learning)
            max_next_q = max(self.weight_matrix.weights[condition.value])

            if self.best_local_improved:
                self.reward = 1
            else:
                self.reward = -0.5

            updated_q = current_q + self.lr * (self.reward + self.gamma_decay * max_next_q - current_q)
            updated_q = max(updated_q, 1e-6)
            # Update weight matrix
            self.weight_matrix.weights[condition.value][operator.value - 1] = updated_q

        return self.weight_matrix.weights

    # ------------------------------------------------------------------------------------------------

    def mimetism_learning(self, received_weights, rho):
        # mimetisim learnng will stay same
        """
        Perform mimetism learning by updating self.weight_matrix.weights using multiple sets of received weights.
        For each weight in each received set:
            w_a = (1 - rho) * w_a + rho * w_b
        :param received_weights: A 3D list (or array) of received weights. Each "slice" is a 2D matrix of weights.
        :param rho: The learning rate, a value between 0 and 1.
        :return: Updated weight matrix.
        """
        # Iterate over each set of received weights
        for weight_set in received_weights:
            # Check dimensions match
            if len(weight_set) != len(self.weight_matrix.weights) or len(weight_set[0]) != len(
                    self.weight_matrix.weights[0]):
                raise ValueError("Dimension mismatch between weight_matrix.weights and received weights.")
                # Update self.weight_matrix.weights for each element
            for i in range(len(self.weight_matrix.weights)):  # Rows
                for j in range(len(self.weight_matrix.weights[i])):  # Columns
                    self.weight_matrix.weights[i][j] = (
                            (1 - rho) * self.weight_matrix.weights[i][j] + rho * weight_set[i][j]
                    )

    def stopping_criterion(self, iteration_count):
        """
        Returns true is the current number of iterations is over a set iteration limit.
        """
        # Define a stopping criterion (e.g., a fixed number of iterations)
        return iteration_count > self.num_iterations

    def end_of_di_cycle(self, cycle_count):
        """
        Returns true if the current cycle is equal to or over the length of a DI cycle.
        """
        if cycle_count >= self.di_cycle_length:
            return True
        return False  # Placeholder; replace with actual condition

    def weight_update_callback(self, msg):
        """
        Stores received weight matrices.
        """
        # Callback to process incoming weight matrix updates
        received_weights = self.weight_matrix.unpack_weights(weights_msg=msg, agent_id=self.agent_ID)
        if received_weights is not None:
            self.received_weight_matrices.append(received_weights)

    def finished_coverage_callback(self, msg):
        self.finished_robots[int(msg.robot_id)] = bool(msg.finished)
        if all(self.finished_robots):
            print("Coverage Complete")
            self.destroy_node()


    def solution_update_callback(self, msg):
        """
        Receives a processes a proposed best coalition solution from another agent.
        """
        if not self.am_i_failed:
            # Callback to process incoming solution updates
            if msg is not None and self.coalition_best_solution is not None:
                received_solution = (msg.order, msg.allocations)

                # Check if the received solution has fewer tasks than the current best solution
                received_task_count = sum(msg.allocations)
                current_task_count = sum(self.coalition_best_solution[1])

                if received_task_count < current_task_count:
                    print(
                        f"Adjusting my solution: I have {current_task_count} tasks, but the received solution has {received_task_count}.")

                    # Remove excess tasks and update allocations
                    self.coalition_best_solution = self.remove_extra_tasks(self.coalition_best_solution,
                                                                           received_task_count)

                # Now check fitness as before
                their_solution_fitness = Fitness.fitness_function_robot_pose(received_solution, self.cost_matrix,
                                                                             [self.robot_cost_matrix[i] for i, purged in
                                                                              enumerate(self.purged_agents) if
                                                                              not purged],
                                                                             self.robot_initial_pose_cost_matrix)

                our_solution_fitness = Fitness.fitness_function_robot_pose(self.coalition_best_solution,
                                                                           self.cost_matrix,
                                                                           [self.robot_cost_matrix[i] for i, purged in
                                                                            enumerate(self.purged_agents) if
                                                                            not purged],
                                                                           self.robot_initial_pose_cost_matrix)

                # If the received solution is better, update the coalition best solution
                if their_solution_fitness < our_solution_fitness:
                    print(f"recieved a better solution: their fitness: {their_solution_fitness}, out fitness: {our_solution_fitness}")
                    self.set_coalition_best_solution(received_solution)
                    self.coalition_best_agent = msg.id

    def remove_extra_tasks(self, solution, target_task_count):
        """
        Removes excess tasks from a solution to match a target task count.
        Also adjusts agent allocations accordingly.
        """
        if solution is None:
            return None

        order, allocations = deepcopy(solution)

        # Remove already covered tasks first
        order = [task for task in order if not self.is_covered[task]]

        # If still too many tasks, remove from the end
        if len(order) > target_task_count:
            order = order[:target_task_count]

        # Adjust allocations to reflect the new task count
        new_allocations = []
        counter = 0

        for count in allocations:
            # Assign the same proportion of tasks from the new order
            new_task_set = order[counter:counter + count]
            new_allocations.append(len(new_task_set))
            counter += len(new_task_set)

        return (order, new_allocations)

    def global_pose_callback(self, msg):
        """
        Receives and processes a global pose from another agent.
        Updating the internal representation of the state.
        If coverage has yet to start and each agent has a pose, calculate the robot cost matrix and start coverage.
        If this agent has reached its goal, complete that task.
        If an agent has finished its tasks, register its completed coverage.
        """
        try:
            agent = msg.robot_id #todo: carry agent id
            # Store the first received pose for each agent
            if self.initial_robot_poses[agent] is None:
                self.initial_robot_poses[agent] = (msg.x_position, msg.y_position)

            task = deepcopy(self.current_task)

            if msg:
                self.robot_poses[agent] = (msg.x_position, msg.y_position)
            if all(pose is not None for pose in
                   self.robot_poses) and self.is_loop_started is False:
                self.new_robot_cost_matrix = self.calculate_robot_cost_matrix()
                self.is_new_robot_cost_matrix = True
                self.calculate_robot_inital_pose_cost_matrix()
                if self.population is None:
                    if self.is_generating is False:
                        self.is_generating = True
                        self.population = self.generate_population()
                        self.current_solution = self.select_solution()
                self.run_timer = self.create_timer(0.1, self.run_step, callback_group=self.me_cb_group)
                self.is_loop_started = True

            if self.task_poses is not None and task is not None:
                x = msg.x_position
                y = msg.y_position

                goal_x = self.task_poses[task][0]
                goal_y = self.task_poses[task][1]

                distance = math.sqrt(((x - goal_x) ** 2)+((y - goal_y) ** 2))

                if agent == self.true_agent_ID and distance < 0.4:
                    print(f"[INFO] Agent {self.agent_ID} covered position {task}.")
                    self.task_covered = task
                    self.is_new_task_covered = True
                    return

            if self.coalition_best_solution is not None and self.coalition_best_solution[1][self.agent_ID - 1] == 0:
                x = msg.x_position
                y = msg.y_position
                goal_x = self.initial_robot_poses[self.true_agent_ID][0]
                goal_y = self.initial_robot_poses[self.true_agent_ID][1]

                distance = math.sqrt(((x - goal_x) ** 2)+((y - goal_y) ** 2))

                if agent == self.true_agent_ID and distance < 0.4:
                    print(f"[INFO] Agent {self.agent_ID} finished coverage!.")
                    msg = FinishedCoverage()
                    msg.finished = True
                    msg.robot_id = self.agent_ID
                    self.finished_coverage_pub.publish(msg)

        except Exception as e:
            error_message = (
                f"[ERROR] Exception in global_pose_callback for agent {agent}:\n"
                f"    Error Type: {type(e).__name__}\n"
                f"    Error Message: {e}\n"
                f"    Stack Trace:\n{traceback.format_exc()}"
            )
            print(f"{self.coalition_best_solution}")
            print(error_message)
            # print(error_message)

    def handle_covered_task(self, current_task):
        """
        Handles a task covered by this agent.
        Updating all it's solutions to reflect the new state, and publishing it's new environmental representation.
        *** TODO: This shouldn't always publish! ***
        """
        if self.is_loop_started:
            print(f"Agent {self.agent_ID} is processing covered task")
            self.is_covered[current_task] = True
            rep = EnvironmentalRepresentation()
            rep.agent_id = self.true_agent_ID
            rep.is_covered = self.is_covered
            self.environmental_representation_publisher.publish(rep)
            if not self.is_covered[current_task]:
                self.is_covered[current_task] = True  # Mark task as covered

            # Function to update a given solution
            def update_solution(solution):
                if solution is None:
                    return None

                order, allocations = solution
                counter = 0
                for agent_idx, task_count in enumerate(allocations):
                    if current_task in order[counter:counter + task_count]:
                        order.remove(current_task)  # Remove task from order
                        allocations[agent_idx] -= 1  # Decrease allocation count
                        break
                    counter += task_count
                return order, allocations

                # Update population

            for idx, solution in enumerate(self.population):
                self.population[idx] = update_solution(solution)

            # Update current_solution and coalition_best_solution if they exist
            self.current_solution = update_solution(self.current_solution)
            self.coalition_best_solution = update_solution(self.coalition_best_solution)
            self.assign_next_task(self.coalition_best_solution)
            print(f"Task {current_task} is covered: {self.is_covered[current_task]}")

    def environmental_representation_callback(self, msg):
        """
        On receiving an environmental representation, handle the coverage.
        If an environmental representation is received from a failed agent, set it be revived.
        """
        if not self.am_i_failed:
            agent_id = msg.agent_id  # Extract the sender agent's ID
            self.last_env_rep_timestamps[agent_id] = time()  # Store the current time as the last received time

            # Persist True values from msg.is_covered
            for i in range(len(msg.is_covered)):
                if msg.is_covered[i] and not self.is_covered[i]:  # If new value is True, persist it
                    self.handle_covered_task(i)
                    self.is_covered[i] = True
            if self.failed_agents[agent_id - 1]:
                self.agent_to_revive = agent_id

    def check_stale_agents(self):
        """
        If a message hasn't been received by an agent within a threshold period, set the agent as failed and to be
        purged.
        """
        if not self.am_i_failed:
            current_time = time()
            timeout_threshold = 15  # Define a threshold (e.g., 10 seconds)

            for agent_id, last_time in self.last_env_rep_timestamps.items():
                if current_time - last_time > timeout_threshold and self.failed_agents[agent_id - 1] is False:
                    self.is_agent_tobe_purged = True
                    self.last_purge_agent_true_id = agent_id
                    self.failed_agents[agent_id - 1] = True
                    if self.true_agent_ID > agent_id:
                        self.agent_ID = self.agent_ID - 1
                    self.get_logger().warning(
                        f"Agent {agent_id} has not sent an update for {current_time - last_time:.2f} seconds.")

    def environmental_representation_timer_callback(self):
        """
        Publish this agents environmental representation.
        """
        if self.am_i_failed:
            self.environmental_representation_timer.cancel()
            self.environmental_representation_timer = None

        rep = EnvironmentalRepresentation()
        rep.agent_id = self.true_agent_ID
        rep.is_covered = self.is_covered
        self.environmental_representation_publisher.publish(rep)

    def publish_goal_pose(self):
        """
        Publish this agents current goal pose to the flight controller.
        *** Feels like this is violating some good practice (High-level/Low-level) ***
        """
        if self.am_i_failed:
            self.run_goal_publisher_timer.cancel()
            self.run_goal_publisher_timer = None
        if self.current_task is not None and self.task_poses:
            goal_pose = SimplePosition()

            # Set position (Latitude, Longitude, Altitude)
            goal_pose.robot_id = self.agent_ID
            goal_pose.x_position = float(self.task_poses[self.current_task][0])
            goal_pose.y_position = float(self.task_poses[self.current_task][1])
            self.goal_pose_publisher.publish(goal_pose)
        else:
            try:
                if self.am_i_failed:
                    goal_pose = SimplePosition()

                    # Set position (Latitude, Longitude, Altitude)
                    goal_pose.x_position = self.robot_poses[
                        self.true_agent_ID][0]
                    goal_pose.y_position = self.robot_poses[
                        self.true_agent_ID][1]

                    self.goal_pose_publisher.publish(goal_pose)
                    return

                if self.initial_robot_poses[self.true_agent_ID] is not None:
                    goal_pose = SimplePosition()

                    # Set position (Latitude, Longitude, Altitude)
                    goal_pose.x_position = self.initial_robot_poses[
                        self.true_agent_ID][0]
                    goal_pose.y_position = self.initial_robot_poses[
                        self.true_agent_ID][1]

                    self.goal_pose_publisher.publish(goal_pose)
            except:
                print("Goal pose error")

    def select_random_solution(self):
        """
        Sample a random solution from the population.
        """
        temp_solution = sample(population=self.population, k=1)[0]
        if temp_solution != self.current_solution:
            return temp_solution

    def robot_cost_matrix_recalculation(self):
        """Recalculate and update the robot cost matrix."""
        if all(pose is not None for pose in self.robot_poses):
            self.new_robot_cost_matrix = self.calculate_robot_cost_matrix()
            self.is_new_robot_cost_matrix = True

    def regular_solution_publish_timer(self):
        """
        Publishes the coalition best solution.
        """
        if self.am_i_failed:
            self.solution_publisher_timer.cancel()
            self.solution_publisher_timer = None
        if self.coalition_best_solution is not None:
            solution = Solution()
            solution.id = self.agent_ID
            solution.order = self.coalition_best_solution[0]
            solution.allocations = self.coalition_best_solution[1]
            self.solution_publisher.publish(solution)

    def remove_covered_tasks_from_solution(self, solution):
        """
        Removes tasks from a solution that are already covered.
        Adjusts allocations accordingly.
        """
        if solution is None:
            return None

        order, allocations = deepcopy(solution)  # Avoid modifying original data

        # Check if any task in order is actually covered
        if not any(self.is_covered[task] for task in order):
            return solution  # No need to modify if no tasks are covered

        # If necessary, remove covered tasks
        new_order = [task for task in order if not self.is_covered[task]]

        # Recalculate allocations based on new order
        new_allocations = []
        counter = 0

        for count in allocations:
            # Get remaining tasks for this agent
            new_task_set = new_order[counter:counter + count]
            new_allocations.append(len(new_task_set))
            counter += len(new_task_set)

        return new_order, new_allocations

    def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """
        print(f"Agent {self.agent_ID} runstep")
        if self.am_i_failed:
            return

        if self.is_new_robot_cost_matrix:
            self.robot_cost_matrix = self.new_robot_cost_matrix
            self.is_new_robot_cost_matrix = False

        if self.am_i_failed:
            print(f"Im agent {self.true_agent_ID} and im still here!")
        if self.is_new_task_covered:
            self.handle_covered_task(self.task_covered)
            self.task_covered = -1
            self.is_new_task_covered = False

        if self.current_solution is not None:

            # Ensure the current solution and coalition best solution do not contain covered tasks
            if self.current_solution:
                self.current_solution = self.remove_covered_tasks_from_solution(self.current_solution)
            if self.coalition_best_solution:
                self.coalition_best_solution = self.remove_covered_tasks_from_solution(self.coalition_best_solution)

            if self.is_agent_tobe_purged and not self.purged_agents[self.last_purge_agent_true_id - 1] and len(
                    self.current_solution[1]) > self.failed_agents.count(False):
                print("Purging agent")
                self.purge_agent(self.last_purge_agent_true_id)
                self.purged_agents[self.last_purge_agent_true_id - 1] = True
                self.is_agent_tobe_purged = None

            if self.agent_to_revive:
                if self.agent_ID >= self.agent_to_revive:
                    print(f"I was robot: {self.agent_ID}, now becoming robot: {self.agent_ID + 1}")
                    self.agent_ID = self.agent_ID + 1
                self.unpurge_agent(self.agent_to_revive)
                self.agent_to_revive = None

            num_false = self.failed_agents.count(False)

            if self.stopping_criterion(self.iteration_count):
                self.get_logger().info("Stopping criterion met. Shutting down.")
                self.run_timer.cancel()
                return

            condition = ConditionFunctions.perceive_condition(self.previous_experience)

            if self.no_improvement_attempt_count >= self.no_improvement_attempts:
                self.current_solution = self.select_random_solution()
                self.no_improvement_attempt_count = 0

            operator = OperatorFunctions.choose_operator(self.weight_matrix.weights, condition)
            c_new = None
            try:
                c_new = run_with_timeout(
                    OperatorFunctions.apply_op,
                    args=(
                        operator,
                        self.current_solution,
                        self.population,
                        self.cost_matrix,
                        [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged],
                        self.robot_initial_pose_cost_matrix
                    ),
                    timeout=5.0  # seconds, adjust as needed
                )
                if c_new is None:
                    print(f"[TIMEOUT] Operator {operator} took too long. Skipping this step.")
                    self.no_improvement_attempt_count += 1
                    return

            except Exception as e:
                print(f"Issue with applying operator: {e}")
                print(f"New solution came out empty:\n old solution: {self.current_solution}\n"
                      f"operator applied {operator}")

            new_solution_fitness = Fitness.fitness_function_robot_pose(c_new, self.cost_matrix,
                                                                       [self.robot_cost_matrix[i] for i, purged in
                                                            enumerate(self.purged_agents) if not purged],
                                                                       self.robot_initial_pose_cost_matrix)

            current_solution_fitness = Fitness.fitness_function_robot_pose(self.current_solution, self.cost_matrix,
                                                                           [self.robot_cost_matrix[i] for i, purged in
                                                            enumerate(self.purged_agents) if not purged],
                                                                           self.robot_initial_pose_cost_matrix)

            if self.local_best_solution:
                local_best_solution_fitness = Fitness.fitness_function_robot_pose(self.local_best_solution,
                                                                                  self.cost_matrix,
                                                                                  [self.robot_cost_matrix[i] for i,
                                                                                  purged in enumerate(self.purged_agents)
                                                                                   if not purged],
                                                                                  self.robot_initial_pose_cost_matrix)

            if self.coalition_best_solution:
                coalition_best_solution_fitness = Fitness.fitness_function_robot_pose(self.coalition_best_solution,
                                                                                      self.cost_matrix,
                                                                                      [self.robot_cost_matrix[i] for i,
                                                                                      purged in
                                                                                       enumerate(self.purged_agents)
                                                                                       if not purged],
                                                                                      self.robot_initial_pose_cost_matrix,
                                                                                      islog=False)

            if c_new:
                gain = new_solution_fitness - current_solution_fitness

                self.update_experience(condition, operator, gain)

                if self.local_best_solution is None or \
                        new_solution_fitness < local_best_solution_fitness:
                    self.local_best_solution = deepcopy(c_new)
                    self.best_local_improved = True
                    self.no_improvement_attempt_count = 0
                else:
                    self.no_improvement_attempt_count += 1

                if self.coalition_best_solution is None or new_solution_fitness < coalition_best_solution_fitness:
                    self.set_coalition_best_solution(c_new)
                    self.coalition_best_agent = self.agent_ID
                    self.best_coalition_improved = True
                    solution = Solution()
                    solution.id = self.agent_ID
                    solution.order = self.coalition_best_solution[0]
                    solution.allocations = self.coalition_best_solution[1]
                    self.solution_publisher.publish(solution)

                self.current_solution = c_new
                self.di_cycle_count += 1

                condition = ConditionFunctions.perceive_condition(self.previous_experience)
                if condition == Condition.C_4:
                    print("*********************** Im Condition C4 ***********************")
                if self.end_of_di_cycle(self.di_cycle_count) or condition == Condition.C_4:

                    learning_method_switch = {
                        LearningMethod.FERREIRA: self.individual_learning_old,
                        LearningMethod.Q_LEARNING: self.individual_learning
                    }

                    condition = ConditionFunctions.perceive_condition(self.previous_experience)

                    # Call the appropriate function based on the current learning method
                    learning_function = learning_method_switch.get(LearningMethod(self.learning_method))
                    if learning_function:
                        self.weight_matrix.weights = learning_function()
                    else:
                        self.get_logger().error(f"Unknown learning method: {self.learning_method}")

                    self.best_local_improved = False

                    if self.best_coalition_improved:
                        self.best_coalition_improved = False

                        msg = Weights()
                        msg_dict = self.weight_matrix.pack_weights(self.agent_ID)
                        msg.id = msg_dict["id"]
                        msg.rows = msg_dict["rows"]
                        msg.cols = msg_dict["cols"]
                        msg.weights = msg_dict["weights"]

                        self.weight_publisher.publish(msg)

                    if self.received_weight_matrices:
                        self.mimetism_learning(self.received_weight_matrices, self.rho)
                        self.received_weight_matrices = []

                    self.previous_experience = []
                    self.di_cycle_count = 0

                self.iteration_count += 1
            else:
                print("Something went wrong with applying the operator and resulted in a None.")


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


# Deneme

import concurrent.futures

def run_with_timeout(func, args=(), kwargs=None, timeout=1.0):
    """
    Runs a function with a timeout. Returns the result or None on timeout.
    """
    kwargs = kwargs or {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return None

def main(args=None):
    rclpy.init(args=args)

    temp_node = Node("parameter_loader")
    temp_node.declare_parameter("agent_id", 1)
    temp_node.declare_parameter("runtime", 60.0)
    temp_node.declare_parameter("learning_method", "Ferreira et al.")

    agent_id = temp_node.get_parameter("agent_id").value
    runtime = temp_node.get_parameter("runtime").value
    learning_method = temp_node.get_parameter("learning_method").value
    temp_node.destroy_node()

    node_name = f"cbm_population_agent_{agent_id}"
    agent = CBMPopulationAgentOnlineSimpleSimulation(
        pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
        num_iterations=9999999, num_solution_attempts=21, agent_id=agent_id,
        node_name=node_name, learning_method=learning_method
    )
    print("CBMPopulationAgentOnlineSimpleSimulation has been initialized.")

    def shutdown_callback():
        agent.get_logger().info("LLM-Interface-agent Runtime completed. Shutting down.")
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if runtime != -1:
        # Create a timer for shutdown
        print("Hit shutdown_callback")
        agent.create_timer(runtime, shutdown_callback)

    # asyncio.run(agent.generate_population())

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

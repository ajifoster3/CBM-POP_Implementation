import json
import random
import sys
from builtin_interfaces.msg import Time
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
from copy import deepcopy
from random import sample
from cbm_pop.Condition import ConditionFunctions
from cbm_pop.Fitness import Fitness
from cbm_pop.Operator_Fuctions import OperatorFunctions
from cbm_pop.WeightMatrix import WeightMatrix
from cbm_pop.Problem import Problem
from rclpy.node import Node
from std_msgs.msg import String, Float32
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
from cbm_pop.path_cost_calculator import calculate_drone_distance
from cbm_pop_interfaces.msg import Solution, Weights, EnvironmentalRepresentation
from enum import Enum
from geographic_msgs.msg import GeoPoseStamped
from math import radians, cos, sin, asin, sqrt
from pygeodesy.geoids import GeoidPGM
from std_msgs.msg import Bool


class LearningMethod(Enum):
    FERREIRA = "Ferreira_et_al."
    Q_LEARNING = "Q-Learning"


class CBMPopulationAgentOnline(Node):

    def __init__(self, pop_size, eta, rho, di_cycle_length, epsilon, num_tasks, num_tsp_agents, num_iterations,
                 num_solution_attempts, agent_id, node_name: str, cost_matrix, learning_method):
        super().__init__(node_name)
        self.geoid = GeoidPGM('/home/ajifoster3/PycharmProjects/ros2_ws/src/CBM-POP_Implementation/egm96-5.pgm')
        self.pop_size = pop_size
        self.eta = eta
        self.rho = rho
        self.di_cycle_length = di_cycle_length
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.num_tasks = 5
        self.num_tsp_agents = 5
        self.agent_best_solution = None
        self.coalition_best_solution = None
        self.local_best_solution = None
        self.coalition_best_agent = None
        self.num_intensifiers = 2
        self.num_diversifiers = 4
        self.population = self.generate_population()
        self.weight_matrix = WeightMatrix(self.num_intensifiers, self.num_diversifiers)
        self.previous_experience = []
        self.no_improvement_attempts = num_solution_attempts
        self.agent_ID = agent_id  # This will change as teamsize changes
        self.true_agent_ID = self.agent_ID  # This is permanent
        self.received_weight_matrices = []
        self.learning_method = learning_method

        # Iteration state
        self.iteration_count = 0
        self.di_cycle_count = 0
        self.no_improvement_attempt_count = 0
        self.best_coalition_improved = False
        self.best_local_improved = False

        # Runtime data
        self.initial_robot_poses = [None] * (self.num_tsp_agents)
        self.robot_poses = [None] * (self.num_tsp_agents)

        self.current_task = None
        with open("/home/ajifoster3/Downloads/all_geoposes_wind_turbine.json", "r") as file:
            data = json.load(file)
        self.task_poses = [
            {
                "latitude": item["GeoPose"]["position"]["latitude"],
                "longitude": item["GeoPose"]["position"]["longitude"],
                "altitude": item["GeoPose"]["position"]["altitude"],
                "orientation_x": item["GeoPose"]["orientation"]["x"],
                "orientation_y": item["GeoPose"]["orientation"]["y"],
                "orientation_z": item["GeoPose"]["orientation"]["z"],
                "orientation_w": item["GeoPose"]["orientation"]["w"]
            }
            for item in data
        ]

        self.num_tasks = len(self.task_poses)
        self.is_covered = [False] * self.num_tasks
        self.cost_matrix = self.calculate_cost_matrix()
        self.robot_cost_matrix = [None] * self.num_tasks
        self.current_solution = self.select_solution()
        self.last_purge_agent_true_id = None
        self.is_agent_tobe_purged = False
        self.failed_agents = [False] * (self.num_tsp_agents)
        self.purged_agents = [False] * (self.num_tsp_agents)

        self.ros_timer = None
        self.agent_timeouts = [False] * (self.num_tsp_agents)

        cb_group = ReentrantCallbackGroup()
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
        # List to store subscriptions
        self.global_pose_subscribers = []

        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id + 1}/global_pose'

            # Wrap the callback to include agent_id
            sub = self.create_subscription(
                GeoPoseStamped,
                topic,
                lambda msg, agent=(id + 1): self.global_pose_callback(msg, agent),
                10,
                callback_group=cb_group
            )

            self.global_pose_subscribers.append(sub)  # Store subscription to prevent garbage collection

        # Publisher for goal position
        self.goal_pose_publisher = self.create_publisher(
            GeoPoseStamped,
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

            self.global_pose_subscribers.append(sub)  # Store subscription to prevent garbage collection

        # Timer for periodic execution of the run loop
        self.run_goal_publisher_timer = self.create_timer(0.5, self.publish_goal_pose, callback_group=cb_group)
        self.run_cost_matrix_recalculation = self.create_timer(15, self.robot_cost_matrix_recalculation,
                                                               callback_group=cb_group)

        self.environmental_representaion_subscriber = self.create_subscription(
            EnvironmentalRepresentation,
            '/environmental_representation',
            self.environmental_representation_callback,
            10,
            callback_group=cb_group
        )

        self.environmental_representaion_publisher = self.create_publisher(
            EnvironmentalRepresentation,
            '/environmental_representation',
            10
        )

        self.environmental_representation_timer = self.create_timer(5, self.environmental_representation_timer,
                                                                    callback_group=cb_group)

        self.solution_publisher_timer = self.create_timer(2, self.regular_solution_publish_timer,
                                                                    callback_group=cb_group)

        # Q_learning
        # Q_learning parameter
        self.lr = 0.1  # RL learning Rate
        self.reward = 0  # RL reward initial 0
        self.new_reward = 0  # RL tarafından secilecek. initial 0
        self.gamma_decay = 0.99  # it can be change interms of iteration

        self.run_timer = None
        self.is_loop_started = False
        self.task_covered = -1
        self.is_new_task_covered = False

    def calculate_cost_matrix(self):
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_tasks, num_tasks))
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    # Extract positions
                    lat1, lon1, alt1 = self.task_poses[i]["latitude"], self.task_poses[i]["longitude"], \
                        self.task_poses[i]["altitude"]
                    lat2, lon2, alt2 = self.task_poses[j]["latitude"], self.task_poses[j]["longitude"], \
                        self.task_poses[j]["altitude"]

                    # Compute horizontal and vertical distances
                    total_horizontal_distance = self.haversine(lat1, lon1, 0, lat2, lon2, 0)
                    total_vertical_distance = alt2 - alt1

                    # Compute cost using trajectory generation function
                    cost = calculate_drone_distance(
                        total_horizontal_distance, total_vertical_distance, 3, 1.45, 11)

                    cost_map[i][j] = cost
        return cost_map

    def calculate_robot_cost_matrix(self):
        # Filter out None values from robot_poses

        valid_robot_poses = [pose for pose in self.robot_poses if pose is not None]
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        # Initialize cost matrix
        cost_map = np.zeros((num_robots, num_tasks))

        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                # Extract positions
                lat1, lon1, alt1 = robot_pose.position.latitude, robot_pose.position.longitude, robot_pose.position.altitude
                lat2, lon2, alt2 = task_pose["latitude"], task_pose["longitude"], task_pose["altitude"]

                # Compute horizontal and vertical distances
                total_horizontal_distance = self.haversine(lat1, lon1, 0, lat2, lon2, 0)
                total_vertical_distance = alt2 - alt1

                # Compute cost using trajectory generation function
                cost = calculate_drone_distance(
                    total_horizontal_distance, total_vertical_distance, 3, 1.45, 11)

                cost_map[i][j] = cost
        self.robot_cost_matrix = cost_map

    def kill_robot_callback(self, msg, failed_agent_true_id):
        if failed_agent_true_id == self.true_agent_ID:
            if msg.data:
                goal_pose = GeoPoseStamped()

                # Set header
                goal_pose.header.stamp = Time()
                goal_pose.header.frame_id = "world"  # Change frame if needed

                # Set position (Latitude, Longitude, Altitude)
                goal_pose.pose.position.latitude = self.initial_robot_poses[self.true_agent_ID - 1].position.latitude
                goal_pose.pose.position.longitude = self.initial_robot_poses[self.true_agent_ID - 1].position.longitude
                goal_pose.pose.position.altitude = self.initial_robot_poses[self.true_agent_ID].position.altitude  # Example altitude

                # Set orientation (Quaternion)
                goal_pose.pose.orientation.x = self.initial_robot_poses[self.true_agent_ID - 1].orientation.x
                goal_pose.pose.orientation.y = self.initial_robot_poses[self.true_agent_ID - 1].orientation.y
                goal_pose.pose.orientation.z = self.initial_robot_poses[self.true_agent_ID - 1].orientation.z
                goal_pose.pose.orientation.w = self.initial_robot_poses[self.true_agent_ID - 1].orientation.w
                self.goal_pose_publisher.publish(goal_pose)
                print("Shutting down node...")
                self.destroy_node()
                rclpy.shutdown()
        else:
            self.is_agent_tobe_purged = True
            self.last_purge_agent_true_id = failed_agent_true_id
            self.failed_agents[failed_agent_true_id - 1] = True
            if self.true_agent_ID > failed_agent_true_id:
                self.agent_ID = self.agent_ID - 1

    def purge_agent(self, purge_agent_true_id):
        """
        Removes the specified agent (1-indexed) from all solutions, reassigning their tasks
        to the previous agent (or the last agent if it's agent 1).
        """
        print("Purging")
        agent_idx = purge_agent_true_id - 1  # Convert to 0-indexed for allocations
        num_agents = len(self.population[0][1]) if self.population else 0

        # Function to update a given solution
        def update_solution(solution):
            if solution is None:
                return None

            order, allocations = solution

            if num_agents <= 1:  # Edge case: If only one agent, nothing to purge
                return (order, allocations)

            # Determine the agent receiving the tasks
            new_owner_idx = agent_idx - 1 if agent_idx > 0 else num_agents - 1

            # Compute the start and end index of tasks for the purged agent
            counter = 0
            for i in range(agent_idx):
                counter += allocations[i]
            start_idx = counter
            end_idx = start_idx + allocations[agent_idx]

            # Extract the tasks of the purged agent
            purged_tasks = order[start_idx:end_idx]

            # Remove the purged agent's tasks and allocation
            del allocations[agent_idx]
            del order[start_idx:end_idx]

            # Append the tasks to the receiving agent
            counter = 0
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

    def generate_population(self):
        """
        Randomly generates a population of size `pop_size`
        :return: Population of solutions of size `pop_size`
        """
        # Generate initial population where each solution is a list of task allocations to agents
        population = []

        for _ in range(self.pop_size):
            # Create a list of task indexes and shuffle it for a random allocation
            allocation = list(range(self.num_tasks))
            random.shuffle(allocation)

            # Generate non-zero task counts for each agent that sum to number_tasks
            # Start with each agent assigned at least 1 task
            counts = [1] * self.num_tsp_agents
            for _ in range(self.num_tasks - self.num_tsp_agents):
                counts[random.randint(0, self.num_tsp_agents - 1)] += 1

            # Add both allocation and counts to the population
            population.append((allocation, counts))

        return population

    def set_coalition_best_solution(self, solution):
        self.coalition_best_solution = solution

        self.assign_next_task(solution)

    def assign_next_task(self, solution):
        try:
            if self.agent_ID - 1 >= len(solution[1]) or self.agent_ID < 0:
                raise ValueError("Invalid robot_id")

            if solution[1][self.agent_ID - 1] != 0:
                # Compute the start index for the given robot
                start_index = sum(solution[1][:self.agent_ID - 1])

                # Get the number of tasks assigned to the robot
                num_tasks = solution[1][self.agent_ID - 1]

                # Extract the tasks assigned to the robot
                agent_tasks = solution[0][start_index:start_index + num_tasks]

                # Find the first uncovered task
                for task in agent_tasks:
                    if not self.is_covered[task]:  # Check if the task is not covered
                        self.current_task = task  # Assign the first uncovered task
                        return

                # If no uncovered task is found, mark the robot as finished
                print(f"Agent {self.agent_ID} has completed all assigned tasks.")
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
        Updates the weight matrix (Q(s, a)) using the Q-learning formula.
        :return: Updated weight matrix
        """
        for experience in self.previous_experience:
            condition, operator, gain = experience

            # Current Q value
            current_q = self.weight_matrix.weights[condition.value][operator.value - 1]

            # Estimate future rewards (no explicit next_state, assume single-step Q-learning)
            max_next_q = max(
                self.weight_matrix.weights[condition.value])  # Max Q for current state (proxy for next state)

            # Q-learning update formula
            if gain > 0:
                self.reward = -0.5
            elif gain == 0:
                self.reward = 0
            else:
                self.reward = 1

            updated_q = current_q + self.lr * (self.reward + self.gamma_decay * max_next_q - current_q)

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

            # Check if the received solution has more allocations than the current best
            if len(msg.allocations) > len(self.coalition_best_solution[1]):
                return  # Ignore this solution if it has more elements

            if len(msg.allocations) < len(self.coalition_best_solution[1]):
                return

            their_solution_fitness = Fitness.fitness_function_robot_pose(solution, self.cost_matrix,
                                                   [self.robot_cost_matrix[i] for i, purged in
                                                    enumerate(self.purged_agents) if not purged])

            our_solution_fitness = Fitness.fitness_function_robot_pose(self.coalition_best_solution, self.cost_matrix,
                                                        [self.robot_cost_matrix[i] for i, purged in
                                                         enumerate(self.purged_agents) if not purged])

            # Evaluate fitness only if the number of allocations is valid
            if their_solution_fitness < our_solution_fitness:
                print("Recieved a new best solution.")
                self.set_coalition_best_solution(solution)
                self.coalition_best_agent = msg.id

    def global_pose_callback(self, msg, agent):

        # Store the first received pose for each agent
        if self.initial_robot_poses[agent - 1] is None:
            self.initial_robot_poses[agent - 1] = deepcopy(msg.pose)

        task = deepcopy(self.current_task)

        self.robot_poses[agent - 1] = deepcopy(msg.pose)
        if all(pose is not None for pose in self.robot_poses) and self.is_loop_started is False:
            self.calculate_robot_cost_matrix()
            self.run_timer = self.create_timer(0.1, self.run_step, callback_group=self.me_cb_group)
            self.is_loop_started = True
        if self.task_poses is not None and task is not None:
            global_lat = msg.pose.position.latitude
            global_lon = msg.pose.position.longitude
            global_alt = msg.pose.position.altitude
            geoid_height = self.geoid.height(global_lat, global_lon)
            global_alt = global_alt - geoid_height

            goal_lat = self.task_poses[task]["latitude"]
            goal_lon = self.task_poses[task]["longitude"]
            goal_alt = self.task_poses[task]["altitude"]

            if agent == self.true_agent_ID and self.haversine(global_lat, global_lon, global_alt, goal_lat, goal_lon,
                                                              goal_alt) < 0.4:
                print(f"Agent {self.agent_ID} covered position {task}.")

                self.task_covered = task
                self.is_new_task_covered = True

    def handle_covered_task(self, current_task):
        self.is_covered[current_task] = True
        rep = EnvironmentalRepresentation()
        rep.is_covered = self.is_covered
        self.environmental_representaion_publisher.publish(rep)
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
                return (order, allocations)

            # Update population
            for idx, solution in enumerate(self.population):
                self.population[idx] = update_solution(solution)

            # Update current_solution and coalition_best_solution if they exist
            self.current_solution = update_solution(self.current_solution)
            self.coalition_best_solution = update_solution(self.coalition_best_solution)
        self.assign_next_task(self.coalition_best_solution)
        print(f"Task {current_task} is covered: {self.is_covered[current_task]}")

    def environmental_representation_callback(self, msg):
        # Ensure is_covered list is long enough
        if len(self.is_covered) < len(msg.is_covered):
            self.is_covered = self.is_covered + [False] * (len(msg.is_covered) - len(self.is_covered))

        # Persist True values from msg.is_covered
        for i in range(len(msg.is_covered)):
            if msg.is_covered[i] and not self.is_covered[i]:  # If new value is True, persist it
                self.handle_covered_task(i)
                self.is_covered[i] = True

    def environmental_representation_timer(self):
        rep = EnvironmentalRepresentation()
        rep.is_covered = self.is_covered
        self.environmental_representaion_publisher.publish(rep)

    def publish_goal_pose(self):
        if self.current_task is not None:
            goal_pose = GeoPoseStamped()

            # Set header
            goal_pose.header.stamp = Time()
            goal_pose.header.frame_id = "world"  # Change frame if needed

            # Set position (Latitude, Longitude, Altitude)
            goal_pose.pose.position.latitude = self.task_poses[self.current_task]["latitude"]
            goal_pose.pose.position.longitude = self.task_poses[self.current_task]["longitude"]
            goal_pose.pose.position.altitude = self.task_poses[self.current_task]["altitude"]  # Example altitude

            # Set orientation (Quaternion)
            goal_pose.pose.orientation.x = self.task_poses[self.current_task]["orientation_x"]
            goal_pose.pose.orientation.y = self.task_poses[self.current_task]["orientation_y"]
            goal_pose.pose.orientation.z = self.task_poses[self.current_task]["orientation_z"]
            goal_pose.pose.orientation.w = self.task_poses[self.current_task]["orientation_w"]
            self.goal_pose_publisher.publish(goal_pose)
        else:
            if self.initial_robot_poses[self.agent_ID - 1] is not None:
                goal_pose = GeoPoseStamped()

                # Set header
                goal_pose.header.stamp = Time()
                goal_pose.header.frame_id = "world"  # Change frame if needed

                # Set position (Latitude, Longitude, Altitude)
                goal_pose.pose.position.latitude = self.initial_robot_poses[self.agent_ID - 1].position.latitude
                goal_pose.pose.position.longitude = self.initial_robot_poses[self.agent_ID - 1].position.longitude
                goal_pose.pose.position.altitude = self.initial_robot_poses[
                    self.agent_ID - 1].position.altitude  # Example altitude

                # Set orientation (Quaternion)
                goal_pose.pose.orientation.x = self.initial_robot_poses[self.agent_ID - 1].orientation.x
                goal_pose.pose.orientation.y = self.initial_robot_poses[self.agent_ID - 1].orientation.y
                goal_pose.pose.orientation.z = self.initial_robot_poses[self.agent_ID - 1].orientation.z
                goal_pose.pose.orientation.w = self.initial_robot_poses[self.agent_ID - 1].orientation.w
                self.goal_pose_publisher.publish(goal_pose)

    @staticmethod
    def haversine(lat1, lon1, alt1, lat2, lon2, alt2):

        R = 6378160  # Earth radius in miles (for km use 6372.8)

        # Convert lat/lon from degrees to radians
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)

        # Haversine formula for surface distance
        a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
        c = 2 * asin(sqrt(a))
        haversine_distance = R * c  # Great-circle distance in miles

        # Compute altitude difference in miles
        alt_diff = alt2 - alt1  # Altitude difference in miles

        # Compute 3D distance using Pythagorean theorem
        distance_3D = sqrt(haversine_distance ** 2 + alt_diff ** 2)

        return distance_3D

    def select_random_solution(self):
        temp_solution = sample(population=self.population, k=1)[0]
        if temp_solution != self.current_solution:
            return temp_solution

    def robot_cost_matrix_recalculation(self):
        if all(pose is not None for pose in self.robot_poses):
            self.calculate_robot_cost_matrix()

    def regular_solution_publish_timer(self):
        if self.coalition_best_solution is not None:
            solution = Solution()
            solution.id = self.agent_ID
            solution.order = self.coalition_best_solution[0]
            solution.allocations = self.coalition_best_solution[1]
            self.solution_publisher.publish(solution)

    def run_step(self):
        """
        A single step of the `run` method, executed periodically by the ROS2 timer.
        """

        if self.is_new_task_covered:
            self.handle_covered_task(self.task_covered)
            self.task_covered = -1
            self.is_new_task_covered = False



        if self.current_solution is not None:


            if self.is_agent_tobe_purged and not self.purged_agents[self.last_purge_agent_true_id] and len(
                    self.current_solution[1]) > self.failed_agents.count(False):
                print("Purging agent")
                self.purge_agent(self.last_purge_agent_true_id)
                self.purged_agents[self.last_purge_agent_true_id] = True

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
                c_new = OperatorFunctions.apply_op(
                    operator,
                    self.current_solution,
                    self.population,
                    self.cost_matrix,
                    [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged]
                )
            except Exception as e:
                print(f"Issue with applying operator: {e}")
            if c_new:
                gain = Fitness.fitness_function_robot_pose(c_new, self.cost_matrix,
                                                           [self.robot_cost_matrix[i] for i, purged in
                                                            enumerate(self.purged_agents) if not purged]) - \
                       Fitness.fitness_function_robot_pose(self.current_solution, self.cost_matrix,
                                                           [self.robot_cost_matrix[i] for i, purged in
                                                            enumerate(self.purged_agents) if not purged])
                self.update_experience(condition, operator, gain)

                if self.local_best_solution is None or \
                        Fitness.fitness_function_robot_pose(c_new, self.cost_matrix,
                                                            [self.robot_cost_matrix[i] for i, purged in
                                                             enumerate(self.purged_agents) if
                                                             not purged]) < Fitness.fitness_function_robot_pose(
                    self.local_best_solution, self.cost_matrix,
                    [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged]):
                    self.local_best_solution = deepcopy(c_new)
                    self.best_local_improved = True
                    self.no_improvement_attempt_count = 0
                else:
                    self.no_improvement_attempt_count += 1

                if self.coalition_best_solution is None or \
                        Fitness.fitness_function_robot_pose(c_new, self.cost_matrix,
                                                            [self.robot_cost_matrix[i] for i, purged in
                                                             enumerate(self.purged_agents) if
                                                             not purged]) < Fitness.fitness_function_robot_pose(
                    self.coalition_best_solution, self.cost_matrix,
                    [self.robot_cost_matrix[i] for i, purged in enumerate(self.purged_agents) if not purged]):
                    self.set_coalition_best_solution(deepcopy(c_new))
                    self.coalition_best_agent = self.agent_ID
                    self.best_coalition_improved = True
                    solution = Solution()
                    solution.id = self.agent_ID
                    solution.order = self.coalition_best_solution[0]
                    solution.allocations = self.coalition_best_solution[1]
                    self.solution_publisher.publish(solution)

                self.current_solution = c_new
                self.di_cycle_count += 1

                if self.end_of_di_cycle(self.di_cycle_count):
                    if self.best_local_improved:
                        learning_method_switch = {
                            LearningMethod.FERREIRA: self.individual_learning_old,
                            LearningMethod.Q_LEARNING: self.individual_learning
                        }

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


def main(args=None):
    try:
        rclpy.init(args=args)

        temp_node = Node("parameter_loader")
        temp_node.declare_parameter("agent_id", 1)
        temp_node.declare_parameter("problem_filename", "resources/150_Task_Problem.csv")
        temp_node.declare_parameter("runtime", 60.0)
        temp_node.declare_parameter("learning_method", "Ferreira et al.")

        agent_id = temp_node.get_parameter("agent_id").value
        problem_filename = temp_node.get_parameter("problem_filename").value
        runtime = temp_node.get_parameter("runtime").value
        learning_method = temp_node.get_parameter("learning_method").value
        temp_node.destroy_node()

        problem = Problem()
        problem.load_cost_matrix(problem_filename, "csv")
        num_tasks = 2

        node_name = f"cbm_population_agent_{agent_id}"
        agent = CBMPopulationAgentOnline(
            pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
            num_tasks=num_tasks, num_tsp_agents=20, num_iterations=9999999999,
            num_solution_attempts=20, agent_id=agent_id, node_name=node_name,
            cost_matrix=problem.cost_matrix, learning_method=learning_method
        )

        def shutdown_callback():
            agent.get_logger().info("Runtime completed. Shutting down.")
            agent.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

        agent.create_timer(runtime, shutdown_callback)

        try:
            rclpy.spin(agent)
        except KeyboardInterrupt:
            pass
        finally:
            if rclpy.ok():
                agent.destroy_node()
                rclpy.shutdown()

    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)


if __name__ == '__main__':
    main()


# Deneme


def main(args=None):
    rclpy.init(args=args)

    temp_node = Node("parameter_loader")
    temp_node.declare_parameter("agent_id", 1)
    temp_node.declare_parameter("problem_filename", "resources/150_Task_Problem.csv")
    temp_node.declare_parameter("runtime", 60.0)
    temp_node.declare_parameter("learning_method", "Ferreira et al.")

    agent_id = temp_node.get_parameter("agent_id").value
    problem_filename = temp_node.get_parameter("problem_filename").value
    runtime = temp_node.get_parameter("runtime").value
    learning_method = temp_node.get_parameter("learning_method").value
    temp_node.destroy_node()

    problem = Problem()
    problem.load_cost_matrix(problem_filename, "csv")
    num_tasks = len(problem.cost_matrix)

    node_name = f"cbm_population_agent_{agent_id}"
    agent = CBMPopulationAgentOnline(
        pop_size=10, eta=0.1, rho=0.1, di_cycle_length=5, epsilon=0.01,
        num_tasks=num_tasks, num_tsp_agents=5, num_iterations=9999999,
        num_solution_attempts=20, agent_id=agent_id, node_name=node_name,
        cost_matrix=problem.cost_matrix, learning_method=learning_method
    )
    print("CBMPopulationAgentOnline has been initialized.")

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

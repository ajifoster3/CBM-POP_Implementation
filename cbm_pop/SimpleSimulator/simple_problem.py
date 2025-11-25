import math
from enum import Enum

import numpy as np


class ProblemClass(Enum):
    SimpleGrid = "Simple_Grid"
    RandomClusters = "Random_Clusters"
    RandomSpread = "Random_Spread"


class SimpleProblem:
    def __init__(self, problem_class, grid_size=15, problem_seed=1):
        self.task_poses = None
        self.initial_robot_cost_matrix = None
        self.current_robot_cost_matrix = None
        self.problem_class = problem_class

        try:
            size = int(grid_size)
        except (TypeError, ValueError):
            raise ValueError("grid_size must be an integer") from None

        if size < 1:
            raise ValueError("grid_size must be >= 1")

        self.grid_size = size

        if problem_class == ProblemClass.SimpleGrid:
            self.task_poses = [(i + 0.5, j + 0.5) for i in range(size) for j in range(size)]
        elif problem_class == ProblemClass.RandomClusters:
            self.task_poses = self.generate_random_clusters(size, problem_seed)
        elif problem_class == ProblemClass.RandomSpread:
            self.task_poses = self.generate_random_spread(size, problem_seed)
        else:
            raise NotImplementedError(f"Unsupported problem class: {problem_class}")
        self.cost_matrix = self.calculate_cost_matrix()
        self.num_tasks = len(self.task_poses)

    def generate_random_clusters(self, size, seed):
        """
        Generates random clusters for task positions with a random number of points
        distributed between clusters. The total number of points is `size`, and the
        separation between clusters is random.
        """
        np.random.seed(seed)  # For reproducibility
        num_clusters = 5  # Example number of clusters
        cluster_radius = 2.0  # Example cluster radius

        cluster_centers = [(np.random.uniform(0, size), np.random.uniform(0, size)) for _ in range(num_clusters)]
        task_poses = []

        # Generate random number of tasks for each cluster
        remaining_tasks = size*size
        tasks_per_cluster = []

        for i in range(num_clusters):
            if i == num_clusters - 1:  # Last cluster takes the remaining tasks
                tasks_per_cluster.append(remaining_tasks)
            else:
                num_tasks_in_cluster = np.random.randint(1, remaining_tasks // (num_clusters - i) + 1)
                tasks_per_cluster.append(num_tasks_in_cluster)
                remaining_tasks -= num_tasks_in_cluster

        # Assign random tasks to each cluster
        for i, center in enumerate(cluster_centers):
            num_tasks_in_cluster = tasks_per_cluster[i]
            for _ in range(num_tasks_in_cluster):
                x_offset = np.random.uniform(-cluster_radius, cluster_radius)
                y_offset = np.random.uniform(-cluster_radius, cluster_radius)
                x_task = np.clip(center[0] + x_offset, 0, size)
                y_task = np.clip(center[1] + y_offset, 0, size)
                task_poses.append((x_task, y_task))

        return task_poses

    def generate_random_spread(self, size, seed):
        np.random.seed(seed)  # For reproducibility

        task_poses = []
        total_tasks = size * size

        # Generate random task positions within the grid
        for _ in range(total_tasks):
            x_task = np.random.uniform(0, size)
            y_task = np.random.uniform(0, size)
            task_poses.append((x_task, y_task))

        return task_poses

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
                    x1, y1 = self.task_poses[i][0], self.task_poses[i][1]
                    x2, y2 = self.task_poses[j][0], self.task_poses[j][1]
                    x_dist = abs(x2 - x1)
                    y_dist = abs(y2 - y1)
                    cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))
                    cost_map[i][j] = cost
        return cost_map

    def update_robot_cost_matrix(self, robot_poses):
        """
        Returns a cost map representing the traversal cost from each robot_pose to each task_pose calculated using
        drone_distance.
        """
        valid_robot_poses = [pose for pose in robot_poses if pose is not None]
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_robots, num_tasks))
        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                x1, y1 = robot_pose
                x2, y2 = task_pose
                x_dist = x2 - x1
                y_dist = y2 - y1
                cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))
                cost_map[i][j] = cost
        self.current_robot_cost_matrix = cost_map

    def initialize_robot_initial_pose_cost_matrix(self,initial_robot_poses):
        """
        Returns a cost map representing the traversal cost from each initial_robot_pose to each each task_pose calculated
        using drone_distance.
        """
        valid_robot_poses = initial_robot_poses
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_robots, num_tasks))
        for i, robot_pose in enumerate(valid_robot_poses):
            for j, task_pose in enumerate(self.task_poses):
                x1, y1 = robot_pose
                x2, y2 = task_pose
                x_dist = x2 - x1
                y_dist = y2 - y1
                cost = math.sqrt((x_dist ** 2) + (y_dist ** 2))
                cost_map[i][j] = cost
        self.initial_robot_cost_matrix = cost_map
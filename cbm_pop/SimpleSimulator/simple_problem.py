import math
from enum import Enum

import numpy as np


class ProblemClass(Enum):
    SimpleGrid = 1


class SimpleProblem:
    def __init__(self, problem_class):
        self.task_poses = None
        self.initial_robot_cost_matrix = None
        self.current_robot_cost_matrix = None
        if problem_class == ProblemClass.SimpleGrid:
            self.task_poses = [(i + 0.5, j + 0.5) for i in range(15) for j in range(15)]
        self.cost_matrix = self.calculate_cost_matrix()
        self.num_tasks = len(self.task_poses)


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
import json
import math
from enum import Enum
import numpy as np
from math import radians, sin, asin, cos, sqrt
from pygeodesy.geoids import GeoidPGM

class ProblemClass(Enum):
    FourWhichWay = "FourWhichWay"


class RealisticProblem:
    def __init__(self, problem_class):
        self.geoid = GeoidPGM('/home/ajifoster3/Documents/Software/ros_ws/src/CBM-POP_Implementation/egm96-5.pgm')
        self.task_poses = None
        self.initial_robot_cost_matrix = None
        self.current_robot_cost_matrix = None
        self.problem_class = problem_class

        # ---- Load task poses from JSON ----
        if problem_class == ProblemClass.FourWhichWay:
            filename = "/home/ajifoster3/Downloads/Poses/4Whichway_geoposes.json"
        else:
            raise NotImplementedError(f"Unsupported problem class: {problem_class}")

        with open(filename, "r") as file:
            print(f"using {filename}")
            data = json.load(file)

        self.task_poses = []
        for item in data:
            lat = item["GeoPose"]["position"]["latitude"]
            lon = item["GeoPose"]["position"]["longitude"]
            alt = item["GeoPose"]["position"]["altitude"]

            geoid_offset = self.geoid.height(lat, lon)

            self.task_poses.append(
                {
                    "latitude": lat,
                    "longitude": lon,
                    "altitude": alt + geoid_offset,  # corrected altitude
                    "orientation_x": item["GeoPose"]["orientation"]["x"],
                    "orientation_y": item["GeoPose"]["orientation"]["y"],
                    "orientation_z": item["GeoPose"]["orientation"]["z"],
                    "orientation_w": item["GeoPose"]["orientation"]["w"],
                }
            )

        self.num_tasks = len(self.task_poses)

        # Precompute full task–task cost matrix
        self.cost_matrix = self.calculate_cost_matrix()

    # =====================================================================
    # Distance helpers
    # =====================================================================
    @staticmethod
    def haversine(lat1, lon1, alt1, lat2, lon2, alt2):
        """
        3D distance between (lat1, lon1, alt1) and (lat2, lon2, alt2).

        - Horizontal: great-circle distance on a sphere (Haversine formula)
        - Vertical: simple difference in altitude
        - Returns: distance in metres
        """
        R = 6_378_160.0  # Earth radius in metres

        # Convert lat/lon from degrees to radians
        dLat = radians(lat2 - lat1)
        dLon = radians(lon2 - lon1)
        lat1 = radians(lat1)
        lat2 = radians(lat2)

        # Haversine formula for surface distance
        a = sin(dLat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dLon / 2) ** 2
        c = 2 * asin(sqrt(a))
        haversine_distance = R * c  # Great-circle distance in metres

        # Altitude difference (assumed metres, consistent with GeoPose / task_poses)
        alt_diff = alt2 - alt1

        # 3D distance
        distance_3D = math.sqrt(haversine_distance ** 2 + alt_diff ** 2)
        return distance_3D

    # =====================================================================
    # Task–task cost matrix (precomputed)
    # =====================================================================
    def calculate_cost_matrix(self):
        """
        Cost from each task to every other task using the drone-distance model.
        """
        num_tasks = len(self.task_poses)
        cost_map = np.zeros((num_tasks, num_tasks), dtype=float)

        for i in range(num_tasks):
            lat1 = self.task_poses[i]["latitude"]
            lon1 = self.task_poses[i]["longitude"]
            alt1 = self.task_poses[i]["altitude"]

            for j in range(num_tasks):
                if i == j:
                    continue

                lat2 = self.task_poses[j]["latitude"]
                lon2 = self.task_poses[j]["longitude"]
                alt2 = self.task_poses[j]["altitude"]

                # Separate horizontal & vertical for use in calculate_drone_distance
                horizontal = self.haversine(lat1, lon1, 0.0, lat2, lon2, 0.0)
                vertical = alt2 - alt1

                cost = self.calculate_drone_distance(
                    horizontal_distance=horizontal,
                    vertical_distance=vertical,
                    ascending_speed=3.0,
                    descending_speed=1.45,
                    horizontal_speed=11.0,
                )
                cost_map[i, j] = cost

        return cost_map

    # =====================================================================
    # Robot–task cost matrices (current & initial)
    # =====================================================================
    def update_robot_cost_matrix(self, robot_poses):
        """
        Build cost map from EACH robot's current GeoPose to EACH task pose using
        the drone-distance model.

        robot_poses: list[GeoPose] (possibly containing None entries)
        """
        valid_robot_poses = [pose for pose in robot_poses if pose is not None]
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        cost_map = np.zeros((num_robots, num_tasks), dtype=float)

        for i, robot_pose in enumerate(valid_robot_poses):
            lat1 = robot_pose.position.latitude
            lon1 = robot_pose.position.longitude
            alt1 = robot_pose.position.altitude

            for j, task_pose in enumerate(self.task_poses):
                lat2 = task_pose["latitude"]
                lon2 = task_pose["longitude"]
                alt2 = task_pose["altitude"]

                horizontal = self.haversine(lat1, lon1, 0.0, lat2, lon2, 0.0)
                vertical = alt2 - alt1

                cost = self.calculate_drone_distance(
                    horizontal_distance=horizontal,
                    vertical_distance=vertical,
                    ascending_speed=3.0,
                    descending_speed=1.45,
                    horizontal_speed=11.0,
                )
                cost_map[i, j] = cost

        self.current_robot_cost_matrix = cost_map

    def initialize_robot_initial_pose_cost_matrix(self, initial_robot_poses):
        """
        Build cost map from EACH robot's *initial* GeoPose to EACH task pose using
        the drone-distance model.

        initial_robot_poses: list[GeoPose]
        """
        valid_robot_poses = [pose for pose in initial_robot_poses if pose is not None]
        num_robots = len(valid_robot_poses)
        num_tasks = len(self.task_poses)

        cost_map = np.zeros((num_robots, num_tasks), dtype=float)

        for i, robot_pose in enumerate(valid_robot_poses):
            lat1 = robot_pose.position.latitude
            lon1 = robot_pose.position.longitude
            alt1 = robot_pose.position.altitude

            for j, task_pose in enumerate(self.task_poses):
                lat2 = task_pose["latitude"]
                lon2 = task_pose["longitude"]
                alt2 = task_pose["altitude"]

                horizontal = self.haversine(lat1, lon1, 0.0, lat2, lon2, 0.0)
                vertical = alt2 - alt1

                cost = self.calculate_drone_distance(
                    horizontal_distance=horizontal,
                    vertical_distance=vertical,
                    ascending_speed=3.0,
                    descending_speed=1.45,
                    horizontal_speed=11.0,
                )
                cost_map[i, j] = cost

        self.initial_robot_cost_matrix = cost_map

    # =====================================================================
    # Drone trajectory "distance" model
    # =====================================================================
    def calculate_drone_distance(
        self,
        horizontal_distance,
        vertical_distance,
        ascending_speed,
        descending_speed,
        horizontal_speed,
    ):
        """
        Calculate an effective travel distance for a drone that moves vertically and
        horizontally with different speeds.

        We treat the path as:
          1. A simultaneous segment where both vertical and horizontal motion occur
             until one axis finishes.
          2. A remaining segment purely along the longer axis.
        """

        try:
            # Vertical speed depends on direction
            vertical_speed = ascending_speed if vertical_distance >= 0 else descending_speed

            # Times along each axis
            time_vertical = abs(vertical_distance) / vertical_speed if vertical_speed > 0 else 0.0
            time_horizontal = horizontal_distance / horizontal_speed if horizontal_speed > 0 else 0.0

            # Time over which both axes move together
            common_time = min(time_vertical, time_horizontal)

            # Distances in the common phase
            vertical_common = vertical_speed * common_time
            horizontal_common = horizontal_speed * common_time

            segment1_distance = math.sqrt(vertical_common ** 2 + horizontal_common ** 2)

            # Remaining distance in whichever axis hasn't finished
            remaining_distance = 0.0
            if time_vertical > time_horizontal:
                # Still going vertically
                remaining_distance = vertical_speed * (time_vertical - time_horizontal)
            elif time_horizontal > time_vertical:
                # Still going horizontally
                remaining_distance = horizontal_speed * (time_horizontal - time_vertical)

            total_distance = segment1_distance + remaining_distance
            return total_distance

        except Exception as e:
            print(f"Error in calculate_drone_distance: {e}")
            return -1.0

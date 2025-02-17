import math

import numpy as np

def horiz_accel_func(time, velocity, remaining_distance):
    return min(0.2 * (1.3 * remaining_distance) ** 4, 4.0)


def vert_accel_func(time, velocity, remaining_distance):
    return min(0.2 * (1.3 * remaining_distance) ** 4, 8.0)


def descend_accel_func(time, velocity, remaining_distance):
    return min(0.2 * (1.3 * remaining_distance) ** 4, 4.0)

def calculate_drone_distance(vertical_distance, horizontal_distance, ascending_speed, descending_speed,
                             horizontal_speed):
    """
    Calculate the total drone travel distance considering vertical and horizontal movement.

    :param vertical_distance: Vertical distance to be traveled (positive for ascending, negative for descending)
    :param horizontal_distance: Horizontal distance to be traveled
    :param ascending_speed: Speed of ascent
    :param descending_speed: Speed of descent
    :param horizontal_speed: Speed of horizontal movement
    :param output_file: File to log the calculation details (default: "drone_distances.txt")
    :return: Total distance traveled by the drone
    """
    try:
        # Determine vertical speed based on direction
        vertical_speed = ascending_speed if vertical_distance >= 0 else descending_speed

        # Compute time required for each movement
        time_vertical = abs(vertical_distance) / vertical_speed
        time_horizontal = horizontal_distance / horizontal_speed

        # Find the common time where both movements happen simultaneously
        common_time = min(time_vertical, time_horizontal)

        # Compute distances covered in the common time
        vertical_distance_common = vertical_speed * common_time
        horizontal_distance_common = horizontal_speed * common_time

        # Compute first segment distance
        segment1_distance = math.sqrt(vertical_distance_common ** 2 + horizontal_distance_common ** 2)

        # Compute remaining distance in the longer direction
        remaining_distance = 0.0
        if time_vertical > time_horizontal:
            remaining_distance = vertical_speed * (time_vertical - time_horizontal)  # Remaining vertical distance
        elif time_horizontal > time_vertical:
            remaining_distance = horizontal_speed * (
                        time_horizontal - time_vertical)  # Remaining horizontal distance

        # Compute total distance
        total_distance = segment1_distance + remaining_distance

        return total_distance
    except Exception as e:
        print(f"Error: {e}")
        return -1.0

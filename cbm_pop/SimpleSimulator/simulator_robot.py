import math
import sys

from cbm_pop_interfaces.msg import SimplePosition, FinishedCoverage
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from rclpy.node import Node


class SimulatorRobot(Node):
    def __init__(self, robot_id, robot_starting_position, robot_speed, num_robots):
        super().__init__(f'simulator_robot_{robot_id}')
        self.robot_id = robot_id
        self.robot_position = robot_starting_position
        self.robot_speed = robot_speed
        self.current_goal_pose = None
        self.is_finished = False
        self.finished_robots = [False] * num_robots

        self.global_pose_publisher = self.create_publisher(
            SimplePosition,
            f'/central_control/uas_{robot_id}/global_pose',
            10)

        self.goal_pose_subscriber = self.create_subscription(
            SimplePosition,
            f'/central_control/uas_{robot_id}/goal_pose',
            self.goal_pose_callback,
            10)

        self.finished_coverage_sub = self.create_subscription(
            FinishedCoverage,
            f'/central_control/finished_coverage',
            self.finished_coverage_callback,
            10
        )

        # Create a timer to tick every 0.1 seconds (10 Hz)
        self.timer_period = 0.5  # seconds
        self.timer = self.create_timer(self.timer_period, self.publish_position)

    def move_robot(self):
        if self.current_goal_pose:
            robot_position_x = self.robot_position[-1][0]
            robot_position_y = self.robot_position[-1][1]

            goal_position_x = self.current_goal_pose[0]
            goal_position_y = self.current_goal_pose[1]

            difference_x = goal_position_x - robot_position_x
            difference_y = goal_position_y - robot_position_y

            distance = math.hypot(difference_x, difference_y)

            if distance > 1e-3:  # a small tolerance to avoid floating point noise
                if distance <= self.robot_speed:
                    new_x, new_y = goal_position_x, goal_position_y
                else:
                    angle = math.atan2(difference_y, difference_x)
                    new_x = robot_position_x + math.cos(angle) * self.robot_speed
                    new_y = robot_position_y + math.sin(angle) * self.robot_speed

                self.robot_position.append((new_x, new_y))
            else:
                self.robot_position.append(self.robot_position[-1])
        else:
            self.robot_position.append(self.robot_position[-1])

    def finished_coverage_callback(self, msg):
        self.finished_robots[int(msg.robot_id)] = bool(msg.finished)
        if all(self.finished_robots):
            print("Coverage Complete")
            self.is_finished = True
            self.destroy_node()

    def publish_position(self):
        position = SimplePosition()
        position.robot_id = int(self.robot_id)
        position.x_position = float(self.robot_position[-1][0])
        position.y_position = float(self.robot_position[-1][1])
        self.global_pose_publisher.publish(position)

    def goal_pose_callback(self, msg):
        self.current_goal_pose = (msg.x_position, msg.y_position)

    def get_robot_position(self):
        return self.robot_position

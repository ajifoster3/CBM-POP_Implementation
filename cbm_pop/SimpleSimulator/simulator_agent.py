import math


class SimulatorAgent:
    def __init__(self, robot_starting_position, robot_speed, robot_goal_poses):
        self.robot_position = robot_starting_position
        self.robot_speed = robot_speed
        self.robot_goal_poses = robot_goal_poses

    def move_robot(self):
        robot_position_x = self.robot_position[0][-1]
        robot_position_y = self.robot_position[1][-1]

        goal_position_x = self.robot_goal_poses[0][0]
        goal_position_y = self.robot_goal_poses[0][1]

        difference_x = goal_position_x - robot_position_x
        difference_y = goal_position_y - robot_position_y

        angle = math.atan2(difference_y, difference_x)

        if abs(goal_position_x - robot_position_x) > 0.02 or abs(goal_position_y - robot_position_y) > 0.05:
            self.robot_position[0].append(
                self.robot_position[0][-1] + math.cos(angle) * self.robot_speed)
            self.robot_position[1].append(
                self.robot_position[1][-1] + math.sin(angle) * self.robot_speed)
        else:
            self.robot_position[0].append(self.robot_position[0][-1])
            self.robot_position[1].append(self.robot_position[1][-1])
            if len(self.robot_goal_poses) > 1:
                self.robot_goal_poses.remove(self.robot_goal_poses[0])

    def get_robot_position(self):
        return self.robot_position

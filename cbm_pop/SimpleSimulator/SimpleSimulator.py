import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.ticker as ticker

class SimpleSimulator:
    def __init__(self, tasks, environmental_bounds, robot_starting_positions, robot_speed, obstacles):
        self.environmental_bounds = environmental_bounds
        self.tasks = tasks
        self.robot_starting_positions = robot_starting_positions
        self.robot_speed = robot_speed
        self.obstacles = obstacles

        # Simulate robot positions over time
        T = 100  # total time steps
        self.t = 0
        self.robot_positions = self.robot_starting_positions

        self.robot_goal_pose = [13,4]

        # Setup the plot
        fig, ax = plt.subplots()
        ax.set_xlim(self.environmental_bounds[0], self.environmental_bounds[1])
        ax.set_ylim(self.environmental_bounds[2], self.environmental_bounds[3])
        ax.set_aspect('equal')
        ax.set_title("2-Robot Environment Over Time")
        ax.minorticks_on()
        robot1, = ax.plot([], [], 'ro', label='Robot 1')  # red dot
        robot2, = ax.plot([], [], 'bo', label='Robot 2')  # blue dot
        taskplots, = ax.plot([],[], '.', markersize=0.1, label='Tasks')
        path1, = ax.plot([], [], 'r--', alpha=0.5)  # Robot 1 path
        path2, = ax.plot([], [], 'b--', alpha=0.5)  # Robot 2 path

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))  # every 1 unit on x-axis
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))  # every 1 unit on y-axis

        ax.grid(True, which='major', linestyle='--', linewidth=0.5)
        ax.legend()

        # Initialization function
        def init():
            robot1.set_data([], [])
            robot2.set_data([], [])
            taskplots.set_data(zip(*self.tasks))
            path1.set_data([], [])
            path2.set_data([], [])
            return robot1, robot2, path1, path2

        # Update function for animation
        def update(frame):
            robot1.set_data([self.robot_positions[1][0][self.t]], [self.robot_positions[1][1][self.t]])
            robot2.set_data([self.robot_positions[0][0][self.t]], [self.robot_positions[0][1][self.t]])
            path1.set_data(self.robot_positions[1][0][:self.t], self.robot_positions[1][1][:self.t])
            path2.set_data(self.robot_positions[0][0][:self.t], self.robot_positions[0][1][:self.t])
            self.robot_positions[1][0].append(self.robot_positions[1][0][self.t])
            self.robot_positions[1][1].append(self.robot_positions[1][1][self.t])

            robot_position_x = self.robot_positions[0][0][self.t]
            robot_position_y = self.robot_positions[0][1][self.t]

            goal_position_x = self.robot_goal_pose[0]
            goal_position_y = self.robot_goal_pose[1]

            difference_x = goal_position_x-robot_position_x
            difference_y = goal_position_y-robot_position_y

            angle = math.atan2(difference_y, difference_x)

            self.robot_positions[1][0].append(self.robot_positions[1][0][self.t])
            self.robot_positions[1][1].append(self.robot_positions[1][1][self.t])
            if (not (0.02 > robot_position_x) or not (0.02 < robot_position_x)) and (
                    robot_position_y >= goal_position_y + 0.02 or robot_position_y <= goal_position_y - 0.02):
                    self.robot_positions[0][0].append(self.robot_positions[0][0][self.t] + math.cos(angle) * robot_speed)
                    self.robot_positions[0][1].append(self.robot_positions[0][1][self.t] + math.sin(angle) * robot_speed)
            else:
                self.robot_positions[0][0].append(self.robot_positions[0][0][self.t])
                self.robot_positions[0][1].append(self.robot_positions[0][1][self.t])
            self.t += 1

            return robot1, robot2, path1, path2

        # Create the animation
        ani = FuncAnimation(fig, update, frames=T, init_func=init, blit=True, interval=50)

        plt.show()


if __name__ == '__main__':
    tasks = []
    environmental_bounds = [0, 15, 0, 15]
    for i in range(environmental_bounds[1]):
        for j in range(environmental_bounds[3]):
            tasks.append([i+0.5, j+0.5])


    SimpleSimulator(tasks,environmental_bounds,[[[7], [5]], [[8], [5]]],0.025,1)
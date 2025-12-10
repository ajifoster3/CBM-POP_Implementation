import math
import random
import sys
import traceback
from time import time
import numpy as np
from copy import deepcopy

from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool

# Import Simulation Interfaces
from cbm_pop.SimpleSimulator.simple_problem import SimpleProblem, ProblemClass
from cbm_pop_interfaces.msg import (
    EnvironmentalRepresentation,
    SimplePosition,
    FinishedCoverage,
    CurrentTask
)


class GreedyAgentOnlineSimpleSimulation(Node):

    def __init__(self, agent_id, node_name: str,
                 num_tsp_agents=10,
                 problem_size=15,
                 problem_class=ProblemClass.SimpleGrid,
                 problem_seed=1):

        super().__init__(node_name)

        # --- Configuration ---
        self.agent_ID = agent_id
        self.num_tsp_agents = num_tsp_agents
        self.problem = SimpleProblem(ProblemClass(problem_class), grid_size=problem_size, problem_seed=problem_seed)

        self.get_logger().info(f"Greedy Agent {self.agent_ID} Initialized. Problem Size: {problem_size}")

        # --- State ---
        self.current_task = None
        self.is_covered = [False] * self.problem.num_tasks

        # To track peer activities for "Non-Allocated" check
        self.peer_current_tasks = [-1] * num_tsp_agents

        self.initial_robot_poses = [None] * self.num_tsp_agents
        self.robot_poses = [None] * self.num_tsp_agents

        # Flags
        self.is_loop_started = False
        self.is_finished = False
        self.am_i_failed = False
        self.is_all_poses = False
        self.failed_agents = [False] * self.num_tsp_agents
        self.finished_robots = [False] * self.num_tsp_agents

        # --- ROS Callbacks Groups ---
        self.cb_group = ReentrantCallbackGroup()
        self.me_cb_group = MutuallyExclusiveCallbackGroup()

        # --- Publishers ---
        self.goal_pose_publisher = self.create_publisher(
            SimplePosition,
            f'/central_control/uas_{agent_id}/goal_pose',
            10
        )

        self.environmental_representation_publisher = self.create_publisher(
            EnvironmentalRepresentation,
            '/environmental_representation',
            10
        )

        self.current_task_publisher = self.create_publisher(
            CurrentTask,
            'current_task',
            10
        )

        self.finished_coverage_pub = self.create_publisher(
            FinishedCoverage,
            '/central_control/finished_coverage',
            10
        )

        # --- Subscribers ---

        # 1. Peer Tasks (To avoid allocated tasks)
        self.current_task_subscriber = self.create_subscription(
            CurrentTask,
            'current_task',
            self.current_task_update_callback,
            10
        )

        # 2. Environmental Representation (To know what is done)
        self.environmental_representation_subscriber = self.create_subscription(
            EnvironmentalRepresentation,
            '/environmental_representation',
            self.__environmental_representation_callback,
            40,
            callback_group=self.cb_group
        )

        # 3. Global Poses (To calculate distance)
        for id in range(self.num_tsp_agents):
            topic = f'/central_control/uas_{id}/global_pose'
            self.create_subscription(
                SimplePosition,
                topic,
                lambda msg: self.__global_pose_callback(msg),
                10,
                callback_group=self.cb_group
            )

        # 4. Kill/Revive (Simulation specific)
        self.create_subscription(Bool, f'/central_control/uas_{agent_id}/kill_robot',
                                 lambda msg: self.kill_robot_callback(msg), 10)
        self.create_subscription(Bool, f'/central_control/uas_{agent_id}/revive_robot',
                                 self.revive_robot_callback, 10)

        # 5. Finish Signal
        self.create_subscription(FinishedCoverage, '/central_control/finished_coverage',
                                 self.__finished_coverage_callback, 10)

        # --- Timers ---
        # Main logic loop (Run frequent updates)
        self.run_timer = None  # Started after receiving poses

        # Publish my current goal regularly so others don't take it
        self.create_timer(1.0, self.publish_current_task_allocation, callback_group=self.cb_group)

        # Publish what I know is covered
        self.create_timer(2.0, self.publish_env_rep, callback_group=self.cb_group)

        # Publish physical goal to controller
        self.create_timer(0.5, self.__publish_goal_pose, callback_group=self.cb_group)

    # ================= LOGIC =================

    def run_step(self):
        """
        The main brain of the Greedy Agent.
        """
        if self.am_i_failed or self.is_finished:
            return

        # 1. Check if we need a task
        if self.current_task is None:
            self.select_greedy_task()

        # 2. Check if the task we have is already done by someone else (race condition)
        elif self.is_covered[self.current_task]:
            self.current_task = None
            self.select_greedy_task()

    def select_greedy_task(self):
        """
        Selects the closest task.
        If no tasks remain, ensures current_task is None to trigger Return-to-Home.
        """
        my_pose = self.robot_poses[self.agent_ID]
        if my_pose is None:
            return

        best_task = None
        min_dist = float('inf')
        allocated_tasks = set(self.peer_current_tasks)

        for i in range(self.problem.num_tasks):
            if self.is_covered[i]: continue
            if i in allocated_tasks: continue

            t_x, t_y = self.problem.task_poses[i]
            dist = math.sqrt((my_pose[0] - t_x) ** 2 + (my_pose[1] - t_y) ** 2)

            if dist < min_dist:
                min_dist = dist
                best_task = i

        if best_task is not None:
            self.current_task = best_task
            self.get_logger().info(f"Greedy Selected Task: {self.current_task} (Dist: {min_dist:.2f})")
            self.publish_current_task_allocation()
        else:
            # No tasks available. Set to None to trigger "Return to Home" behavior.
            # Do NOT finish_mission() here.
            self.current_task = None

    def finish_mission(self):
        self.is_finished = True
        self.current_task = None  # Go home or idle
        msg = FinishedCoverage()
        msg.finished = True
        msg.robot_id = self.agent_ID
        self.finished_coverage_pub.publish(msg)
        self.get_logger().info("No available tasks. Mission Complete.")

    # ================= CALLBACKS =================

    def __global_pose_callback(self, msg):
        self.robot_poses[msg.robot_id] = (msg.x_position, msg.y_position)

        if self.initial_robot_poses[msg.robot_id] is None:
            self.initial_robot_poses[msg.robot_id] = (msg.x_position, msg.y_position)

        if not self.is_loop_started and all(p is not None for p in self.robot_poses):
            self.is_loop_started = True
            self.run_timer = self.create_timer(0.1, self.run_step, callback_group=self.me_cb_group)

        # Only check logic for MY robot
        if msg.robot_id != self.agent_ID:
            return

        # CASE A: Working on a Task
        if self.current_task is not None:
            gx, gy = self.problem.task_poses[self.current_task]
            dist = math.sqrt((msg.x_position - gx) ** 2 + (msg.y_position - gy) ** 2)
            if dist < 0.15:
                self.handle_task_completion(self.current_task)

        # CASE B: Returning Home (No task, but not marked finished yet)
        elif self.current_task is None and not self.is_finished:
            # Double check that no tasks are left globally
            remaining = [i for i, c in enumerate(self.is_covered) if not c]

            if not remaining:
                # Calculate distance to HOME (Initial Pose)
                hx, hy = self.initial_robot_poses[self.agent_ID]
                dist_home = math.sqrt((msg.x_position - hx) ** 2 + (msg.y_position - hy) ** 2)

                # If arrived at home, NOW we finish
                if dist_home < 0.15:
                    self.get_logger().info(f"Arrived Home. Reporting finish.")
                    self.finish_mission()

    def handle_task_completion(self, task_id):
        self.get_logger().info(f"Covered Task {task_id}")
        self.is_covered[task_id] = True
        self.current_task = None  # Free to pick new one

        # Broadcast update
        rep = EnvironmentalRepresentation()
        rep.agent_id = self.agent_ID
        rep.is_covered = list(self.is_covered)
        self.environmental_representation_publisher.publish(rep)

    def current_task_update_callback(self, msg):
        """
        Updates the state of peer tasks and resolves conflicts based on Agent ID.
        Rule: Lower Agent ID keeps the task. Higher Agent ID yields.
        """
        # 1. Always update the record of what the peer is doing
        self.peer_current_tasks[msg.agent_id] = msg.current_task

        # If I am failed or finished, I don't care about conflicts
        if self.am_i_failed or self.is_finished:
            return

        # 2. Check for Conflict
        # Condition: I have a task AND the peer is claiming the SAME task
        if (self.current_task is not None and
                msg.agent_id != self.agent_ID and
                msg.current_task == self.current_task):

            # 3. Resolution Logic (Lower ID Wins)
            if self.agent_ID > msg.agent_id:
                # I have the Higher ID -> I Lose -> I Yield
                self.get_logger().warn(
                    f"[CONFLICT] Agent {msg.agent_id} (Lower ID) claimed Task {self.current_task}. "
                    f"Yielding and re-selecting."
                )

                # Drop current task
                self.current_task = None

                # Immediately select a new task
                # (Note: select_greedy_task uses self.peer_current_tasks,
                # which we just updated, so it will avoid the conflict task)
                self.select_greedy_task()

            else:
                # I have the Lower ID -> I Win -> I Keep
                # The other agent will receive my message and yield.
                self.get_logger().info(
                    f"[CONFLICT] Agent {msg.agent_id} claimed Task {self.current_task}. "
                    f"I have Lower ID ({self.agent_ID} < {msg.agent_id}), keeping task."
                )

    def __environmental_representation_callback(self, msg):
        # Merge knowledge
        for i, covered in enumerate(msg.is_covered):
            if covered:
                self.is_covered[i] = True

    def __finished_coverage_callback(self, msg):
        self.finished_robots[int(msg.robot_id)] = True

        # If ALL robots have reported in as finished
        if all(self.finished_robots):
            # The bash script greps for exactly this string:
            self.get_logger().info("Coverage Complete")

            # Optional: Small delay or just shut down
            self.destroy_node()
            sys.exit()

    def publish_current_task_allocation(self):
        if self.current_task is not None:
            msg = CurrentTask()
            msg.agent_id = self.agent_ID
            msg.current_task = self.current_task
            self.current_task_publisher.publish(msg)

    def publish_env_rep(self):
        rep = EnvironmentalRepresentation()
        rep.agent_id = self.agent_ID
        rep.is_covered = list(self.is_covered)
        self.environmental_representation_publisher.publish(rep)

    def __publish_goal_pose(self):
        """
        Publish this agent's current goal pose to the flight controller.
        """
        if self.current_task is not None and self.problem.task_poses:
            goal_pose = SimplePosition()
            goal_pose.robot_id = self.agent_ID
            goal_pose.x_position = float(self.problem.task_poses[self.current_task][0])
            goal_pose.y_position = float(self.problem.task_poses[self.current_task][1])
            self.goal_pose_publisher.publish(goal_pose)
        else:
            try:
                if self.am_i_failed:
                    goal_pose = SimplePosition()
                    goal_pose.robot_id = self.agent_ID
                    goal_pose.x_position = self.initial_robot_poses[self.agent_ID][0]
                    goal_pose.y_position = self.initial_robot_poses[self.agent_ID][1]
                    self.goal_pose_publisher.publish(goal_pose)
                    return

                if self.initial_robot_poses[self.agent_ID] is not None:
                    goal_pose = SimplePosition()
                    goal_pose.robot_id = self.agent_ID
                    goal_pose.x_position = self.initial_robot_poses[self.agent_ID][0]
                    goal_pose.y_position = self.initial_robot_poses[self.agent_ID][1]
                    self.goal_pose_publisher.publish(goal_pose)
            except:
                print("Goal pose error")

    # ================= FAULT INJECTION =================

    def kill_robot_callback(self, msg):
        if msg.data:
            self.get_logger().warn("KILL SIGNAL RECEIVED")

            # 1. Explicitly release the task to peers before shutting down
            if self.current_task is not None:
                self.publish_task_release()

            self.am_i_failed = True
            self.current_task = None

            # Publish current pos as goal to stop
            if self.robot_poses[self.agent_ID]:
                goal = SimplePosition()
                goal.robot_id = self.agent_ID

                # Use current position to freeze in place,
                # or initial position to return home.
                # (Freezing in place is usually better for failure simulation)
                curr_pose = self.robot_poses[self.agent_ID]
                goal.x_position = curr_pose[0]
                goal.y_position = curr_pose[1]

                self.goal_pose_publisher.publish(goal)

    def publish_task_release(self):
        """
        Broadcasts task -1 so peers know I am no longer working on anything.
        """
        msg = CurrentTask()
        msg.agent_id = self.agent_ID
        msg.current_task = -1  # -1 indicates "No Task" / "Unallocated"
        self.current_task_publisher.publish(msg)
        self.get_logger().info("Released current task due to failure.")

    def revive_robot_callback(self, msg):
        if msg.data and self.am_i_failed:
            self.get_logger().warn("REVIVE SIGNAL RECEIVED")
            self.am_i_failed = False
            # Logic will resume in run_step


# ================= MAIN =================

def main(args=None):
    rclpy.init(args=args)

    # Param Loader Node
    temp_node = Node("loader")
    temp_node.declare_parameter("agent_id", 0)
    temp_node.declare_parameter("num_tsp_agents", 3)
    temp_node.declare_parameter("problem_size", 10)

    agent_id = temp_node.get_parameter("agent_id").value
    num_agents = temp_node.get_parameter("num_tsp_agents").value
    prob_size = temp_node.get_parameter("problem_size").value
    temp_node.destroy_node()

    agent = GreedyAgentOnlineSimpleSimulation(
        agent_id=agent_id,
        node_name=f'greedy_agent_{agent_id}',
        num_tsp_agents=num_agents,
        problem_size=prob_size
    )

    executor = MultiThreadedExecutor()
    executor.add_node(agent)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
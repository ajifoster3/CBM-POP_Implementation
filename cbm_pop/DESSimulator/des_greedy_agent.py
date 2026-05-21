"""
DES-compatible greedy agent.

Selects the nearest uncovered, unclaimed task.  No optimisation loop —
the simulation calls select_task() after every state change.
"""

import math
from typing import List, Optional, Tuple

from cbm_pop.SimpleSimulator.simple_problem import SimpleProblem


class DESGreedyAgent:
    def __init__(self, agent_id: int, num_agents: int, problem: SimpleProblem):
        self.agent_id   = agent_id
        self.num_agents = num_agents
        self.problem    = problem

        self.is_alive:      bool          = True
        self.is_covered:    List[bool]    = [False] * problem.num_tasks
        self.current_task:  Optional[int] = None
        self.failed_agents: List[bool]    = [False] * num_agents

        self.robot_poses:         List[Optional[Tuple]] = [None] * num_agents
        self.initial_robot_poses: List[Optional[Tuple]] = [None] * num_agents

    def initialise(self, robot_poses: List[Tuple[float, float]]) -> None:
        for i, pos in enumerate(robot_poses):
            self.robot_poses[i]         = pos
            self.initial_robot_poses[i] = pos

    def select_task(self, claimed_tasks: set) -> None:
        """
        Pick the nearest uncovered task not already in claimed_tasks.
        Updates self.current_task in-place.

        claimed_tasks — set of task IDs already assigned to higher-priority agents.
        """
        if not self.is_alive:
            self.current_task = None
            return

        pos = self.robot_poses[self.agent_id]
        if pos is None:
            self.current_task = None
            return

        best_task = None
        min_dist  = float('inf')
        for t, task_pos in enumerate(self.problem.task_poses):
            if self.is_covered[t] or t in claimed_tasks:
                continue
            dist = math.hypot(pos[0] - task_pos[0], pos[1] - task_pos[1])
            if dist < min_dist:
                min_dist  = dist
                best_task = t

        self.current_task = best_task

    def handle_task_covered(self, task_id: int) -> None:
        if self.is_covered[task_id]:
            return
        self.is_covered[task_id] = True
        if self.current_task == task_id:
            self.current_task = None

    def kill_robot(self, robot_id: int) -> None:
        self.failed_agents[robot_id] = True
        if robot_id == self.agent_id:
            self.is_alive     = False
            self.current_task = None

    def revive_robot(self, robot_id: int) -> None:
        self.failed_agents[robot_id] = False
        if robot_id == self.agent_id:
            self.is_alive = True

"""
DES orchestrator for the greedy baseline.

Only ROBOT_ARRIVAL events are used — there is no optimisation loop.
After every state change (arrival, kill, revive) a global greedy
assignment pass runs: agents are processed in ascending ID order and
each claims the nearest uncovered, unclaimed task.

Logging is identical to DESSimulation so results can be compared
side-by-side.  operator_log.csv / di_cycle_log.csv / weight_matrix_log.csv
will be created with headers but no data rows.
"""

from typing import List, Optional, Tuple

import numpy as np

from cbm_pop.DESSimulator.des_greedy_agent import DESGreedyAgent
from cbm_pop.DESSimulator.des_event import EventQueue, EventType
from cbm_pop.DESSimulator.des_robot import DESRobot
from cbm_pop.SimpleSimulator.simple_problem import SimpleProblem


class DESGreedySimulation:
    def __init__(
        self,
        problem:          SimpleProblem,
        num_agents:       int,
        robot_speed:      float = 1.0,
        seed:             int   = 1,
        max_sim_time:     float = float('inf'),
        logger=None,
        enable_kill:      bool  = False,
        kill_threshold:   float = 0.2,
        num_to_kill:      int   = 1,
        enable_revive:    bool  = False,
        revive_threshold: float = 0.8,
    ):
        self.problem          = problem
        self.num_agents       = num_agents
        self.sim_time         = 0.0
        self.max_sim_time     = max_sim_time
        self.logger           = logger

        self.enable_kill      = enable_kill
        self.kill_threshold   = kill_threshold
        self.num_to_kill      = min(num_to_kill, num_agents)
        self.enable_revive    = enable_revive
        self.revive_threshold = revive_threshold

        self.killed_robots:        set  = set()
        self._is_kill_triggered:   bool = False
        self._is_revive_triggered: bool = False

        self.queue      = EventQueue()
        self.is_covered = [False] * problem.num_tasks

        rng  = np.random.default_rng(seed)
        lo, hi = 0, problem.grid_size
        starts: List[Tuple[float, float]] = [
            (float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))
            for _ in range(num_agents)
        ]

        self.robots: List[DESRobot] = [
            DESRobot(i, starts[i], robot_speed) for i in range(num_agents)
        ]
        self.agents: List[DESGreedyAgent] = [
            DESGreedyAgent(i, num_agents, problem) for i in range(num_agents)
        ]

        for agent in self.agents:
            agent.initialise(starts)

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #

    def run(self, progress_interval: float = 0.0) -> dict:
        self._progress_enabled = progress_interval > 0

        if self.logger:
            self.logger.log_setup(
                task_poses=self.problem.task_poses,
                robot_starts=[r.get_position(0.0) for r in self.robots],
                robot_speed=self.robots[0].speed,
                num_agents=self.num_agents,
            )

        # Initial greedy assignment and robot dispatch
        self._assign_tasks_greedy()
        for robot in self.robots:
            self._reschedule_robot(robot.robot_id)

        exit_reason = 'queue_empty'
        while not self.queue.is_empty():
            event = self.queue.pop()

            if event.time > self.max_sim_time:
                exit_reason = 'max_sim_time'
                break

            self.sim_time = event.time

            if event.type == EventType.ROBOT_ARRIVAL:
                self._on_robot_arrival(event.data)

            if self.logger:
                self.logger.tick(self.sim_time, self.is_covered, self.agents)

            self._check_kill_revive()

            if all(self.is_covered):
                exit_reason = 'complete'
                break

        covered   = sum(self.is_covered)
        remaining = [t for t, c in enumerate(self.is_covered) if not c]
        print(f'[DES_GREEDY_EXIT] reason={exit_reason}  sim_time={self.sim_time:.2f}'
              f'  covered={covered}/{self.problem.num_tasks}'
              f'  uncovered_tasks={remaining[:20]}{"..." if len(remaining) > 20 else ""}',
              flush=True)

        return self._summary()

    # ------------------------------------------------------------------ #
    # Event handler                                                        #
    # ------------------------------------------------------------------ #

    def _on_robot_arrival(self, data: dict) -> None:
        robot_id     = data['robot_id']
        task_id      = data['task_id']
        goal_version = data['goal_version']

        robot = self.robots[robot_id]

        if not robot.is_current_goal_version(goal_version):
            return

        robot._leg_start_pos  = robot._goal
        robot._leg_start_time = self.sim_time
        robot._goal           = None

        if not self.is_covered[task_id]:
            self.is_covered[task_id] = True

            for agent in self.agents:
                agent.handle_task_covered(task_id)

            if self.logger:
                self.logger.task_covered(self.sim_time, task_id, robot_id)

            if self._progress_enabled:
                covered = sum(self.is_covered)
                print(f'  t={self.sim_time:8.2f}  covered={covered}/{self.problem.num_tasks}',
                      flush=True)

        # Recompute assignment for all agents now that coverage changed,
        # then reschedule every robot.
        self._assign_tasks_greedy()
        for r in self.robots:
            self._reschedule_robot(r.robot_id)

    # ------------------------------------------------------------------ #
    # Greedy assignment                                                    #
    # ------------------------------------------------------------------ #

    def _assign_tasks_greedy(self) -> None:
        """
        Global greedy assignment.  Agents processed in ascending ID order
        so lower IDs have priority when tasks are equidistant.
        Each agent picks the nearest uncovered, unclaimed task.
        """
        claimed: set = set()
        for agent in self.agents:
            # Update robot pose from DESRobot before selecting
            agent.robot_poses[agent.agent_id] = self.robots[agent.agent_id].get_position(self.sim_time)
            agent.select_task(claimed)
            if agent.current_task is not None:
                claimed.add(agent.current_task)

    # ------------------------------------------------------------------ #
    # Robot scheduling                                                     #
    # ------------------------------------------------------------------ #

    def _reschedule_robot(self, robot_id: int) -> None:
        if not self.robots[robot_id].is_alive:
            return

        agent = self.agents[robot_id]
        robot = self.robots[robot_id]
        task  = agent.current_task

        if task is None or self.is_covered[task]:
            if robot._goal is not None:
                robot._leg_start_pos  = robot.get_position(self.sim_time)
                robot._leg_start_time = self.sim_time
                robot._goal           = None
                robot._goal_version  += 1
            return

        goal = tuple(self.problem.task_poses[task])

        if robot._goal == goal:
            return

        current_pos        = robot.get_position(self.sim_time)
        arrival, version   = robot.set_goal(goal, self.sim_time)

        if self.logger:
            self.logger.log_robot_leg(
                self.sim_time, robot_id,
                current_pos[0], current_pos[1],
                goal[0], goal[1],
                task, arrival,
            )

        self.queue.push(arrival, EventType.ROBOT_ARRIVAL, {
            'robot_id':     robot_id,
            'task_id':      task,
            'goal_version': version,
        })

    # ------------------------------------------------------------------ #
    # Kill / Revive                                                        #
    # ------------------------------------------------------------------ #

    def _check_kill_revive(self) -> None:
        if not (self.enable_kill or self.enable_revive):
            return
        coverage = sum(self.is_covered) / self.problem.num_tasks
        if (self.enable_kill
                and not self._is_kill_triggered
                and coverage >= self.kill_threshold):
            self._trigger_kill()
        elif (self.enable_revive
                and self._is_kill_triggered
                and not self._is_revive_triggered
                and coverage >= self.revive_threshold):
            self._trigger_revive()

    def _trigger_kill(self) -> None:
        self._is_kill_triggered = True
        targets = list(range(self.num_agents - self.num_to_kill, self.num_agents))

        for robot_id in targets:
            self.killed_robots.add(robot_id)
            stop_pos = self.robots[robot_id].get_position(self.sim_time)
            self.robots[robot_id].kill(self.sim_time)

            for agent in self.agents:
                agent.kill_robot(robot_id)

            if self.logger:
                self.logger.log_robot_leg(
                    sim_time=self.sim_time,
                    robot_id=robot_id,
                    from_x=stop_pos[0], from_y=stop_pos[1],
                    to_x=stop_pos[0],   to_y=stop_pos[1],
                    task_id=-1, arrival_sim_time=self.sim_time,
                )
                self.logger.robot_killed(self.sim_time, robot_id)

        self._assign_tasks_greedy()
        for robot_id in range(self.num_agents):
            if robot_id not in self.killed_robots:
                self._reschedule_robot(robot_id)
        print(f'[KILL]   sim_time={self.sim_time:.2f}  robots={targets}', flush=True)

    def _trigger_revive(self) -> None:
        self._is_revive_triggered = True
        revived = sorted(self.killed_robots)
        for robot_id in revived:
            self.robots[robot_id].revive()
            revived_pos = self.robots[robot_id].get_position(self.sim_time)
            for agent in self.agents:
                agent.revive_robot(robot_id)
                agent.robot_poses[robot_id] = revived_pos
            if self.logger:
                self.logger.robot_revived(self.sim_time, robot_id)
        self.killed_robots.clear()

        self._assign_tasks_greedy()
        for robot_id in range(self.num_agents):
            self._reschedule_robot(robot_id)
        print(f'[REVIVE] sim_time={self.sim_time:.2f}  robots={revived}', flush=True)

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def _summary(self) -> dict:
        covered = sum(self.is_covered)
        return {
            'sim_time':      self.sim_time,
            'tasks_covered': covered,
            'total_tasks':   self.problem.num_tasks,
            'complete':      all(self.is_covered),
        }

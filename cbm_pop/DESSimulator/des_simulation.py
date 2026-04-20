"""
DES orchestrator for CBM-POP.

Time model
----------
  - Each agent's operator wall-time is measured at compute_step() and advances
    sim time before the result is applied.  This preserves the relative
    computational cost of different operators without being tied to wall-clock speed.
  - Robot travel time is computed analytically (distance / speed), preserving
    the physical temporal aspect of the mission.
  - Events are processed in sim-time order via a min-heap priority queue.

Parallelism model
-----------------
  All N agents compute operators in sequence inside the DES loop, but their
  *sim-time completion events* are interleaved in the priority queue, modelling
  the parallel execution seen in the real system.  The measured wall time of
  each operator determines how far ahead in sim time that agent's result lands.
"""

from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from cbm_pop.DESSimulator.des_agent import DESAgent, StepData
from cbm_pop.DESSimulator.des_event import EventQueue, EventType
from cbm_pop.DESSimulator.des_robot import DESRobot
from cbm_pop.SimpleSimulator.simple_problem import SimpleProblem


class DESSimulation:
    def __init__(
        self,
        problem:            SimpleProblem,
        num_agents:         int,
        robot_speed:        float = 1.0,
        seed:               int   = 1,
        agent_kwargs:       Optional[dict] = None,
        max_sim_time:       float = float('inf'),
        compute_time_scale: float = 1.0,
        logger=None,
    ):
        self.problem            = problem
        self.num_agents         = num_agents
        self.sim_time           = 0.0
        self.max_sim_time       = max_sim_time
        self.compute_time_scale = compute_time_scale
        self.logger             = logger

        self.queue           = EventQueue()
        self.is_covered      = [False] * problem.num_tasks
        self._wall_start     = 0.0
        self._progress_enabled = False

        # Reproducible starting positions
        rng  = np.random.default_rng(seed)
        lo, hi = 0, problem.grid_size
        starts: List[Tuple[float, float]] = [
            (float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))
            for _ in range(num_agents)
        ]

        self.robots: List[DESRobot] = [
            DESRobot(i, starts[i], robot_speed) for i in range(num_agents)
        ]

        kwargs = agent_kwargs or {}
        self.agents: List[DESAgent] = [
            DESAgent(i, num_agents, problem, **kwargs, logger=logger)
            for i in range(num_agents)
        ]

        for agent in self.agents:
            agent.initialise(starts)

        self._broadcast_best_initial_solution()

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #

    def run(self, progress_interval: float = 0.0) -> dict:
        """
        progress_interval : unused, kept for CLI compatibility.
                            Progress is now printed on every coalition-best improvement.
                            Pass progress_interval > 0 to enable; 0 = silent.
        """
        import time as _wt
        self._wall_start       = _wt.monotonic()
        self._progress_enabled = progress_interval > 0

        if self.logger:
            self.logger.log_setup(
                task_poses=self.problem.task_poses,
                robot_starts=[r.get_position(0.0) for r in self.robots],
                robot_speed=self.robots[0].speed,
                num_agents=self.num_agents,
            )

        # Bootstrap: schedule first compute_step for every agent at T=0
        for agent in self.agents:
            poses = [r.get_position(0.0) for r in self.robots]
            step  = agent.compute_step(poses)
            self.queue.push(
                step.wall_time * self.compute_time_scale,
                EventType.OPERATOR_COMPLETE,
                {'agent_id': agent.agent_id, 'step': step},
            )

        # Schedule initial robot movements
        for robot in self.robots:
            self._reschedule_robot(robot.robot_id)

        # Main DES loop
        exit_reason = 'queue_empty'
        while not self.queue.is_empty():
            event = self.queue.pop()

            if event.time > self.max_sim_time:
                exit_reason = 'max_sim_time'
                break

            self.sim_time = event.time

            if event.type == EventType.OPERATOR_COMPLETE:
                self._on_operator_complete(event.data)
            elif event.type == EventType.ROBOT_ARRIVAL:
                self._on_robot_arrival(event.data)

            if self.logger:
                self.logger.tick(self.sim_time, self.is_covered, self.agents)

            if all(self.is_covered):
                exit_reason = 'complete'
                break

        covered   = sum(self.is_covered)
        remaining = [t for t, c in enumerate(self.is_covered) if not c]
        print(f'[DES_EXIT] reason={exit_reason}  sim_time={self.sim_time:.2f}'
              f'  covered={covered}/{self.problem.num_tasks}'
              f'  uncovered_tasks={remaining[:20]}{"..." if len(remaining) > 20 else ""}',
              flush=True)

        return self._summary()

    # ------------------------------------------------------------------ #
    # Event handlers                                                       #
    # ------------------------------------------------------------------ #

    def _on_operator_complete(self, data: dict) -> None:
        agent_id = data['agent_id']
        step: StepData = data['step']
        agent  = self.agents[agent_id]
        poses  = [r.get_position(self.sim_time) for r in self.robots]

        # Bring the agent's robot cost matrix up to the current sim time (when the
        # operator completed) before evaluating the result.  Without this, fitness
        # is assessed against positions from when the operator *started*, not when
        # it *finished*.
        for i, pos in enumerate(poses):
            if pos is not None:
                agent.robot_poses[i] = pos
        agent.problem.update_robot_cost_matrix(agent.robot_poses)

        coalition_improved = agent.apply_step_result(step, self.sim_time)

        if coalition_improved:
            weights = (
                agent.weight_matrix.weights
                if agent.is_mimetism_enabled
                else None
            )
            for other in self.agents:
                if other.agent_id != agent_id:
                    other.receive_coalition_best(
                        agent.coalition_best_solution, agent_id, weights
                    )
                    # Re-route if the new coalition assigned this robot a task
                    # and it is currently idle (no pending arrival event).
                    self._reschedule_robot(other.agent_id)
            # Re-route this agent's robot if its assigned task changed
            self._reschedule_robot(agent_id)

        # Schedule next step for this agent
        next_step = agent.compute_step(poses)
        self.queue.push(
            self.sim_time + next_step.wall_time * self.compute_time_scale,
            EventType.OPERATOR_COMPLETE,
            {'agent_id': agent_id, 'step': next_step},
        )

        if coalition_improved and self._progress_enabled:
            import time as _wt
            covered   = sum(self.is_covered)
            wall_now  = _wt.monotonic() - self._wall_start
            total_iters = sum(a.iteration_count for a in self.agents)
            best_fit  = agent._fitness(agent.coalition_best_solution)
            print(
                f'  t={self.sim_time:8.2f}  '
                f'covered={covered}/{self.problem.num_tasks}  '
                f'iters={total_iters}  '
                f'best_fit={best_fit:.3f}  '
                f'wall={wall_now:.1f}s  '
                f'agent={agent_id}',
                flush=True,
            )

        if self.logger:
            fitness = agent._fitness(agent.coalition_best_solution)
            self.logger.operator_complete(
                self.sim_time, agent_id,
                step.operator.name, step.wall_time,
                coalition_improved,
                fitness,
            )
            if coalition_improved:
                self.logger.coalition_best_improved(self.sim_time, agent_id, fitness)

    def _on_robot_arrival(self, data: dict) -> None:
        robot_id     = data['robot_id']
        task_id      = data['task_id']
        goal_version = data['goal_version']

        robot = self.robots[robot_id]

        # Stale event — robot's goal changed after this was scheduled
        if not robot.is_current_goal_version(goal_version):
            return

        if not self.is_covered[task_id]:
            self.is_covered[task_id] = True

            for agent in self.agents:
                agent.handle_task_covered(task_id)

            if self.logger:
                self.logger.task_covered(self.sim_time, task_id, robot_id)

            # Any robot physically heading to the now-covered task needs rerouting.
            # Check the robot's goal position rather than current_task, because
            # handle_task_covered() has already advanced current_task for all agents.
            covered_goal = tuple(self.problem.task_poses[task_id])
            for other in self.robots:
                if other.robot_id != robot_id and other._goal == covered_goal:
                    self._reschedule_robot(other.robot_id)

        # Send this robot to its next task
        self._reschedule_robot(robot_id)

    # ------------------------------------------------------------------ #
    # Robot scheduling                                                     #
    # ------------------------------------------------------------------ #

    def _reschedule_robot(self, robot_id: int) -> None:
        """Send robot to its agent's current_task, if valid."""
        agent = self.agents[robot_id]
        robot = self.robots[robot_id]
        task  = agent.current_task

        if task is None or self.is_covered[task]:
            return

        goal = tuple(self.problem.task_poses[task])

        # Avoid redundant reschedule: skip if already heading to this exact goal
        if robot._goal == goal:
            return

        current_pos = robot.get_position(self.sim_time)
        arrival, version = robot.set_goal(goal, self.sim_time)

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
    # Initialisation helpers                                               #
    # ------------------------------------------------------------------ #

    def _broadcast_best_initial_solution(self) -> None:
        """Find the globally best initial solution and share it to all agents."""
        best_agent = min(self.agents, key=lambda a: a._fitness(a.coalition_best_solution))
        for agent in self.agents:
            if agent.agent_id != best_agent.agent_id:
                agent.receive_coalition_best(
                    best_agent.coalition_best_solution,
                    best_agent.agent_id,
                )

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def _summary(self) -> dict:
        covered = sum(self.is_covered)
        return {
            'sim_time':             self.sim_time,
            'tasks_covered':        covered,
            'total_tasks':          self.problem.num_tasks,
            'complete':             all(self.is_covered),
            'iterations_per_agent': [a.iteration_count for a in self.agents],
            'coalition_fitness':    [
                a._fitness(a.coalition_best_solution) for a in self.agents
            ],
        }

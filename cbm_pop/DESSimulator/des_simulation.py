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
        enable_kill:        bool  = False,
        kill_threshold:     float = 0.2,
        num_to_kill:        int   = 1,
        enable_revive:      bool  = False,
        revive_threshold:   float = 0.8,
    ):
        self.problem            = problem
        self.num_agents         = num_agents
        self.sim_time           = 0.0
        self.max_sim_time       = max_sim_time
        self.compute_time_scale = compute_time_scale
        self.logger             = logger

        self.enable_kill      = enable_kill
        self.kill_threshold   = kill_threshold
        self.num_to_kill      = min(num_to_kill, num_agents)
        self.enable_revive    = enable_revive
        self.revive_threshold = revive_threshold

        self.killed_robots:       set  = set()
        self._is_kill_triggered:  bool = False
        self._is_revive_triggered: bool = False

        self.queue           = EventQueue()
        self.is_covered      = [False] * problem.num_tasks
        self._wall_start     = 0.0
        self._progress_enabled = False
        # Buffers the start of each robot's current leg so only completed
        # (or interrupted) legs are written to the path log.
        self._pending_leg: dict = {}   # robot_id -> (from_pos, start_time, task_id)

        # Reproducible starting positions
        rng  = np.random.default_rng(seed)
        lo, hi = 0, problem.grid_size
        starts: List[Tuple[float, float]] = [
            (float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))
            for _ in range(num_agents)
        ]

        self.robot_starts = starts

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

            self._check_kill_revive()

            if all(self.is_covered):
                exit_reason = 'complete'
                break

        if exit_reason == 'complete':
            self._execute_return_to_depot()

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

        # Stale event fired after this agent was killed — discard silently.
        if not agent.is_alive:
            return

        # Pass None for dead robots so their cost appears infinite in operators.
        poses  = [r.get_position(self.sim_time) if r.is_alive else None for r in self.robots]

        # Bring the agent's robot cost matrix up to the current sim time (when the
        # operator completed) before evaluating the result.  Without this, fitness
        # is assessed against positions from when the operator *started*, not when
        # it *finished*.
        for i, pos in enumerate(poses):
            agent.robot_poses[i] = pos
        agent.problem.update_robot_cost_matrix(agent.robot_poses)

        coalition_improved, weights_to_share = agent.apply_step_result(step, self.sim_time)

        if coalition_improved:
            # Use the post-learning deepcopy from _finish_di_cycle when available;
            # otherwise snapshot current weights so receivers always blend with a
            # fixed copy, never a live reference that drifts over time.
            if agent.is_mimetism_enabled:
                weights = (
                    weights_to_share
                    if weights_to_share is not None
                    else deepcopy(agent.weight_matrix.weights)
                )
            else:
                weights = None
            secondary_improvers = []
            for other in self.agents:
                if other.agent_id != agent_id:
                    _, prepend_adopted = other.receive_coalition_best(
                        agent.coalition_best_solution, agent_id, weights
                    )
                    if prepend_adopted:
                        secondary_improvers.append(other)
                    # Re-route if the new coalition assigned this robot a task
                    # and it is currently idle (no pending arrival event).
                    self._reschedule_robot(other.agent_id)
            # Re-route this agent's robot if its assigned task changed
            self._reschedule_robot(agent_id)

            # Propagate prepend-first-task improvements (mirrors SimpleSimulator
            # publish behaviour: if an agent improved the received solution by
            # prepending its current task, re-broadcast that version to peers).
            # Each round picks the single best solution among all improvers and
            # broadcasts it, avoiding ordering bias from sequential iteration.
            # Cascades until no further prepend improvement is found — safe
            # because each round strictly decreases fitness over a finite set.
            current_improvers = secondary_improvers
            while current_improvers:
                best_improver = min(
                    current_improvers,
                    key=lambda a: a.fitness(a.coalition_best_solution),
                )
                imp_weights = (
                    deepcopy(best_improver.weight_matrix.weights)
                    if best_improver.is_mimetism_enabled
                    else None
                )
                next_improvers = []
                for other in self.agents:
                    if other.agent_id != best_improver.agent_id:
                        _, prepend_adopted = other.receive_coalition_best(
                            best_improver.coalition_best_solution,
                            best_improver.agent_id,
                            imp_weights,
                        )
                        if prepend_adopted:
                            next_improvers.append(other)
                        self._reschedule_robot(other.agent_id)
                self._reschedule_robot(best_improver.agent_id)
                current_improvers = next_improvers

        # Schedule next step for this agent (skip if killed during event handling)
        if agent.is_alive:
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
            best_fit  = agent.fitness(agent.coalition_best_solution)
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
            fitness = agent.fitness(agent.coalition_best_solution)
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

        # Log the completed leg before clearing the goal
        if self.logger and robot_id in self._pending_leg:
            fp, ft, fk = self._pending_leg.pop(robot_id)
            self.logger.log_robot_leg(
                ft, robot_id,
                fp[0], fp[1],
                robot._goal[0], robot._goal[1],
                fk, self.sim_time,
            )

        # Arrived — snapshot position before clearing goal so get_position()
        # returns the arrival coordinates on subsequent calls.
        robot._leg_start_pos  = robot._goal
        robot._leg_start_time = self.sim_time
        robot._goal           = None

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
        if not self.robots[robot_id].is_alive:
            return

        agent = self.agents[robot_id]
        robot = self.robots[robot_id]
        task  = agent.current_task

        if task is None or self.is_covered[task]:
            # If robot is heading somewhere but should be idle, stop it
            if robot._goal is not None:
                current_pos           = robot.get_position(self.sim_time)
                # Log the partial leg that was interrupted
                if self.logger and robot_id in self._pending_leg:
                    fp, ft, fk = self._pending_leg.pop(robot_id)
                    self.logger.log_robot_leg(
                        ft, robot_id,
                        fp[0], fp[1],
                        current_pos[0], current_pos[1],
                        fk, self.sim_time,
                    )
                robot._leg_start_pos  = current_pos
                robot._leg_start_time = self.sim_time
                robot._goal           = None
                robot._goal_version  += 1
            return

        goal = tuple(self.problem.task_poses[task])

        # Avoid redundant reschedule: skip if already heading to this exact goal
        if robot._goal == goal:
            return

        current_pos = robot.get_position(self.sim_time)

        # Log the partial leg just traveled before redirecting
        if self.logger and robot_id in self._pending_leg:
            fp, ft, fk = self._pending_leg.pop(robot_id)
            self.logger.log_robot_leg(
                ft, robot_id,
                fp[0], fp[1],
                current_pos[0], current_pos[1],
                fk, self.sim_time,
            )

        arrival, version = robot.set_goal(goal, self.sim_time)

        if self.logger:
            self._pending_leg[robot_id] = (current_pos, self.sim_time, task)

        self.queue.push(arrival, EventType.ROBOT_ARRIVAL, {
            'robot_id':     robot_id,
            'task_id':      task,
            'goal_version': version,
        })

    def _execute_return_to_depot(self) -> None:
        """Advance sim_time by the longest return leg back to each robot's start."""
        import math
        max_return = 0.0
        for robot in self.robots:
            if not robot.is_alive:
                continue
            pos = robot.get_position(self.sim_time)
            start = self.robot_starts[robot.robot_id]
            dist = math.hypot(pos[0] - start[0], pos[1] - start[1])
            return_time = dist / robot.speed if robot.speed > 0 else 0.0
            if self.logger:
                self.logger.log_robot_leg(
                    self.sim_time, robot.robot_id,
                    pos[0], pos[1],
                    start[0], start[1],
                    -2,
                    self.sim_time + return_time,
                )
            max_return = max(max_return, return_time)
        self.sim_time += max_return

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
        # Ensure we pick a living agent to receive the tasks
        survivors = [a for a in self.agents if a.agent_id not in targets]
        recipient = survivors[0] if survivors else self.agents[0]

        for robot_id in targets:
            self.killed_robots.add(robot_id)

            # 1. Capture the exact location where the robot stops moving
            stop_pos = self.robots[robot_id].get_position(self.sim_time)

            # Log the partial leg that was interrupted by the kill
            if self.logger and robot_id in self._pending_leg:
                fp, ft, fk = self._pending_leg.pop(robot_id)
                self.logger.log_robot_leg(
                    ft, robot_id,
                    fp[0], fp[1],
                    stop_pos[0], stop_pos[1],
                    fk, self.sim_time,
                )

            self.robots[robot_id].kill(self.sim_time)

            # Start the dead robot travelling back to its depot so that
            # get_position() interpolates correctly during the return trip.
            # If revived before arrival it will start from wherever it is.
            depot = tuple(self.robot_starts[robot_id])
            self.robots[robot_id].set_goal(depot, self.sim_time)
            if self.logger:
                self._pending_leg[robot_id] = (stop_pos, self.sim_time, -1)

            for agent in self.agents:
                # Use a specific living recipient for redistribution
                # instead of each agent redistributing to itself.
                agent.failed_agents[robot_id] = True
                if robot_id == agent.agent_id:
                    agent.is_alive = False
                    continue

                def _purge_to_recipient(solution):
                    if solution is None or not solution[0]:
                        return solution
                    order, alloc = list(solution[0]), list(solution[1])
                    if robot_id >= len(alloc) or alloc[robot_id] == 0:
                        return (order, alloc)
                    start = sum(alloc[:robot_id])
                    end = start + alloc[robot_id]
                    purged_tasks = order[start:end]
                    del order[start:end]
                    alloc[robot_id] = 0

                    # Compute recipient index in the new order
                    dest_idx = sum(alloc[:recipient.agent_id])
                    if dest_idx > len(order):
                        dest_idx = len(order)
                    order[dest_idx:dest_idx] = purged_tasks
                    alloc[recipient.agent_id] += len(purged_tasks)
                    return (order, alloc)

                agent.coalition_best_solution = _purge_to_recipient(agent.coalition_best_solution)
                agent.current_solution = _purge_to_recipient(agent.current_solution)
                agent.local_best_solution = _purge_to_recipient(agent.local_best_solution)
                if agent.population:
                    agent.population = [_purge_to_recipient(s) for s in agent.population]
                agent._assign_next_task(agent.coalition_best_solution)

            if self.logger:
                self.logger.robot_killed(self.sim_time, robot_id)

        # Null dead robots' poses so the cost matrix shows infinity for them,
        # preventing operators from assigning tasks to dead robots.
        for agent in self.agents:
            if agent.is_alive:
                for dead_id in targets:
                    agent.robot_poses[dead_id] = None
                agent.problem.update_robot_cost_matrix(agent.robot_poses)

        # Surviving robots may have been assigned extra tasks — reschedule them.
        for robot_id in range(self.num_agents):
            if robot_id not in self.killed_robots:
                self._reschedule_robot(robot_id)
        print(f'[KILL]   sim_time={self.sim_time:.2f}  robots={targets}', flush=True)

    def _trigger_revive(self) -> None:
        self._is_revive_triggered = True
        revived = sorted(self.killed_robots)
        for robot_id in revived:
            revived_pos = self.robots[robot_id].get_position(self.sim_time)
            if self.logger and robot_id in self._pending_leg:
                fp, ft, fk = self._pending_leg.pop(robot_id)
                self.logger.log_robot_leg(
                    ft, robot_id,
                    fp[0], fp[1],
                    revived_pos[0], revived_pos[1],
                    fk, self.sim_time,
                )
            self.robots[robot_id].revive()
            for agent in self.agents:
                agent.revive_robot(robot_id)
                agent.robot_poses[robot_id] = revived_pos
                agent.problem.update_robot_cost_matrix(agent.robot_poses)
            # Re-enter the revived agent into the optimisation loop.
            poses = [r.get_position(self.sim_time) for r in self.robots]
            step  = self.agents[robot_id].compute_step(poses)
            self.queue.push(
                self.sim_time + step.wall_time * self.compute_time_scale,
                EventType.OPERATOR_COMPLETE,
                {'agent_id': robot_id, 'step': step},
            )
            self._reschedule_robot(robot_id)
            if self.logger:
                self.logger.robot_revived(self.sim_time, robot_id)
        self.killed_robots.clear()
        print(f'[REVIVE] sim_time={self.sim_time:.2f}  robots={revived}', flush=True)

    # ------------------------------------------------------------------ #
    # Initialisation helpers                                               #
    # ------------------------------------------------------------------ #

    def _broadcast_best_initial_solution(self) -> None:
        """Find the globally best initial solution and share it to all agents."""
        best_agent = min(self.agents, key=lambda a: a.fitness(a.coalition_best_solution))
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
                a.fitness(a.coalition_best_solution) for a in self.agents
            ],
        }

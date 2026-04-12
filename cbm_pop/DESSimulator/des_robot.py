import math
from typing import Optional, Tuple


class DESRobot:
    """
    Analytically tracks a robot's position during a DES — no movement loop.

    Position at any sim_time is computed by linear interpolation along the
    current leg (start_pos → goal).  When a new goal is assigned the robot's
    position is snapshotted and the leg restarts from there.

    Goal versioning lets the simulation discard stale ROBOT_ARRIVAL events:
    every call to set_goal() increments _goal_version, and each scheduled
    event carries the version that was current when it was created.
    """

    def __init__(self, robot_id: int, start_pos: Tuple[float, float], speed: float):
        self.robot_id = robot_id
        self.speed    = speed

        self._leg_start_pos:  Tuple[float, float] = start_pos
        self._leg_start_time: float = 0.0
        self._goal:           Optional[Tuple[float, float]] = None
        self._goal_version:   int = 0

    def set_goal(
        self, goal: Tuple[float, float], sim_time: float
    ) -> Tuple[float, int]:
        """
        Assign a new goal.  Returns (arrival_sim_time, goal_version).
        Snap current position so future interpolation is correct.
        """
        self._leg_start_pos  = self.get_position(sim_time)
        self._leg_start_time = sim_time
        self._goal           = goal
        self._goal_version  += 1

        dist = math.hypot(
            goal[0] - self._leg_start_pos[0],
            goal[1] - self._leg_start_pos[1],
        )
        arrival = sim_time + (dist / self.speed if self.speed > 0 else 0.0)
        return arrival, self._goal_version

    def get_position(self, sim_time: float) -> Tuple[float, float]:
        if self._goal is None:
            return self._leg_start_pos

        dx = self._goal[0] - self._leg_start_pos[0]
        dy = self._goal[1] - self._leg_start_pos[1]
        dist = math.hypot(dx, dy)

        if dist < 1e-9:
            return self._goal

        elapsed   = max(0.0, sim_time - self._leg_start_time)
        travelled = min(self.speed * elapsed, dist)
        frac      = travelled / dist
        return (
            self._leg_start_pos[0] + dx * frac,
            self._leg_start_pos[1] + dy * frac,
        )

    def is_current_goal_version(self, version: int) -> bool:
        return version == self._goal_version

#!/usr/bin/env python3
import math
import sys
import time
from typing import List, Tuple, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from cbm_pop_interfaces.msg import SimplePosition, FinishedCoverage


class SimulatorRobot(Node):
    """
    A simple simulated robot that:
      - Publishes its current position at a fixed rate.
      - Listens for a goal_pose and moves toward it at a fixed speed.
      - Tracks a finished-coverage flag per robot and declares completion
        after a configurable 'grace window' where all robots remain finished.

    IMPORTANT: This class does NOT call destroy_node() from callbacks.
    Call robot.stop() and then destroy the node from the owning thread.
    """

    def __init__(self,
                 robot_id: int,
                 robot_starting_position: List[Tuple[float, float]],
                 robot_speed: float,
                 num_robots: int,
                 publish_hz: float = 2.0,
                 finish_grace_seconds: float = 10.0):
        super().__init__(f'simulator_robot_{robot_id}')
        print("Initialising robot")
        self.robot_id = int(robot_id)
        self.robot_position: List[Tuple[float, float]] = list(robot_starting_position)
        if not self.robot_position:
            self.robot_position = [(0.0, 0.0)]
        self.robot_speed = float(robot_speed)
        self.current_goal_pose: Optional[Tuple[float, float]] = None
        self.is_finished = False
        self.finished_robots = [False] * int(num_robots)

        # --- grace window tracking ---
        self._all_finished_since: Optional[float] = None
        self._finish_grace_seconds = float(finish_grace_seconds)
        self._grace_timer = None  # periodic timer to check grace window

        # Publishers / Subscribers
        self.global_pose_publisher = self.create_publisher(
            SimplePosition, f'/central_control/uas_{self.robot_id}/global_pose', 10
        )
        self.goal_pose_subscriber = self.create_subscription(
            SimplePosition, f'/central_control/uas_{self.robot_id}/goal_pose',
            self.goal_pose_callback, 10
        )
        self.finished_coverage_sub = self.create_subscription(
            FinishedCoverage, '/central_control/finished_coverage',
            self.finished_coverage_callback, 10
        )

        # Position publishing timer (publish_hz nominal)
        if publish_hz <= 0:
            publish_hz = 2.0
        self._pub_period = 1.0 / publish_hz
        self._pub_timer = self.create_timer(self._pub_period, self.publish_position)

        self.get_logger().info(
            f"SimulatorRobot {self.robot_id} started | speed={self.robot_speed:.3f} m/step | "
            f"publish_hz={publish_hz:.2f} | grace={self._finish_grace_seconds:.1f}s"
        )

    # ---------------- Internal helpers for grace window ---------------- #

    def _start_grace_timer(self):
        """Start a lightweight periodic timer to check the grace window."""
        if self._grace_timer is None:
            # Check 10x per second; cancels once done or invalidated
            self._grace_timer = self.create_timer(0.1, self._check_finish_grace)

    def _cancel_grace_timer(self):
        if self._grace_timer is not None:
            try:
                self._grace_timer.cancel()
            except Exception:
                pass
            self._grace_timer = None

    def _check_finish_grace(self):
        """Timer callback: finalize after grace window, or reset if state changes."""
        if not all(self.finished_robots):
            # Someone flipped back; invalidate window and stop checking
            self._all_finished_since = None
            self._cancel_grace_timer()
            return

        if self._all_finished_since is None:
            # Defensive: if we somehow got here, start the clock now.
            self._all_finished_since = time.monotonic()
            return

        elapsed = time.monotonic() - self._all_finished_since
        if elapsed >= self._finish_grace_seconds:
            self._complete_coverage()

    def _complete_coverage(self):
        """Mark completion and stop our timers so the executor won’t reschedule callbacks."""
        if self.is_finished:
            return
        self.get_logger().info("Coverage Complete")
        self.is_finished = True
        # stop our timers so executor won’t reschedule callbacks
        try:
            if self._pub_timer is not None:
                self._pub_timer.cancel()
        except Exception:
            pass
        self._cancel_grace_timer()
        # NOTE: Do NOT destroy the node here. Let the owning code do it safely.

    # ---------------- ROS Callbacks ---------------- #

    def finished_coverage_callback(self, msg: FinishedCoverage):
        """
        Track finished status for all robots. When everyone is finished,
        start or continue the grace window; if any revert, reset it.
        """
        try:
            rid = int(msg.robot_id)
        except Exception:
            self.get_logger().warn(f"Bad FinishedCoverage.robot_id: {msg.robot_id!r}")
            return
        if 0 <= rid < len(self.finished_robots):
            self.finished_robots[rid] = bool(msg.finished)
        else:
            self.get_logger().warn(f"robot_id {rid} out of range [0,{len(self.finished_robots)-1}]")
            return

        if not all(self.finished_robots):
            # Invalidate window and stop grace timer
            if self._all_finished_since is not None:
                self.get_logger().info("Not all robots finished (resetting grace window)")
            self._all_finished_since = None
            self._cancel_grace_timer()
            return

        # All robots currently finished: start or continue grace window
        now = time.monotonic()
        if self._all_finished_since is None:
            self._all_finished_since = now
            self.get_logger().info("All robots finished — starting grace window")
            self._start_grace_timer()
            return

        # If more messages come in while waiting, we can also complete here
        elapsed = now - self._all_finished_since
        self.get_logger().info(f"Robots finished for {elapsed:.2f}s")
        if elapsed >= self._finish_grace_seconds:
            self._complete_coverage()

    def goal_pose_callback(self, msg: SimplePosition):
        self.current_goal_pose = (float(msg.x_position), float(msg.y_position))

    def publish_position(self):
        if self.is_finished:
            return  # stop publishing after completion
        position = SimplePosition()
        position.robot_id = int(self.robot_id)
        position.x_position = float(self.robot_position[-1][0])
        position.y_position = float(self.robot_position[-1][1])
        self.global_pose_publisher.publish(position)

    # ---------------- Movement / Accessors ---------------- #

    def move_robot(self):
        """
        Integrate one movement step toward current_goal_pose with step size == robot_speed.
        If within 1e-3, snap to goal; otherwise move along bearing.
        """
        last_x, last_y = self.robot_position[-1]

        if self.current_goal_pose:
            gx, gy = self.current_goal_pose
            dx = gx - last_x
            dy = gy - last_y
            dist = math.hypot(dx, dy)

            if dist > 1e-3:  # small tolerance to avoid floating point noise
                if dist <= self.robot_speed:
                    new_x, new_y = gx, gy
                else:
                    angle = math.atan2(dy, dx)
                    new_x = last_x + math.cos(angle) * self.robot_speed
                    new_y = last_y + math.sin(angle) * self.robot_speed
                self.robot_position.append((new_x, new_y))
            else:
                # already at goal, keep position (still append for time history)
                self.robot_position.append((last_x, last_y))
        else:
            # no goal, hold position
            self.robot_position.append((last_x, last_y))

    def get_robot_position(self) -> List[Tuple[float, float]]:
        return self.robot_position

    # ---------------- External teardown API ---------------- #

    def stop(self):
        """
        Stop internal timers and mark finished. Call this from the owner thread
        BEFORE executor shutdown and node destruction.
        """
        self._complete_coverage()
        # Also ensure any other timers are cancelled defensively
        try:
            if self._pub_timer is not None:
                self._pub_timer.cancel()
        except Exception:
            pass
        self._cancel_grace_timer()


# Optional: local demo when run directly (kept minimal; real system instantiates from a controller)
if __name__ == "__main__":
    rclpy.init(args=sys.argv)

    # Minimal demo setup for one robot
    robot = SimulatorRobot(
        robot_id=0,
        robot_starting_position=[(0.0, 0.0)],
        robot_speed=0.1,
        num_robots=1,
        publish_hz=2.0,
        finish_grace_seconds=5.0,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(robot)

    try:
        # Simulate a goal and a finished signal after a short delay
        start = time.monotonic()
        while rclpy.ok():
            # drive the robot locally for demo
            robot.move_robot()

            # set a goal after 0.5s
            if time.monotonic() - start > 0.5 and robot.current_goal_pose is None:
                robot.current_goal_pose = (1.0, 0.0)

            # mark finished for the single robot after 3s
            if time.monotonic() - start > 3.0 and not all(robot.finished_robots):
                msg = FinishedCoverage()
                msg.robot_id = 0
                msg.finished = True
                robot.finished_coverage_callback(msg)

            executor.spin_once(timeout_sec=0.05)

            if robot.is_finished:
                break
    finally:
        # Safe teardown outside callbacks
        try:
            executor.remove_node(robot)
        except Exception:
            pass
        robot.stop()
        robot.destroy_node()
        rclpy.shutdown()

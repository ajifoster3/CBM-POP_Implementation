#!/usr/bin/env python3
import sys
import time
import threading
import argparse
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from cbm_pop.SimpleSimulator.simple_problem import SimpleProblem, ProblemClass
from cbm_pop.SimpleSimulator.simulator_robot import SimulatorRobot as robot
from cbm_pop_interfaces.msg import EnvironmentalRepresentation, Solution
from std_msgs.msg import Bool   # <<< added
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib.colors as mcolors


class SimpleSimulator:
    def __init__(self, tasks, environmental_bounds, robot_starting_positions, obstacles, args):
        self.simulation_done = False
        self.environmental_bounds = environmental_bounds
        self.tasks = tasks
        self.robot_starting_positions = robot_starting_positions
        self.obstacles = obstacles
        self.robots = []
        self.lock = threading.Lock()
        self.finished_robots = [False] * args.num_robots
        self.is_covered = [False] * len(self.tasks)
        self.task_scat = None

        # kill-robot trigger config from arguments
        self.is_kill_enabled = bool(args.enable_kill)
        self.kill_threshold = float(args.kill_threshold)
        self.number_to_kill = int(args.num_to_kill)
        self.is_kill_published = False
        self.kill_pubs = []  # set in start_listeners if enabled

        # revive-robot trigger config from arguments
        self.is_revive_enabled = bool(args.enable_revive)
        self.revive_threshold = float(args.revive_threshold)
        self.number_to_revive = self.number_to_kill
        self.is_revive_published = False
        self.revive_pubs = []

        # Colors for robots
        num_robots = args.num_robots
        cmap_name = 'tab20' if num_robots > 10 else 'tab10'
        cmap = cm.get_cmap(cmap_name, num_robots)
        self.robot_colours = [mcolors.to_hex(cmap(i)) for i in range(num_robots)]

        # coalition best (order, allocations) -> per-task owner (agent id)
        self.task_owner = [-1] * len(self.tasks)   # -1 means unassigned

        # Instantiate robots
        for i in range(len(robot_starting_positions)):
            self.robots.append(robot(i, robot_starting_positions[i], args.speed, args.num_robots))

    def start_simulation_thread(self, on_complete=None):
        def simulation_loop():
            while not (all(r.is_finished for r in self.robots) and all(self.is_covered)):
                with self.lock:
                    for r in self.robots:
                        r.move_robot()
                time.sleep(0.05)
            time.sleep(5.0)
            print("Simulation complete.")
            if on_complete:
                on_complete()

        threading.Thread(target=simulation_loop, daemon=True).start()

    def start_animation(self):
        self.fig, self.ax = plt.subplots()

        # --- Only attempt window operations if a GUI window exists ---
        try:
            manager = getattr(self.fig.canvas, "manager", None)
            window = getattr(manager, "window", None)
            if window is not None:
                backend = (plt.get_backend() or "").lower()
                try:
                    if hasattr(window, "show"):
                        window.show()
                    if "qt" in backend:
                        try:
                            from PyQt6 import QtCore
                        except ImportError:
                            from PyQt5 import QtCore
                        QtCore.QTimer.singleShot(0, getattr(window, "showMinimized", lambda: None))
                    elif "tkagg" in backend and hasattr(window, "after"):
                        window.after(0, lambda: window.wm_state("iconic"))
                    elif hasattr(window, "Iconize"):  # wx
                        window.Iconize(True)
                    else:
                        def _minimize_on_first_draw(event):
                            try:
                                win = getattr(event.canvas.manager, "window", None)
                                if win is None:
                                    return
                                if hasattr(win, "showMinimized"):
                                    win.showMinimized()
                                elif hasattr(win, "wm_state"):
                                    win.wm_state("iconic")
                                elif hasattr(win, "Iconize"):
                                    win.Iconize(True)
                            finally:
                                event.canvas.mpl_disconnect(cid)
                        cid = self.fig.canvas.mpl_connect("draw_event", _minimize_on_first_draw)
                except Exception as e:
                    print(f"[WARN] GUI minimize skipped: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"[WARN] Could not inspect figure manager: {type(e).__name__}: {e}")

        self.ax.set_xlim(self.environmental_bounds[0], self.environmental_bounds[1])
        self.ax.set_ylim(self.environmental_bounds[2], self.environmental_bounds[3])
        self.ax.set_title("Live Robot Positions")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Tasks
        if len(self.tasks) > 0:
            task_xs, task_ys = zip(*self.tasks)
        else:
            task_xs, task_ys = [], []
        self.task_scat = self.ax.scatter(task_xs, task_ys, s=18, linewidths=1.6)

        # Robots scatter with NaN positions
        num_robots = len(self.robots)
        init_positions = np.full((num_robots, 2), np.nan, dtype=float)
        self.robot_scatter = self.ax.scatter(
            init_positions[:, 0],
            init_positions[:, 1],
            c=self.robot_colours,
            s=40,
            label='Robots'
        )

        # Robot labels
        self.robot_labels = [
            self.ax.text(np.nan, np.nan, str(i), fontsize=8, color='black', ha='center')
            for i in range(num_robots)
        ]

        def _finite_xy(x, y) -> bool:
            return x is not None and y is not None and np.isfinite(x) and np.isfinite(y)

        def update(_frame):
            with self.lock:
                positions = np.full((num_robots, 2), np.nan, dtype=float)
                for i, r in enumerate(self.robots):
                    try:
                        x, y = r.get_robot_position()[-1]
                        if _finite_xy(x, y):
                            positions[i] = (x, y)
                            self.robot_labels[i].set_position((x, y + 0.2))
                    except Exception:
                        pass
                self.robot_scatter.set_offsets(positions)

                if len(self.is_covered) == len(self.tasks) and len(self.tasks) > 0:
                    faces = ['blue' if covered else 'red' for covered in self.is_covered]
                    self.task_scat.set_facecolor(faces)

                outline_colors = []
                for owner in self.task_owner:
                    if owner is None or owner < 0 or owner >= num_robots:
                        outline_colors.append('none')
                    else:
                        outline_colors.append(self.robot_colours[owner])
                if len(outline_colors) == len(self.tasks) and len(self.tasks) > 0:
                    self.task_scat.set_edgecolors(outline_colors)

            return self.robot_scatter, self.task_scat, *self.robot_labels

        self.ani = FuncAnimation(self.fig, update, interval=50)

    def start_listeners(self, node):
        # keep a reference to node so we can publish
        self.listener_node = node

        # Coverage updates
        self.coverage_subscriber = node.create_subscription(
            EnvironmentalRepresentation,
            '/environmental_representation',
            self.environmental_representation_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )
        # Coalition best solution updates
        self.solution_subscriber = node.create_subscription(
            Solution,
            'best_solution',
            self.solution_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )

        # publisher for killing a robot — only if enabled
        if self.is_kill_enabled:
            for i in range(self.number_to_kill):
                self.kill_pubs.append(node.create_publisher(
                    Bool,
                    f"/central_control/uas_{len(self.robot_starting_positions)-1-i}/kill_robot",
                    10
                ))

        if self.is_revive_enabled:
            for i in range(self.number_to_revive):
                self.revive_pubs.append(node.create_publisher(
                    Bool,
                    f"/central_control/uas_{len(self.robot_starting_positions) - 1 - i}/revive_robot",
                    10
                ))


    def environmental_representation_callback(self, msg):
        # update local coverage
        with self.lock:
            for i in range(len(msg.is_covered)):
                if msg.is_covered[i]:
                    if i < len(self.is_covered):
                        self.is_covered[i] = True

            # kill logic — only if enabled
            if self.is_kill_enabled and not self.is_kill_published:
                total_tasks = len(self.is_covered)
                if total_tasks > 0:
                    covered = sum(self.is_covered)
                    ratio = covered / total_tasks
                    if ratio >= self.kill_threshold:
                        if self.kill_pubs is not None:
                            for i in self.kill_pubs:
                                for _ in range(4):
                                    kill_msg = Bool()
                                    kill_msg.data = True
                                    i.publish(kill_msg)
                            self.is_kill_published = True

            if self.is_revive_enabled and self.is_kill_published and not self.is_revive_published:
                total_tasks = len(self.is_covered)
                if total_tasks > 0:
                    covered = sum(self.is_covered)
                    ratio = covered / total_tasks
                    if ratio >= self.revive_threshold:
                        if self.revive_pubs is not None:
                            for i in self.revive_pubs:
                                for _ in range(4):
                                    revive_msg = Bool()
                                    revive_msg.data = True
                                    i.publish(revive_msg)
                            self.is_revive_published = True

    def solution_callback(self, msg: Solution):
        """
        Update per-task owner mapping from coalition best (order, allocations).
        """
        try:
            order = list(msg.order)
            alloc = list(msg.allocations)
            owners = [-1] * len(self.tasks)

            cursor = 0
            for agent_id, count in enumerate(alloc):
                start, end = cursor, cursor + count
                for t in order[start:end]:
                    if 0 <= t < len(owners):
                        owners[t] = agent_id
                cursor = end

            with self.lock:
                self.task_owner = owners
        except Exception as e:
            print(f"[solution_callback] failed to parse coalition best: {e}")


def main():
    print("Initializing ROS...")
    parser = argparse.ArgumentParser(description="Run CBM-POP Simulation")
    parser.add_argument('--num_robots', type=int, default=2, help='Number of robots')
    parser.add_argument('--speed', type=float, default=0.05, help='Robot speed')
    parser.add_argument('--problem_size', type=int, default=None,
                        help='Side length of the square grid (number of cells per side)')
    parser.add_argument('--env_size', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--no_gui', action='store_true', help='Run without GUI (headless)')
    parser.add_argument('--problem_class', type=str, default='simple_grid', help='How tasks are distributed')
    parser.add_argument('--problem_seed', type=int, default='simple_grid', help='Random seed for task distribution')

    # <<< new params for kill behaviour
    parser.add_argument('--enable_kill', action='store_true',
                        help='Enable publishing a kill message once coverage reaches threshold')
    parser.add_argument('--kill_threshold', type=float, default=0.20,
                        help='Coverage fraction (0-1) at which to publish kill (default 0.20)')
    parser.add_argument('--num_to_kill', type=float, default=1,
                        help='Number of robots to kill if kill enabled')
    parser.add_argument('--enable_revive', action='store_true',
                        help='Enable publishing a revival message once coverage reaches a threshold')
    parser.add_argument('--revive_threshold', type=float, default=0.80,
                        help='Coverage fraction (0-1) at which to publish revive (default 0.80)')
    # >>>

    args = parser.parse_args()

    if args.no_gui:
        matplotlib.use("Agg")

    rclpy.init(args=sys.argv)

    problem_size = args.problem_size if args.problem_size is not None else args.env_size
    if problem_size is None:
        problem_size = 15

    problem_size = int(problem_size)
    if problem_size < 1:
        raise ValueError("problem_size must be >= 1")

    args.problem_size = problem_size

    problem_class = args.problem_class if args.problem_class is not None else "Simple_Grid"
    if isinstance(problem_class, str):
        try:
            problem_class = ProblemClass(problem_class)
        except ValueError:
            raise ValueError(
                f"Invalid learning method '{problem_class}'. Must be one of: "
                f"{[e.value for e in ProblemClass]}"
            )
    problem_seed = args.problem_seed if args.problem_seed is not None else 1
    problem = SimpleProblem(problem_class=ProblemClass(problem_class), grid_size=problem_size, problem_seed=problem_seed)
    tasks = problem.task_poses

    env_bounds = [0, problem_size, 0, problem_size]
    xmin, xmax, ymin, ymax = env_bounds

    starts = [[(float(np.random.uniform(xmin, xmax)), float(np.random.uniform(ymin, ymax)))]
              for _ in range(args.num_robots)]

    simulator = SimpleSimulator(tasks, env_bounds, starts, obstacles=1, args=args)
    print("Simulator created.")

    executor = MultiThreadedExecutor()
    for r in simulator.robots:
        executor.add_node(r)

    listener_node = rclpy.create_node('sim_listeners')
    simulator.start_listeners(listener_node)
    executor.add_node(listener_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def shutdown():
        print("All robots finished.")
        try:
            path_lengths = []
            for r in simulator.robots:
                pts = np.array(r.get_robot_position(), dtype=float)
                if len(pts) >= 2 and np.all(np.isfinite(pts)):
                    path_lengths.append(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                else:
                    path_lengths.append(0.0)

            longest_path = max(path_lengths) if path_lengths else 0.0
            average_path = float(np.mean(path_lengths)) if path_lengths else 0.0

            print(f"Longest robot path length: {longest_path:.2f}")
            print(f"Average robot path length: {average_path:.2f}")
        except Exception as e:
            print(f"[WARN] Failed to compute path lengths: {e}")

        try:
            plt.close('all')
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)

    simulator.start_simulation_thread(on_complete=shutdown)
    simulator.start_animation()

    try:
        has_window = bool(getattr(getattr(simulator.fig.canvas, "manager", None), "window", None))
    except Exception:
        has_window = False

    try:
        while rclpy.ok():
            if has_window and not args.no_gui:
                plt.pause(0.1)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted. Manual shutdown.")
        shutdown()


if __name__ == '__main__':
    main()

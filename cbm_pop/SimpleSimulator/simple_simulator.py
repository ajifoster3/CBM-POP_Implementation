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
from std_msgs.msg import Bool
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

        self.num_robots_total = args.num_robots

        # split_mode: True when --robot_id is passed (one process per robot).
        # Original all-in-one behaviour when robot_id is None.
        self.robot_id = getattr(args, 'robot_id', None)
        self.split_mode = self.robot_id is not None

        self.finished_robots = [False] * args.num_robots
        self.is_covered = [False] * len(self.tasks)
        self.task_scat = None

        # kill-robot trigger config
        self.is_kill_enabled = bool(args.enable_kill)
        self.kill_threshold = float(args.kill_threshold)
        self.number_to_kill = int(args.num_to_kill)
        self.is_kill_published = False
        self.kill_pubs = []

        # revive-robot trigger config
        self.is_revive_enabled = bool(args.enable_revive)
        self.revive_threshold = float(args.revive_threshold)
        self.number_to_revive = self.number_to_kill
        self.is_revive_published = False
        self.revive_pubs = []

        # Full colour palette always built — needed for task-ownership colouring
        # even when this process only simulates one robot.
        num_robots = args.num_robots
        cmap_name = 'tab20' if num_robots > 10 else 'tab10'
        cmap = cm.get_cmap(cmap_name, num_robots)
        self.robot_colours = [mcolors.to_hex(cmap(i)) for i in range(num_robots)]

        self.task_owner = [-1] * len(self.tasks)

        # In split mode only instantiate this process's robot.
        if self.split_mode:
            rid = self.robot_id
            self.robots.append(
                robot(rid, robot_starting_positions[rid], args.speed, args.num_robots)
            )
        else:
            for i, pos in enumerate(robot_starting_positions):
                self.robots.append(robot(i, pos, args.speed, args.num_robots))

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
        # Visualization is only available in non-split (all-robots) mode.
        # In split mode start_animation is never called, but guard here for safety.
        if self.split_mode:
            return

        self.fig, self.ax = plt.subplots()

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
                    elif hasattr(window, "Iconize"):
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

        if len(self.tasks) > 0:
            task_xs, task_ys = zip(*self.tasks)
        else:
            task_xs, task_ys = [], []
        self.task_scat = self.ax.scatter(task_xs, task_ys, s=18, linewidths=1.6)

        num_robots = len(self.robots)
        init_positions = np.full((num_robots, 2), np.nan, dtype=float)
        self.robot_scatter = self.ax.scatter(
            init_positions[:, 0], init_positions[:, 1],
            c=self.robot_colours, s=40, label='Robots'
        )

        self.robot_labels = [
            self.ax.text(np.nan, np.nan, str(i), fontsize=8, color='black', ha='center')
            for i in range(num_robots)
        ]

        def _finite_xy(x, y):
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
        self.listener_node = node

        # ---- TOPICS ARE RELATIVE so ROS2 namespace resolution applies ----
        self.coverage_subscriber = node.create_subscription(
            EnvironmentalRepresentation,
            'environmental_representation',
            self.environmental_representation_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )
        self.solution_subscriber = node.create_subscription(
            Solution,
            'best_solution',
            self.solution_callback,
            10,
            callback_group=ReentrantCallbackGroup()
        )

        # In split mode only robot 0 handles kill/revive to avoid N duplicate
        # publishes on the same topics. In non-split mode the single process
        # always handles it, matching original behaviour.
        handles_kill_revive = (not self.split_mode) or (self.robot_id == 0)

        if self.is_kill_enabled and handles_kill_revive:
            for i in range(self.number_to_kill):
                self.kill_pubs.append(node.create_publisher(
                    Bool,
                    f"central_control/uas_{self.num_robots_total - 1 - i}/kill_robot",
                    10
                ))

        if self.is_revive_enabled and handles_kill_revive:
            for i in range(self.number_to_revive):
                self.revive_pubs.append(node.create_publisher(
                    Bool,
                    f"central_control/uas_{self.num_robots_total - 1 - i}/revive_robot",
                    10
                ))

    def environmental_representation_callback(self, msg):
        with self.lock:
            for i in range(len(msg.is_covered)):
                if msg.is_covered[i] and i < len(self.is_covered):
                    self.is_covered[i] = True

            handles_kill_revive = (not self.split_mode) or (self.robot_id == 0)

            if handles_kill_revive and self.is_kill_enabled and not self.is_kill_published:
                total_tasks = len(self.is_covered)
                if total_tasks > 0 and (sum(self.is_covered) / total_tasks) >= self.kill_threshold:
                    for pub in self.kill_pubs:
                        for _ in range(4):
                            kill_msg = Bool()
                            kill_msg.data = True
                            pub.publish(kill_msg)
                    self.is_kill_published = True

            if handles_kill_revive and self.is_revive_enabled \
                    and self.is_kill_published and not self.is_revive_published:
                total_tasks = len(self.is_covered)
                if total_tasks > 0 and (sum(self.is_covered) / total_tasks) >= self.revive_threshold:
                    for pub in self.revive_pubs:
                        for _ in range(4):
                            revive_msg = Bool()
                            revive_msg.data = True
                            pub.publish(revive_msg)
                    self.is_revive_published = True

    def solution_callback(self, msg: Solution):
        try:
            order = list(msg.order)
            alloc = list(msg.allocations)
            owners = [-1] * len(self.tasks)
            cursor = 0
            for agent_id, count in enumerate(alloc):
                for t in order[cursor:cursor + count]:
                    if 0 <= t < len(owners):
                        owners[t] = agent_id
                cursor += count
            with self.lock:
                self.task_owner = owners
        except Exception as e:
            print(f"[solution_callback] failed to parse coalition best: {e}")


def main():
    print("Initializing ROS...")
    parser = argparse.ArgumentParser(description="Run CBM-POP Simulation")
    parser.add_argument('--num_robots', type=int, default=2)
    parser.add_argument('--speed', type=float, default=0.05)
    parser.add_argument('--problem_size', type=int, default=None)
    parser.add_argument('--env_size', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--no_gui', action='store_true')
    parser.add_argument('--problem_class', type=str, default='simple_grid')
    parser.add_argument('--problem_seed', type=int, default=1)

    # Split-mode: pass --robot_id <i> to have this process simulate only robot i.
    # Omit entirely to use the original all-robots-in-one-process behaviour.
    parser.add_argument('--robot_id', type=int, default=None,
                        help='Index of the single robot this process simulates (split/HPC mode). '
                             'Omit to simulate all robots in one process (original behaviour).')

    parser.add_argument('--enable_kill', action='store_true')
    parser.add_argument('--kill_threshold', type=float, default=0.20)
    parser.add_argument('--num_to_kill', type=float, default=1)
    parser.add_argument('--enable_revive', action='store_true')
    parser.add_argument('--revive_threshold', type=float, default=0.80)

    # Use parse_known_args so --ros-args (namespace, remappings etc.) pass
    # through to rclpy.init() without argparse raising an error.
    args, _ = parser.parse_known_args()

    split_mode = args.robot_id is not None

    if split_mode and not args.no_gui:
        print("[WARN] Split mode (--robot_id) forces --no_gui. "
              "Run without --robot_id for live visualisation.")
        args.no_gui = True

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
                f"Invalid problem class '{problem_class}'. "
                f"Must be one of: {[e.value for e in ProblemClass]}"
            )

    problem_seed = args.problem_seed if args.problem_seed is not None else 1
    problem = SimpleProblem(
        problem_class=ProblemClass(problem_class),
        grid_size=problem_size,
        problem_seed=problem_seed,
    )
    tasks = problem.task_poses

    env_bounds = [0, problem_size, 0, problem_size]
    xmin, xmax, ymin, ymax = env_bounds

    # Seed before generating starts so every split process produces
    # the exact same starting positions for all robots.
    np.random.seed(problem_seed)
    starts = [
        [(float(np.random.uniform(xmin, xmax)), float(np.random.uniform(ymin, ymax)))]
        for _ in range(args.num_robots)
    ]

    simulator = SimpleSimulator(tasks, env_bounds, starts, obstacles=1, args=args)
    label = f"(split mode: robot {args.robot_id})" if split_mode else "(all robots)"
    print(f"Simulator created {label}.")

    executor = MultiThreadedExecutor()
    for r in simulator.robots:
        executor.add_node(r)

    # Unique listener node name avoids name collisions when N processes share
    # the same ROS domain.
    listener_name = f"sim_listeners_{args.robot_id}" if split_mode else "sim_listeners"
    listener_node = rclpy.create_node(listener_name)
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

    # Visualisation only in non-split mode
    has_window = False
    if not split_mode:
        simulator.start_animation()
        try:
            has_window = bool(
                getattr(getattr(simulator.fig.canvas, "manager", None), "window", None)
            )
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
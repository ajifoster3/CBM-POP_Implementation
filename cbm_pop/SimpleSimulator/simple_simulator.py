import sys
import time
import threading
import argparse
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from cbm_pop.SimpleSimulator.simulator_robot import SimulatorRobot as robot
from cbm_pop_interfaces.msg import EnvironmentalRepresentation, Solution
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
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

        # --- colours available immediately for callbacks ---
        num_robots = args.num_robots
        cmap_name = 'tab20' if num_robots > 10 else 'tab10'
        cmap = cm.get_cmap(cmap_name, num_robots)
        self.robot_colours = [mcolors.to_hex(cmap(i)) for i in range(num_robots)]

        # coalition best (order, allocations) -> per-task owner (agent id)
        self.task_owner = [-1] * len(self.tasks)   # -1 means unassigned

        for i in range(len(robot_starting_positions)):
            self.robots.append(robot(i, robot_starting_positions[i], args.speed, args.num_robots))

    def start_simulation_thread(self, on_complete=None):
        def simulation_loop():
            while not (all(robot.is_finished for robot in self.robots) and all(self.is_covered)):
                with self.lock:
                    for robot in self.robots:
                        robot.move_robot()
                time.sleep(0.05)
            time.sleep(5.0)
            print("Simulation complete.")
            if on_complete:
                on_complete()

        threading.Thread(target=simulation_loop, daemon=True).start()

    def start_animation(self):
        self.fig, self.ax = plt.subplots()

        # --- robust "start minimized" across common backends ---
        try:
            mgr = plt.get_current_fig_manager()
            backend = plt.get_backend().lower()

            if "qt" in backend:  # Qt5Agg / Qt6Agg
                try:
                    from PyQt6 import QtCore
                except ImportError:
                    from PyQt5 import QtCore
                mgr.window.show()
                QtCore.QTimer.singleShot(0, mgr.window.showMinimized)

            elif "tkagg" in backend:  # Tkinter
                mgr.window.update_idletasks()
                mgr.window.after(0, lambda: mgr.window.wm_state('iconic'))

            elif "wx" in backend:
                import wx
                wx.CallAfter(mgr.window.Iconize, True)

            else:
                def _minimize_on_first_draw(event):
                    try:
                        win = event.canvas.manager.window
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
            print(f"Could not minimize figure window: {e} (backend={plt.get_backend()})")
        # --- end minimized block ---

        self.ax.set_xlim(self.environmental_bounds[0], self.environmental_bounds[1])
        self.ax.set_ylim(self.environmental_bounds[2], self.environmental_bounds[3])
        self.ax.set_title("Live Robot Positions")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Tasks
        task_xs, task_ys = zip(*self.tasks)
        # Use facecolor for covered/uncovered; outline colour shows agent assignment
        self.task_scat = self.ax.scatter(task_xs, task_ys, s=18, label='Tasks',
                                         linewidths=1.6)  # linewidth for visible outline

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

        def update(frame):
            with self.lock:
                # update robot positions
                positions = np.full((num_robots, 2), np.nan, dtype=float)
                for i, r in enumerate(self.robots):
                    try:
                        x, y = r.get_robot_position()[-1]
                        positions[i] = (x, y)
                        self.robot_labels[i].set_position((x, y + 0.2))
                    except IndexError:
                        pass
                self.robot_scatter.set_offsets(positions)

                # task facecolors (covered/uncovered)
                task_faces = ['blue' if covered else 'red' for covered in self.is_covered]
                self.task_scat.set_facecolor(task_faces)

                # task edgecolors (assignment outline by agent)
                outline_colors = []
                for owner in self.task_owner:
                    if owner is None or owner < 0 or owner >= num_robots:
                        outline_colors.append('none')  # no outline if unassigned
                    else:
                        outline_colors.append(self.robot_colours[owner])
                self.task_scat.set_edgecolors(outline_colors)

            return self.robot_scatter, self.task_scat, *self.robot_labels

        self.ani = FuncAnimation(self.fig, update, interval=50)

    def start_listeners(self, node):
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

    def environmental_representation_callback(self, msg):
        with self.lock:
            for i in range(len(msg.is_covered)):
                if msg.is_covered[i]:
                    self.is_covered[i] = True

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
    parser.add_argument('--env_size', type=int, default=5, help='Environment size (square)')
    args = parser.parse_args()

    rclpy.init(args=sys.argv)

    tasks = [(i + 0.5, j + 0.5) for i in range(15) for j in range(15)]
    env_bounds = [0, 15, 0, 15]
    xmin, xmax, ymin, ymax = env_bounds

    starts = [[(np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax))] for _ in range(args.num_robots)]

    simulator = SimpleSimulator(tasks, env_bounds, starts, obstacles=1, args=args)
    print("Simulator created.")

    executor = MultiThreadedExecutor()
    for r in simulator.robots:
        executor.add_node(r)

    # Node to listen for ROS topics (coverage + coalition best)
    listener_node = rclpy.create_node('sim_listeners')
    simulator.start_listeners(listener_node)
    executor.add_node(listener_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def shutdown():
        print("All robots finished.")
        path_lengths = [np.sum(
            np.linalg.norm(np.diff(np.array(r.get_robot_position()), axis=0), axis=1)
        ) for r in simulator.robots]

        longest_path = max(path_lengths)
        average_path = np.mean(path_lengths)

        print(f"Longest robot path length: {longest_path:.2f}")
        print(f"Average robot path length: {average_path:.2f}")

        plt.close('all')
        executor.shutdown()
        rclpy.shutdown()
        sys.exit(0)

    simulator.start_simulation_thread(on_complete=shutdown)
    simulator.start_animation()

    try:
        while rclpy.ok():
            plt.pause(0.1)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted. Manual shutdown.")
        shutdown()


if __name__ == '__main__':
    main()

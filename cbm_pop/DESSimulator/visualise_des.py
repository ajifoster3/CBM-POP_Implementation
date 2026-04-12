#!/usr/bin/env python3
"""
Visualise DES simulation logs.

Usage (static path view):
    python -m cbm_pop.DESSimulator.visualise_des --log_dir /path/to/logs

Usage (animation):
    python -m cbm_pop.DESSimulator.visualise_des --log_dir /path/to/logs --animate

The script reads:
  setup.json          — task poses, robot starts, robot speed
  robot_path_log.csv  — movement legs
  event_log.csv       — task-covered events (optional)
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


# ── colour cycle for robots ──────────────────────────────────────────────────

_ROBOT_COLOURS = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
    '#ff7f00', '#a65628', '#f781bf', '#999999',
]


def _robot_colour(robot_id: int) -> str:
    return _ROBOT_COLOURS[robot_id % len(_ROBOT_COLOURS)]


# ── data loading ─────────────────────────────────────────────────────────────

def load_logs(log_dir: str):
    with open(os.path.join(log_dir, 'setup.json')) as f:
        setup = json.load(f)

    path_csv = os.path.join(log_dir, 'robot_path_log.csv')
    paths = pd.read_csv(path_csv)

    event_csv = os.path.join(log_dir, 'event_log.csv')
    events = None
    if os.path.exists(event_csv):
        events = pd.read_csv(event_csv)
        events = events[events['event_type'] == 'TASK_COVERED']

    return setup, paths, events


# ── interpolation helper ─────────────────────────────────────────────────────

def robot_position_at(paths_for_robot: pd.DataFrame, t: float):
    """
    Return (x, y) for a single robot at simulation time t by linear
    interpolation along its recorded legs.
    """
    df = paths_for_robot
    if df.empty:
        return None

    # Legs that have already started (sim_time <= t)
    started = df[df['sim_time'] <= t]
    if started.empty:
        # Before the robot has moved — return first known start position
        row = df.iloc[0]
        return float(row['from_x']), float(row['from_y'])

    row = started.iloc[-1]
    leg_start  = float(row['sim_time'])
    leg_end    = float(row['arrival_sim_time'])
    from_x, from_y = float(row['from_x']), float(row['from_y'])
    to_x,   to_y   = float(row['to_x']),   float(row['to_y'])

    if leg_end <= leg_start or t >= leg_end:
        return to_x, to_y

    alpha = (t - leg_start) / (leg_end - leg_start)
    return from_x + alpha * (to_x - from_x), from_y + alpha * (to_y - from_y)


# ── static plot ──────────────────────────────────────────────────────────────

def plot_static(log_dir: str) -> None:
    setup, paths, events = load_logs(log_dir)

    task_poses   = np.array(setup['task_poses'])
    robot_starts = np.array(setup['robot_starts'])
    num_robots   = setup['num_agents']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_title('Robot paths (full simulation)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Tasks
    ax.scatter(task_poses[:, 0], task_poses[:, 1],
               marker='x', s=80, c='black', zorder=5, label='Tasks')
    for i, (x, y) in enumerate(task_poses):
        ax.text(x, y, f' {i}', fontsize=7, color='black', zorder=6)

    # Robot starts
    ax.scatter(robot_starts[:, 0], robot_starts[:, 1],
               marker='o', s=60, c='grey', zorder=5, label='Start')

    # Paths
    for rid in range(num_robots):
        df = paths[paths['robot_id'] == rid].sort_values('sim_time')
        if df.empty:
            continue
        colour = _robot_colour(rid)
        xs = list(df['from_x']) + [df.iloc[-1]['to_x']]
        ys = list(df['from_y']) + [df.iloc[-1]['to_y']]
        ax.plot(xs, ys, '-', color=colour, linewidth=1.2,
                alpha=0.7, label=f'Robot {rid}')
        ax.plot(xs[0], ys[0], 'o', color=colour, markersize=6)
        ax.plot(xs[-1], ys[-1], 's', color=colour, markersize=6)

    # Task-covered markers
    if events is not None:
        for _, ev in events.iterrows():
            tid = int(ev['task_id'])
            ax.scatter(*task_poses[tid], marker='*', s=200,
                       c='gold', zorder=7, edgecolors='black', linewidths=0.5)

    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


# ── animation ────────────────────────────────────────────────────────────────

def animate(log_dir: str, fps: int = 30, speed: float = 1.0) -> None:
    """
    Animate robot movement.

    speed : sim-seconds shown per real second (e.g. speed=10 means 10s of sim
            time passes per real second of animation).
    """
    setup, paths, events = load_logs(log_dir)

    task_poses   = np.array(setup['task_poses'])
    robot_starts = np.array(setup['robot_starts'])
    num_robots   = setup['num_agents']

    t_max = paths['arrival_sim_time'].max() if not paths.empty else 1.0

    # Pre-split path data per robot
    robot_paths = {
        rid: paths[paths['robot_id'] == rid].sort_values('sim_time').reset_index(drop=True)
        for rid in range(num_robots)
    }

    # Task-covered lookup: task_id -> sim_time
    covered_at = {}
    if events is not None:
        for _, ev in events.iterrows():
            covered_at[int(ev['task_id'])] = float(ev['sim_time'])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Static task markers
    task_scatters = []
    for i, (x, y) in enumerate(task_poses):
        sc = ax.scatter(x, y, marker='x', s=80, c='black', zorder=5)
        ax.text(x, y, f' {i}', fontsize=7, color='black', zorder=6)
        task_scatters.append(sc)

    # Robot dots and trail lines
    robot_dots  = []
    robot_trails = []
    for rid in range(num_robots):
        colour = _robot_colour(rid)
        dot,  = ax.plot([], [], 'o', color=colour, markersize=8, zorder=10)
        trail, = ax.plot([], [], '-', color=colour, linewidth=1.0,
                         alpha=0.5, zorder=4)
        robot_dots.append(dot)
        robot_trails.append(trail)

    # Sim-time label
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        fontsize=9, verticalalignment='top')

    # Trail history per robot
    trail_xs: list[list] = [[] for _ in range(num_robots)]
    trail_ys: list[list] = [[] for _ in range(num_robots)]

    dt_real   = 1.0 / fps          # real seconds per frame
    dt_sim    = dt_real * speed    # sim seconds per frame
    num_frames = max(1, int(t_max / dt_sim) + 1)

    def init():
        for dot, trail in zip(robot_dots, robot_trails):
            dot.set_data([], [])
            trail.set_data([], [])
        time_text.set_text('')
        return robot_dots + robot_trails + [time_text]

    def update(frame):
        t = frame * dt_sim
        time_text.set_text(f't = {t:.2f} s')

        # Colour covered tasks gold
        for tid, sc in enumerate(task_scatters):
            if tid in covered_at and covered_at[tid] <= t:
                sc.set_color('gold')
                sc.set_sizes([150])

        for rid in range(num_robots):
            pos = robot_position_at(robot_paths[rid], t)
            if pos is None:
                continue
            x, y = pos
            trail_xs[rid].append(x)
            trail_ys[rid].append(y)
            robot_dots[rid].set_data([x], [y])
            robot_trails[rid].set_data(trail_xs[rid], trail_ys[rid])

        return robot_dots + robot_trails + task_scatters + [time_text]

    ani = animation.FuncAnimation(
        fig, update, frames=num_frames,
        init_func=init, blit=False,
        interval=int(dt_real * 1000),
    )

    plt.tight_layout()
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Visualise DES simulation logs')
    p.add_argument('--log_dir', required=True,
                   help='Directory containing setup.json and CSV logs')
    p.add_argument('--animate', action='store_true',
                   help='Show animation instead of static plot')
    p.add_argument('--speed', type=float, default=5.0,
                   help='Sim-seconds per real second for animation (default: 5)')
    p.add_argument('--fps', type=int, default=30,
                   help='Animation frames per second (default: 30)')
    args = p.parse_args()

    if not os.path.isdir(args.log_dir):
        print(f'Error: {args.log_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    if args.animate:
        animate(args.log_dir, fps=args.fps, speed=args.speed)
    else:
        plot_static(args.log_dir)


if __name__ == '__main__':
    main()
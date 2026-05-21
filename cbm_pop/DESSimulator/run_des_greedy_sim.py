#!/usr/bin/env python3
"""
Entry point for the greedy-baseline Discrete Event Simulation.

Run directly:
    python -m cbm_pop.DESSimulator.run_des_greedy_sim --num_agents 5 --problem_size 10

Logs are written in the same format as run_des_sim.py so results can be
compared side-by-side.
"""

import argparse
import sys
import time

from cbm_pop.DESSimulator.des_logger import DESLogger
from cbm_pop.DESSimulator.des_greedy_simulation import DESGreedySimulation
from cbm_pop.SimpleSimulator.simple_problem import ProblemClass, SimpleProblem


def parse_args():
    p = argparse.ArgumentParser(description='CBM-POP Greedy DES Baseline')

    p.add_argument('--num_agents',    type=int,   default=10)
    p.add_argument('--problem_size',  type=int,   default=20)
    p.add_argument('--problem_class', type=str,   default='Simple_Grid')
    p.add_argument('--problem_seed',  type=int,   default=1)
    p.add_argument('--speed',         type=float, default=1.0,
                   help='Robot speed (grid units / sim-second)')
    p.add_argument('--max_sim_time',  type=float, default=float('inf'),
                   help='Hard cap on sim time (default: unlimited)')

    # Kill / Revive
    p.add_argument('--enable_kill',      action='store_true')
    p.add_argument('--kill_threshold',   type=float, default=0.2)
    p.add_argument('--num_to_kill',      type=int,   default=1)
    p.add_argument('--enable_revive',    action='store_true')
    p.add_argument('--revive_threshold', type=float, default=0.8)

    # Output
    p.add_argument('--output_dir',        type=str,   default='log/des_greedy_sim/',
                   help='Directory for CSV logs (default: log/des_greedy_sim/)')
    p.add_argument('--progress_interval', type=float, default=0.5,
                   help='Print a progress line on each task coverage (0 = off)')

    return p.parse_args()


def main():
    args = parse_args()

    problem_class = ProblemClass(args.problem_class)
    problem = SimpleProblem(
        problem_class,
        grid_size=args.problem_size,
        problem_seed=args.problem_seed,
    )

    logger = None
    if args.output_dir:
        logger = DESLogger(args.output_dir, problem.num_tasks)

    sim = DESGreedySimulation(
        problem=problem,
        num_agents=args.num_agents,
        robot_speed=args.speed,
        seed=args.problem_seed,
        max_sim_time=args.max_sim_time,
        logger=logger,
        enable_kill=args.enable_kill,
        kill_threshold=args.kill_threshold,
        num_to_kill=args.num_to_kill,
        enable_revive=args.enable_revive,
        revive_threshold=args.revive_threshold,
    )

    print(f'Problem: {args.problem_class}  size={args.problem_size}  '
          f'tasks={problem.num_tasks}  agents={args.num_agents}')
    print('Running greedy DES...')
    sys.stdout.flush()

    wall_start = time.monotonic()
    try:
        summary = sim.run(progress_interval=args.progress_interval)
    except BaseException as exc:
        import traceback
        import os
        print(f'\n[FATAL] {type(exc).__name__} raised in sim.run():', flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        try:
            os.fsync(sys.stdout.fileno())
        except Exception:
            pass
        if logger:
            try:
                logger.close()
            except Exception:
                pass
        code = exc.code if isinstance(exc, SystemExit) else 1
        sys.exit(code if code is not None else 1)
    wall_time = time.monotonic() - wall_start

    if logger:
        logger.close()

    print()
    print(f'  sim time      : {summary["sim_time"]:.4f} s')
    print(f'  wall time     : {wall_time:.2f} s')
    print(f'  tasks covered : {summary["tasks_covered"]} / {summary["total_tasks"]}')
    print(f'  complete      : {summary["complete"]}')
    sys.stdout.flush()
    try:
        import os
        os.fsync(sys.stdout.fileno())
    except Exception:
        pass

    sys.exit(0 if summary['complete'] else 1)


if __name__ == '__main__':
    main()

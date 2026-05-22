#!/usr/bin/env python3
"""
Entry point for the CBM-POP Discrete Event Simulation.

Run directly:
    python -m cbm_pop.DESSimulator.run_des_sim --num_agents 5 --problem_size 10

Or, if registered as a console script in setup.py:
    des_sim --num_agents 5 --problem_size 10
"""

import argparse
import os
import sys
import time

from cbm_pop.DESSimulator.des_logger import DESLogger
from cbm_pop.DESSimulator.des_simulation import DESSimulation
from cbm_pop.SimpleSimulator.simple_problem import ProblemClass, SimpleProblem


def parse_args():
    p = argparse.ArgumentParser(description='CBM-POP Discrete Event Simulation')

    # Problem
    p.add_argument('--num_agents',    type=int,   default=10)
    p.add_argument('--problem_size',  type=int,   default=20)
    p.add_argument('--problem_class', type=str,   default='Simple_Grid')
    p.add_argument('--problem_seed',  type=int,   default=1)
    p.add_argument('--speed',         type=float, default=1.0,
                   help='Robot speed (grid units / sim-second)')
    p.add_argument('--max_sim_time',  type=float, default=float('inf'),
                   help='Hard cap on sim time (default: unlimited)')
    p.add_argument('--compute_time_scale', type=float, default=1.0,
                   help='Multiplier applied to measured operator wall time before '
                        'advancing sim time (default: 1.0 = real measured time)')

    # Agent
    p.add_argument('--method',             type=str,   default='UCB',
                   choices=['Q-Learning', 'Q-Learning-Step', 'Q-Learning-Separate',
                            'Q-Learning-improveoncurrent', 'Q-Learning-Step-improveoncurrent',
                            'Q-Learning-Separate-improveoncurrent',
                            'Ferreira_et_al.', 'UCB', 'Uniform'])
    p.add_argument('--pop_size',           type=int,   default=10)
    p.add_argument('--di_cycle_length',    type=int,   default=10)
    p.add_argument('--num_solution_attempts', type=int, default=21)
    p.add_argument('--lr',                 type=float, default=0.22)
    p.add_argument('--gamma_decay',        type=float, default=0.95)
    p.add_argument('--positive_reward',    type=float, default=7.0)
    p.add_argument('--negative_reward',    type=float, default=-8.0)
    p.add_argument('--rho',                type=float, default=0.5)
    p.add_argument('--eta',                type=float, default=0.1,
                   help='Ferreira et al. discount factor')
    p.add_argument('--ucb_c',              type=float, default=1.414,
                   help='UCB exploration constant')
    p.add_argument('--ucb_window',         type=int,   default=200,
                   help='UCB sliding window size')
    p.add_argument('--time_discount',      action='store_true',
                   help='Enable time-discounted rewards')
    p.add_argument('--time_discount_lambda', type=float, default=0.1,
                   help='Time discount strength λ (default: 0.1)')
    p.add_argument('--is_free_weight_matrix', action='store_true',
                   help='Use free (all-ones) weight matrix instead of classical')
    p.add_argument('--init_method', type=str, default='voronoi',
                   choices=['voronoi', 'greedy', 'random'],
                   help='Population initialisation method (default: voronoi)')
    p.add_argument('--random_init', action='store_true',
                   help='Initialise population randomly (legacy alias for --init_method random)')
    p.add_argument('--no_knn',             action='store_true',
                   help='Disable KNN operator')
    p.add_argument('--no_mimetism',        action='store_true')
    p.add_argument('--no_inject',          action='store_true',
                   help='Disable inject-best-on-cycle')
    p.add_argument('--inject_best_prob',   type=float, default=0.9)
    p.add_argument('--no_append_first_task', action='store_true',
                   help='Disable prepending current task when receiving coalition best')

    # Kill / Revive
    p.add_argument('--enable_kill',      action='store_true',
                   help='Kill the highest-numbered robot(s) at kill_threshold coverage')
    p.add_argument('--kill_threshold',   type=float, default=0.2,
                   help='Coverage fraction at which kill is triggered (default: 0.2)')
    p.add_argument('--num_to_kill',      type=int,   default=1,
                   help='Number of robots to kill (default: 1)')
    p.add_argument('--enable_revive',    action='store_true',
                   help='Revive killed robots at revive_threshold coverage')
    p.add_argument('--revive_threshold', type=float, default=0.8,
                   help='Coverage fraction at which revive is triggered (default: 0.8)')

    # Output
    p.add_argument('--output_dir', type=str, default="log/des_sim/",
                   help='Directory for CSV logs (default: log/des_sim/)')
    p.add_argument('--progress_interval', type=float, default=0.5,
                   help='Print a progress line every this many sim-time units (0 = off)')

    return p.parse_args()


def main():
    args = parse_args()

    problem_class = ProblemClass(args.problem_class)
    problem = SimpleProblem(
        problem_class,
        grid_size=args.problem_size,
        problem_seed=args.problem_seed,
    )

    agent_kwargs = dict(
        method=args.method,
        pop_size=args.pop_size,
        di_cycle_length=args.di_cycle_length,
        num_solution_attempts=args.num_solution_attempts,
        lr=args.lr,
        gamma_decay=args.gamma_decay,
        positive_reward=args.positive_reward,
        negative_reward=args.negative_reward,
        rho=args.rho,
        eta=args.eta,
        ucb_c=args.ucb_c,
        ucb_window=args.ucb_window,
        time_discount=args.time_discount,
        time_discount_lambda=args.time_discount_lambda,
        is_free_weight_matrix=args.is_free_weight_matrix,
        init_method='random' if args.random_init else args.init_method,
        initialise_with_heuristic=not args.random_init,
        is_knn_enabled=not args.no_knn,
        is_mimetism_enabled=not args.no_mimetism,
        is_inject_best_on_cycle=not args.no_inject,
        inject_best_prob=args.inject_best_prob,
        is_append_first_task=not args.no_append_first_task,
    )

    logger = None
    if args.output_dir:
        logger = DESLogger(args.output_dir, problem.num_tasks)

    sim = DESSimulation(
        problem=problem,
        num_agents=args.num_agents,
        robot_speed=args.speed,
        seed=args.problem_seed,
        agent_kwargs=agent_kwargs,
        max_sim_time=args.max_sim_time,
        compute_time_scale=args.compute_time_scale,
        logger=logger,
        enable_kill=args.enable_kill,
        kill_threshold=args.kill_threshold,
        num_to_kill=args.num_to_kill,
        enable_revive=args.enable_revive,
        revive_threshold=args.revive_threshold,
    )

    print(f'Problem: {args.problem_class}  size={args.problem_size}  '
          f'tasks={problem.num_tasks}  agents={args.num_agents}')
    print('Running DES...')
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
    wall_time  = time.monotonic() - wall_start

    if logger:
        logger.close()

    print()
    print(f'  sim time      : {summary["sim_time"]:.4f} s')
    print(f'  wall time     : {wall_time:.2f} s')
    print(f'  tasks covered : {summary["tasks_covered"]} / {summary["total_tasks"]}')
    print(f'  complete      : {summary["complete"]}')
    print(f'  iterations    : {summary["iterations_per_agent"]}')
    print(f'  coalition fit : '
          f'{[f"{f:.3f}" for f in summary["coalition_fitness"]]}')
    sys.stdout.flush()
    try:
        import os
        os.fsync(sys.stdout.fileno())
    except Exception:
        pass

    sys.exit(0 if summary['complete'] else 1)


if __name__ == '__main__':
    main()

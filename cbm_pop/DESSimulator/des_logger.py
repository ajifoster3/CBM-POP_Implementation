import csv
import json
import os
import time as _wall
from pathlib import Path
from typing import List, Optional


class DESLogger:
    """
    Writes logs to output_dir:

      setup.json          — task positions, robot starts, speed
      coverage_log.csv    — sim_time, wall_time, tasks_covered, coverage_fraction
      operator_log.csv    — sim_time, wall_time, agent_id, operator, op_wall_time,
                            coalition_improved, coalition_fitness
      event_log.csv       — sim_time, wall_time, event_type, robot_id, task_id
      robot_path_log.csv  — one row per movement leg:
                            sim_time, robot_id, from_x, from_y, to_x, to_y,
                            task_id, arrival_sim_time
    """

    def __init__(self, output_dir: str, num_tasks: int):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._wall_start   = _wall.monotonic()
        self._num_tasks    = num_tasks
        self._last_covered = -1

        def _open(name, headers):
            f = open(os.path.join(output_dir, name), 'w', newline='')
            w = csv.writer(f)
            w.writerow(headers)
            return f, w

        self._cov_f, self._cov_w = _open(
            'coverage_log.csv',
            ['sim_time', 'wall_time', 'tasks_covered', 'coverage_fraction'],
        )
        self._op_f, self._op_w = _open(
            'operator_log.csv',
            ['sim_time', 'wall_time', 'agent_id', 'operator',
             'op_wall_time', 'coalition_improved', 'coalition_fitness'],
        )
        self._ev_f, self._ev_w = _open(
            'event_log.csv',
            ['sim_time', 'wall_time', 'event_type', 'robot_id', 'task_id', 'fitness'],
        )
        self._path_f, self._path_w = _open(
            'robot_path_log.csv',
            ['sim_time', 'robot_id', 'from_x', 'from_y',
             'to_x', 'to_y', 'task_id', 'arrival_sim_time'],
        )
        self._di_f, self._di_w = _open(
            'di_cycle_log.csv',
            ['sim_time', 'wall_time', 'agent_id', 'di_cycle_num',
             'best_local_improved', 'best_coalition_improved',
             'num_experiences', 'total_gain', 'operators_json'],
        )
        self._wm_f, self._wm_w = _open(
            'weight_matrix_log.csv',
            ['sim_time', 'wall_time', 'agent_id', 'di_cycle_num', 'weights_json'],
        )
        self._output_dir = output_dir

    def log_setup(
        self,
        task_poses: list,
        robot_starts: list,
        robot_speed: float,
        num_agents: int,
    ) -> None:
        """Write static problem geometry to setup.json."""
        data = {
            'num_agents':   num_agents,
            'num_tasks':    len(task_poses),
            'robot_speed':  robot_speed,
            'task_poses':   [[float(x), float(y)] for x, y in task_poses],
            'robot_starts': [[float(x), float(y)] for x, y in robot_starts],
        }
        with open(os.path.join(self._output_dir, 'setup.json'), 'w') as f:
            json.dump(data, f, indent=2)

    def log_robot_leg(
        self,
        sim_time:        float,
        robot_id:        int,
        from_x:          float,
        from_y:          float,
        to_x:            float,
        to_y:            float,
        task_id:         int,
        arrival_sim_time: float,
    ) -> None:
        self._path_w.writerow([
            f'{sim_time:.6f}', robot_id,
            f'{from_x:.6f}', f'{from_y:.6f}',
            f'{to_x:.6f}',   f'{to_y:.6f}',
            task_id, f'{arrival_sim_time:.6f}',
        ])

    def _wt(self) -> str:
        return f'{_wall.monotonic() - self._wall_start:.4f}'

    def tick(self, sim_time: float, is_covered: List[bool], agents) -> None:
        covered = sum(is_covered)
        if covered != self._last_covered:
            self._cov_w.writerow([
                f'{sim_time:.6f}', self._wt(),
                covered, f'{covered / self._num_tasks:.4f}',
            ])
            self._last_covered = covered

    def operator_complete(
        self,
        sim_time:            float,
        agent_id:            int,
        operator_name:       str,
        op_wall_time:        float,
        coalition_improved:  bool,
        coalition_fitness:   float,
    ) -> None:
        self._op_w.writerow([
            f'{sim_time:.6f}', self._wt(), agent_id, operator_name,
            f'{op_wall_time:.6f}', int(coalition_improved),
            f'{coalition_fitness:.6f}',
        ])

    def task_covered(self, sim_time: float, task_id: int, robot_id: int) -> None:
        self._ev_w.writerow([
            f'{sim_time:.6f}', self._wt(), 'TASK_COVERED', robot_id, task_id, '',
        ])

    def di_cycle_complete(
        self,
        sim_time:               float,
        agent_id:               int,
        di_cycle_num:           int,
        best_local_improved:    bool,
        best_coalition_improved: bool,
        experiences:            list,       # list of [condition, op_name, gain]
        weights:                list,       # weight_matrix.weights (list of lists)
    ) -> None:
        total_gain = sum(e[2] for e in experiences) if experiences else 0.0

        op_counts: dict = {}
        for _, op_name, _ in experiences:
            op_counts[op_name] = op_counts.get(op_name, 0) + 1

        wt = self._wt()
        self._di_w.writerow([
            f'{sim_time:.6f}', wt, agent_id, di_cycle_num,
            int(best_local_improved), int(best_coalition_improved),
            len(experiences), f'{total_gain:.6f}',
            json.dumps(op_counts),
        ])
        self._wm_w.writerow([
            f'{sim_time:.6f}', wt, agent_id, di_cycle_num,
            json.dumps([[round(w, 6) for w in row] for row in weights]),
        ])

    def coalition_best_improved(
        self,
        sim_time: float,
        agent_id: int,
        fitness:  float,
    ) -> None:
        self._ev_w.writerow([
            f'{sim_time:.6f}', self._wt(), 'COALITION_BEST', agent_id, '', f'{fitness:.6f}',
        ])

    def close(self) -> None:
        for f in (self._cov_f, self._op_f, self._ev_f,
                  self._path_f, self._di_f, self._wm_f):
            try:
                f.close()
            except Exception:
                pass

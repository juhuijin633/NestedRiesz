#!/usr/bin/env python3
"""DiD simulation pipeline.

  python code/RUN.py              # local: all 9 sims sequentially, then summary.csv
  sbatch code/submit.sbatch       # cluster: array 0-8 simulate; collect separately via
                                  #   python code/collect_results.py
"""

import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
os.chdir(APP_DIR)

from collect_results import collect_results
from run_simulation import run_simulation

NUM_SIM_TASKS = 9


def main():
    array_task = os.environ.get("SLURM_ARRAY_TASK_ID")

    if array_task is not None:
        task_id = int(array_task)
        if not (0 <= task_id < NUM_SIM_TASKS):
            raise ValueError(
                f"Unexpected SLURM_ARRAY_TASK_ID={task_id} (expected 0-{NUM_SIM_TASKS - 1})"
            )
        run_simulation(task_id=task_id)
        return

    for task_id in range(NUM_SIM_TASKS):
        run_simulation(task_id=task_id)
    collect_results()


if __name__ == "__main__":
    main()

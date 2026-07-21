#!/usr/bin/env python3
"""Surrogacy application pipeline.

  python code/RUN.py

Runs, in order:
  1. calc_auto.py   for earn and employ  -> results/auto_{earn,employ}.csv
  2. calc_manual.py for earn and employ  -> results/manual_{earn,employ}.pkl
  3. make_figs_tables.py                 -> figures and summary tables
"""

import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
APP_DIR = CODE_DIR.parent

CALC_AUTO = CODE_DIR / "1. calc_auto.py"
CALC_MANUAL = CODE_DIR / "2. calc_manual.py"
MAKE_FIGS = CODE_DIR / "3. make_figs_tables.py"
APPLICATIONS = ("earn", "employ")


def run_script(script: Path, application: str | None = None) -> None:
    env = os.environ.copy()
    if application is not None:
        env["SURROGACY_APPLICATION"] = application

    label = script.name
    if application is not None:
        label = f"{label} ({application})"

    print(f"\n{'=' * 72}\nRunning {label}\n{'=' * 72}\n", flush=True)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=APP_DIR,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def main() -> None:
    os.chdir(APP_DIR)
    os.makedirs(APP_DIR / "results", exist_ok=True)

    for application in APPLICATIONS:
        run_script(CALC_AUTO, application)

    for application in APPLICATIONS:
        run_script(CALC_MANUAL, application)

    run_script(MAKE_FIGS)

    print(f"\nPipeline finished. Outputs in {APP_DIR / 'results'}/")


if __name__ == "__main__":
    main()

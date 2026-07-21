#!/usr/bin/env python3
"""Collect Monte Carlo simulation tables and write summary outputs.

Loads per-iteration .pt files, computes bias / RMSE / coverage / interval length,
and writes to results/:
  - results/summary_tables.txt  (human-readable tables, one block per configuration)
  - results/summary_tables.csv  (long-format rows for all configurations)

Run from Simulation_1_TimeTreatment/:
  python code/collect_results.py
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
RESULTS_DIR = APP_DIR / "results"

# -----------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------
N_VALUES = [500, 1000, 2000]
TMAX_DEFAULT = 500

# Table row order; maps to keys in each result_{t}.pt (see run_sim.py / run_sim_nonlinear.py).
METHODS = [
    "Oracle",
    "Manual-Lasso (Bradic Original)",  # estimateBradic (R script)
    "Manual-Lasso (Bradic)",           # DML_dynamic, nu_score='S-DRL', logistic prop
    "Manual-RF (Bradic)",
    "Manual-NN (Bradic)",
    "Manual-Lasso",                    # DML_dynamic, nu_score='regression', logistic prop
    "Manual-RF",
    "Manual-NN",
    "Manual-Lasso-Lasso",              # DML_dynamic, S-DRL, matched lasso propensity
    "Manual-RF-RF",
    "Manual-NN-NN",
    "Auto-Lasso",
    "Auto-RF",
    "Auto-NN",
]

# Each config: estimand, label, result_dir, func, lower, upper, theta_key
# Truncation [lower, upper] applies to truncated_* propensity models; logistic uses None.
CONFIGS = [
    # --- Main E[Y(1,1)] simulations (submit.sh; default truncation 0.1 / 0.9) ---
    {
        "estimand": "E[Y(1,1)]",
        "label": "Linear DGP + truncated logistic",
        "result_dir": "results/linear",
        "func": "truncated_logistic",
        "lower": 0.1,
        "upper": 0.9,
        "theta_key": "theta_true",
    },
    {
        "estimand": "E[Y(1,1)]",
        "label": "Nonlinear DGP + truncated adversarial",
        "result_dir": "results/nonlinear",
        "func": "truncated_adv",
        "lower": 0.1,
        "upper": 0.9,
        "theta_key": "theta_true",
    },
    {
        "estimand": "E[Y(1,1)]",
        "label": "Linear DGP + truncated adversarial",
        "result_dir": "results/linear",
        "func": "truncated_adv",
        "lower": 0.1,
        "upper": 0.9,
        "theta_key": "theta_true",
    },
    {
        "estimand": "E[Y(1,1)]",
        "label": "Linear DGP + logistic",
        "result_dir": "results/linear",
        "func": "logistic",
        "lower": None,
        "upper": None,
        "theta_key": "theta_true",
    },
    # --- Appendix: alternate truncation levels (submit_nonATE.sh) ---
    {
        "estimand": "E[Y(1,1)]",
        "label": "Linear DGP + truncated logistic",
        "result_dir": "results/linear",
        "func": "truncated_logistic",
        "lower": 0.3,
        "upper": 0.7,
        "theta_key": "theta_true",
    },
    {
        "estimand": "E[Y(1,1)]",
        "label": "Nonlinear DGP + truncated adversarial",
        "result_dir": "results/nonlinear",
        "func": "truncated_adv",
        "lower": 0.3,
        "upper": 0.7,
        "theta_key": "theta_true",
    },
    {
        "estimand": "E[Y(1,1)]",
        "label": "Linear DGP + truncated adversarial",
        "result_dir": "results/linear",
        "func": "truncated_adv",
        "lower": 0.3,
        "upper": 0.7,
        "theta_key": "theta_true",
    },
    # --- ATE simulations (submit_ate.sh; legacy result paths use func name only) ---
    {
        "estimand": "ATE",
        "label": "Linear DGP + truncated logistic",
        "result_dir": "results_ate",
        "func": "truncated_logistic",
        "lower": 0.1,
        "upper": 0.9,
        "theta_key": "ate_true",
        "legacy_subdir": True,
    },
    {
        "estimand": "ATE",
        "label": "Nonlinear DGP + truncated adversarial",
        "result_dir": "results_nonlinear_ate",
        "func": "truncated_adv",
        "lower": 0.1,
        "upper": 0.9,
        "theta_key": "ate_true",
        "legacy_subdir": True,
    },
    {
        "estimand": "ATE",
        "label": "Linear DGP + truncated adversarial",
        "result_dir": "results_ate",
        "func": "truncated_adv",
        "lower": 0.1,
        "upper": 0.9,
        "theta_key": "ate_true",
        "legacy_subdir": True,
    },
    {
        "estimand": "ATE",
        "label": "Linear DGP + logistic",
        "result_dir": "results_ate",
        "func": "logistic",
        "lower": None,
        "upper": None,
        "theta_key": "ate_true",
        "legacy_subdir": True,
    },
]


def _resolve_se(r: dict, prefix: str) -> float:
    for key in (f"{prefix}_se", f"{prefix}_sig"):
        if key in r:
            val = r[key]
            return float(val) if not hasattr(val, "item") else float(val.item())
    return float("nan")


def _vec_get(r: dict, key: str, j: int, legacy: str | None = None) -> float:
    if key in r:
        val = r[key]
    elif legacy and legacy in r:
        val = r[legacy]
    else:
        return float("nan")

    if isinstance(val, (list, tuple)):
        x = val[j]
    elif torch.is_tensor(val):
        x = val[j].item()
    else:
        return float("nan")

    if isinstance(x, float) and math.isnan(x):
        return float("nan")
    return float(x)


def extract_method_series(results: list[dict], index: int):
    """Return (theta_i, se_i) vectors across Monte Carlo iterations."""
    thetas = []
    ses = []

    for r in results:
        if index == 0:
            thetas.append(r["oracle_theta"])
            ses.append(_resolve_se(r, "oracle"))
        elif index == 1:
            thetas.append(r.get("bradic_theta", float("nan")))
            ses.append(_resolve_se(r, "bradic"))
        elif index in (2, 3, 4):
            j = index - 2
            thetas.append(_vec_get(r, "manual_drl_theta", j, legacy="manual_bradic_theta"))
            ses.append(_vec_get(r, "manual_drl_se", j, legacy="manual_bradic_sig"))
        elif index in (5, 6, 7):
            j = index - 5
            thetas.append(_vec_get(r, "manual_reg_theta", j, legacy="manual_theta"))
            ses.append(_vec_get(r, "manual_reg_se", j, legacy="manual_sig"))
        elif index in (8, 9, 10):
            j = index - 8
            thetas.append(_vec_get(r, "manual_matched_theta", j))
            ses.append(_vec_get(r, "manual_matched_se", j))
        else:
            j = index - 11
            thetas.append(_vec_get(r, "auto_theta", j, legacy="pred_theta"))
            ses.append(_vec_get(r, "auto_se", j, legacy="pred_sig"))

    return (
        torch.tensor(thetas, dtype=torch.float64),
        torch.tensor(ses, dtype=torch.float64),
    )


def truncation_label(lower, upper) -> str:
    if lower is None or upper is None:
        return "none (full support)"
    return f"[{lower:g}, {upper:g}]"


def result_subdirs(func: str, lower, upper, legacy_subdir: bool = False) -> list[str]:
    """Return candidate subdirectories to search, most specific first."""
    if legacy_subdir or func == "logistic" or lower is None:
        return [func]
    bounded = f"{func}_{lower}_{upper}"
    return [bounded, func]


def load_results(
    result_dir: str,
    N: int,
    func: str,
    lower,
    upper,
    tmax: int,
    legacy_subdir: bool = False,
):
    base = APP_DIR / result_dir / f"N{N}"
    for subdir in result_subdirs(func, lower, upper, legacy_subdir):
        results, missing = [], []
        subdir_path = base / subdir
        for t in range(tmax):
            path = subdir_path / f"result_{t}.pt"
            try:
                try:
                    results.append(torch.load(path, weights_only=False))
                except TypeError:
                    results.append(torch.load(path))
            except FileNotFoundError:
                missing.append(t)
        if results:
            return results, missing, subdir
    return [], list(range(tmax)), None


def compute_table(results, N: int, theta_key: str = "theta_true"):
    n = len(results)
    theta_true = results[0][theta_key]
    n_methods = len(METHODS)

    pred_theta = torch.zeros(n, n_methods)
    pred_sig = torch.zeros(n, n_methods)

    for k in range(n_methods):
        pred_theta[:, k], pred_sig[:, k] = extract_method_series(results, k)

    bias = torch.nanmean(pred_theta - theta_true, 0)
    rmse = torch.sqrt(torch.nanmean((pred_theta - theta_true) ** 2, 0))
    ub = pred_theta + 1.96 * pred_sig / (N ** 0.5)
    lb = pred_theta - 1.96 * pred_sig / (N ** 0.5)
    coverage = torch.nanmean(((theta_true >= lb) & (theta_true <= ub)).float(), 0)
    interval_length = torch.nanmean(ub - lb, 0)

    summary_df = pd.DataFrame(
        {
            "Method": METHODS,
            "Bias": bias.tolist(),
            "RMSE": rmse.tolist(),
            "Coverage": coverage.tolist(),
            "Interval Length": interval_length.tolist(),
        }
    )
    return summary_df, float(theta_true)


def format_block(
    cfg: dict,
    N: int,
    tmax: int,
    df: pd.DataFrame,
    theta_true: float,
    n_results: int,
    n_missing: int,
    subdir: str | None,
) -> str:
    lines = [
        "",
        "=" * 72,
        f"Estimand: {cfg['estimand']}",
        f"Configuration: {cfg['label']}",
        f"Propensity: {cfg['func']}",
        f"Truncation: {truncation_label(cfg['lower'], cfg['upper'])}",
        f"Result dir: {cfg['result_dir']}/N{N}/{subdir or '(not found)'}",
        f"N = {N}  |  tmax = {tmax}",
        f"True parameter = {theta_true:.4f}  |  Iterations: {n_results} / {tmax}",
    ]
    if n_missing:
        lines.append(f"Warning: {n_missing} missing iterations")
    lines.extend(["=" * 72, df.to_string(index=False, float_format="{:.4f}".format)])
    return "\n".join(lines)


def collect_all(configs, N_values, tmax: int, verbose: bool = True):
    text_blocks: list[str] = []
    csv_rows: list[dict] = []

    header = [
        "Simulation summary",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"N values: {N_values}",
        f"tmax (requested iterations per config): {tmax}",
        f"Configurations: {len(configs)}",
    ]
    text_blocks.append("\n".join(header))

    for cfg in configs:
        for N in N_values:
            results, missing, subdir = load_results(
                cfg["result_dir"],
                N,
                cfg["func"],
                cfg["lower"],
                cfg["upper"],
                tmax,
                legacy_subdir=cfg.get("legacy_subdir", False),
            )

            if not results:
                msg = (
                    f"\n[{cfg['estimand']} | {cfg['label']} | "
                    f"{cfg['func']} | truncation {truncation_label(cfg['lower'], cfg['upper'])} | "
                    f"N={N}] — no results found, skipping"
                )
                text_blocks.append(msg)
                if verbose:
                    print(msg)
                continue

            df, theta_true = compute_table(results, N, theta_key=cfg["theta_key"])
            block = format_block(
                cfg, N, tmax, df, theta_true, len(results), len(missing), subdir
            )
            text_blocks.append(block)
            if verbose:
                print(block)

            for _, row in df.iterrows():
                csv_rows.append(
                    {
                        "estimand": cfg["estimand"],
                        "configuration": cfg["label"],
                        "propensity": cfg["func"],
                        "truncation_lower": cfg["lower"],
                        "truncation_upper": cfg["upper"],
                        "truncation": truncation_label(cfg["lower"], cfg["upper"]),
                        "result_dir": cfg["result_dir"],
                        "result_subdir": subdir,
                        "N": N,
                        "tmax": tmax,
                        "n_iterations": len(results),
                        "n_missing": len(missing),
                        "true_parameter": theta_true,
                        "method": row["Method"],
                        "bias": row["Bias"],
                        "rmse": row["RMSE"],
                        "coverage": row["Coverage"],
                        "interval_length": row["Interval Length"],
                    }
                )

    return "\n".join(text_blocks), pd.DataFrame(csv_rows)


def main():
    parser = argparse.ArgumentParser(description="Collect simulation MC summary tables.")
    parser.add_argument(
        "--tmax",
        type=int,
        default=TMAX_DEFAULT,
        help=f"Number of iterations to load per configuration (default: {TMAX_DEFAULT})",
    )
    parser.add_argument(
        "--txt-out",
        type=Path,
        default=RESULTS_DIR / "summary_tables.txt",
        help="Path for human-readable summary tables",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=RESULTS_DIR / "summary_tables.csv",
        help="Path for long-format CSV summary",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the CSV summary",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print tables to stdout (still writes output files)",
    )
    args = parser.parse_args()

    text, csv_df = collect_all(CONFIGS, N_VALUES, args.tmax, verbose=not args.quiet)

    args.txt_out.parent.mkdir(parents=True, exist_ok=True)
    args.txt_out.write_text(text + "\n", encoding="utf-8")
    if not args.quiet:
        print(f"\nWrote text summary to: {args.txt_out}")

    if not args.no_csv:
        if csv_df.empty:
            if not args.quiet:
                print("No results collected; CSV not written.", file=sys.stderr)
        else:
            args.csv_out.parent.mkdir(parents=True, exist_ok=True)
            csv_df.to_csv(args.csv_out, index=False)
            if not args.quiet:
                print(f"Wrote CSV summary to: {args.csv_out} ({len(csv_df)} rows)")

    if args.quiet:
        print(f"Wrote text summary to: {args.txt_out}")
        if not args.no_csv and not csv_df.empty:
            print(f"Wrote CSV summary to: {args.csv_out} ({len(csv_df)} rows)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate DiD simulation .pt outputs into results/summary.csv."""

import math
import os
import sys

import pandas as pd
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
os.chdir(APP_DIR)

RESULTS_DIR = os.path.join(APP_DIR, "results")
Ns = [500, 1000, 2000]
propensity_models = ["logistic", "truncated_logistic", "truncated_adv"]
method_names = ["OLS", "Linear_old", "LASSO", "RF", "Net"]


def result_paths(N, model_name):
    file_suffix = f"final_N:{N}_{model_name}"
    return [
        os.path.join(RESULTS_DIR, f"{file_suffix}_pred_theta.pt"),
        os.path.join(RESULTS_DIR, f"{file_suffix}_pred_sig.pt"),
        os.path.join(RESULTS_DIR, f"{file_suffix}_ATT_calculations.pt"),
    ]


def missing_configs():
    missing = []
    for N in Ns:
        for model_name in propensity_models:
            if not all(os.path.exists(p) for p in result_paths(N, model_name)):
                missing.append((N, model_name))
    return missing


def all_results_present():
    return not missing_configs()


def _load_pt(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def collect_results():
    rows = []

    for N in Ns:
        for model_name in propensity_models:
            paths = result_paths(N, model_name)
            theta_path, sig_path, att_path = paths

            if not all(os.path.exists(p) for p in paths):
                print(f"Missing results for N={N}, model={model_name} — skipping")
                continue

            pred_theta = _load_pt(theta_path)
            pred_sig = _load_pt(sig_path)
            ATT = _load_pt(att_path)["ATT"]

            for k, name in enumerate(method_names):
                theta_k = pred_theta[:, k]
                sig_k = pred_sig[:, k]

                se_k = sig_k / math.sqrt(N)

                bias = torch.mean(theta_k - ATT).item()
                rmse = torch.sqrt(torch.mean((theta_k - ATT) ** 2)).item()
                ci_low = theta_k - 1.96 * se_k
                ci_high = theta_k + 1.96 * se_k
                coverage = torch.mean(((ci_low <= ATT) & (ATT <= ci_high)).float()).item()
                interval_length = torch.mean(2 * 1.96 * se_k).item()
                rows.append(
                    {
                        "N": N,
                        "model": model_name,
                        "method": name,
                        "bias": round(bias, 4),
                        "rmse": round(rmse, 4),
                        "coverage": round(coverage, 4),
                        "interval_length": round(interval_length, 4),
                    }
                )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSaved to {summary_path}")
    return df


if __name__ == "__main__":
    collect_results()

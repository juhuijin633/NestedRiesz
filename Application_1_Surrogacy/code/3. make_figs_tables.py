#!/usr/bin/env python
"""Build surrogacy application figures and summary tables from pipeline outputs."""

import os
import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CODE_DIR = Path(__file__).resolve().parent
APP_DIR = CODE_DIR.parent
os.chdir(APP_DIR)
os.makedirs("results", exist_ok=True)

river_data = pd.read_csv("data/river_data.csv")
other_data = pd.read_csv("data/others_data.csv")

pretreat_vars = (
    [f"paid{i}" for i in range(1, 5)]
    + [f"tcpp{i}" for i in range(1, 11)]
    + [f"tcprn{i}" for i in range(1, 11)]
)
covariates = [
    "xsexf", "xhsdip", "xchld05", "single",
    "grd1720", "grade16", "grd1315", "grade12", "grde911", "white",
    "hisp", "black", "age",
] + pretreat_vars


def create_dataset(quarters, application):
    if application not in ["earn", "employ"]:
        raise ValueError
    Y_observed = other_data[f"Y_{application}"]
    Y_experimental = river_data[f"Y_{application}"]
    if application == "employ":
        s_columns = (
            [f"{application}{i}" for i in range(1, quarters + 1)]
            + [f"aid{i}" for i in range(1, quarters + 1)]
            + [f"earn{i}" for i in range(1, quarters + 1)]
        )
    else:
        s_columns = (
            [f"{application}{i}" for i in range(1, quarters + 1)]
            + [f"aid{i}" for i in range(1, quarters + 1)]
        )
    D_exp = river_data["e"]
    D_obs = other_data["e"]
    return {
        "Y_observed": Y_observed,
        "Y_experimental": Y_experimental,
        "D_exp": D_exp,
        "D_obs": D_obs,
        "names_z": s_columns,
        "n_total": len(D_obs) + len(D_exp),
    }


def ci(estimates, stds, z=1.96):
    estimates = np.asarray(estimates)
    stds = np.asarray(stds)
    return estimates + z * stds, estimates - z * stds


def parse_auto_csv(path):
    """Read auto_{earn,employ}.csv; handles torch tensor strings or plain floats."""
    df = pd.read_csv(path)

    def _parse_val(s):
        s = str(s)
        m = re.search(r"tensor\(([-\d\.eE\+]+)", s)
        if m:
            return float(m.group(1))
        return float(s)

    atts = [_parse_val(s) for s in df.iloc[:, ::2].values.flatten()]
    stds = [_parse_val(s) for s in df.iloc[:, 1::2].values.flatten()]
    return atts, stds


label_map = {
    "lasso": "Manual-Lasso",
    "rf": "Manual-RF",
    "net": "Manual-NN",
}
MANUAL_ORDER = ["Manual-Lasso", "Manual-RF", "Manual-NN"]

AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 18
COLOR_BASELINE = "black"
COLOR_MANUAL = "skyblue"
COLOR_AUTO = "green"
CI_LINEWIDTH = 4


def _collect_estimates(ylabel, baseline_labels, baseline_estimates, baseline_lower, baseline_upper,
                       manual_labels, manual_means, manual_lower, manual_upper,
                       auto_labels, auto_estimates, auto_lower, auto_upper, n):
    z = 1.96
    rows = []

    def se_from_ci(est, upper):
        return (upper - est) / z

    for lbl, est, up in zip(baseline_labels, baseline_estimates, baseline_upper):
        rows.append({"group": "Baseline", "method": lbl, "estimate": est, "SE": se_from_ci(est, up)})
    for lbl, est, up in zip(manual_labels, manual_means, manual_upper):
        rows.append({"group": "Manual", "method": lbl, "estimate": est, "SE": se_from_ci(est, up)})
    for lbl, est, up in zip(auto_labels, auto_estimates, auto_upper):
        rows.append({"group": "Auto", "method": lbl, "estimate": est, "SE": se_from_ci(est, up)})

    print(f"\nEstimates and standard errors — {ylabel}:")
    print(f"{'group':<14} {'method':<14} {'estimate':>14} {'SE':>14}")
    for row in rows:
        print(f"{row['group']:<14} {row['method']:<14} {row['estimate']:>14.4f} {row['SE']:>14.4f}")

    return pd.DataFrame(rows)


def plot_application(true_effect, true_se, obs_effect, obs_se, ATT, std, n,
                     nnpiv_results, ylabel, exclude_manual_nn_from_scale=False,
                     ylim=None, yticks=None, save_fig_path=None, save_table_path=None,
                     print_table=True):
    ATT = np.asarray(ATT)
    std = np.asarray(std)

    benchmark_bounds = ci(true_effect, true_se)
    observational_bounds = ci(obs_effect, obs_se)
    auto_bounds = ci(ATT, std / np.sqrt(n))

    baseline_labels = ["Benchmark", "Observational"]
    baseline_estimates = np.array([true_effect, obs_effect])
    baseline_lower = np.array([benchmark_bounds[1], observational_bounds[1]])
    baseline_upper = np.array([benchmark_bounds[0], observational_bounds[0]])

    auto_labels = ["Auto-Lasso", "Auto-RF", "Auto-NN"]
    auto_estimates = ATT
    auto_lower = auto_bounds[1]
    auto_upper = auto_bounds[0]

    manual_results = {label_map[k]: v for k, v in nnpiv_results.items() if k in label_map}
    manual_means = np.array([manual_results[name][0] for name in MANUAL_ORDER])
    manual_bounds_arr = np.array([manual_results[name][2] for name in MANUAL_ORDER])
    manual_lower = manual_bounds_arr[:, 0]
    manual_upper = manual_bounds_arr[:, 1]
    manual_labels = MANUAL_ORDER

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, label in enumerate(baseline_labels):
        ax.errorbar(
            x=[label], y=[baseline_estimates[i]],
            yerr=[[baseline_estimates[i] - baseline_lower[i]], [baseline_upper[i] - baseline_estimates[i]]],
            fmt='o', capsize=5, markersize=14, color=COLOR_BASELINE, ecolor='gray', elinewidth=CI_LINEWIDTH,
        )

    ax.errorbar(
        x=manual_labels, y=manual_means,
        yerr=[manual_means - manual_lower, manual_upper - manual_means],
        fmt='o', capsize=5, markersize=14, color=COLOR_MANUAL, ecolor='gray', elinewidth=CI_LINEWIDTH,
    )

    ax.errorbar(
        x=auto_labels, y=auto_estimates,
        yerr=[auto_estimates - auto_lower, auto_upper - auto_estimates],
        fmt='o', capsize=5, markersize=14, color=COLOR_AUTO, ecolor='gray', elinewidth=CI_LINEWIDTH,
    )

    ax.set_xlabel('Estimators', fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='x', labelsize=TICK_LABEL_SIZE, rotation=30)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
    plt.setp(ax.get_xticklabels(), ha='right')
    ax.grid(True, linestyle='--', alpha=0.6)

    if ylim is not None:
        ax.set_ylim(*ylim)
    elif exclude_manual_nn_from_scale:
        scale_lower = np.concatenate((
            baseline_lower, auto_lower,
            manual_lower[np.array(manual_labels) != "Manual-NN"],
        ))
        scale_upper = np.concatenate((
            baseline_upper, auto_upper,
            manual_upper[np.array(manual_labels) != "Manual-NN"],
        ))
        pad = 0.05 * (scale_upper.max() - scale_lower.min())
        ax.set_ylim(0, scale_upper.max() + pad)
    else:
        scale_lower = np.concatenate((baseline_lower, auto_lower, manual_lower))
        scale_upper = np.concatenate((baseline_upper, auto_upper, manual_upper))
        pad = 0.05 * (scale_upper.max() - scale_lower.min())
        ax.set_ylim(0, scale_upper.max() + pad)

    if yticks is not None:
        ax.set_yticks(yticks)

    fig.tight_layout()
    if save_fig_path is not None:
        fig.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.show()

    table = _collect_estimates(
        ylabel, baseline_labels, baseline_estimates, baseline_lower, baseline_upper,
        manual_labels, manual_means, manual_lower, manual_upper,
        auto_labels, auto_estimates, auto_lower, auto_upper, n,
    ) if print_table else None
    if save_table_path is not None:
        table.to_csv(save_table_path, index=False)
    return table


# ---------------------------------------------------------------
# Earnings (Quarter 6)
# ---------------------------------------------------------------
ds_earn = create_dataset(6, "earn")
true_effect_earn, true_se_earn = 248.0, 31.5
obs_effect_earn, obs_se_earn = 327.1, 36.6
n = ds_earn["n_total"]

ATT_6, std_6 = parse_auto_csv("results/auto_earn.csv")

with open("results/manual_earn.pkl", "rb") as f:
    results_earn = pickle.load(f)

plot_application(
    true_effect_earn, true_se_earn, obs_effect_earn, obs_se_earn,
    ATT_6, std_6, n, results_earn,
    ylabel="Effect on Earnings",
    ylim=(0, 430),
    yticks=[0, 100, 200, 300, 400],
    save_fig_path="results/fig_earn.png",
    save_table_path="results/results_earn.csv",
)

plot_application(
    true_effect_earn, true_se_earn, obs_effect_earn, obs_se_earn,
    ATT_6, std_6, n, results_earn,
    ylabel="Effect on Earnings",
    save_fig_path="results/fig_earn_fullview.png",
    print_table=False,
)

# ---------------------------------------------------------------
# Employment (Quarter 6)
# ---------------------------------------------------------------
ds_employ = create_dataset(6, "employ")
true_effect_employ, true_se_employ = 0.063, 0.006
obs_effect_employ, obs_se_employ = 0.117, 0.010

ATT_6_employ, std_6_employ = parse_auto_csv("results/auto_employ.csv")

with open("results/manual_employ.pkl", "rb") as f:
    results_employ = pickle.load(f)

plot_application(
    true_effect_employ, true_se_employ, obs_effect_employ, obs_se_employ,
    ATT_6_employ, std_6_employ, n, results_employ,
    ylabel="Effect on Employment",
    ylim=(0, 0.14),
    yticks=[0, 0.04, 0.08, 0.12],
    save_fig_path="results/fig_employ.png",
    save_table_path="results/results_employ.csv",
)

plot_application(
    true_effect_employ, true_se_employ, obs_effect_employ, obs_se_employ,
    ATT_6_employ, std_6_employ, n, results_employ,
    ylabel="Effect on Employment",
    save_fig_path="results/fig_employ_fullview.png",
    print_table=False,
)

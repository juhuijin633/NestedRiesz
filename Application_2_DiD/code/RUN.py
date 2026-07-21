#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
APP_DIR = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
os.chdir(APP_DIR)


# In[3]:


# pip install pd  # notebook magic; not valid Python


# In[4]:


# Parameters

from utils.dynamicRieszFunctions import estimateDynamicRiesz_all
from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiDLinear import estimateDiDLinear
from utils.estimateDiD_OLS import estimateDiD_OLS
import torch
import pandas as pd
import time
from torch.distributions import Normal
from utils.dgp import DiD_DGP
import statsmodels.api as sm

from utils.application import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd

lasso_cv_settings = {
    'b_degree' : 1,
    'cv_folds' : 5,
    'random_state' : 42
}

lasso_a_settings = {
    'lambda_val' : 10,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' : "CV",
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

lasso_f_settings = {
    'lambda_val' : 10,
    'beta_start' : None,
    'D_LB' : 0,
    'D_add' : 0.2,
    'c1' :  "CV",
    'c2' : 0.1,
    'tol' : 1e-5,
    'max_iter' : 100,
    'b_degree' : 1,
    'control' : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}

rf_a_settings = {
    'poly_degree' : 0,
    'l2' : 0,
    'n_estimators' : 10,
    'criterion' : "mse",
    'max_depth' : None,
    'min_samples_split' : 10,
    'min_samples_leaf' : 5,
    'min_weight_fraction_leaf' : 0.,
    'min_var_fraction_leaf' : None,
    'min_var_leaf_on_val' : False,
    'max_features' : "auto",
    'min_impurity_decrease' : 0.,
    'max_samples' : .45,
    'min_balancedness_tol' : .45,
    'honest' : True,
    'inference' : True,
    'fit_intercept' : True,
    'subforest_size' : 2,
    'n_jobs' : 1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}
rf_f_settings = {
    'poly_degree' : 1, # 1 or 2?
    'l2' : 0,
    'n_estimators' : 10,
    'criterion' : "mse",
    'max_depth' : None,
    'min_samples_split' : 10,
    'min_samples_leaf' : 5,
    'min_weight_fraction_leaf' : 0.,
    'min_var_fraction_leaf' : None,
    'min_var_leaf_on_val' : False,
    'max_features' : "auto",
    'min_impurity_decrease' : 0.,
    'max_samples' : .45,
    'min_balancedness_tol' : .45,
    'honest' : True,
    'inference' : True,
    'fit_intercept' : True,
    'subforest_size' : 2,
    'n_jobs' : 1,
    'random_state' : None,
    'verbose' : 0,
    'warm_start' : False
}

net_a_settings = {
    'test_split' : 0,
    'learner_lr' : 1e-4,
    'learner_l2' : 1e-3,
    'learner_l1' : 0,
    'n_epochs' : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta' : 1e-3,
    'bs' : 64,
    'optimizer' : 'adam',
    'warm_start' : False,
    'logger' : None,
    'model_dir' : '.',
    'device' : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden' : 100,
    'drop_prob' : 0,
    'degree' : 2,
    'interaction_only' : True,
    'n_common' : 200,
    'act_func' : 'elu'
}

net_f_settings = {
    'test_split' : 0,
    'learner_lr' : 1e-4,
    'learner_l2' : 1e-3,
    'learner_l1' : 0,
    'n_epochs' : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta' : 1e-3,
    'bs' : 64,
    'optimizer' : 'adam',
    'warm_start' : False,
    'logger' : None,
    'model_dir' : '.',
    'device' : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden' : 100,
    'drop_prob' : 0,
    'degree' : 2,
    'interaction_only' : True,
    'n_common' : 200,
    'act_func' : 'elu'
}

# In[5]:

def dynamic_riesz_results(start_year: int,
                          effect_year: int,
                          treatment_year: int = 2004, 
                          baseline_2001: bool = True,
                          folds: int = 5,
                          seed: int = 0) -> pd.DataFrame:

    data_app = application_data()
    data = data_app.get_data(start_year, effect_year,treatment_year= treatment_year ,baseline_2001=baseline_2001)
    X1, X2 = data['X1'], data['X2']
    Y1, Y2 = data['Y1'], data['Y2']
    Z, D   = data['Z'],  data['D']

    Z_df  = pd.DataFrame(Z.numpy())
    X1_df = pd.DataFrame(X1.numpy())


    X_cov = pd.concat(
        [pd.DataFrame(D.numpy(), columns=["D"]), Z_df, X1_df],
        axis=1
    )
    Ydiff = np.asarray(Y2) - np.asarray(Y1)
    X_cov = sm.add_constant(X_cov)
    ols2 = sm.OLS(Ydiff, X_cov).fit(cov_type="HC1")
    rows = []
    rows.append({
        "method": "Static_OLS",
        "ATT":    float(ols2.params["D"]),
        "SE":     float(ols2.bse["D"])
    })

    lin_out = estimateDiD_OLS(Y1, Y2, D, Z, X1, X2,seed=seed)
    rows.append({
            "method": "Dynamic_OLS",
            "ATT":    lin_out[0],
            "SE":     lin_out[1]}
            )

    for estimator, method_id in (
        ("LASSO", "Dynamic_Auto_Lasso"),
        ("RF", "Dynamic_Auto_RF"),
        ("Net", "Dynamic_Auto_NN"),
    ):
        ATT, STD, *_ = estimateDynamicRiesz(
            Y1, Y2, D, Z, X1, X2,
            folds,
            method_a=estimator,
            method_f=estimator,
            seed=seed
        )
        se = STD / np.sqrt(len(Y1))
        rows.append({
            "method": method_id,
            "ATT":    ATT.item(),
            "SE":     se.item()
        })

    return pd.DataFrame(rows)



def plot_att_estimates_with_ci(out_df, att_manual, std_manual, manual_label, year, manual_linear_att, manual_linear_std):

    manual_df = pd.DataFrame({
        'method': [manual_label],
        'ATT':    [att_manual],
        'SE':     [std_manual]
    })
    
    static_manual_linear = pd.DataFrame({
        'method': ["Static_Manual_Linear"],
        'ATT':    [manual_linear_att],
        'SE':     [manual_linear_std]
    })

    # Combine all estimates
    plot_df = pd.concat([static_manual_linear, manual_df, out_df], ignore_index=True)

    # Explicit ordering: static block, then dynamic block
    method_order = [
        "Static_OLS",
        "Static_Manual_Linear",
        manual_label,
        "Dynamic_OLS",
        "Dynamic_Auto_Lasso",
        "Dynamic_Auto_RF",
        "Dynamic_Auto_NN",
    ]

    plot_df["method"] = pd.Categorical(plot_df["method"], categories=method_order, ordered=True)
  

    plot_df = plot_df.sort_values("method")
    # Plot
    fig, ax = plt.subplots()
    ax.errorbar(
        plot_df['method'],
        plot_df['ATT'],
        yerr=1.96 * plot_df['SE'],  
        fmt='o',
        capsize=5
    )
    ax.axhline(0, linestyle='--', color='gray')
    ax.set_ylabel('ATT Estimate')
    ax.set_title(f'ATT Estimates with 95% Confidence Intervals: {year}')
    ax.set_ylim(-0.13, 0.02)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()





# Rerunning the below will take time; the results are cached. If one would like to confirm the results, they can be rerun below.

# # 2004

# In[6]:


# Automatic Debiased Machine Learning for Covariate Shifts
out_2004 = dynamic_riesz_results(2003,2004)



# In[7]:


att_2014, std_2014 =-0.022,0.019
att_SantAnna, std_SantAnna = -0.0303     , 0.0225    


# In[8]:


out = plot_att_estimates_with_ci(out_2004, att_2014, std_2014, 'RF Static', "2004",att_SantAnna, std_SantAnna )


# In[13]:


out


# ## 2005

# In[4]:


out_2005 = dynamic_riesz_results(2003,2005)
att_2015, std_2015 = -0.049,0.020
att_SantAnna, std_SantAnna= -0.0247,     0.0217  


# In[5]:


plot_att_estimates_with_ci(out_2005, att_2015, std_2015, 'RF Static', "2005", att_SantAnna, std_SantAnna )


# ## 2006

# In[6]:


out_2006 = dynamic_riesz_results(2003,2006)
att_2006, std_2006 = -0.051, 0.020

att_SantAnna, std_SantAnna =  -0.0497,     0.0212 


# In[15]:


out_2006 = dynamic_riesz_results(2003,2006)
att_2006, std_2006 = -0.051, 0.020

att_SantAnna, std_SantAnna =  -0.0497,     0.0212 


# In[7]:


plot_att_estimates_with_ci(out_2006, att_2006, std_2006, 'RF Static', "2006", att_SantAnna, std_SantAnna )


# ## 2007

# In[8]:


out_2007 = dynamic_riesz_results(2003,2007)
att_2007, std_2007 = -0.064, 0.023
att_SantAnna, std_SantAnna= -0.0709    , 0.0232  


# In[9]:


plot_att_estimates_with_ci(out_2007, att_2007, std_2007, 'RF Static', "2007", att_SantAnna, std_SantAnna )


# In[ ]:





# All results computed here.

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiD_OLS import estimateDiD_OLS
from utils.application import *

# ---------------------------------------------------------------
# Hardcoded "Static" results (computed elsewhere; pasted in here)
# Source: Chernozhukov, Newey, Singh, Syrgkanis (2023/2024),
# "Automatic Debiased Machine Learning for Covariate Shifts" (arxiv.org/abs/2307.04527)
#   Auto-RF     <- (RF, RF-RR)             [Tables 1-2]
#   Auto-Lasso  <- (Lasso-3d, Lasso-RR-3d) [Tables 1-2]
#   Auto-NN     <- (RF, NN-RR)             [Tables 1-2]
# Manual-Linear <- Sant'Anna-Zhao DR (pasted per year)
# ---------------------------------------------------------------
STATIC_RESULTS = {
    2004: {"Auto-RF": (-0.022, 0.019), "Manual-Linear": (-0.0303, 0.0225),
           "Auto-Lasso": (-0.024, 0.020), "Auto-NN": (-0.022, 0.019)},
    2005: {"Auto-RF": (-0.049, 0.020), "Manual-Linear": (-0.0247, 0.0217),
           "Auto-Lasso": (-0.045, 0.021), "Auto-NN": (-0.046, 0.020)},
    2006: {"Auto-RF": (-0.051, 0.020), "Manual-Linear": (-0.0497, 0.0212),
           "Auto-Lasso": (-0.052, 0.021), "Auto-NN": (-0.052, 0.020)},
    2007: {"Auto-RF": (-0.064, 0.023), "Manual-Linear": (-0.0709, 0.0232),
           "Auto-Lasso": (-0.060, 0.025), "Auto-NN": (-0.064, 0.023)},
}

STATIC_METHOD_IDS = [
    "Static_OLS",
    "Static_Manual_Linear",
    "Static_Auto_RF",
    "Static_Auto_Lasso",
    "Static_Auto_NN",
]
DYNAMIC_METHOD_IDS = [
    "Dynamic_OLS",
    "Dynamic_Auto_Lasso",
    "Dynamic_Auto_RF",
    "Dynamic_Auto_NN",
]
RESULT_METHOD_ORDER = STATIC_METHOD_IDS + DYNAMIC_METHOD_IDS

STATIC_INTERNAL_TO_METHOD_ID = {
    "OLS": "Static_OLS",
    "Manual-Linear": "Static_Manual_Linear",
    "Auto-RF": "Static_Auto_RF",
    "Auto-Lasso": "Static_Auto_Lasso",
    "Auto-NN": "Static_Auto_NN",
}
DYNAMIC_INTERNAL_TO_METHOD_ID = {
    "OLS": "Dynamic_OLS",
    "Auto-Lasso": "Dynamic_Auto_Lasso",
    "Auto-RF": "Dynamic_Auto_RF",
    "Auto-NN": "Dynamic_Auto_NN",
}

PLOT_LABELS = {
    "Static_OLS": "OLS",
    "Static_Manual_Linear": "Manual-Linear",
    "Static_Auto_RF": "Auto-RF",
    "Static_Auto_Lasso": "Auto-Lasso",
    "Static_Auto_NN": "Auto-NN",
    "Dynamic_OLS": "OLS",
    "Dynamic_Auto_Lasso": "Auto-Lasso",
    "Dynamic_Auto_RF": "Auto-RF",
    "Dynamic_Auto_NN": "Auto-NN",
}

COLOR_OLS = "black"
COLOR_MANUAL_LINEAR = "skyblue"
COLOR_STATIC = "green"
COLOR_DYNAMIC = "green"

TITLE_SIZE = 24
AXIS_LABEL_SIZE = 26
TICK_LABEL_SIZE = 20
BRACKET_LABEL_SIZE = 20

RESULTS_DIR = "results"
DYNAMIC_CACHE_DIR = os.path.join(RESULTS_DIR, "dynamic_cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DYNAMIC_CACHE_DIR, exist_ok=True)

_LEGACY_DYNAMIC_METHODS = {
    "OLS YDiff on Z, X1, and D",
    "Caetano",
    "LASSO",
    "RF",
    "Net",
}

_DYNAMIC_METHOD_RENAME = {
    "OLS YDiff on Z, X1, and D": "Static_OLS",
    "Caetano": "Dynamic_OLS",
    "LASSO": "Dynamic_Auto_Lasso",
    "RF": "Dynamic_Auto_RF",
    "Net": "Dynamic_Auto_NN",
}

DYNAMIC_CACHE_METHODS = {
    "Static_OLS",
    "Dynamic_OLS",
    "Dynamic_Auto_Lasso",
    "Dynamic_Auto_RF",
    "Dynamic_Auto_NN",
}


def _normalize_dynamic_method_names(df):
    df = df.copy()
    df["method"] = df["method"].astype(str).replace(_DYNAMIC_METHOD_RENAME)
    return df


def _dynamic_cache_path(year):
    return os.path.join(DYNAMIC_CACHE_DIR, f"dynamic_{year}.csv")


def _results_path(year):
    return os.path.join(RESULTS_DIR, f"results_{year}.csv")


def _is_legacy_dynamic_df(df):
    return set(df["method"].astype(str)).issubset(_LEGACY_DYNAMIC_METHODS)


def _is_full_results_df(df):
    if "group" not in df.columns or "method" not in df.columns:
        return False
    return list(df["method"]) == RESULT_METHOD_ORDER


def dynamic_riesz_results(start_year, effect_year, treatment_year=2004,
                           baseline_2001=True, folds=5, seed=0, verbose=True):
    data_app = application_data()
    data = data_app.get_data(start_year, effect_year, treatment_year=treatment_year,
                              baseline_2001=baseline_2001)
    X1, X2 = data['X1'], data['X2']
    Y1, Y2 = data['Y1'], data['Y2']
    Z, D = data['Z'], data['D']

    Z_df = pd.DataFrame(Z.numpy())
    X1_df = pd.DataFrame(X1.numpy())

    X_cov = pd.concat([pd.DataFrame(D.numpy(), columns=["D"]), Z_df, X1_df], axis=1)
    Ydiff = np.asarray(Y2) - np.asarray(Y1)
    X_cov = sm.add_constant(X_cov)

    steps = ["Static OLS", "Dynamic OLS", "Auto-Lasso", "Auto-RF", "Auto-NN"]
    pbar = tqdm(steps, desc=f"Year {effect_year}", disable=not verbose)

    rows = []

    pbar.set_postfix_str("Static OLS")
    ols2 = sm.OLS(Ydiff, X_cov).fit(cov_type="HC1")
    rows.append({
        "method": "Static_OLS",
        "ATT": float(ols2.params["D"]),
        "SE": float(ols2.bse["D"])
    })
    pbar.update(1)

    pbar.set_postfix_str("Dynamic OLS")
    lin_out = estimateDiD_OLS(Y1, Y2, D, Z, X1, X2, seed=seed)
    rows.append({"method": "Dynamic_OLS", "ATT": lin_out[0], "SE": lin_out[1]})
    pbar.update(1)

    for estimator, method_id, step_label in (
        ("LASSO", "Dynamic_Auto_Lasso", "Auto-Lasso"),
        ("RF", "Dynamic_Auto_RF", "Auto-RF"),
        ("Net", "Dynamic_Auto_NN", "Auto-NN"),
    ):
        pbar.set_postfix_str(step_label)
        ATT, STD, *_ = estimateDynamicRiesz(
            Y1, Y2, D, Z, X1, X2, folds,
            method_a=estimator, method_f=estimator, seed=seed
        )
        se = STD / np.sqrt(len(Y1))
        rows.append({"method": method_id, "ATT": ATT.item(), "SE": se.item()})
        pbar.update(1)

    pbar.close()
    return pd.DataFrame(rows)


def _load_or_compute_dynamic(year, force_recompute=False, verbose=True):
    cache_path = _dynamic_cache_path(year)
    legacy_path = _results_path(year)

    if os.path.exists(cache_path) and not force_recompute:
        cached = _normalize_dynamic_method_names(pd.read_csv(cache_path))
        if set(cached["method"]).issubset(DYNAMIC_CACHE_METHODS):
            cached.to_csv(cache_path, index=False)
            if verbose:
                print(f"[{year}] loaded dynamic estimates from cache: {cache_path}")
            return cached

    if os.path.exists(legacy_path) and not force_recompute:
        legacy_df = pd.read_csv(legacy_path)
        if _is_legacy_dynamic_df(legacy_df):
            legacy_df = _normalize_dynamic_method_names(legacy_df)
            legacy_df.to_csv(cache_path, index=False)
            if verbose:
                print(f"[{year}] migrated legacy dynamic cache to: {cache_path}")
            return legacy_df

    out_df = dynamic_riesz_results(2003, year, verbose=verbose)
    out_df.to_csv(cache_path, index=False)
    if verbose:
        print(f"[{year}] computed and cached dynamic estimates to: {cache_path}")
    return out_df


def build_results_df(year, dynamic_df):
    dynamic_df = _normalize_dynamic_method_names(dynamic_df)
    dynamic_df["ATT"] = pd.to_numeric(dynamic_df["ATT"], errors="coerce")
    dynamic_df["SE"] = pd.to_numeric(dynamic_df["SE"], errors="coerce")

    static_block = dynamic_df[
        dynamic_df["method"] == "Static_OLS"
    ].copy()
    static_block = pd.concat([
        static_block,
        pd.DataFrame([
            {
                "method": STATIC_INTERNAL_TO_METHOD_ID[name],
                "ATT": float(vals[0]),
                "SE": float(vals[1]),
            }
            for name, vals in STATIC_RESULTS[year].items()
        ])
    ], ignore_index=True)
    static_block = static_block.set_index("method").reindex(STATIC_METHOD_IDS).reset_index()
    static_block["group"] = "Static"

    dynamic_block = dynamic_df[
        dynamic_df["method"].isin(DYNAMIC_METHOD_IDS)
    ].copy()
    dynamic_block = dynamic_block.set_index("method").reindex(DYNAMIC_METHOD_IDS).reset_index()
    dynamic_block["group"] = "Dynamic"

    results_df = pd.concat([static_block, dynamic_block], ignore_index=True)
    results_df = results_df.set_index("method").reindex(RESULT_METHOD_ORDER).reset_index()
    return results_df[["group", "method", "ATT", "SE"]]


def get_or_compute_results(year, force_recompute=False, verbose=True):
    results_path = _results_path(year)
    if os.path.exists(results_path) and not force_recompute:
        cached = pd.read_csv(results_path)
        if _is_full_results_df(cached):
            if verbose:
                print(f"[{year}] loaded from cache: {results_path}")
            return cached

    dynamic_df = _load_or_compute_dynamic(
        year, force_recompute=force_recompute, verbose=verbose
    )
    results_df = build_results_df(year, dynamic_df)
    results_df.to_csv(results_path, index=False)
    if verbose:
        print(f"[{year}] wrote full results table to: {results_path}")
    return results_df


def add_group_bracket(ax, x0, x1, label, y=-0.34, h=0.04, text_offset=0.03):
    trans = ax.get_xaxis_transform()
    ax.plot(
        [x0, x0, x1, x1],
        [y + h, y, y, y + h],
        transform=trans,
        clip_on=False,
        linewidth=2.0,
        color="black",
    )
    ax.text(
        (x0 + x1) / 2,
        y - text_offset,
        label,
        transform=trans,
        ha="center",
        va="top",
        fontsize=BRACKET_LABEL_SIZE,
        clip_on=False,
    )


def build_plot_df(year, force_recompute=False, verbose=True):
    return get_or_compute_results(
        year, force_recompute=force_recompute, verbose=verbose
    )


def _point_colors(plot_df):
    colors = []
    for _, row in plot_df.iterrows():
        method = row["method"]
        if method in ("Static_OLS", "Dynamic_OLS"):
            colors.append(COLOR_OLS)
        elif method == "Static_Manual_Linear":
            colors.append(COLOR_MANUAL_LINEAR)
        elif row["group"] == "Static":
            colors.append(COLOR_STATIC)
        else:
            colors.append(COLOR_DYNAMIC)
    return colors


def plot_att_estimates(year, force_recompute=False, verbose=True):
    plot_df = build_plot_df(year, force_recompute=force_recompute, verbose=verbose)

    if not pd.api.types.is_numeric_dtype(plot_df["ATT"]) or not pd.api.types.is_numeric_dtype(plot_df["SE"]):
        print("dtype problem detected:")
        print(plot_df.dtypes)
        print(plot_df[["method", "ATT", "SE"]])
        raise TypeError("ATT or SE column is not numeric — see printed dataframe above.")

    x = np.arange(len(plot_df))
    colors = _point_colors(plot_df)
    x_labels = [PLOT_LABELS[method] for method in plot_df["method"]]

    fig, ax = plt.subplots(figsize=(10, 8.5))

    for xi, att, se, c in zip(x, plot_df['ATT'].to_numpy(), plot_df['SE'].to_numpy(), colors):
        ax.errorbar(
            xi, att, yerr=1.96 * se,
            fmt='o', capsize=5, markersize=14,
            color=c, ecolor='gray', elinewidth=4
        )

    ax.axhline(0, linestyle='--', color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.set_xlabel('Estimators', fontsize=AXIS_LABEL_SIZE, labelpad=45)
    ax.set_ylabel('Effect on Employment', fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
    ax.set_ylim(-0.13, 0.02)
    ax.set_yticks([0, -0.05, -0.10])
    ax.grid(axis='y', color='gray', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    n_static = len(STATIC_METHOD_IDS)
    n_total = len(plot_df)
    add_group_bracket(ax, 0, n_static - 1, "Static", y=-0.40)
    add_group_bracket(ax, n_static, n_total - 1, "Dynamic", y=-0.40)

    fig.subplots_adjust(bottom=0.40, left=0.18)
    fig_path = os.path.join(RESULTS_DIR, f"fig_{year}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"[{year}] figure saved to: {fig_path}")
    plt.show()

    print(f"\nEstimates and standard errors (year {year}):")
    print(f"{'group':<8} {'method':<24} {'ATT':>10} {'SE':>10}")
    for _, row in plot_df.iterrows():
        print(
            f"{row['group']:<8} {row['method']:<24} "
            f"{row['ATT']:>10.4f} {row['SE']:>10.4f}"
        )

    return plot_df


# Final output: figures and full results CSVs for each post-treatment year.
for year in (2004, 2005, 2006, 2007):
    plot_df = plot_att_estimates(year)
    print(plot_df)


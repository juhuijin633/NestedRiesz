import os
import sys

# Limit BLAS/OpenMP threads before heavy imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, APP_DIR)
os.chdir(APP_DIR)

import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from utils.dynamicRieszBradic import estimateBradic
from utils.dynamicRieszFunctions import estimateDynamicRiesz_all
from utils.dynamicRieszManual import estimateManual_all, estimateManual_matched_all

threadpool_limits(1)

# -----------------------------------------------------------------------
# Arguments: iteration index, N, and treatment probability function name
# Usage: python run_sim.py <t> <N> <func_name>
# -----------------------------------------------------------------------
t         = int(sys.argv[1])
N         = int(sys.argv[2])
func_name = sys.argv[3]
lower     = float(sys.argv[4]) if len(sys.argv) > 4 else 0.10
upper     = float(sys.argv[5]) if len(sys.argv) > 5 else 0.90

result_subdir = func_name if func_name == 'logistic' else f'{func_name}_{lower}_{upper}'

def seed_everything(seed: int = 123) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

# -----------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------
dimX1 = 5
dimX2 = 5
folds = 4

# -----------------------------------------------------------------------
# Estimator settings
# -----------------------------------------------------------------------
lasso_a_settings = {
    'estimate'   : True,
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB'       : 0,
    'D_add'      : 0.2,
    'c1'         : 0.1,
    'c2'         : 0.1,
    'tol'        : 1e-5,
    'max_iter'   : 100,
    'b_degree'   : 1,
    'control'    : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}
lasso_f_settings = {
    'estimate'   : True,
    'lambda_val' : 0,
    'beta_start' : None,
    'D_LB'       : 0,
    'D_add'      : 0.2,
    'c1'         : 0.1,
    'c2'         : 0.1,
    'tol'        : 1e-5,
    'max_iter'   : 100,
    'b_degree'   : 1,
    'control'    : {'maxIter': 1000, 'optTol': 1e-5, 'zeroThreshold': 1e-6}
}
rf_a_settings = {
    'estimate'                : True,
    'poly_degree'             : 0,
    'l2'                      : 0,
    'n_estimators'            : 100,
    'criterion'               : "mse",
    'max_depth'               : None,
    'min_samples_split'       : 10,
    'min_samples_leaf'        : 5,
    'min_weight_fraction_leaf': 0.,
    'min_var_fraction_leaf'   : None,
    'min_var_leaf_on_val'     : False,
    'max_features'            : "auto",
    'min_impurity_decrease'   : 0.,
    'max_samples'             : .45,
    'min_balancedness_tol'    : .45,
    'honest'                  : True,
    'inference'               : True,
    'fit_intercept'           : True,
    'subforest_size'          : 4,
    'n_jobs'                  : -1,
    'random_state'            : None,
    'verbose'                 : 0,
    'warm_start'              : False
}
rf_f_settings = {
    'estimate'                : True,
    'poly_degree'             : 1,
    'l2'                      : 0,
    'n_estimators'            : 100,
    'criterion'               : "mse",
    'max_depth'               : None,
    'min_samples_split'       : 10,
    'min_samples_leaf'        : 5,
    'min_weight_fraction_leaf': 0.,
    'min_var_fraction_leaf'   : None,
    'min_var_leaf_on_val'     : False,
    'max_features'            : "auto",
    'min_impurity_decrease'   : 0.,
    'max_samples'             : .45,
    'min_balancedness_tol'    : .45,
    'honest'                  : True,
    'inference'               : True,
    'fit_intercept'           : True,
    'subforest_size'          : 4,
    'n_jobs'                  : -1,
    'random_state'            : None,
    'verbose'                 : 0,
    'warm_start'              : False
}
net_a_settings = {
    'estimate'         : True,
    'test_split'       : 0,
    'learner_lr'       : 1e-4,
    'learner_l2'       : 1e-3,
    'learner_l1'       : 0,
    'n_epochs'         : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta'  : 1e-3,
    'bs'               : 64,
    'optimizer'        : 'adam',
    'warm_start'       : False,
    'logger'           : None,
    'model_dir'        : '.',
    'device'           : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden'         : 100,
    'drop_prob'        : 0,
    'degree'           : 2,
    'interaction_only' : True,
    'n_common'         : 200,
    'act_func'         : 'elu'
}
net_f_settings = {
    'estimate'         : True,
    'test_split'       : 0,
    'learner_lr'       : 1e-4,
    'learner_l2'       : 1e-3,
    'learner_l1'       : 0,
    'n_epochs'         : 100,
    'earlystop_rounds' : 20,
    'earlystop_delta'  : 1e-3,
    'bs'               : 64,
    'optimizer'        : 'adam',
    'warm_start'       : False,
    'logger'           : None,
    'model_dir'        : '.',
    'device'           : torch.cuda.current_device() if torch.cuda.is_available() else None,
    'n_hidden'         : 100,
    'drop_prob'        : 0,
    'degree'           : 2,
    'interaction_only' : True,
    'n_common'         : 200,
    'act_func'         : 'elu'
}

# Hyperparameters for the manual estimators (to feed to NNPIV package DML_dynamic)

# these are the hyperparameters fed to sparse_l1vsl1 in NNPIV for outcome models 
lasso_hyperparams = {
    "B": 100,
    "lambda_theta": 0.1,
    "eta_theta": 0.1,
    "eta_w": 0.1,
    "n_iter": 10000,
    "tol": 0.0001,
    "sparsity": None,
}

# these are the hyperparameters fed to Ensemble IV in NNPIV for outcome models 

# these are the hyperparameters fed to AGMM in NNPIV for outcome models 
nn_hyperparams = {
            "learner_lr": 1e-4,
            "adversary_lr": 1e-4,
            "learner_l2": 1e-3,
            "adversary_l2": 1e-4,
            "adversary_norm_reg": 1e-3,
            "n_epochs": 20,
            "bs": 128,
            "train_learner_every": 1,
            "train_adversary_every": 1,
            "ols_weight": 0.0,
            "verbose": 0,
        }



seed_everything(123 + t)

# -----------------------------------------------------------------------
# Treatment probability functions (lower/upper set from CLI args above)
# -----------------------------------------------------------------------
def logistic(x):
    return torch.exp(x) / (1 + torch.exp(x))

def truncated_logistic(x):
    return lower + (upper - lower) * logistic(x)

def func_link(x):
    return (x > 0) * torch.abs(x) / (torch.abs(x) + 1) + (x < 0) / (torch.abs(x) + 1)

def truncated_link(x):
    return lower + (upper - lower) * func_link(x)

def truncated_adv(x):
    return lower + (upper - lower) * (x > 0).float()

def double_nonlinear(x):
    return lower + (upper - lower) * ((x < -0.5) + (x < 0.5)).float()

func_map = {
    'logistic'          : logistic,
    'truncated_logistic': truncated_logistic,
    'truncated_link'    : truncated_link,
    'truncated_adv'     : truncated_adv,
    'double_nonlinear'  : double_nonlinear,
}

treatment_probability_func = func_map[func_name]

# -----------------------------------------------------------------------
# DGP: coefficients
# -----------------------------------------------------------------------
beta_pi1_0    = 0
beta_pi1_S1   = torch.tensor([1, 1, 1]      + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_pi2_0_0  = 0
beta_pi2_0_S1 = torch.tensor([0.5, 0, -0.5] + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_pi2_0_S2 = torch.tensor([0.5, 0, 0.5]  + [0] * (dimX2 - 3), dtype=torch.float32).reshape(-1, 1)
beta_pi2_1_0  = 0
beta_pi2_1_S1 = torch.tensor([1, 1, 0]      + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_pi2_1_S2 = torch.tensor([1, -1, 0]     + [0] * (dimX2 - 3), dtype=torch.float32).reshape(-1, 1)
beta_g0_0     = 1
beta_g0_S1    = torch.tensor([1, 1, -1]     + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_g0_S2    = torch.tensor([1, 1, 1]      + [0] * (dimX2 - 3), dtype=torch.float32).reshape(-1, 1)
beta_g1_0     = -1
beta_g1_S1    = torch.tensor([-1, 1, -1]    + [0] * (dimX1 - 3), dtype=torch.float32).reshape(-1, 1)
beta_g1_S2    = torch.tensor([-1, -1, 1]    + [0] * (dimX2 - 3), dtype=torch.float32).reshape(-1, 1)

# -----------------------------------------------------------------------
# DGP: generate one dataset of size N
# -----------------------------------------------------------------------
S1    = torch.randn(N, dimX1)
pi1   = treatment_probability_func(beta_pi1_0 + S1 @ beta_pi1_S1).reshape(-1, 1)
A1    = torch.bernoulli(pi1).int().reshape(-1, 1)

delta1 = torch.randn(N, 1)
delta2 = torch.randn(N, dimX2)

S2   = S1 + A1 * (1 + delta1) + delta2
S2_1 = S1 + 1 + delta1 + delta2
S2_0 = S1 + delta2

pi2_0 = treatment_probability_func(beta_pi2_0_0 + S1 @ beta_pi2_0_S1 + S2_0 @ beta_pi2_0_S2)
pi2_1 = treatment_probability_func(beta_pi2_1_0 + S1 @ beta_pi2_1_S1 + S2_1 @ beta_pi2_1_S2)
pi2   = (1 - A1) * pi2_0 + A1 * pi2_1
A2    = torch.bernoulli(pi2).int()

g = (
    (A1 + A2 == 0).float() * (beta_g0_0 + S1 @ beta_g0_S1 + S2 @ beta_g0_S2) +
    (A1 * A2 == 1).float() * (beta_g1_0 + S1 @ beta_g1_S1 + S2 @ beta_g1_S2)
)
Y = g + torch.randn(N, 1)

# -----------------------------------------------------------------------
# True nuisances (for oracle)
# -----------------------------------------------------------------------
mu2_1 = beta_g1_0 + S1 @ beta_g1_S1 + S2_1 @ beta_g1_S2
mu1_1 = beta_g1_0 + S1 @ (beta_g1_S1 + beta_g1_S2) + beta_g1_S2.sum()
theta1 = beta_g1_0 + beta_g1_S2.sum()

# -----------------------------------------------------------------------
# Estimation
# -----------------------------------------------------------------------
X       = torch.hstack((S1, S2))
X_index = torch.tensor([S1.shape[1] - 1, S1.shape[1] + S2.shape[1] - 1])
D       = torch.hstack((A1, A2))
d       = torch.ones(D.shape)

time0 = time.time()

# Oracle
seed_everything(123 + t)
RR2_oracle       = A1 * A2 / (pi1 * pi2_1)
RR1_oracle       = A1 / pi1
theta_hat_oracle = RR2_oracle * Y - (RR1_oracle - 1) * mu1_1 - (RR2_oracle - RR1_oracle) * mu2_1
oracle_theta     = torch.mean(theta_hat_oracle).item()
oracle_sig       = torch.sqrt(torch.mean((theta_hat_oracle - torch.mean(theta_hat_oracle))**2)).item()

# Manual-Lasso (Bradic R script)
seed_everything(123 + t)
try:
    bradic_result = estimateBradic(Y, X, D, X_index, folds)
    bradic_theta = float(bradic_result[6])   # pred_psi1_t[0]: D-DRL E[Y(1,1)]
    bradic_se = float(torch.sqrt(torch.tensor(float(bradic_result[9]))))
except Exception as e:
    print(f"  Bradic R failed: {e}")
    bradic_theta = float("nan")
    bradic_se = float("nan")

# Manual Lasso / RF / NN via DML_dynamic (nu_score = S-DRL vs regression)
seed_everything(123 + t)
manual_drl_pt, manual_drl_se = estimateManual_all(
    Y, A1, A2, S1, S2,
    nu_score="S-DRL",
    folds=folds,
    lasso_hyperparams=lasso_hyperparams,
    nn_hyperparams=nn_hyperparams,
    random_seed=123 + t,
)

seed_everything(123 + t)
manual_reg_pt, manual_reg_se = estimateManual_all(
    Y, A1, A2, S1, S2,
    nu_score="regression",
    folds=folds,
    lasso_hyperparams=lasso_hyperparams,
    nn_hyperparams=nn_hyperparams,
    random_seed=123 + t,
)

# Manual Lasso-Lasso / RF-RF / NN-NN (matched propensity + outcome learners, S-DRL)
seed_everything(123 + t)
manual_matched_pt, manual_matched_se = estimateManual_matched_all(
    Y, A1, A2, S1, S2,
    folds=folds,
    lasso_hyperparams=lasso_hyperparams,
    nn_hyperparams=nn_hyperparams,
    random_seed=123 + t,
)

# Auto-Lasso / Auto-RF / Auto-NN (Dynamic Riesz)
seed_everything(123 + t)
auto_pt, auto_sg, _, _, _, _ = estimateDynamicRiesz_all(
    Y, X, D, d, X_index, folds,
    lasso_a_settings=lasso_a_settings,
    lasso_f_settings=lasso_f_settings,
    rf_a_settings=rf_a_settings,
    rf_f_settings=rf_f_settings,
    net_a_settings=net_a_settings,
    net_f_settings=net_f_settings
)

print(f"[N={N}, {func_name}] Iteration {t} done in {time.time() - time0:.1f}s")

# -----------------------------------------------------------------------
# Save results (results/linear/N{N}/{func_name}/result_{t}.pt)
# -----------------------------------------------------------------------
os.makedirs(f'results/linear/N{N}/{result_subdir}', exist_ok=True)
torch.save({
    "iteration": t,
    "N": N,
    "func_name": func_name,
    "lower": lower,
    "upper": upper,
    "theta_true": theta1.item(),
    # Oracle
    "oracle_theta": oracle_theta,
    "oracle_se": oracle_sig,
    # Manual-Lasso via Bradic R (estimateBradic)
    "bradic_theta": bradic_theta,
    "bradic_se": bradic_se,
    # Manual Lasso/RF/NN via DML_dynamic, nu_score='S-DRL'
    "manual_drl_theta": torch.tensor(manual_drl_pt, dtype=torch.float64),
    "manual_drl_se": torch.tensor(manual_drl_se, dtype=torch.float64),
    # Manual Lasso/RF/NN via DML_dynamic, nu_score='regression'
    "manual_reg_theta": torch.tensor(manual_reg_pt, dtype=torch.float64),
    "manual_reg_se": torch.tensor(manual_reg_se, dtype=torch.float64),
    # Manual Lasso-Lasso / RF-RF / NN-NN (matched propensity, S-DRL)
    "manual_matched_theta": torch.tensor(manual_matched_pt, dtype=torch.float64),
    "manual_matched_se": torch.tensor(manual_matched_se, dtype=torch.float64),
    # Auto Dynamic Riesz: [Lasso, RF, NN]
    "auto_theta": auto_pt,
    "auto_se": auto_sg,
}, f"results/linear/N{N}/{result_subdir}/result_{t}.pt")
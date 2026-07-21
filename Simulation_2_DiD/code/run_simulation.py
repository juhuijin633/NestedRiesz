import os
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
os.chdir(APP_DIR)
os.makedirs(os.path.join(APP_DIR, "results"), exist_ok=True)

import torch
from tqdm import tqdm

from utils.dgp import DiD_DGP
from utils.dynamicRieszFunctions import estimateDynamicRiesz
from utils.estimateDiDLinear import estimateDiDLinear
from utils.estimateDiD_OLS import estimateDiD_OLS

Ns = [500, 1000, 2000]
tmax = 500  # Monte Carlo replications
dimX = 3
dimZ = 2
seed = 123  # seed for the DGP
folds = 5
phi_Y = 1.0

lower = 0.30
upper = 0.70

lasso_a_settings = {
    "lambda_val": 0,
    "beta_start": None,
    "D_LB": 0,
    "D_add": 0.2,
    "c1": "CV",
    "c2": 0.1,
    "tol": 1e-5,
    "max_iter": 100,
    "b_degree": 1,
    "control": {"maxIter": 1000, "optTol": 1e-5, "zeroThreshold": 1e-6},
}

lasso_f_settings = {
    "lambda_val": 0,
    "beta_start": None,
    "D_LB": 0,
    "D_add": 0.2,
    "c1": "CV",
    "c2": 0.1,
    "tol": 1e-5,
    "max_iter": 100,
    "b_degree": 1,
    "control": {"maxIter": 1000, "optTol": 1e-5, "zeroThreshold": 1e-6},
}

rf_a_settings = {
    "poly_degree": 0,
    "l2": 0,
    "n_estimators": 100,
    "criterion": "mse",
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "min_weight_fraction_leaf": 0.0,
    "min_var_fraction_leaf": None,
    "min_var_leaf_on_val": False,
    "max_features": "auto",
    "min_impurity_decrease": 0.0,
    "max_samples": 0.45,
    "min_balancedness_tol": 0.45,
    "honest": True,
    "inference": True,
    "fit_intercept": True,
    "subforest_size": 4,
    "n_jobs": 1,
    "random_state": None,
    "verbose": 0,
    "warm_start": False,
}

rf_f_settings = {
    "poly_degree": 1,
    "l2": 0,
    "n_estimators": 100,
    "criterion": "mse",
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "min_weight_fraction_leaf": 0.0,
    "min_var_fraction_leaf": None,
    "min_var_leaf_on_val": False,
    "max_features": "auto",
    "min_impurity_decrease": 0.0,
    "max_samples": 0.45,
    "min_balancedness_tol": 0.45,
    "honest": True,
    "inference": True,
    "fit_intercept": True,
    "subforest_size": 4,
    "n_jobs": 1,
    "random_state": None,
    "verbose": 0,
    "warm_start": False,
}

net_a_settings = {
    "test_split": 0,
    "learner_lr": 1e-4,
    "learner_l2": 1e-3,
    "learner_l1": 0,
    "n_epochs": 100,
    "earlystop_rounds": 20,
    "earlystop_delta": 1e-3,
    "bs": 64,
    "optimizer": "adam",
    "warm_start": False,
    "logger": None,
    "model_dir": ".",
    "device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    "n_hidden": 100,
    "drop_prob": 0,
    "degree": 2,
    "interaction_only": True,
    "n_common": 200,
    "act_func": "elu",
}

net_f_settings = {
    "test_split": 0,
    "learner_lr": 1e-4,
    "learner_l2": 1e-3,
    "learner_l1": 0,
    "n_epochs": 100,
    "earlystop_rounds": 20,
    "earlystop_delta": 1e-3,
    "bs": 64,
    "optimizer": "adam",
    "warm_start": False,
    "logger": None,
    "model_dir": ".",
    "device": torch.cuda.current_device() if torch.cuda.is_available() else None,
    "n_hidden": 100,
    "drop_prob": 0,
    "degree": 2,
    "interaction_only": True,
    "n_common": 200,
    "act_func": "elu",
}


def logistic(x):
    return torch.exp(x) / (1 + torch.exp(x))


def truncated_logistic(x):
    return lower + (upper - lower) * logistic(x)


def truncated_adv(x):
    return lower + (upper - lower) * (x > 0)


propensity_models = {
    "logistic": logistic,
    "truncated_logistic": truncated_logistic,
    "truncated_adv": truncated_adv,
}

method_names = ["OLS", "Linear_old", "LASSO", "RF", "Net"]


def run_simulation(task_id=None):
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    n_idx = task_id // 3
    model_idx = task_id % 3

    N = Ns[n_idx]
    model_name, prop_func = list(propensity_models.items())[model_idx]

    from collect_results import result_paths

    for path in result_paths(N, model_name):
        if os.path.exists(path):
            os.remove(path)

    print(f"\nRunning with propensity model: {model_name}, N={N}")
    file_suffix = f"final_N:{N}_{model_name}"
    dgp = DiD_DGP(
        dim_X=dimX,
        dim_Z=dimZ,
        alpha_1=1,
        gamma_1=torch.ones(dimZ),
        gamma_2=torch.ones(dimZ),
        g=prop_func,
        beta_2=2,
        delta_3=2,
    )

    data = dgp.generate(n=N * tmax, seed=seed)
    X1, X2 = data["X1"], data["X2"]
    Y1, Y2 = data["Y1"], data["Y2"]

    Y2 = Y2 + phi_Y * (Y1 > 0).float()

    Z, D = data["Z"], data["D"]

    ATT_calculations = dgp.simulate_ATT(n=1000000)
    torch.save(ATT_calculations, f"results/{file_suffix}_ATT_calculations.pt")
    ATT = ATT_calculations["ATT"]

    pred_theta = torch.zeros(tmax, 5)
    pred_sig = torch.zeros(tmax, 5)

    for t in tqdm(range(tmax)):
        torch.manual_seed(t)
        X1_sub = X1[t * N : (t + 1) * N, :]
        X2_sub = X2[t * N : (t + 1) * N, :]
        Y1_sub = Y1[t * N : (t + 1) * N].view(-1, 1)
        Y2_sub = Y2[t * N : (t + 1) * N].view(-1, 1)
        D_sub = D[t * N : (t + 1) * N].view(-1, 1)
        Z_sub = Z[t * N : (t + 1) * N, :]

        pred_theta[t, 0], pred_sig[t, 0] = estimateDiD_OLS(
            Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, seed=t
        )

        pred_theta[t, 1], pred_sig[t, 1] = estimateDiDLinear(
            Y1_sub, Y2_sub, D_sub, Z_sub, X1_sub, X2_sub, seed=t
        )

        pred_theta[t, 2], pred_sig[t, 2], *_ = estimateDynamicRiesz(
            Y1_sub,
            Y2_sub,
            D_sub,
            Z_sub,
            X1_sub,
            X2_sub,
            folds,
            method_a="LASSO",
            lasso_a_settings=lasso_a_settings,
            method_f="LASSO",
            lasso_f_settings=lasso_f_settings,
            seed=t,
        )

        pred_theta[t, 3], pred_sig[t, 3], *_ = estimateDynamicRiesz(
            Y1_sub,
            Y2_sub,
            D_sub,
            Z_sub,
            X1_sub,
            X2_sub,
            folds,
            method_a="RF",
            rf_a_settings=rf_a_settings,
            method_f="RF",
            rf_f_settings=rf_f_settings,
            seed=t,
        )

        pred_theta[t, 4], pred_sig[t, 4], *_ = estimateDynamicRiesz(
            Y1_sub,
            Y2_sub,
            D_sub,
            Z_sub,
            X1_sub,
            X2_sub,
            folds,
            method_a="Net",
            net_a_settings=net_a_settings,
            method_f="Net",
            net_f_settings=net_f_settings,
            seed=t,
        )

    torch.save(pred_theta, f"results/{file_suffix}_pred_theta.pt")
    torch.save(pred_sig, f"results/{file_suffix}_pred_sig.pt")

    print(f"\nResults: N={N}, Model={model_name}")
    print(f"{'Method':<12} {'Bias':>8} {'RMSE':>8} {'Coverage':>10} {'Interval Length':>16}")
    print("-" * 58)
    for k, name in enumerate(method_names):
        theta_k = pred_theta[:, k]
        sig_k = pred_sig[:, k]
        # OLS returns an HC0 SE; other methods return asymptotic sigma.
        se_k = sig_k if name == "OLS" else sig_k / (N ** 0.5)
        bias = torch.mean(theta_k - ATT).item()
        rmse = torch.sqrt(torch.mean((theta_k - ATT) ** 2)).item()
        ci_low = theta_k - 1.96 * se_k
        ci_high = theta_k + 1.96 * se_k
        coverage = torch.mean(((ci_low <= ATT) & (ATT <= ci_high)).float()).item()
        interval_length = torch.mean(2 * 1.96 * se_k).item()
        print(
            f"{name:<12} {bias:>8.4f} {rmse:>8.4f} "
            f"{coverage:>10.2f} {interval_length:>16.4f}"
        )


if __name__ == "__main__":
    run_simulation()

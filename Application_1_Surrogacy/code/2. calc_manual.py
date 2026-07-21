#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
APP_DIR = CODE_DIR.parent
os.chdir(APP_DIR)

import pandas as pd
import torch
import torch.nn as nn 
import numpy as np
from time import perf_counter

# Limit BLAS/OpenMP threads to avoid oversubscription when joblib/pytorch also parallelize
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from threadpoolctl import threadpool_limits
threadpool_limits(1)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from sklearn.linear_model import LogisticRegression
from nnpiv.neuralnet.agmm import AGMM  
from nnpiv.ensemble import EnsembleIV
from nnpiv.linear import sparse_l1vsl1

from nnpiv.semiparametrics import DML_longterm

# In[58]:

def seed_everything(seed: int = 123) -> None:
    """Set seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(123)

# In[59]:

# ---------- System / resource banner ----------
def print_system_resources():
    import sys, os, platform, datetime
    try:
        import psutil
    except Exception:
        psutil = None
    from threadpoolctl import threadpool_info
    import numpy as np
    import sklearn
    try:
        import torch
    except Exception:
        torch = None

    print("="*72)
    print(f"Session started: {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"Python: {sys.version.split()[0]}  |  OS: {platform.system()} {platform.release()} ({platform.machine()})")
    # CPU / RAM
    logical = os.cpu_count()
    physical = None
    if psutil:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    line = f"CPU cores: {logical}"
    if physical:
        line += f" (physical: {physical})"
    print(line)
    if psutil:
        try:
            mem_gb = psutil.virtual_memory().total / (1024**3)
            print(f"RAM: {mem_gb:.1f} GB")
        except Exception:
            pass

    # BLAS/OpenMP threading + libraries
    env_keys = ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]
    env_summary = ", ".join([f"{k}={os.environ.get(k,'-')}" for k in env_keys])
    print(f"Thread env: {env_summary}")
    try:
        infos = threadpool_info()
        if infos:
            print("Threadpool libraries:")
            for lib in infos:
                libname = lib.get("internal_api", lib.get("prefix", ""))
                nthreads = lib.get("num_threads", "?")
                fname = lib.get("filename", "")
                print(f"  - {libname:8s}  threads={nthreads}  lib={os.path.basename(fname)}")
    except Exception:
        pass

    # Packages
    print(f"NumPy: {np.__version__}  |  scikit-learn: {sklearn.__version__}")

    # GPU (PyTorch)
    if torch is not None:
        cuda_ok = torch.cuda.is_available()
        print(f"PyTorch: {torch.__version__}  |  CUDA available: {cuda_ok}")
        if cuda_ok:
            ngpu = torch.cuda.device_count()
            print(f"GPUs: {ngpu}")
            for i in range(ngpu):
                name = torch.cuda.get_device_name(i)
                cap  = torch.cuda.get_device_capability(i)
                print(f"  - [{i}] {name}  capability={cap[0]}.{cap[1]}")
            try:
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
    else:
        print("PyTorch: not installed")

    print("="*72)
    
# Print system resources
print_system_resources()

# In[60]:

import warnings
from sklearn.exceptions import ConvergenceWarning

# ignore ConvergenceWarning coming from sklearn.linear_model._logistic
warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    module=r"sklearn\.linear_model\._logistic"
)

# In[61]:

river_data = pd.read_csv("data/river_data.csv") # experimental
other_data = pd.read_csv("data/others_data.csv") # observational

# Define pretreatment variables
pretreat_vars = (
    [f"paid{i}" for i in range(1, 5)] +         # 4 lagged values for aid
    [f"tcpp{i}" for i in range(1, 11)] +        # 10 lagged values for employment
    [f"tcprn{i}" for i in range(1, 11)]         # 10 lagged values for earnings
)

# List of used covariates
covariates = [
    "xsexf", "xhsdip", "xchld05", "single",
    "grd1720", "grade16", "grd1315", "grade12", "grde911", "white",
    "hisp", "black", "age"
] + pretreat_vars

def create_dataset(quarters, application):
    if application not in ["earn", "employ"]:
        raise ValueError
    Y_obsorved = other_data[f"Y_{application}"]
    Y_experimental = river_data[f"Y_{application}"] # not used in fitting, only to evaluate
    if application == "employ":
        s_colums = (
            [f"{application}{i}" for i in range(1, quarters+ 1)] +
            [f"aid{i}" for i in range(1, quarters+ 1)] +
            [f"earn{i}" for i in range(1, quarters + 1)]
        )
    elif application == "earn":
        s_colums = (
            [f"{application}{i}" for i in range(1, quarters+ 1)] +
            [f"aid{i}" for i in range(1, quarters + 1)]
        )

    S_obs = other_data[s_colums]
    S_exp = river_data[s_colums]
    D_exp = river_data["e"]
    D_obs = other_data["e"]
    X_obs = other_data[covariates]
    X_exp = river_data[covariates]
    Y_all = pd.concat([Y_obsorved, Y_experimental], axis=0).reset_index(drop=True)
    X_all = pd.concat([X_obs, X_exp], axis=0).reset_index(drop=True)
    S_all = pd.concat([S_obs, S_exp], axis=0).reset_index(drop=True)
    D_all = pd.concat([D_obs, D_exp], axis=0).reset_index(drop=True)
    G_all = pd.concat([
        pd.Series(np.ones(len(D_obs))),
        pd.Series(np.zeros(len(D_exp)))
    ], axis=0).reset_index(drop=True)

    Y_all_torch = torch.tensor(Y_all.values, dtype=torch.float64).view(-1, 1)
    X_all_torch = torch.tensor(X_all.values, dtype=torch.float64)
    S_all_torch = torch.tensor(S_all.values, dtype=torch.float64)
    D_all_torch = torch.tensor(D_all.values, dtype=torch.float64).view(-1, 1)
    G_all_torch = torch.tensor(G_all.values, dtype=torch.float64).view(-1, 1)
    return {"Y_obsorved": Y_obsorved,"Y_experimental": Y_experimental, "S_obs": S_obs, "S_exp":S_exp,
            "D_exp": D_exp, "D_ops":D_obs, "X_obs": X_obs, "X_exp": X_exp, "Y_all" : Y_all_torch, "X_all": X_all_torch, "S_all": S_all_torch, "D_all": D_all_torch, "G_all": G_all_torch,
            "names_x":covariates, "names_z":s_colums }

q = 6
application = os.environ.get("SURROGACY_APPLICATION", "earn")
ds = create_dataset(q, application)
Y_all = ds["Y_all"]
X_all = ds["X_all"]
S_all = ds["S_all"]
D_all = ds["D_all"]
G_all = ds["G_all"]

D_changed = D_all.clone()
Y_changed = Y_all.clone()

###################
D_changed[G_all.bool()] = 0
Y_changed[(1 - G_all).bool()] = 0


# In[62]:


# Fit on Y_changed, D_changed, X_all, G_all, S_all
# DML takes numpy arrays as inputs.

Y_changed_np = Y_changed.numpy()
D_changed_np = D_changed.numpy()
X_all_np = X_all.numpy()
G_all_np = G_all.numpy()
S_all_np = S_all.numpy()

print(Y_changed_np.shape)
print(D_changed_np.shape)
print(X_all_np.shape)
print(G_all_np.shape)
print(S_all_np.shape)

# In[63]:

results = {}

# In[66]:

# Random Forest

t0 = perf_counter()
dml_rf = DML_longterm(Y_changed_np, D_changed_np, S_all_np, G_all_np, X1 = X_all_np,
                   longterm_model='surrogacy',
                   model1=[EnsembleIV(n_iter=200, max_abs_value=2), EnsembleIV(n_iter=200, max_abs_value=2)],
                   nn_1=[False, False],
                   sample_G='G=0',
                   n_folds=5, n_rep=1, inner_n_jobs=1, CHIM=False,
                   prop_score=LogisticRegression(max_iter=2000))

t1 = perf_counter()
theta, var, ci = dml_rf.dml()
t2 = perf_counter()

print(theta, var, ci)
print(f"setup: {t1-t0:.3f}s | dml(): {t2-t1:.3f}s | total: {t2-t0:.3f}s")
results["rf"]  = (theta, var, ci)

# In[68]:
# Sparse linear model with L1 penalty

t0 = perf_counter()
mod = sparse_l1vsl1(B=100, lambda_theta=.1,
                            eta_theta=.1,
                            eta_w=.1,
                            n_iter=10000, tol=.0001, sparsity=None)

dml_sparsel1 = DML_longterm(Y_changed_np, D_changed_np, S_all_np, G_all_np, X1 = X_all_np,
                   longterm_model='surrogacy',
                   model1=[mod, mod],
                   nn_1=[False, False],
                   sample_G='G=0',
                   n_folds=5, n_rep=1, inner_n_jobs=1, CHIM=False,
                   prop_score=LogisticRegression(max_iter=2000))

t1 = perf_counter()
theta, var, ci = dml_sparsel1.dml()
t2 = perf_counter()

print(theta, var, ci)
print(f"setup: {t1-t0:.3f}s | dml(): {t2-t1:.3f}s | total: {t2-t0:.3f}s")
results["lasso"]  = (theta, var, ci)

# Neural network with adversarial training

t0 = perf_counter()
p = 0        # dropout prob
n_hidden = 100   # hidden layer width
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

fitargs_seq = {
    "n_epochs": 100, "bs": 64,
    "learner_lr": 1e-4, "adversary_lr": 1e-4,
    "learner_l2": 1e-3, "adversary_l2": 1e-3,
    "adversary_norm_reg": 1e-1,
    "device": DEVICE,
}

def _get_learner(n_t):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_t, n_hidden), nn.LeakyReLU(),
                        nn.Dropout(p=p),
                        nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),
                        nn.Dropout(p=p),
                        nn.Linear(n_hidden, 1))

def _get_adversary(n_z):
    return nn.Sequential(nn.Dropout(p=p), nn.Linear(n_z, n_hidden), nn.LeakyReLU(),
                        nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(),
                        nn.Dropout(p=p),
                        nn.Linear(n_hidden, 1))

m1_dim = S_all_np.shape[1] + X_all_np.shape[1]
m2_dim = X_all_np.shape[1] 
m1 = AGMM(_get_learner(m1_dim), _get_adversary(m1_dim))
m2 = AGMM(_get_learner(m2_dim), _get_adversary(m2_dim))

dml_agmm = DML_longterm(Y_changed_np, D_changed_np, S_all_np, G_all_np, X1 = X_all_np,
                   longterm_model='surrogacy',
                   model1=[m1, m2], sample_G='G=0',
                   nn_1=[True, True], fitargs1=[fitargs_seq, fitargs_seq],
                   n_folds=5, n_rep=1, inner_n_jobs=1, CHIM=False,
                   prop_score=LogisticRegression(max_iter=2000))

t1 = perf_counter()
theta, var, ci = dml_agmm.dml()
t2 = perf_counter()

print(theta, var, ci)
print(f"setup: {t1-t0:.3f}s | dml(): {t2-t1:.3f}s | total: {t2-t0:.3f}s")
results["net"]  = (theta, var, ci)

# In[72]:

import os
import pickle

os.makedirs("results", exist_ok=True)
with open(f"results/manual_{application}.pkl", "wb") as f:
    pickle.dump(results, f)



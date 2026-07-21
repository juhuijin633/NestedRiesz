"""Manual dynamic estimators via nnpiv DML_dynamic (Lasso / RF / NN)."""

from __future__ import annotations

import importlib
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

_NNPIV_GIT = "git+https://github.com/isaacmeza/NNPIV.git"


def _ensure_nnpiv() -> None:
    """Install or upgrade nnpiv from GitHub if DML_dynamic is unavailable."""
    try:
        importlib.import_module("nnpiv.semiparametrics")
        from nnpiv.semiparametrics import DML_dynamic  # noqa: F401
        return
    except ImportError:
        pass

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", _NNPIV_GIT],
        )
    except subprocess.CalledProcessError as exc:
        raise ImportError(
            "nnpiv with DML_dynamic is required but could not be installed automatically. "
            f"Run: pip install --upgrade {_NNPIV_GIT}"
        ) from exc

    for name in list(sys.modules):
        if name == "nnpiv" or name.startswith("nnpiv."):
            del sys.modules[name]

    try:
        from nnpiv.semiparametrics import DML_dynamic  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "nnpiv was installed but DML_dynamic is still unavailable. "
            f"Run: pip install --upgrade {_NNPIV_GIT}"
        ) from exc


_ensure_nnpiv()

from nnpiv.ensemble import EnsembleIV
from nnpiv.linear import sparse_l1vsl1
from nnpiv.neuralnet.agmm import AGMM
from nnpiv.semiparametrics import DML_dynamic


def _to_numpy(*arrays):
    out = []
    for arr in arrays:
        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        out.append(np.asarray(arr))
    return out


def _as_1d(y):
    y = np.asarray(y)
    return y.reshape(-1)


def _as_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def _dml_se(theta, var, ci):
    """DML_dynamic.dml() returns (theta, var, ci); store SE = sqrt(var)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    var = np.atleast_1d(np.asarray(var, dtype=float))
    se = np.sqrt(np.maximum(var, 0.0))
    if theta.size == 1:
        return float(theta[0]), float(se[0])
    return float(theta[0]), float(se[0])


def _get_learner(n_t: int, n_hidden: int = 32, p: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=p),
        nn.Linear(n_t, n_hidden),
        nn.LeakyReLU(),
        nn.Linear(n_hidden, 1),
    )


def _get_adversary(n_z: int, n_hidden: int = 32, p: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Dropout(p=p),
        nn.Linear(n_z, n_hidden),
        nn.LeakyReLU(),
        nn.Linear(n_hidden, 1),
    )


def _get_prop_score_model(
    method: str,
    *,
    lasso_hyperparams: dict | None = None,
    nn_hyperparams: dict | None = None,
    rf_n_iter: int = 100,
    rf_max_abs_value: float = 4,
    random_seed: int | None = None,
):
    """Classifier for DML_dynamic prop_score (must implement fit + predict_proba)."""
    rs = random_seed

    if method == "Lasso":
        # L1 logistic mirrors the lasso outcome learner (sparse_l1vsl1 is regression-only).
        return LogisticRegression(
            penalty="l1",
            solver="saga",
            max_iter=5000,
            random_state=rs,
        )
    if method == "RF":
        # sklearn RF classifier — same family as EnsembleIV's internal learner.
        return RandomForestClassifier(
            n_estimators=max(rf_n_iter, 40),
            max_depth=2,
            criterion="gini",
            bootstrap=True,
            min_samples_leaf=40,
            min_impurity_decrease=0.001,
            random_state=rs,
            n_jobs=1,
        )
    if method == "Net":
        # MLPClassifier mirrors the AGMM hidden width used for outcome models.
        hidden = 32
        if nn_hyperparams and "hidden" in nn_hyperparams:
            hidden = int(nn_hyperparams["hidden"])
        return MLPClassifier(
            hidden_layer_sizes=(hidden,),
            activation="relu",
            max_iter=500,
            random_state=rs,
        )

    raise ValueError(f"Unknown method: {method!r}. Expected 'Lasso', 'RF', or 'Net'.")


def estimateManual(
    Y,
    D1,
    D2,
    X1,
    X2,
    *,
    nu_score: str,
    method: str,
    folds: int,
    lasso_hyperparams: dict | None = None,
    nn_hyperparams: dict | None = None,
    rf_n_iter: int = 100,
    rf_max_abs_value: float = 4,
    random_seed: int | None = None,
    logistic_prop_score: bool = True,
):
    Y, D1, D2, X1, X2 = _to_numpy(Y, D1, D2, X1, X2)
    Y = _as_1d(Y)
    D1 = _as_1d(D1)
    D2 = _as_1d(D2)
    X1 = _as_2d(X1)
    X2 = _as_2d(X2)

    if logistic_prop_score:
        prop_score = LogisticRegression(max_iter=2000, random_state=random_seed)
    else:
        prop_score = _get_prop_score_model(
            method,
            lasso_hyperparams=lasso_hyperparams,
            nn_hyperparams=nn_hyperparams,
            rf_n_iter=rf_n_iter,
            rf_max_abs_value=rf_max_abs_value,
            random_seed=random_seed,
        )

    common = dict(
        estimator="MR",
        treatment_path=(1, 1),
        nu_score=nu_score,
        prop_score=prop_score,
        n_folds=folds,
        n_rep=1,
        inner_n_jobs=1,
        verbose=False,
    )
    if random_seed is not None:
        common["random_seed"] = random_seed

    if method == "Lasso":
        if lasso_hyperparams is None:
            raise ValueError("lasso_hyperparams required for method='Lasso'")
        hp = dict(lasso_hyperparams)
        dml = DML_dynamic(
            Y, D1, D2, X1, X2,
            model1=[sparse_l1vsl1(**hp), sparse_l1vsl1(**hp)],
            nn_1=[False, False],
            fitargs1=[None, None],
            **common,
        )
    elif method == "RF":
        dml = DML_dynamic(
            Y, D1, D2, X1, X2,
            model1=[
                EnsembleIV(n_iter=rf_n_iter, max_abs_value=rf_max_abs_value),
                EnsembleIV(n_iter=rf_n_iter, max_abs_value=rf_max_abs_value),
            ],
            nn_1=[False, False],
            fitargs1=[None, None],
            **common,
        )
    elif method == "Net":
        if nn_hyperparams is None:
            raise ValueError("nn_hyperparams required for method='Net'")
        hp = dict(nn_hyperparams)
        dml = DML_dynamic(
            Y, D1, D2, X1, X2,
            model1=[
                AGMM(_get_learner(X1.shape[1] + X2.shape[1]), _get_adversary(X1.shape[1] + X2.shape[1])),
                AGMM(_get_learner(X1.shape[1]), _get_adversary(X1.shape[1])),
            ],
            nn_1=[True, True],
            fitargs1=[hp, hp],
            **common,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}. Expected 'Lasso', 'RF', or 'Net'.")

    theta, var, ci = dml.dml()
    return _dml_se(theta, var, ci)


def estimateManual_all(
    Y,
    D1,
    D2,
    X1,
    X2,
    *,
    nu_score: str,
    folds: int,
    lasso_hyperparams: dict,
    nn_hyperparams: dict,
    random_seed: int | None = None,
    logistic_prop_score: bool = True,
    rf_n_iter: int = 100,
    rf_max_abs_value: float = 4,
):
    """Run Manual-Lasso, Manual-RF, Manual-NN for one nu_score setting."""
    if nu_score not in ("S-DRL", "regression"):
        raise ValueError("nu_score must be 'S-DRL' or 'regression'")

    points = []
    ses = []
    for method in ("Lasso", "RF", "Net"):
        pt, se = estimateManual(
            Y, D1, D2, X1, X2,
            nu_score=nu_score,
            method=method,
            folds=folds,
            lasso_hyperparams=lasso_hyperparams,
            nn_hyperparams=nn_hyperparams,
            rf_n_iter=rf_n_iter,
            rf_max_abs_value=rf_max_abs_value,
            random_seed=random_seed,
            logistic_prop_score=logistic_prop_score,
        )
        points.append(pt)
        ses.append(se)

    return tuple(points), tuple(ses)


def estimateManual_matched_all(
    Y,
    D1,
    D2,
    X1,
    X2,
    *,
    folds: int,
    lasso_hyperparams: dict,
    nn_hyperparams: dict,
    random_seed: int | None = None,
    rf_n_iter: int = 100,
    rf_max_abs_value: float = 4,
):
    """Manual Lasso-Lasso, RF-RF, NN-NN: matched propensity + outcome learners (S-DRL)."""
    return estimateManual_all(
        Y, D1, D2, X1, X2,
        nu_score="S-DRL",
        folds=folds,
        lasso_hyperparams=lasso_hyperparams,
        nn_hyperparams=nn_hyperparams,
        random_seed=random_seed,
        logistic_prop_score=False,
        rf_n_iter=rf_n_iter,
        rf_max_abs_value=rf_max_abs_value,
    )

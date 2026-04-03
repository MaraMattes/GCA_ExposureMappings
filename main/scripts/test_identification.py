from __future__ import annotations
import itertools
import math
import warnings
from typing import Callable, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.linalg import toeplitz
from scipy.stats import norm
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LassoCV, LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression


# -----------------------------------------------------------------------------
# 1. Shared ML wrappers (Python analogue of MLfunct and MLmean)
# -----------------------------------------------------------------------------

def _fit_model(
    y: np.ndarray,
    X: np.ndarray,
    method: str = "lasso",
    is_binary: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit a predictive model and return a prediction function.

    Parameters
    ----------
    y : 1‑d array
        Outcome to be predicted.
    X : 2‑d array
        Feature matrix (n × p).
    method : {"lasso", "randomforest", "xgboost", "svm", "ensemble", "parametric"}
        Modelling strategy analogue to the R switch.
    is_binary : bool, default False
        Treat *y* as Bernoulli and use a classifier.

    Returns
    -------
    Callable
        Function mapping a *new* 2‑d design matrix to a 1‑d prediction vector.
    """
    # Choose estimator ---------------------------------------------------------
    if method == "lasso":
        if is_binary:
            est = LogisticRegressionCV(
                Cs=10,
                penalty="l1",
                solver="saga",
                max_iter=10000,
                n_jobs=1,
            )
        else:
            est = LassoCV(cv=5, n_jobs=1)

    elif method == "randomforest":
        if is_binary:
            est = RandomForestClassifier(n_estimators=300, min_samples_leaf=5)
        else:
            est = RandomForestRegressor(n_estimators=300, min_samples_leaf=5)

    elif method == "svm":
        if is_binary:
            est = SVC(probability=is_binary) 
        else: est = SVR()

    elif method == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError("Install xgboost or pick another MLmethod.") from exc
        if is_binary:
            est = XGBClassifier(use_label_encoder=False, eval_metric="logloss") 
        else: est = XGBRegressor()

    elif method == "parametric":
        est = LinearRegression()


    else:
        raise ValueError(f"Unknown MLmethod: {method}")

    # Fit and return prediction function --------------------------------------
    est.fit(X, y)
    if is_binary and hasattr(est, "predict_proba"):
        return lambda Xnew: est.predict_proba(Xnew)[:, 1]
    return lambda Xnew: est.predict(Xnew)


def ml_mean(
    y: np.ndarray,
    X: np.ndarray,
    method: str = "lasso",
    k: int = 3,
) -> np.ndarray:
    """k‑fold cross‑fitted predictions (analogue of MLmean in R)."""
    n = y.shape[0]
    out = np.empty(n)
    is_binary = np.array_equal(np.unique(y), [0, 1])
    kfold = KFold(n_splits=k, shuffle=True)
    for train_idx, test_idx in kfold.split(np.arange(n)):
        pred_fn = _fit_model(y[train_idx], X[train_idx], method, is_binary)
        out[test_idx] = pred_fn(X[test_idx])
    return out


# -----------------------------------------------------------------------------
# 2. Influence‑function builder
# -----------------------------------------------------------------------------

def compute_psi( 
    y: np.ndarray,
    mu1: np.ndarray, 
    mu0: np.ndarray, 
    Zb: np.ndarray, 
    Phat: np.ndarray, 
    epsilon: float, 
    reweight: bool = False, 
) -> Tuple[np.ndarray, np.ndarray]:
    n, L = Zb.shape 
    delta_mu = mu1 - mu0
    r1 = y - mu1
    r0 = y - mu0

    psi = np.zeros(n)
    w_full = np.zeros(n)

    for l in range(L):
        p_l = Phat[:, l] # propensity scores for bin l
        A = np.nonzero((p_l >= epsilon) & (p_l <= 1 - epsilon))[0] 
        I1 = A[Zb[A, l] == 1] 
        I0 = A[Zb[A, l] == 0] 

        w1_raw = 1 / p_l[I1] if I1.size else np.empty(0)
        w0_raw = 1 / (1 - p_l[I0]) if I0.size else np.empty(0)

        if reweight:
            w1 = w1_raw / w1_raw.mean() if w1_raw.size else np.empty(0)
            w0 = w0_raw / w0_raw.mean() if w0_raw.size else np.empty(0)
        else:
            w1, w0 = w1_raw, w0_raw

        # Update influence function and weight record -------------------------
        if I1.size:
            psi[I1] += (
                delta_mu[I1] ** 2
                + 2 * delta_mu[I1] * (r1[I1] * w1)
                + delta_mu[I1]
                + r1[I1] * w1
            )
            w_full[I1] = w1
        if I0.size:
            psi[I0] += (
                delta_mu[I0] ** 2
                - 2 * delta_mu[I0] * (r0[I0] * w0)
                + delta_mu[I0]
                - r0[I0] * w0
            )
            w_full[I0] = w0

    return psi, w_full


# -----------------------------------------------------------------------------
# 3. Trim + reweight test with analytic standard error 
# -----------------------------------------------------------------------------

def test_trim_reweight(
    y: np.ndarray, # outcome
    d: np.ndarray, # treatment
    z: np.ndarray, # instrument
    MLmethod: str = "lasso",
    k: int = 2,
    L: int = 4,
    epsilon: float = 0.05, # trimming parameter
):
    n = y.shape[0]

    # Bin the instrument ------------------------------------------------------
    if np.unique(z).size < 10:
        # Treat as categorical with dummy coding, drop last level
        levels = np.unique(z)
        levels = levels[:-1]  # drop reference level to mimic R behaviour
        Zb = np.column_stack([(z == lev).astype(int) for lev in levels])
    else:
        cuts = np.quantile(z, np.linspace(0, 1, L + 1))
        Zp = np.digitize(z, cuts, right=True) - 1 
        Zb = np.column_stack([(Zp == l).astype(int) for l in range(L)]) 

    L_eff = Zb.shape[1]

    # Propensity predictions for each bin indicator ---------------------------
    Phat = np.column_stack(
        [
            ml_mean(Zb[:, l].astype(float), d.reshape(-1, 1), MLmethod, k)
            for l in range(L_eff)
        ]
    )

    # Outcome regressions ------------------------------------------------------
    mu1 = ml_mean(y, np.column_stack((d, z)), MLmethod, k)
    mu0 = ml_mean(y, d.reshape(-1,1), MLmethod, k) 

    # Influence function + weights -------------------------------------------
    psi, w_full = compute_psi(y, mu1, mu0, Zb, Phat, epsilon, reweight=True)
    keep = w_full > 0
    w_k = w_full[keep]
    n_eff = (w_k.sum() ** 2) / (w_k**2).sum()

    theta_hat = psi[keep].mean()
    sigma_hat = psi[keep].std(ddof=1)
    se = sigma_hat / math.sqrt(n_eff)
    p_value = 2 * norm.sf(abs(theta_hat / se))

    return {
        "teststat": theta_hat,
        "se": se,
        "pval": p_value,
        "n_eff": n_eff,
    }

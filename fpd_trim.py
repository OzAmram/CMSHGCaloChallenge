"""NLL-trimmed FPD: drop the top-Mahalanobis fraction of events per
batch under each sample's own MLE Gaussian, then run jetnet's
Chong-Forsyth 1/N extrapolation.

The default trims the most extreme 1% of events per batch, motivated by
a diagnosis of the heavy-tailed FPD feature distributions on HGCal
showers and fixed by an empirical sweep over the trim fraction.

Public API:
    trim_top_nll(X, trim_pct)  -> X with top trim_pct events dropped
    fpd_one(X, Y, trim_pct)    -> single batch FPD with trim
    fpd_trimmed_intercept(X, Y, trim_pct, **opts)
                               -> (intercept, SE, batches, perbatch)
                                  same signature/units as jetnet.evaluation.fpd
"""
from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit

import jetnet.evaluation.gen_metrics as _g

DEFAULT_TRIM_PCT = 0.01


def trim_top_nll(X: np.ndarray, trim_pct: float) -> np.ndarray:
    """Drop the top trim_pct fraction of events ranked by Mahalanobis²
    under the sample's own MLE Gaussian fit. trim_pct=0 is a no-op."""
    if trim_pct <= 0:
        return X
    mu = X.mean(0)
    S = np.cov(X, rowvar=False)
    d = S.shape[0]
    Sr = S + 1e-10 * np.trace(S) / d * np.eye(d)
    try:
        Sinv = linalg.inv(Sr)
    except linalg.LinAlgError:
        Sinv = linalg.pinv(Sr, rtol=1e-6)
    delta = X - mu
    nll = np.einsum("ij,jk,ik->i", delta, Sinv, delta)
    keep = int(round(len(X) * (1 - trim_pct)))
    return X[np.argpartition(nll, keep)[:keep]]


def fpd_one(X: np.ndarray, Y: np.ndarray, trim_pct: float) -> float:
    """Single-batch Fréchet distance with per-sample NLL trim."""
    Xt = trim_top_nll(X, trim_pct)
    Yt = trim_top_nll(Y, trim_pct)
    return _g._calculate_frechet_distance(
        Xt.mean(0), np.cov(Xt, rowvar=False),
        Yt.mean(0), np.cov(Yt, rowvar=False))


def fpd_trimmed_intercept(X: np.ndarray, Y: np.ndarray,
                          trim_pct: float = DEFAULT_TRIM_PCT, *,
                          min_samples: int = 20_000,
                          max_samples: int = 50_000,
                          num_batches: int = 20,
                          num_points: int = 10,
                          normalise: bool = True,
                          seed: int = 42,
                          return_perbatch: bool = False):
    """Drop-in replacement for jetnet.evaluation.fpd with NLL trimming.

    Returns (intercept, SE) by default. If return_perbatch=True, returns
    (intercept, SE, batches, perbatch) where perbatch.shape ==
    (num_points, num_batches).
    """
    X = np.asarray(X); Y = np.asarray(Y)
    if normalise:
        X, Y = _g._normalise_features(X, Y)
    batches = (1 / np.linspace(1.0 / min_samples, 1.0 / max_samples,
                               num_points)).astype("int32")
    rng = np.random.default_rng(seed)
    perbatch = np.zeros((len(batches), num_batches))
    for i, N in enumerate(batches):
        for k in range(num_batches):
            i1 = rng.choice(len(X), size=N)
            i2 = rng.choice(len(Y), size=N)
            perbatch[i, k] = fpd_one(X[i1], Y[i2], trim_pct)
    means = perbatch.mean(axis=1)
    params, covs = curve_fit(lambda x, a, b: a + b * x,
                              1 / batches, means,
                              bounds=([0, 0], [np.inf, np.inf]))
    intercept = float(params[0])
    se = float(np.sqrt(np.diag(covs)[0]))
    if return_perbatch:
        return intercept, se, batches, perbatch
    return intercept, se

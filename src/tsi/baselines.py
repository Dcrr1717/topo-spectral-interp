"""Non-topological baselines for activation trajectories.

* ``linear_cka``: linear centred kernel alignment (Kornblith et al. 2019),
  used as "CKA to the final-model representation".
* ``linear_probe_accuracy``: held-out accuracy of a closed-form ridge
  classifier that decodes the answer ``c`` from the activations (2-fold).
* ``activation_stats``: raw statistics (mean, variance, ...).
* ``pca_stats``: PCA effective dimension (participation ratio, entropy rank)
  and explained-variance fractions.
* Weight norms are computed from the model in ``tsi.model.weight_norms``.
"""
from __future__ import annotations

from typing import List

import numpy as np

ACT_STAT_NAMES: List[str] = ["mean", "total_var", "mean_unit_var", "mean_abs",
                             "frac_zero", "frac_dead_units", "mean_norm"]
PCA_STAT_NAMES: List[str] = ["participation_ratio", "entropy_eff_rank", "evr_top1",
                             "evr_top5", "evr_top10"]


def _center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representations of the same samples.

    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) on column-centred
    matrices (Kornblith et al. 2019, eq. 5). Returns 0 for a constant input.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples")
    Xc, Yc = _center(X), _center(Y)
    xty = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    xx = np.linalg.norm(Xc.T @ Xc, "fro")
    yy = np.linalg.norm(Yc.T @ Yc, "fro")
    if xx < 1e-24 or yy < 1e-24:
        return 0.0
    return float(xty / (xx * yy))


def linear_probe_accuracy(X: np.ndarray, y: np.ndarray, n_classes: int, alpha: float = 1.0,
                          seed: int = 0) -> float:
    """2-fold held-out accuracy of a ridge classifier (one-hot targets, closed
    form) decoding ``y`` from the standardized activations ``X``."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = [perm[: n // 2], perm[n // 2:]]
    correct = 0
    for k in range(2):
        tr, te = folds[1 - k], folds[k]
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-8] = 1.0
        Xtr = np.c_[(X[tr] - mu) / sd, np.ones(len(tr))]
        Xte = np.c_[(X[te] - mu) / sd, np.ones(len(te))]
        T = np.zeros((len(tr), n_classes))
        T[np.arange(len(tr)), y[tr]] = 1.0
        reg = alpha * np.eye(Xtr.shape[1])
        reg[-1, -1] = 0.0
        W = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ T)
        correct += int((np.argmax(Xte @ W, axis=1) == y[te]).sum())
    return correct / n


def activation_stats(X: np.ndarray) -> np.ndarray:
    """Raw activation statistics of a (samples, units) matrix."""
    X = np.asarray(X, dtype=np.float64)
    unit_var = X.var(axis=0)
    return np.array([X.mean(), X.var(), unit_var.mean(), np.abs(X).mean(),
                     float((np.abs(X) < 1e-8).mean()), float((X.std(axis=0) <= 1e-6).mean()),
                     np.linalg.norm(X, axis=1).mean()])


def pca_stats(X: np.ndarray) -> np.ndarray:
    """PCA effective dimension of a (samples, units) matrix.

    participation ratio = (sum l)^2 / sum l^2 and the entropy effective rank
    exp(H(l / sum l)) of the covariance eigenvalues l, plus the explained
    variance of the top 1 / 5 / 10 components.
    """
    Xc = _center(X)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2
    tot = lam.sum()
    if tot < 1e-24:
        return np.zeros(len(PCA_STAT_NAMES))
    p = lam / tot
    nz = p[p > 0]
    return np.array([tot ** 2 / (lam ** 2).sum(), float(np.exp(-(nz * np.log(nz)).sum())),
                     p[:1].sum(), p[:5].sum(), p[:10].sum()])

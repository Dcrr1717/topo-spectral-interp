"""Label-free change-point detection on feature trajectories.

The same detector is applied, unchanged, to every signal (descriptor views,
baselines, and test accuracy itself, which gives the reference grokking time).

* ``single_changepoint``: least-squares single change in the mean of a
  multivariate series (binary segmentation, one split). Returns the split
  index and the fraction of variance explained by a step at that split
  (``r2_step``, our "cleanness" score: 1 = a perfect step, 0 = no step).
* ``onset_index``: the first time the series has moved a fraction ``q`` of
  the way from its pre-change level to its post-change level, measured along
  the change direction (a leading-indicator / early-warning time).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def standardize_columns(Z: np.ndarray) -> np.ndarray:
    """z-score every column; constant columns are dropped."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    Z = np.nan_to_num(Z)
    sd = Z.std(axis=0)
    keep = sd > 1e-12 * (1.0 + np.abs(Z).max(axis=0))
    if not keep.any():
        return np.zeros((Z.shape[0], 0))
    return (Z[:, keep] - Z[:, keep].mean(axis=0)) / sd[keep]


@dataclass(frozen=True)
class ChangePoint:
    index: int        # first index of the right segment
    r2_step: float    # 1 - SSE(step) / SST
    onset10: int      # first index where the projection reaches 10 % of the step
    onset50: int      # first index where the projection reaches 50 % of the step


def single_changepoint(Z: np.ndarray, min_seg: int = 2) -> tuple[int, float]:
    """Least-squares single change point of the mean of the rows of ``Z``.

    ``Z`` (T, d) should already be standardized. Returns (tau, r2): rows
    ``[:tau]`` and ``[tau:]`` are the two segments.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    T = Z.shape[0]
    if T < 2 * min_seg or Z.shape[1] == 0:
        return T // 2, 0.0
    sst = float(((Z - Z.mean(axis=0)) ** 2).sum())
    if sst < 1e-24:
        return T // 2, 0.0
    cs = np.cumsum(Z, axis=0)
    cs2 = float((Z ** 2).sum())
    total = cs[-1]
    best_tau, best_sse = min_seg, np.inf
    for tau in range(min_seg, T - min_seg + 1):
        left = cs[tau - 1]
        right = total - left
        # SSE = sum z^2 - |S_left|^2 / n_left - |S_right|^2 / n_right
        sse = cs2 - float(left @ left) / tau - float(right @ right) / (T - tau)
        if sse < best_sse - 1e-12:
            best_tau, best_sse = tau, sse
    return best_tau, float(max(0.0, 1.0 - best_sse / sst))


def step_projection(Z: np.ndarray, tau: int) -> np.ndarray:
    """Project the rows of ``Z`` on the change direction (mean right - mean
    left), scaled so that the left mean maps to 0 and the right mean to 1."""
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    mu_l, mu_r = Z[:tau].mean(axis=0), Z[tau:].mean(axis=0)
    d = mu_r - mu_l
    nd = float(d @ d)
    if nd < 1e-24:
        return np.zeros(Z.shape[0])
    return np.asarray((Z - mu_l) @ d / nd)


def running_median(x: np.ndarray, k: int = 5) -> np.ndarray:
    """Centred running median with edge truncation (k odd)."""
    h = k // 2
    return np.array([np.median(x[max(0, i - h): i + h + 1]) for i in range(len(x))])


def onset_index(u: np.ndarray, q: float, k: int = 5) -> int:
    """First index at which the running median of ``u`` reaches ``q``."""
    m = running_median(np.asarray(u, dtype=np.float64), k)
    hit = np.flatnonzero(m >= q)
    return int(hit[0]) if len(hit) else len(u) - 1


def detect(Z: np.ndarray, min_seg: int = 2, smooth: int = 5) -> ChangePoint:
    """Standardize, find the change point, and the 10 % / 50 % onsets."""
    Zs = standardize_columns(Z)
    tau, r2 = single_changepoint(Zs, min_seg=min_seg)
    if Zs.shape[1] == 0:
        return ChangePoint(index=tau, r2_step=0.0, onset10=tau, onset50=tau)
    u = step_projection(Zs, tau)
    return ChangePoint(index=tau, r2_step=r2, onset10=onset_index(u, 0.1, smooth),
                       onset50=onset_index(u, 0.5, smooth))

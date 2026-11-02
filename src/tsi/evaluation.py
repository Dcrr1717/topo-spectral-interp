"""Grouped cross-validation and paired bootstrap margins.

`evaluate_grouped` and `paired_bootstrap_margin` are ported from the GEM
benchmark (gem-geophysical-descriptor ``src/eval/classify.py`` and
``src/eval/hard_benchmark.py``). The protocol is unchanged: repeated
StratifiedGroupKFold, LogReg + RandomForest, best classifier per feature set,
and a 95 % CI for accuracy(A) - accuracy(B) from a group (training-run)
bootstrap that is paired because every feature set sees the same folds.
`evaluate_grouped_regression` / `paired_bootstrap_r2` are the regression
analogue (Ridge + RandomForestRegressor, out-of-fold R^2) used for the
early-warning test.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _classifiers(seed: int, n_estimators: int) -> tuple[tuple[str, Any], ...]:
    return (("logreg", LogisticRegression(max_iter=5000)),
            ("rf", RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                          n_jobs=-1)))


def evaluate_grouped(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0,
                     n_splits: int = 4, n_repeats: int = 3,
                     n_estimators: int = 200) -> Dict[str, Any]:
    """Repeated StratifiedGroupKFold: a training run is never split between
    train and test folds. Returns per-classifier accuracy, macro-F1, the
    out-of-fold predictions of every repeat (``oof``) and the best classifier."""
    X = np.nan_to_num(np.asarray(X, dtype=float))
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_splits = min(n_splits, len(np.unique(groups)))
    out: Dict[str, Any] = {}
    for name, clf in _classifiers(seed, n_estimators):
        fold_acc, rep_acc, rep_f1, oof_all = [], [], [], []
        for r in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
            oof = np.empty_like(y)
            for tr, te in cv.split(X, y, groups):
                pipe = make_pipeline(StandardScaler(), clf)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipe.fit(X[tr], y[tr])
                oof[te] = pipe.predict(X[te])
                fold_acc.append(accuracy_score(y[te], oof[te]))
            rep_acc.append(accuracy_score(y, oof))
            rep_f1.append(f1_score(y, oof, average="macro"))
            oof_all.append(oof)
        out[name] = {"accuracy": float(np.mean(rep_acc)),
                     "accuracy_std": float(np.std(fold_acc)),
                     "accuracy_std_repeats": float(np.std(rep_acc)),
                     "f1_macro": float(np.mean(rep_f1)),
                     "oof": np.stack(oof_all)}
    out["best_classifier"] = max(("logreg", "rf"), key=lambda k: out[k]["accuracy"])
    out["best_accuracy"] = out[out["best_classifier"]]["accuracy"]
    return out


def paired_bootstrap_margin(res_a: Dict[str, Any], res_b: Dict[str, Any], y: np.ndarray,
                            groups: np.ndarray, n_boot: int = 2000,
                            seed: int = 0) -> Dict[str, Any]:
    """95% CI of accuracy(A) - accuracy(B) using each set's best classifier,
    resampling training runs (groups) with replacement; paired because both
    sets share the same folds."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    ug, ginv = np.unique(groups, return_inverse=True)
    G = len(ug)
    n_g = np.bincount(ginv, minlength=G).astype(float)

    def correct_counts(res: Dict[str, Any]) -> np.ndarray:
        oof = np.asarray(res[res["best_classifier"]]["oof"])      # (repeats, N)
        return np.stack([np.bincount(ginv, weights=(o == y).astype(float), minlength=G)
                         for o in oof])                            # (repeats, G)
    ca, cb = correct_counts(res_a), correct_counts(res_b)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)  # (B, G)
    tot = W @ n_g
    acc_a = (W @ ca.T) / tot[:, None]
    acc_b = (W @ cb.T) / tot[:, None]
    diff = (acc_a - acc_b).mean(axis=1)
    point = float((ca.sum(1) / n_g.sum()).mean() - (cb.sum(1) / n_g.sum()).mean())
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {"margin": point, "ci95": [float(lo), float(hi)],
            "p_le_0": float(max((diff <= 0).mean(), 1.0 / n_boot))}


# ---------------------------------------------------------------------------
# Regression analogue (early-warning: time to grokking)
# ---------------------------------------------------------------------------
def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float(((y - yhat) ** 2).sum()) / ss if ss > 0 else 0.0


def evaluate_grouped_regression(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                                seed: int = 0, n_estimators: int = 200) -> Dict[str, Any]:
    """Leave-one-run-out (GroupKFold with one fold per run) Ridge and RF
    regression; out-of-fold R^2 and MAE of the best regressor."""
    X = np.nan_to_num(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    cv = GroupKFold(n_splits=len(np.unique(groups)))
    out: Dict[str, Any] = {}
    models: Sequence[tuple[str, Any]] = (
        ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ("rf", RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                     min_samples_leaf=2, n_jobs=-1)))
    for name, reg in models:
        oof = np.zeros_like(y)
        for tr, te in cv.split(X, y, groups):
            pipe = make_pipeline(StandardScaler(), reg)
            pipe.fit(X[tr], y[tr])
            oof[te] = pipe.predict(X[te])
        out[name] = {"r2": _r2(y, oof), "mae": float(np.abs(y - oof).mean()), "oof": oof}
    out["best_model"] = max(("ridge", "rf"), key=lambda k: out[k]["r2"])
    out["best_r2"] = out[out["best_model"]]["r2"]
    out["best_mae"] = out[out["best_model"]]["mae"]
    return out


def paired_bootstrap_r2(res_a: Dict[str, Any], res_b: Dict[str, Any], y: np.ndarray,
                        groups: np.ndarray, n_boot: int = 2000,
                        seed: int = 0) -> Dict[str, Any]:
    """95% CI of R^2(A) - R^2(B) (best model of each), resampling runs."""
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    ug, ginv = np.unique(groups, return_inverse=True)
    G = len(ug)
    ea = (y - res_a[res_a["best_model"]]["oof"]) ** 2
    eb = (y - res_b[res_b["best_model"]]["oof"]) ** 2
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, G, size=G)
        w = np.bincount(pick, minlength=G)[ginv].astype(float)
        mu = float((w * y).sum() / w.sum())
        sst = float((w * (y - mu) ** 2).sum())
        diffs[i] = (float((w * eb).sum()) - float((w * ea).sum())) / sst if sst > 0 else 0.0
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"margin": float(res_a["best_r2"] - res_b["best_r2"]), "ci95": [float(lo), float(hi)],
            "p_le_0": float(max((diffs <= 0).mean(), 1.0 / n_boot))}


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                        seed: int = 0) -> Dict[str, Any]:
    """Paired run-level bootstrap of mean(a - b) (one value per run)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": float(d.mean()), "ci95": [float(lo), float(hi)]}

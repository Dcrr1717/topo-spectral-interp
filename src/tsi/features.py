"""Per-checkpoint, per-site features of a run: descriptor views and baselines.

For every logged checkpoint t and activation site s of one run this computes:

topological-spectral descriptor (``tsi.descriptor.authentic``)
  ``x1``   functional graph of the units (R^10)
  ``x2``   Vietoris-Rips persistent homology of the activation cloud (R^12)
  ``x3``   label-free fusion [X1/|X1|, X2/|X2|] (R^22)
baselines (``tsi.baselines``)
  ``act_stats``  raw activation statistics (R^7)
  ``pca_stats``  PCA effective dimension and explained variance (R^5)
  ``cka_final``  linear CKA to the same site of the final checkpoint (R^1)
  ``probe_acc``  2-fold linear-probe accuracy for the answer c (R^1)
  ``weight_norm`` parameter norms, the same for every site (R^5)
training curves (reference only, never used as a predictor of the labels
they define): train / held-out loss and accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .baselines import (ACT_STAT_NAMES, PCA_STAT_NAMES, activation_stats, linear_cka,
                        linear_probe_accuracy, pca_stats)
from .descriptor.authentic import X1_NAMES, X2_NAMES, desc_x3, graph_invariants, \
    x2_rips_pointcloud
from .train import WEIGHT_NORM_NAMES, RunRecord

FEATURE_NAMES: Dict[str, List[str]] = {
    "x1": X1_NAMES,
    "x2": X2_NAMES,
    "x3": [f"x1n_{n}" for n in X1_NAMES] + [f"x2n_{n}" for n in X2_NAMES],
    "act_stats": ACT_STAT_NAMES,
    "pca_stats": PCA_STAT_NAMES,
    "cka_final": ["cka_to_final"],
    "probe_acc": ["linear_probe_acc"],
    "weight_norm": [f"wnorm_{n}" for n in WEIGHT_NORM_NAMES],
}
SITE_VIEWS = ["x1", "x2", "x3", "act_stats", "pca_stats", "cka_final", "probe_acc"]


@dataclass(frozen=True)
class DescriptorConfig:
    theta: float = 0.40
    max_n: int = 80
    corr: str = "pearson"
    rips_dim: int = 15
    n_pts: int = 50
    probe_alpha: float = 1.0
    n_snapshots: int = 6  # checkpoints whose first n_pts activations are kept for figures


@dataclass
class RunFeatures:
    """Compact per-run output: no full activations."""

    meta: Dict[str, Any]
    sites: List[str]
    steps: np.ndarray
    curves: Dict[str, np.ndarray]  # train/test loss/acc, (T,)
    feats: Dict[str, np.ndarray]   # view -> (T, S, d); weight_norm -> (T, 1, 5)
    snapshots: Dict[str, np.ndarray]  # site -> (n_snap, n_pts, units)
    snapshot_steps: np.ndarray
    snapshot_labels: np.ndarray  # answer c of the first n_pts probe inputs


def run_features(rec: RunRecord, dcfg: DescriptorConfig) -> RunFeatures:
    T = len(rec.steps)
    sites = rec.sites
    S = len(sites)
    p = rec.config.p
    out: Dict[str, np.ndarray] = {v: np.zeros((T, S, len(FEATURE_NAMES[v]))) for v in SITE_VIEWS}
    for j, s in enumerate(sites):
        A = rec.acts[s].astype(np.float64)
        final = A[-1]
        for t in range(T):
            a = A[t]
            x1 = graph_invariants(a, theta=dcfg.theta, max_n=dcfg.max_n, corr=dcfg.corr)
            x2 = x2_rips_pointcloud(a, r=dcfg.rips_dim, n_pts=dcfg.n_pts)
            out["x1"][t, j] = x1
            out["x2"][t, j] = x2
            out["x3"][t, j] = desc_x3(x1, x2)
            out["act_stats"][t, j] = activation_stats(a)
            out["pca_stats"][t, j] = pca_stats(a)
            out["cka_final"][t, j, 0] = linear_cka(a, final)
            out["probe_acc"][t, j, 0] = linear_probe_accuracy(
                a, rec.probe_labels, n_classes=p, alpha=dcfg.probe_alpha,
                seed=rec.config.seed)
    out["weight_norm"] = rec.weight_norm[:, None, :].astype(np.float64)
    snap_t = snapshot_indices(rec.train_acc, rec.test_acc, dcfg.n_snapshots)
    snapshots = {s: rec.acts[s][snap_t, : dcfg.n_pts].astype(np.float32) for s in sites}
    curves = {"train_loss": rec.train_loss, "test_loss": rec.test_loss,
              "train_acc": rec.train_acc, "test_acc": rec.test_acc}
    meta = rec.config.to_dict()
    meta["probe_frac_train"] = float(rec.probe_is_train.mean())
    return RunFeatures(meta=meta, sites=sites, steps=rec.steps, curves=curves, feats=out,
                       snapshots=snapshots, snapshot_steps=rec.steps[snap_t],
                       snapshot_labels=rec.probe_labels[: dcfg.n_pts])


def snapshot_indices(train_acc: np.ndarray, test_acc: np.ndarray, n: int) -> np.ndarray:
    """Checkpoints kept for figures: init, memorization, mid-plateau, held-out
    50 % and 90 %, final (evenly spaced checkpoints fill in what is missing)."""
    T = len(train_acc)

    def first(mask: np.ndarray) -> int | None:
        hit = np.flatnonzero(mask)
        return int(hit[0]) if len(hit) else None
    i_mem, i50, i90 = first(train_acc >= 0.99), first(test_acc >= 0.5), first(test_acc >= 0.9)
    idx = [0, T - 1]
    if i_mem is not None:
        idx.append(i_mem)
        if i50 is not None and i50 > i_mem:
            idx.append((i_mem + i50) // 2)
    idx += [i for i in (i50, i90) if i is not None]
    for i in np.linspace(0, T - 1, n).round().astype(int):
        if len(set(idx)) >= n:
            break
        idx.append(int(i))
    return np.array(sorted(set(idx)))[: max(n, 2)]


def save_run_features(rf: RunFeatures, path: str) -> None:
    arrays: Dict[str, Any] = {"steps": rf.steps, "snapshot_steps": rf.snapshot_steps,
                              "sites": np.asarray(rf.sites)}
    for k, v in rf.curves.items():
        arrays[f"curve__{k}"] = v
    for k, v in rf.feats.items():
        arrays[f"feat__{k}"] = v
    for k, v in rf.snapshots.items():
        arrays[f"snap__{k}"] = v
    arrays["snapshot_labels"] = rf.snapshot_labels
    import json
    arrays["meta_json"] = np.asarray(json.dumps(rf.meta))
    np.savez_compressed(path, **arrays)


def load_run_features(path: str) -> RunFeatures:
    import json
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))
        sites = [str(s) for s in d["sites"]]
        curves = {k[len("curve__"):]: d[k] for k in d.files if k.startswith("curve__")}
        feats = {k[len("feat__"):]: d[k] for k in d.files if k.startswith("feat__")}
        snaps = {k[len("snap__"):]: d[k] for k in d.files if k.startswith("snap__")}
        return RunFeatures(meta=meta, sites=sites, steps=d["steps"], curves=curves,
                           feats=feats, snapshots=snaps, snapshot_steps=d["snapshot_steps"],
                           snapshot_labels=d["snapshot_labels"])

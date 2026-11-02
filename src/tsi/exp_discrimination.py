"""Experiment 2: representation discrimination from activation descriptors.

Samples are (run, checkpoint, site) triples from runs that grokked, taken at
checkpoints at or after memorization (train accuracy >= 0.99). Three tasks:

``layer``  which activation site (embed / resid_mid / mlp / resid_post ...)
``task``   which operation the run was trained on (add / sub / mul)
``phase``  memorized (held-out acc < 0.5) vs grokked (held-out acc >= 0.95)

Protocol, ported unchanged from the GEM benchmark: repeated
StratifiedGroupKFold, with groups = training runs (a run is never split
across folds). Classifiers are LogReg + RandomForest, and the better one is
kept per feature set. Margins carry a paired 95 % bootstrap CI over runs.
Verdicts:

* vs specified: GO iff best topology accuracy - best single baseline accuracy
  >= go_margin (and the CI excludes 0);
* vs union: the same against all baselines concatenated;
* incremental: baselines_union + topology vs baselines_union;
* headroom = 1 - acc(baselines_union); if < go_margin a GO over the union is
  arithmetically impossible.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .evaluation import evaluate_grouped, paired_bootstrap_margin
from .features import RunFeatures
from .featuresets import (BASELINES, FEATURE_SETS, INCREMENTAL, TOPOLOGY, phase_labels,
                          run_phases)


def _site_features(rf: RunFeatures, views: List[str], t: np.ndarray, j: int) -> np.ndarray:
    cols = []
    for v in views:
        arr = rf.feats[v]
        cols.append(arr[t, 0] if v == "weight_norm" else arr[t, j])
    return np.hstack(cols)


def build_samples(runs: List[RunFeatures], task: str, max_ckpts: int,
                  seed: int) -> Dict[str, Any]:
    """Feature matrices (one per feature set), labels and groups for a task."""
    rng = np.random.default_rng(seed)
    ops = sorted({r.meta["op"] for r in runs})
    X: Dict[str, List[np.ndarray]] = {k: [] for k in FEATURE_SETS}
    y: List[np.ndarray] = []
    g: List[np.ndarray] = []
    sites = runs[0].sites
    for gi, rf in enumerate(runs):
        ph = run_phases(rf)
        assert ph.i_mem is not None
        idx = np.arange(ph.i_mem, len(rf.steps))
        if task == "phase":
            lab = phase_labels(rf)
            parts = []
            for c in (0, 1):
                ic = idx[lab[idx] == c]
                if len(ic) > max_ckpts // 2:
                    ic = np.sort(rng.choice(ic, max_ckpts // 2, replace=False))
                parts.append(ic)
            idx = np.concatenate(parts)
            labels_t = lab[idx]
        else:
            if len(idx) > max_ckpts:
                idx = idx[np.linspace(0, len(idx) - 1, max_ckpts).round().astype(int)]
            labels_t = np.zeros(len(idx), dtype=int)
        if len(idx) == 0:
            continue
        for j in range(len(sites)):
            if task == "layer":
                yy = np.full(len(idx), j)
            elif task == "task":
                yy = np.full(len(idx), ops.index(rf.meta["op"]))
            else:
                yy = labels_t
            y.append(yy)
            g.append(np.full(len(idx), gi))
            for name, (_, views, _) in FEATURE_SETS.items():
                X[name].append(_site_features(rf, views, idx, j))
    class_names = {"layer": sites, "task": ops, "phase": ["memorized", "grokked"]}[task]
    return {"X": {k: np.vstack(v) for k, v in X.items()}, "y": np.concatenate(y),
            "groups": np.concatenate(g), "class_names": class_names}


def confusion(y: np.ndarray, yhat: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y, yhat), 1)
    return cm


def run_task(runs: List[RunFeatures], task: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    ev = cfg.get("discrimination", {})
    seed = int(cfg.get("seed", 0))
    go = float(ev.get("go_margin", 0.05))
    n_boot = int(ev.get("bootstrap", 2000))
    S = build_samples(runs, task, int(ev.get("max_ckpts_per_run", 40)), seed)
    y, groups = S["y"], S["groups"]
    n_cls = len(S["class_names"])
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        return {"status": "skipped: one class or one run"}
    res = {}
    for name in FEATURE_SETS:
        res[name] = evaluate_grouped(S["X"][name], y, groups, seed=seed,
                                     n_splits=int(ev.get("n_splits", 4)),
                                     n_repeats=int(ev.get("n_repeats", 3)),
                                     n_estimators=int(ev.get("n_estimators", 200)))
    acc = {k: v["best_accuracy"] for k, v in res.items()}
    bt = max(TOPOLOGY, key=lambda k: acc[k])
    bb = max(BASELINES, key=lambda k: acc[k])
    bi = max(INCREMENTAL, key=lambda k: acc[k])

    def cmp(a: str, b: str) -> Dict[str, Any]:
        c = paired_bootstrap_margin(res[a], res[b], y, groups, n_boot, seed)
        return {"a": a, "b": b, "accuracy_a": acc[a], "accuracy_b": acc[b],
                "margin": acc[a] - acc[b], "ci95": c["ci95"]}

    def rule(c: Dict[str, Any]) -> str:
        return "GO" if c["margin"] >= go and c["ci95"][0] > 0 else "NO-GO"

    c_b, c_u, c_i = cmp(bt, bb), cmp(bt, "baselines_union"), cmp(bi, "baselines_union")
    cms = {}
    for k in (bt, bb, "baselines_union"):
        oof = res[k][res[k]["best_classifier"]]["oof"][0]
        cms[k] = confusion(y, oof, n_cls).tolist()
    chance = float(np.bincount(y).max() / len(y))
    return {"n": int(len(y)), "n_runs": int(len(np.unique(groups))), "n_classes": n_cls,
            "class_names": S["class_names"], "majority_rate": chance,
            "table": {k: {"accuracy": v["best_accuracy"], "classifier": v["best_classifier"],
                          "logreg": v["logreg"]["accuracy"], "rf": v["rf"]["accuracy"],
                          "f1_macro": v[v["best_classifier"]]["f1_macro"],
                          "accuracy_std_repeats": v[v["best_classifier"]]["accuracy_std_repeats"]}
                      for k, v in res.items()},
            "best_topology": bt, "best_baseline": bb, "best_incremental": bi,
            "headroom": 1.0 - acc["baselines_union"],
            "go_arithmetically_possible_vs_union": bool(1.0 - acc["baselines_union"] >= go),
            "topology_vs_best_baseline": c_b, "topology_vs_union": c_u, "incremental": c_i,
            "verdict_vs_best_baseline": rule(c_b), "verdict_vs_union": rule(c_u),
            "verdict_incremental": rule(c_i), "confusion": cms}


def run_discrimination(runs: List[RunFeatures], cfg: Dict[str, Any]) -> Dict[str, Any]:
    grok = [r for r in runs if r.meta["kind"] == "grok" and run_phases(r).grokked]
    out: Dict[str, Any] = {"n_grok_runs": len(grok),
                           "go_margin": float(cfg.get("discrimination", {}).get("go_margin",
                                                                                0.05))}
    if len(grok) < 2:
        out["status"] = "skipped: fewer than 2 grokking runs"
        return out
    tasks = cfg.get("discrimination", {}).get("tasks", ["layer", "task", "phase"])
    out["tasks"] = {}
    for t in tasks:
        if t == "task" and len({r.meta["op"] for r in grok}) < 2:
            out["tasks"][t] = {"status": "skipped: a single operation"}
            continue
        print(f"[exp2] task {t}", flush=True)
        out["tasks"][t] = run_task(grok, t, cfg)
    verdicts = [v.get(k) for v in out["tasks"].values()
                for k in ("verdict_vs_best_baseline", "verdict_incremental")]
    out["overall"] = "GO" if "GO" in verdicts else "NO-GO"
    return out

"""Experiment 1: does the descriptor detect the grokking transition?

Per grokking run the analysis window is [t_mem, t_end], where t_mem is the
first checkpoint with train accuracy >= 0.99. Only train accuracy is needed to
define it. The reference grokking time t_ref is the change point that the
SAME label-free detector (``tsi.changepoint.detect``) finds in held-out
accuracy. Every signal (descriptor views and baselines, one trajectory per
run) then goes through the identical detector.

Questions and pre-registered Go/No-Go rules (all margins on run-level means,
95 % CI from a paired bootstrap over runs; GO needs margin >= go_margin AND a
CI that excludes 0):

1. timing       |t_cp - t_ref| / window. Does the descriptor place the
                transition more accurately than the best baseline?
2. cleanness    R^2 of a single step. Is the descriptor trajectory more
                step-like at the transition than the best baseline's?
3. early warning  supervised. On memorization-plateau checkpoints
                (t_mem <= t < first held-out acc >= 0.5) predict the steps
                remaining until grokking (held-out acc >= 0.9). The test is
                leave-one-run-out and compares out-of-fold R^2.
4. specificity  descriptive. The same detector runs on control runs that
                memorize but never grok. A specific signal shows no clean step
                there.

For every criterion the incremental test is (best baseline + topological view)
vs the best baseline. Does topology add signal to the strongest baseline?
That row and the one against the best baseline set the overall verdict.
``baselines_union + topology`` vs ``baselines_union`` is reported as a
secondary row only. A concatenation of heterogeneous baselines is a WORSE
change-point detector than its best member, so beating it does not show that
topology adds anything.

Descriptive extras: signed lag, 10 % / 50 % onset lead vs held-out accuracy
(negative = the signal moves first), per-site tables.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .changepoint import detect
from .evaluation import (bootstrap_mean_diff, evaluate_grouped_regression,
                         paired_bootstrap_r2)
from .features import MODEL_LEVEL, RunFeatures
from .featuresets import (BASELINES, FEATURE_SETS, INCREMENTAL, TOPOLOGY, run_phases,
                          signal_matrix)

REFERENCE_SIGNALS = ["test_loss", "train_loss"]


def _window_signals(rf: RunFeatures, i0: int) -> Dict[str, Dict[str, np.ndarray]]:
    """name -> site -> (T_window, d) trajectories."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for name, (_, views, _) in FEATURE_SETS.items():
        out[name] = {"all": signal_matrix(rf, views, "all")[i0:]}
        if not all(v in MODEL_LEVEL for v in views):
            for s in rf.sites:
                out[name][s] = signal_matrix(rf, views, s)[i0:]
    for c in REFERENCE_SIGNALS:
        out[c] = {"all": np.log10(rf.curves[c][i0:] + 1e-12)[:, None]}
    for b in BASELINES:  # every single baseline + every topological view (all sites)
        for t in TOPOLOGY:
            views = FEATURE_SETS[b][1] + FEATURE_SETS[t][1]
            out[f"{b}+{t}"] = {"all": signal_matrix(rf, views, "all")[i0:]}
    return out


def analyse_run(rf: RunFeatures, min_seg: int, smooth: int) -> Dict[str, Any]:
    ph = run_phases(rf)
    assert ph.i_mem is not None
    i0 = ph.i_mem
    st = rf.steps[i0:].astype(float)
    span = max(st[-1] - st[0], 1.0)
    ref = detect(rf.curves["test_acc"][i0:], min_seg=min_seg, smooth=smooth)
    t_ref = st[ref.index]
    rec: Dict[str, Any] = {
        "run_id": rf.meta["run_id"], "op": rf.meta["op"], "kind": rf.meta["kind"],
        "t_mem": float(st[0]), "t_end": float(st[-1]), "window": float(span),
        "t_ref_cp": float(t_ref), "ref_r2": ref.r2_step,
        "t_ref_onset10": float(st[ref.onset10]), "t_ref_onset50": float(st[ref.onset50]),
        "t_grok50": float(rf.steps[ph.i_grok50]) if ph.i_grok50 is not None else None,
        "t_grok90": float(rf.steps[ph.i_grok90]) if ph.i_grok90 is not None else None,
        "signals": {}}
    for name, by_site in _window_signals(rf, i0).items():
        rec["signals"][name] = {}
        for site, Z in by_site.items():
            cp = detect(Z, min_seg=min_seg, smooth=smooth)
            rec["signals"][name][site] = {
                "t_cp": float(st[cp.index]), "nlag": float((st[cp.index] - t_ref) / span),
                "r2_step": cp.r2_step,
                "onset10_nlead": float((st[cp.onset10] - st[ref.onset10]) / span),
                "onset50_nlead": float((st[cp.onset50] - st[ref.onset50]) / span)}
    return rec


def _summary(recs: List[Dict[str, Any]], name: str, site: str) -> Dict[str, Any]:
    rows = [r["signals"][name][site] for r in recs if site in r["signals"].get(name, {})]
    nlag = np.array([x["nlag"] for x in rows])
    return {"n_runs": len(rows),
            "mean_abs_nlag": float(np.abs(nlag).mean()),
            "median_abs_nlag": float(np.median(np.abs(nlag))),
            "mean_nlag": float(nlag.mean()),
            "frac_within_0.05": float((np.abs(nlag) <= 0.05).mean()),
            "mean_r2_step": float(np.mean([x["r2_step"] for x in rows])),
            "mean_onset10_nlead": float(np.mean([x["onset10_nlead"] for x in rows])),
            "mean_onset50_nlead": float(np.mean([x["onset50_nlead"] for x in rows]))}


def _per_run(recs: List[Dict[str, Any]], name: str, key: str,
             site: str = "all") -> np.ndarray:
    v = np.array([r["signals"][name][site][key] for r in recs], dtype=float)
    return np.abs(v) if key == "nlag" else v


def _compare(recs: List[Dict[str, Any]], a: str, b: str, key: str, lower_is_better: bool,
             n_boot: int, seed: int) -> Dict[str, Any]:
    va, vb = _per_run(recs, a, key), _per_run(recs, b, key)
    # positive margin = A better than B
    bs = bootstrap_mean_diff(vb, va, n_boot, seed) if lower_is_better else \
        bootstrap_mean_diff(va, vb, n_boot, seed)
    return {"a": a, "b": b, "margin": bs["mean"], "ci95": bs["ci95"]}


def _rule(cmp: Dict[str, Any], go_margin: float) -> str:
    return "GO" if cmp["margin"] >= go_margin and cmp["ci95"][0] > 0 else "NO-GO"


def early_warning(runs: List[RunFeatures], n_boot: int, seed: int,
                  n_estimators: int) -> Dict[str, Any]:
    """Regress steps-to-grokking from plateau checkpoints, leave-one-run-out."""
    X: Dict[str, List[np.ndarray]] = {k: [] for k in FEATURE_SETS}
    X["epoch"] = []
    y_all, g_all = [], []
    for gi, rf in enumerate(runs):
        ph = run_phases(rf)
        if not ph.grokked or ph.i_grok50 is None or ph.i_mem is None:
            continue
        idx = np.arange(ph.i_mem, ph.i_grok50)
        if len(idx) < 2:
            continue
        assert ph.i_grok90 is not None
        y_all.append(rf.steps[ph.i_grok90] - rf.steps[idx])
        g_all.append(np.full(len(idx), gi))
        X["epoch"].append(rf.steps[idx][:, None].astype(float))
        for name, (_, views, _) in FEATURE_SETS.items():
            X[name].append(signal_matrix(rf, views, "all")[idx])
    if len(g_all) < 3:
        return {"status": "skipped: fewer than 3 grokking runs with a plateau"}
    y = np.concatenate(y_all).astype(float)
    groups = np.concatenate(g_all)
    Xs = {k: np.vstack(v) for k, v in X.items()}
    Xs["baselines_union+epoch"] = np.hstack([Xs["baselines_union"], Xs["epoch"]])
    Xs["baselines_union+epoch+x1x2"] = np.hstack([Xs["baselines_union+epoch"],
                                                  Xs["x1x2"]])
    Xs["baselines_union+epoch+x3"] = np.hstack([Xs["baselines_union+epoch"], Xs["x3"]])
    res = {k: evaluate_grouped_regression(v, y, groups, seed=seed, n_estimators=n_estimators)
           for k, v in Xs.items()}
    base_pool = BASELINES + ["epoch"]
    bt = max(TOPOLOGY, key=lambda k: res[k]["best_r2"])
    bb = max(base_pool, key=lambda k: res[k]["best_r2"])
    for t in TOPOLOGY:
        Xs[f"{bb}+{t}"] = np.hstack([Xs[bb], Xs[t]])
        res[f"{bb}+{t}"] = evaluate_grouped_regression(Xs[f"{bb}+{t}"], y, groups, seed=seed,
                                                       n_estimators=n_estimators)
    b_t = max([f"{bb}+{t}" for t in TOPOLOGY], key=lambda k: res[k]["best_r2"])
    cmp_b = paired_bootstrap_r2(res[bt], res[bb], y, groups, n_boot, seed)
    u, u_t = "baselines_union+epoch", max(["baselines_union+epoch+x1x2",
                                            "baselines_union+epoch+x3"],
                                           key=lambda k: res[k]["best_r2"])
    cmp_u = paired_bootstrap_r2(res[bt], res[u], y, groups, n_boot, seed)
    cmp_iu = paired_bootstrap_r2(res[u_t], res[u], y, groups, n_boot, seed)
    cmp_i = paired_bootstrap_r2(res[b_t], res[bb], y, groups, n_boot, seed)
    table = {k: {"r2": v["best_r2"], "mae_steps": v["best_mae"], "model": v["best_model"]}
             for k, v in res.items()}
    return {"n_samples": int(len(y)), "n_runs": int(len(np.unique(groups))),
            "target": "steps until held-out accuracy >= 0.9",
            "target_mean": float(y.mean()), "target_std": float(y.std()),
            "table": table, "best_topology": bt, "best_baseline": bb,
            "topology_vs_best_baseline": {"a": bt, "b": bb, **cmp_b},
            "topology_vs_union": {"a": bt, "b": u, **cmp_u},
            "incremental": {"a": b_t, "b": bb, **cmp_i},
            "incremental_union": {"a": u_t, "b": u, **cmp_iu},
            "oof_example": {"y": y.tolist(), "groups": groups.tolist(),
                            bt: res[bt][res[bt]["best_model"]]["oof"].tolist(),
                            bb: res[bb][res[bb]["best_model"]]["oof"].tolist()}}


def run_detection(runs: List[RunFeatures], cfg: Dict[str, Any]) -> Dict[str, Any]:
    ev = cfg.get("detection", {})
    min_seg, smooth = int(ev.get("min_seg", 2)), int(ev.get("smooth", 5))
    n_boot, seed = int(ev.get("bootstrap", 2000)), int(cfg.get("seed", 0))
    go = float(ev.get("go_margin", 0.05))
    grok = [r for r in runs if r.meta["kind"] == "grok" and run_phases(r).grokked]
    ctrl = [r for r in runs if r.meta["kind"] == "control" and run_phases(r).i_mem is not None]
    out: Dict[str, Any] = {
        "n_grok_runs": len(grok), "n_control_runs": len(ctrl),
        "not_grokked": [r.meta["run_id"] for r in runs
                        if r.meta["kind"] == "grok" and not run_phases(r).grokked],
        "go_margin": go}
    if len(grok) < 2:
        out["status"] = "skipped: fewer than 2 grokking runs"
        return out
    recs = [analyse_run(r, min_seg, smooth) for r in grok]
    out["runs"] = recs
    names = list(FEATURE_SETS) + REFERENCE_SIGNALS
    sites = ["all"] + grok[0].sites
    out["summary"] = {n: {s: _summary(recs, n, s) for s in sites
                          if s in recs[0]["signals"][n]} for n in names}
    combos = [f"{b}+{t}" for b in BASELINES for t in TOPOLOGY]
    out["combo_summary"] = {c: _summary(recs, c, "all") for c in combos}
    summ = {**out["summary"], **{c: {"all": v} for c, v in out["combo_summary"].items()}}
    # trivial "prior" detector: leave-one-run-out mean position of t_ref
    pos = np.array([(r["t_ref_cp"] - r["t_mem"]) / r["window"] for r in recs])
    prior = np.array([abs(np.delete(pos, i).mean() - pos[i]) for i in range(len(pos))])
    out["prior_detector"] = {"mean_abs_nlag": float(prior.mean()),
                             "median_abs_nlag": float(np.median(prior))}

    def best(pool: List[str], key: str, lower: bool) -> str:
        vals = {k: summ[k]["all"][key] for k in pool}
        return min(vals, key=lambda k: vals[k]) if lower else max(vals, key=lambda k: vals[k])

    verdicts: Dict[str, Any] = {}
    for crit, key, lower in (("timing", "nlag", True), ("cleanness", "r2_step", False)):
        skey = "mean_abs_nlag" if key == "nlag" else "mean_r2_step"
        bt, bb = best(TOPOLOGY, skey, lower), best(BASELINES, skey, lower)
        c_b = _compare(recs, bt, bb, key, lower, n_boot, seed)
        c_u = _compare(recs, bt, "baselines_union", key, lower, n_boot, seed)
        # primary incremental test: does topology add to the BEST baseline?
        # (a concatenated union is a worse change-point detector than its best
        # member, so "union + topology vs union" can be GO without topology
        # adding anything over that member; it is reported as a secondary row)
        bi = best([f"{bb}+{t}" for t in TOPOLOGY], skey, lower)
        c_i = _compare(recs, bi, bb, key, lower, n_boot, seed)
        bu = best(INCREMENTAL, skey, lower)
        c_iu = _compare(recs, bu, "baselines_union", key, lower, n_boot, seed)
        verdicts[crit] = {"metric": skey, "best_topology": bt, "best_baseline": bb,
                          "topology_vs_best_baseline": c_b,
                          "topology_vs_union": c_u, "incremental": c_i,
                          "incremental_union": c_iu,
                          "verdict_vs_best_baseline": _rule(c_b, go),
                          "verdict_vs_union": _rule(c_u, go),
                          "verdict_incremental": _rule(c_i, go),
                          "verdict_incremental_union_secondary": _rule(c_iu, go)}
    ew = early_warning(runs, n_boot, seed, int(ev.get("n_estimators", 200)))
    out["early_warning"] = ew
    if "table" in ew:
        verdicts["early_warning"] = {
            "metric": "out-of-fold R^2 (steps to grokking)",
            "best_topology": ew["best_topology"], "best_baseline": ew["best_baseline"],
            "topology_vs_best_baseline": ew["topology_vs_best_baseline"],
            "topology_vs_union": ew["topology_vs_union"], "incremental": ew["incremental"],
            "incremental_union": ew["incremental_union"],
            "verdict_vs_best_baseline": _rule(ew["topology_vs_best_baseline"], go),
            "verdict_vs_union": _rule(ew["topology_vs_union"], go),
            "verdict_incremental": _rule(ew["incremental"], go),
            "verdict_incremental_union_secondary": _rule(ew["incremental_union"], go)}
    out["verdicts"] = verdicts
    out["overall"] = "GO" if any(
        v[k] == "GO" for v in verdicts.values()
        for k in ("verdict_vs_best_baseline", "verdict_incremental")) else "NO-GO"
    if ctrl:
        crecs = [analyse_run(r, min_seg, smooth) for r in ctrl]
        out["controls"] = {
            "runs": [r["run_id"] for r in crecs],
            "mean_r2_step": {n: float(np.mean([r["signals"][n]["all"]["r2_step"]
                                               for r in crecs])) for n in names},
            "grok_mean_r2_step": {n: out["summary"][n]["all"]["mean_r2_step"] for n in names}}
    return out

"""Figures for the two experiments (matplotlib, PNG, Agg backend).

descriptor_trajectory.png   descriptor + baseline trajectories, grokking marked
cka_vs_descriptor.png       CKA-to-final vs descriptor-distance-to-final
detection_timing.png        change-point error and step R^2 per signal
confusion_matrices.png      best topology vs best baseline, every task
discrimination_accuracy.png accuracy of every feature set, every task
activation_topology.png     activation clouds and H1 diagrams, memorized vs grokked
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .changepoint import standardize_columns  # noqa: E402
from .features import FEATURE_NAMES, RunFeatures  # noqa: E402
from .featuresets import FEATURE_SETS, run_phases, signal_matrix  # noqa: E402

# validated categorical slots (dataviz reference palette, light mode)
C_TOPO, C_BASE, C_UNION, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
C_TEXT, C_MUTED = "#0b0b0b", "#52514e"
TIER_COLOR = {"topology": C_TOPO, "baseline": C_BASE, "union": C_UNION,
              "incremental": C_UNION}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
          "#e34948"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C_MUTED, "axes.labelcolor": C_TEXT,
                     "xtick.color": C_MUTED, "ytick.color": C_MUTED, "axes.grid": True,
                     "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "figure.facecolor": "white", "axes.facecolor": "white",
                     "legend.frameon": False})


_PRETTY = {"x1": "X1", "x2": "X2", "x1x2": "X1+X2", "x3": "X3"}


def pretty(name: str) -> str:
    """Display name of a feature set ("x1x2" -> "X1+X2"; avoids the font's
    "1x2" -> "1×2" ligature and matches the paper's notation)."""
    return " + ".join(_PRETTY.get(p, p) for p in name.split("+"))


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(x)


def _mark(ax: Any, rf: RunFeatures, det_run: Dict[str, Any] | None) -> None:
    ph = run_phases(rf)
    if ph.i_mem is not None:
        ax.axvline(rf.steps[ph.i_mem], color=C_MUTED, ls="--", lw=1)
    if ph.i_grok90 is not None:
        ax.axvline(rf.steps[ph.i_grok90], color="#e34948", ls=":", lw=1.2)
    if det_run is not None:
        ax.axvline(det_run["t_ref_cp"], color="#e34948", ls="-", lw=1, alpha=0.6)


def _representatives(runs: List[RunFeatures]) -> List[RunFeatures]:
    out, seen = [], set()
    for r in runs:
        if r.meta["kind"] == "grok" and run_phases(r).grokked and r.meta["op"] not in seen:
            out.append(r)
            seen.add(r.meta["op"])
    return out


def _site(rf: RunFeatures, prefer: str) -> str:
    cands = [s for s in rf.sites if s.startswith(prefer)]
    return cands[-1] if cands else rf.sites[-1]


def fig_trajectory(runs: List[RunFeatures], det: Dict[str, Any], path: Path) -> None:
    reps = _representatives(runs)
    if not reps:
        return
    det_runs = {r["run_id"]: r for r in det.get("runs", [])}
    rows = ["accuracy", "X1 functional graph (mlp)", "X2 Rips PH (resid_post)",
            "baselines", "change-point projection"]
    fig, axes = plt.subplots(len(rows), len(reps), figsize=(4.2 * len(reps), 11),
                             squeeze=False, sharex="col")
    for c, rf in enumerate(reps):
        st = rf.steps
        dr = det_runs.get(rf.meta["run_id"])
        mlp, rp = _site(rf, "mlp"), _site(rf, "resid_post")
        ax = axes[0, c]
        ax.plot(st, rf.curves["train_acc"], color=C_REF, label="train acc")
        ax.plot(st, rf.curves["test_acc"], color=C_TEXT, label="held-out acc")
        ax.set_title(f"{rf.meta['op']}  (seed {rf.meta['seed']}, p={rf.meta['p']})")
        ax.set_ylim(-0.03, 1.03)
        ax = axes[1, c]
        j = rf.sites.index(mlp)
        for k, n in enumerate(["density", "avg_clustering", "betw_mean"]):
            ax.plot(st, _norm(rf.feats["x1"][:, j, FEATURE_NAMES["x1"].index(n)]),
                    color=SERIES[k], label=n)
        ax = axes[2, c]
        j = rf.sites.index(rp)
        for k, n in enumerate(["H1_count", "H1_pers_max", "H0_pers_mean"]):
            ax.plot(st, _norm(rf.feats["x2"][:, j, FEATURE_NAMES["x2"].index(n)]),
                    color=SERIES[k], label=n)
        ax = axes[3, c]
        ax.plot(st, _norm(rf.feats["weight_norm"][:, 0, -1]), color=SERIES[0], label="|W| total")
        ax.plot(st, _norm(rf.feats["cka_final"][:, rf.sites.index(mlp), 0]), color=SERIES[1],
                label="CKA to final (mlp)")
        ax.plot(st, _norm(rf.feats["probe_acc"][:, rf.sites.index(rp), 0]), color=SERIES[2],
                label="probe acc (resid_post)")
        ax.plot(st, _norm(rf.feats["pca_stats"][:, rf.sites.index(mlp), 0]), color=SERIES[3],
                label="PCA part. ratio (mlp)")
        ax = axes[4, c]
        ph = run_phases(rf)
        i0 = ph.i_mem or 0
        for name, col in (("x1x2", C_TOPO), ("baselines_union", C_UNION), ("cka_final", C_BASE)):
            Z = standardize_columns(signal_matrix(rf, FEATURE_SETS[name][1], "all")[i0:])
            if Z.shape[1] == 0:
                continue
            from .changepoint import single_changepoint, step_projection
            tau, _ = single_changepoint(Z)
            ax.plot(st[i0:], step_projection(Z, tau), color=col, label=pretty(name))
        ta = rf.curves["test_acc"][i0:]
        ax.plot(st[i0:], _norm(ta), color=C_TEXT, lw=1, ls="--", label="held-out acc (scaled)")
        ax.set_xlabel("training step")
        for r in range(len(rows)):
            _mark(axes[r, c], rf, dr)
            if c == 0:
                axes[r, c].set_ylabel(rows[r], fontsize=8)
            if c == len(reps) - 1:
                axes[r, c].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Descriptor trajectories through training (one run per operation)\n"
                 "dashed gray: memorization (train acc >= 0.99); red line: change point of "
                 "held-out accuracy; red dotted: held-out acc >= 0.9; rows 2-4 min-max scaled",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _dist_to_final(rf: RunFeatures, name: str) -> np.ndarray:
    Z = standardize_columns(signal_matrix(rf, FEATURE_SETS[name][1], "all"))
    d = np.linalg.norm(Z - Z[-1], axis=1)
    return d / d.max() if d.max() > 0 else d


def fig_cka_vs_descriptor(runs: List[RunFeatures], det: Dict[str, Any], path: Path) -> None:
    grok = [r for r in runs if r.meta["kind"] == "grok" and run_phases(r).grokked]
    if not grok:
        return
    ref = {r["run_id"]: r["t_ref_cp"] for r in det.get("runs", [])}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for rf in grok:
        t0 = ref.get(rf.meta["run_id"], float(rf.steps[run_phases(rf).i_grok90 or 0]))
        x = rf.steps - t0
        cka = rf.feats["cka_final"][:, :, 0].mean(axis=1)
        axes[0].plot(x, cka, color=C_BASE, alpha=0.45, lw=1)
        axes[0].plot(x, 1 - _dist_to_final(rf, "x1x2"), color=C_TOPO, alpha=0.45, lw=1)
        axes[1].plot(x, rf.curves["test_acc"], color=C_TEXT, alpha=0.35, lw=1)
        axes[1].plot(x, rf.feats["probe_acc"][:, :, 0].max(axis=1), color=C_UNION, alpha=0.45,
                     lw=1)
    axes[0].plot([], [], color=C_BASE, label="CKA to final (mean over sites)")
    axes[0].plot([], [], color=C_TOPO, label="1 - descriptor distance to final (X1+X2)")
    axes[1].plot([], [], color=C_TEXT, label="held-out accuracy")
    axes[1].plot([], [], color=C_UNION, label="linear-probe acc (best site)")
    for ax in axes[:2]:
        ax.axvline(0, color="#e34948", lw=1)
        ax.set_xlabel("steps relative to the held-out-accuracy change point")
        ax.legend(fontsize=7, loc="lower right")
    axes[0].set_title("Similarity to the final model, every grokking run")
    axes[1].set_title("Outcome-level references")
    xs, ys, cs = [], [], []
    for rf in grok:
        xs.append(rf.feats["cka_final"][:, :, 0].mean(axis=1))
        ys.append(_dist_to_final(rf, "x1x2"))
        cs.append(rf.curves["test_acc"])
    sc = axes[2].scatter(np.concatenate(xs), np.concatenate(ys), c=np.concatenate(cs),
                         cmap="Blues", s=10, edgecolors="none", vmin=0, vmax=1)
    fig.colorbar(sc, ax=axes[2], label="held-out accuracy")
    axes[2].set_xlabel("CKA to final (mean over sites)")
    axes[2].set_ylabel("descriptor distance to final (X1+X2)")
    axes[2].set_title("CKA vs descriptor, all checkpoints")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_detection(det: Dict[str, Any], path: Path) -> None:
    if "runs" not in det:
        return
    names = [n for n in det["summary"]]
    data = [[abs(r["signals"][n]["all"]["nlag"]) for r in det["runs"]] for n in names]
    order = np.argsort([np.mean(d) for d in data])
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.32 * len(names) + 1.8))
    ax = axes[0]
    for k, i in enumerate(order):
        n = names[i]
        col = TIER_COLOR.get(FEATURE_SETS[n][0], C_REF) if n in FEATURE_SETS else C_REF
        ax.scatter(data[i], np.full(len(data[i]), k), color=col, s=14, alpha=0.7)
        ax.scatter([np.mean(data[i])], [k], color=col, marker="|", s=250)
    ax.axvline(det["prior_detector"]["mean_abs_nlag"], color=C_MUTED, ls=":", lw=1)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([pretty(names[i]) for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("|change point - held-out-acc change point| / window (dots = runs, bar = mean)"
                  "\ndotted: trivial prior detector")
    ax.set_title("Exp 1 - timing error (lower is better)")
    ax = axes[1]
    r2g = [det["summary"][n]["all"]["mean_r2_step"] for n in names]
    ctrl = det.get("controls", {}).get("mean_r2_step", {})
    y = np.arange(len(names))
    ax.barh(y - 0.2, [r2g[i] for i in order], height=0.38, color=C_TOPO, label="grokking runs")
    if ctrl:
        ax.barh(y + 0.2, [ctrl[names[i]] for i in order], height=0.38, color=C_REF,
                label="non-grokking controls")
    ax.set_yticks(y)
    ax.set_yticklabels([pretty(names[i]) for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("step R^2 (1 = perfect single step)")
    ax.set_title("Cleanness of the transition and specificity")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_confusion(dis: Dict[str, Any], path: Path) -> None:
    tasks = {t: r for t, r in dis.get("tasks", {}).items() if "confusion" in r}
    if not tasks:
        return
    fig, axes = plt.subplots(len(tasks), 2, figsize=(9, 3.9 * len(tasks)), squeeze=False)
    for i, (t, r) in enumerate(tasks.items()):
        for k, name in enumerate((r["best_topology"], r["best_baseline"])):
            cm = np.asarray(r["confusion"][name], dtype=float)
            cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax = axes[i, k]
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            ax.grid(False)
            n = len(r["class_names"])
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(r["class_names"], rotation=40, ha="right", fontsize=7)
            ax.set_yticklabels(r["class_names"], fontsize=7)
            for a in range(n):
                for b in range(n):
                    ax.text(b, a, f"{cmn[a, b]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if cmn[a, b] > 0.6 else C_TEXT)
            acc = r["table"][name]["accuracy"]
            ax.set_title(f"{t}: {pretty(name)} (acc {acc:.3f})", fontsize=9)
            ax.set_xlabel("predicted")
            ax.set_ylabel("true")
    fig.suptitle("Exp 2 - grouped-CV confusion matrices (row-normalised, first repeat)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_discrimination(dis: Dict[str, Any], path: Path) -> None:
    tasks = {t: r for t, r in dis.get("tasks", {}).items() if "table" in r}
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.6 * len(tasks), 4.6), squeeze=False)
    names = list(FEATURE_SETS)
    for i, (t, r) in enumerate(tasks.items()):
        ax = axes[0, i]
        acc = [r["table"][n]["accuracy"] for n in names]
        ax.barh(range(len(names)), acc, color=[TIER_COLOR[FEATURE_SETS[n][0]] for n in names],
                height=0.7)
        ax.axvline(r["table"]["baselines_union"]["accuracy"], color=C_UNION, ls=":", lw=1)
        ax.axvline(r["majority_rate"], color=C_MUTED, ls="--", lw=1)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([pretty(n) for n in names] if i == 0 else [], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        m = r["topology_vs_best_baseline"]
        ax.set_title(f"{t}: topo - best baseline {m['margin']:+.3f}\n"
                     f"CI [{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]", fontsize=9)
        ax.set_xlabel("grouped-CV accuracy (dashed: majority class)")
    for tier, col in (("topology", C_TOPO), ("baseline", C_BASE),
                      ("union / incremental", C_UNION)):
        axes[0, -1].barh([], [], color=col, label=tier)
    axes[0, -1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Exp 2 - representation discrimination (groups = training runs)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_activation_topology(runs: List[RunFeatures], path: Path) -> None:
    reps = _representatives(runs)
    if not reps:
        return
    rf = reps[0]
    site = _site(rf, "resid_post")
    snaps = rf.snapshots[site]
    ph = run_phases(rf)
    st = list(rf.snapshot_steps)
    pick = []
    if ph.i_mem is not None and ph.i_grok50 is not None:
        mid = (rf.steps[ph.i_mem] + rf.steps[ph.i_grok50]) / 2
        pick.append(int(np.argmin(np.abs(np.array(st) - mid))))
    pick.append(len(st) - 1)
    labels = rf.snapshot_labels
    from ripser import ripser
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(2, len(pick), figsize=(4.6 * len(pick), 8.4), squeeze=False)
    for k, i in enumerate(pick):
        A = snaps[i].astype(np.float64)
        Ar = PCA(n_components=min(15, A.shape[1], A.shape[0] - 1),
                 random_state=42).fit_transform(A)
        Ar = (Ar - Ar.min()) / (Ar.max() - Ar.min() + 1e-8)
        ax = axes[0, k]
        sc = ax.scatter(Ar[:, 0], Ar[:, 1], c=labels, cmap="twilight", s=22, edgecolors="none")
        test_i = int(np.argmin(np.abs(rf.steps - st[i])))
        ax.set_title(f"step {st[i]} (held-out acc {rf.curves['test_acc'][test_i]:.2f})\n"
                     f"{site}, first {len(A)} probe inputs, PCA 1-2", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        dg = ripser(Ar, maxdim=1)["dgms"]
        ax = axes[1, k]
        h0 = dg[0][np.isfinite(dg[0][:, 1])]
        ax.scatter(h0[:, 0], h0[:, 1], color=C_REF, s=12, label=f"H0 ({len(h0)})")
        ax.scatter(dg[1][:, 0], dg[1][:, 1], color=C_TOPO, s=18, label=f"H1 ({len(dg[1])})")
        lim = max(1e-3, float(np.max(np.r_[h0[:, 1], dg[1][:, 1]])) if len(dg[1]) else
                  float(h0[:, 1].max()))
        ax.plot([0, lim], [0, lim], color=C_MUTED, lw=0.8)
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.set_title("Vietoris-Rips persistence diagram (X2 input)", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.subplots_adjust(hspace=0.38)
    fig.colorbar(sc, ax=axes[0, :].tolist(), label="answer c")
    fig.suptitle(f"Activation cloud before vs after grokking ({rf.meta['op']}, "
                 f"seed {rf.meta['seed']})", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def make_figures(runs: List[RunFeatures], result: Dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    det, dis = result["detection"], result["discrimination"]
    fig_trajectory(runs, det, out / "descriptor_trajectory.png")
    fig_cka_vs_descriptor(runs, det, out / "cka_vs_descriptor.png")
    fig_detection(det, out / "detection_timing.png")
    fig_confusion(dis, out / "confusion_matrices.png")
    fig_discrimination(dis, out / "discrimination_accuracy.png")
    fig_activation_topology(runs, out / "activation_topology.png")

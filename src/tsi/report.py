"""Markdown report of a pipeline run (``results.md``)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import featuresets
from .featuresets import FEATURE_SETS


def _ci(c: Dict[str, Any]) -> str:
    lo, hi = c["ci95"]
    return f"{c['margin']:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _fmt(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def _detection(det: Dict[str, Any]) -> List[str]:
    L = ["## Experiment 1 - grokking detection", ""]
    if "verdicts" not in det:
        return L + [f"Skipped: {det.get('status')}", ""]
    L += [f"Grokking runs: {det['n_grok_runs']}; controls (memorize, never grok): "
          f"{det['n_control_runs']}. Not grokked within budget: "
          f"{', '.join(det['not_grokked']) or 'none'}.", "",
          f"**Overall: {det['overall']}** (GO needs margin >= {det['go_margin']:+.2f} with the "
          "95 % CI above 0, against the best single baseline or as an increment over "
          "`baselines_union`).", "",
          "| criterion | best topology | best baseline | topology - best baseline | verdict |"
          " topology - union | incremental (union+topo - union) | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for crit, v in det["verdicts"].items():
        L.append(f"| {crit} ({v['metric']}) | {v['best_topology']} | {v['best_baseline']} | "
                 f"{_ci(v['topology_vs_best_baseline'])} | {v['verdict_vs_best_baseline']} | "
                 f"{_ci(v['topology_vs_union'])} | {v['incremental']['a']}: "
                 f"{_ci(v['incremental'])} | {v['verdict_incremental']} |")
    L += ["", "Margins are oriented so that **positive = topology better** (smaller timing "
          "error, larger step R^2, larger early-warning R^2).", "",
          "### Change-point timing and cleanness (all sites concatenated)", "",
          "`nlag` = (detected change point - held-out-accuracy change point) / window; "
          "window = [t_mem, t_end]. `onset10 lead` < 0 means the signal starts moving before "
          "held-out accuracy does.", "",
          "| signal | tier | mean abs nlag | median abs nlag | mean nlag | within 0.05 | "
          "mean step R^2 | onset10 lead | onset50 lead |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, by_site in det["summary"].items():
        s = by_site["all"]
        tier = FEATURE_SETS[n][0] if n in FEATURE_SETS else "reference (label-adjacent)"
        L.append(f"| {n} | {tier} | {s['mean_abs_nlag']:.3f} | {s['median_abs_nlag']:.3f} | "
                 f"{s['mean_nlag']:+.3f} | {s['frac_within_0.05']:.2f} | {s['mean_r2_step']:.3f}"
                 f" | {s['mean_onset10_nlead']:+.3f} | {s['mean_onset50_nlead']:+.3f} |")
    pr = det["prior_detector"]
    L += [f"| prior (leave-one-run-out mean position) | trivial | {pr['mean_abs_nlag']:.3f} | "
          f"{pr['median_abs_nlag']:.3f} | | | | | |", ""]
    sites = [s for s in next(iter(det["summary"].values())) if s != "all"]
    L += ["### Per-site timing error (mean abs nlag) / step R^2", "",
          "| signal | " + " | ".join(sites) + " |", "|---|" + "---|" * len(sites)]
    for n, by_site in det["summary"].items():
        if all(s in by_site for s in sites):
            L.append(f"| {n} | " + " | ".join(
                f"{by_site[s]['mean_abs_nlag']:.3f} / {by_site[s]['mean_r2_step']:.2f}"
                for s in sites) + " |")
    ew = det.get("early_warning", {})
    if "table" in ew:
        L += ["", "### Early warning: steps remaining until grokking (plateau checkpoints)", "",
              f"{ew['n_samples']} plateau checkpoints from {ew['n_runs']} runs, "
              f"leave-one-run-out; target mean {ew['target_mean']:.0f} +- "
              f"{ew['target_std']:.0f} steps.", "",
              "| feature set | out-of-fold R^2 | MAE (steps) | model |", "|---|---|---|---|"]
        for k, v in sorted(ew["table"].items(), key=lambda kv: -kv[1]["r2"]):
            L.append(f"| {k} | {v['r2']:.3f} | {v['mae_steps']:.0f} | {v['model']} |")
    if "controls" in det:
        c = det["controls"]
        L += ["", "### Specificity: step R^2 in grokking runs vs non-grokking controls", "",
              f"Controls: {', '.join(c['runs'])}. A signal specific to grokking should have a "
              "much lower step R^2 in the controls.", "",
              "| signal | grokking runs | controls |", "|---|---|---|"]
        for n in c["mean_r2_step"]:
            L.append(f"| {n} | {c['grok_mean_r2_step'][n]:.3f} | {c['mean_r2_step'][n]:.3f} |")
    L += ["", "### Per-run reference times (steps)", "",
          "| run | t_mem | t_ref (change point of held-out acc) | t_onset | t_grok | "
          "best-topology t_cp | best-baseline t_cp |", "|---|---|---|---|---|---|---|"]
    bt, bb = det["verdicts"]["timing"]["best_topology"], det["verdicts"]["timing"]["best_baseline"]
    for r in det["runs"]:
        L.append(f"| {r['run_id']} | {r['t_mem']:.0f} | {r['t_ref_cp']:.0f} | "
                 f"{_fmt(r['t_grok50'])} | {_fmt(r['t_grok90'])} | "
                 f"{r['signals'][bt]['all']['t_cp']:.0f} | {r['signals'][bb]['all']['t_cp']:.0f} |")
    return L + [""]


def _discrimination(dis: Dict[str, Any]) -> List[str]:
    L = ["## Experiment 2 - representation discrimination", ""]
    if "tasks" not in dis:
        return L + [f"Skipped: {dis.get('status')}", ""]
    L += [f"**Overall: {dis['overall']}** (GO margin {dis['go_margin']:+.2f}, CI above 0).", "",
          "| task | n / runs | best topology | best baseline | topology - best baseline | "
          "verdict | topology - union | headroom | incremental | verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in dis["tasks"].items():
        if "table" not in r:
            L.append(f"| {t} | skipped: {r.get('status')} |||||||||")
            continue
        tb = r["table"]
        L.append(f"| {t} | {r['n']} / {r['n_runs']} | {r['best_topology']} "
                 f"{tb[r['best_topology']]['accuracy']:.3f} | {r['best_baseline']} "
                 f"{tb[r['best_baseline']]['accuracy']:.3f} | {_ci(r['topology_vs_best_baseline'])}"
                 f" | {r['verdict_vs_best_baseline']} | {_ci(r['topology_vs_union'])} | "
                 f"{r['headroom']:.3f} | {r['best_incremental']}: {_ci(r['incremental'])} | "
                 f"{r['verdict_incremental']} |")
    for t, r in dis["tasks"].items():
        if "table" not in r:
            continue
        L += ["", f"### Task `{t}` ({r['n_classes']} classes: {', '.join(r['class_names'])}; "
              f"majority rate {r['majority_rate']:.3f})", "",
              "| feature set | tier | accuracy | LogReg | RF | macro-F1 |", "|---|---|---|---|---|---|"]
        for k, v in sorted(r["table"].items(), key=lambda kv: -kv[1]["accuracy"]):
            L.append(f"| {k} | {FEATURE_SETS[k][0]} | {v['accuracy']:.3f} | {v['logreg']:.3f} | "
                     f"{v['rf']:.3f} | {v['f1_macro']:.3f} |")
    return L + [""]


def write_report(result: Dict[str, Any], path: Path) -> None:
    th = featuresets.THRESHOLDS
    L = ["# Results - topological-spectral descriptor on grokking transformers", "",
         "Generated by `python -m tsi.run`. Interpretation: `docs/RESULTS.md`.", "",
         "## Runs", "",
         f"t_mem: first train acc >= {th.mem}; t_onset: first held-out acc >= {th.onset}; "
         f"t_grok: first held-out acc >= {th.grok}. grokked = delayed generalization "
         "(t_onset after t_mem) and t_grok reached.", "",
         "| run | op | kind | train frac | final train | final held-out | t_mem | t_onset | "
         "t_grok | grokked |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["runs"]:
        L.append(f"| {r['run_id']} | {r['op']} | {r['kind']} | {_fmt(r.get('train_frac'))} | "
                 f"{r['final_train_acc']:.3f} | "
                 f"{r['final_test_acc']:.3f} | {_fmt(r['t_mem'])} | {_fmt(r['t_grok50'])} | "
                 f"{_fmt(r['t_grok90'])} | {r['grokked']} |")
    L += [""] + _detection(result["detection"]) + _discrimination(result["discrimination"])
    L += ["## Feature sets", "", "| name | tier | description |", "|---|---|---|"]
    L += [f"| {k} | {v[0]} | {v[2]} |" for k, v in FEATURE_SETS.items()]
    s = result["seconds"]
    L += ["", f"Runtime: runs {s['runs']:.0f} s, analysis {s['total_analysis']:.0f} s.", ""]
    path.write_text("\n".join(L))

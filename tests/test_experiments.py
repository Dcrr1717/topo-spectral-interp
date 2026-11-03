"""Both experiments, the report and the figures on synthetic run trajectories."""
from pathlib import Path
import json
from typing import Any, Dict, List

import numpy as np

from tsi.exp_detection import run_detection
from tsi.exp_discrimination import run_discrimination
from tsi.features import FEATURE_NAMES, MODEL_LEVEL, RunFeatures
from tsi.featuresets import phase_labels, run_phases
from tsi.figures import make_figures
from tsi.report import write_report
from tsi.run import _jsonable

SITES = ["embed", "resid_mid_0", "mlp_0", "resid_post_0"]


def fake_run(op: str, seed: int, t_grok: int, kind: str = "grok", T: int = 60) -> RunFeatures:
    rng = np.random.default_rng(seed + 100 * ["add", "sub", "mul"].index(op))
    steps = np.arange(T) * 50
    train = np.minimum(1.0, np.arange(T) / 5.0)
    grok = kind == "grok"
    step_fn = (np.arange(T) >= t_grok).astype(float) if grok else np.zeros(T)
    test = 0.03 + 0.97 / (1 + np.exp(-(np.arange(T) - t_grok) / 1.5)) if grok else \
        np.full(T, 0.03)
    feats = {}
    for v, names in FEATURE_NAMES.items():
        S = 1 if v in MODEL_LEVEL else len(SITES)
        base = rng.standard_normal((1, S, len(names)))
        sig = 4.0 * step_fn[:, None, None] if v in ("x1", "x2", "x3") else \
            np.linspace(0, 1, T)[:, None, None]
        site_offset = np.arange(S)[None, :, None] * (0.0 if v in MODEL_LEVEL else 6.0)
        task_offset = 0.5 * ["add", "sub", "mul"].index(op)
        feats[v] = base + sig + site_offset + task_offset + \
            0.05 * rng.standard_normal((T, S, len(names)))
    snaps = {s: rng.standard_normal((3, 50, 8)).astype(np.float32) for s in SITES}
    meta = {"run_id": f"{kind}_{op}_s{seed}", "op": op, "seed": seed, "kind": kind, "p": 11}
    return RunFeatures(meta=meta, sites=SITES, steps=steps,
                       curves={"train_acc": train, "test_acc": test,
                               "train_loss": 1 - train + 1e-3, "test_loss": 1.5 - test},
                       feats=feats, snapshots=snaps, snapshot_steps=steps[[0, 30, T - 1]],
                       snapshot_labels=rng.integers(0, 11, 50))


CFG: Dict[str, Any] = {"seed": 0,
       "detection": {"bootstrap": 200, "n_estimators": 20},
       "discrimination": {"n_splits": 3, "n_repeats": 1, "n_estimators": 20,
                          "max_ckpts_per_run": 12, "bootstrap": 200}}


def runs() -> List[RunFeatures]:
    out = [fake_run(op, s, t_grok=20 + 4 * s + 3 * k)
           for k, op in enumerate(["add", "sub", "mul"]) for s in range(3)]
    return out + [fake_run("add", 9, 0, kind="control")]


def test_phase_helpers() -> None:
    r = fake_run("add", 0, 30)
    ph = run_phases(r)
    assert ph.i_mem == 5 and ph.grokked and ph.i_grok90 is not None
    lab = phase_labels(r)
    assert lab[0] == -1 and lab[10] == 0 and lab[-1] == 1


def test_detection_on_synthetic_step() -> None:
    det = run_detection(runs(), CFG)
    assert det["n_grok_runs"] == 9 and det["n_control_runs"] == 1
    t = det["verdicts"]["timing"]
    # the topological views carry an exact step at the grokking time; the baselines ramp
    assert t["best_topology"] in ("x1", "x2", "x1x2", "x3")
    assert det["summary"]["x1"]["all"]["mean_abs_nlag"] < 0.05
    assert t["topology_vs_best_baseline"]["margin"] > 0
    assert det["verdicts"]["cleanness"]["verdict_vs_best_baseline"] == "GO"
    assert "early_warning" in det["verdicts"]
    assert det["controls"]["mean_r2_step"]["x1"] < det["summary"]["x1"]["all"]["mean_r2_step"]


def test_discrimination_on_synthetic() -> None:
    dis = run_discrimination(runs(), CFG)
    assert set(dis["tasks"]) == {"layer", "task", "phase"}
    lay = dis["tasks"]["layer"]
    assert lay["n_classes"] == 4 and lay["table"]["x1"]["accuracy"] > 0.9
    ph = dis["tasks"]["phase"]
    assert ph["table"]["x1x2"]["accuracy"] > 0.9
    assert np.asarray(ph["confusion"][ph["best_topology"]]).sum() == ph["n"]


def test_report_and_figures(tmp_path: Path) -> None:
    rs = runs()
    det, dis = run_detection(rs, CFG), run_discrimination(rs, CFG)
    from tsi.run import run_summary
    result = {"config": CFG, "runs": run_summary(rs), "detection": det, "discrimination": dis,
              "seconds": {"runs": 0.0, "total_analysis": 0.0}}
    json.dumps(_jsonable(result))
    write_report(result, tmp_path / "r.md")
    text = (tmp_path / "r.md").read_text()
    assert "Experiment 1" in text and "Experiment 2" in text
    make_figures(rs, result, tmp_path / "fig")
    made = {p.name for p in (tmp_path / "fig").iterdir()}
    assert {"descriptor_trajectory.png", "cka_vs_descriptor.png", "detection_timing.png",
            "confusion_matrices.png", "discrimination_accuracy.png",
            "activation_topology.png"} <= made


def test_skips_without_grokking() -> None:
    rs = [fake_run("add", 0, 0, kind="control")]
    assert "status" in run_detection(rs, CFG)
    assert "status" in run_discrimination(rs, CFG)

"""End-to-end pipeline: train runs -> features -> experiments -> report -> figures.

Usage::

    python -m tsi.run --config configs/smoke.yaml   # ~1-2 min, offline CI
    python -m tsi.run --config configs/full.yaml    # ~1 h on 4 CPU cores

Per-run features are cached under ``cache_dir`` (keyed by a hash of the run
and descriptor configuration), so re-running only redoes the analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from .exp_detection import run_detection
from .exp_discrimination import run_discrimination
from .features import (DescriptorConfig, RunFeatures, load_run_features, run_features,
                       save_run_features)
from .train import RunConfig, train_run


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)
    return cfg


def run_configs(cfg: Dict[str, Any]) -> List[RunConfig]:
    """Grokking runs (ops x seeds) plus optional non-grokking controls.

    ``op_overrides: {op: {key: value}}`` changes training settings for one
    operation (e.g. its train fraction); controls inherit them before their
    own ``overrides``."""
    base = dict(cfg.get("training", {}))
    if "betas" in base:
        base["betas"] = tuple(base["betas"])
    per_op = cfg.get("op_overrides") or {}

    def settings(op: str) -> Dict[str, Any]:
        return {**base, **per_op.get(op, {})}
    out = [RunConfig(op=op, seed=int(s), kind="grok", **settings(op))
           for op in cfg["ops"] for s in cfg["seeds"]]
    ctrl = cfg.get("controls")
    if ctrl:
        out += [RunConfig(op=op, seed=int(s), kind="control",
                          **{**settings(op), **ctrl.get("overrides", {})})
                for op in ctrl.get("ops", cfg["ops"]) for s in ctrl.get("seeds", [0])]
    return out


def _key(rc: RunConfig, dc: DescriptorConfig) -> str:
    blob = json.dumps({"run": rc.to_dict(), "desc": dc.__dict__, "v": 1}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _worker(args: Tuple[RunConfig, DescriptorConfig, str, bool]) -> str:
    rc, dc, path, verbose = args
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    t0 = time.time()
    with threadpool_limits(limits=1):  # one BLAS thread per worker: no oversubscription
        rec = train_run(rc, verbose=verbose)
        t1 = time.time()
        rf = run_features(rec, dc)
    save_run_features(rf, path)
    print(f"[run] {rc.run_id}: final train {rec.train_acc[-1]:.3f} test {rec.test_acc[-1]:.3f}"
          f" | train {t1 - t0:.0f}s, features {time.time() - t1:.0f}s", flush=True)
    return path


def collect_runs(cfg: Dict[str, Any], verbose: bool = False) -> List[RunFeatures]:
    cache = Path(cfg.get("cache_dir", "data/cache"))
    cache.mkdir(parents=True, exist_ok=True)
    dc = DescriptorConfig(**cfg.get("descriptor", {}))
    jobs, paths = [], []
    for rc in run_configs(cfg):
        path = str(cache / f"{rc.run_id}_{_key(rc, dc)}.npz")
        paths.append(path)
        if not os.path.exists(path):
            jobs.append((rc, dc, path, verbose))
    workers = int(cfg.get("n_workers", os.cpu_count() or 1))
    if jobs:
        print(f"[runs] training {len(jobs)} runs on {min(workers, len(jobs))} workers", flush=True)
        if workers <= 1 or len(jobs) == 1:
            for j in jobs:
                _worker(j)
        else:
            with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
                for _ in pool.imap_unordered(_worker, jobs):
                    pass
    return [load_run_features(p) for p in paths]


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def run_summary(runs: List[RunFeatures]) -> List[Dict[str, Any]]:
    from .featuresets import run_phases
    rows = []
    for r in runs:
        ph = run_phases(r)
        rows.append({"run_id": r.meta["run_id"], "op": r.meta["op"], "kind": r.meta["kind"],
                     "seed": r.meta["seed"], "final_train_acc": float(r.curves["train_acc"][-1]),
                     "final_test_acc": float(r.curves["test_acc"][-1]),
                     "t_mem": int(r.steps[ph.i_mem]) if ph.i_mem is not None else None,
                     "t_grok50": int(r.steps[ph.i_grok50]) if ph.i_grok50 is not None else None,
                     "t_grok90": int(r.steps[ph.i_grok90]) if ph.i_grok90 is not None else None,
                     "grokked": ph.grokked})
    return rows


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = Path(cfg.get("out_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runs = collect_runs(cfg, verbose=args.verbose)
    t_runs = time.time() - t0
    print("[exp1] grokking detection", flush=True)
    det = run_detection(runs, cfg)
    print("[exp2] representation discrimination", flush=True)
    dis = run_discrimination(runs, cfg)
    result = {"config": cfg, "runs": run_summary(runs), "detection": det,
              "discrimination": dis,
              "seconds": {"runs": t_runs, "total_analysis": time.time() - t0 - t_runs}}
    (out_dir / "results.json").write_text(json.dumps(_jsonable(result), indent=1))
    from .report import write_report
    write_report(result, out_dir / "results.md")
    if not args.no_figures:
        from .figures import make_figures
        make_figures(runs, result, out_dir / "figures")
    print(f"[done] detection: {det.get('overall')} | discrimination: {dis.get('overall')} "
          f"| {time.time() - t0:.0f}s -> {out_dir}", flush=True)
    return result


if __name__ == "__main__":
    main()

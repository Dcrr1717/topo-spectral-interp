# topo-spectral-interp

A topological-spectral multiview descriptor of **transformer activations**,
tested on the **grokking** transition of small transformers trained from scratch
on modular arithmetic.

The descriptor is the authentic X1 / X2 / X3 of Reinoso & Escudero (2026),
ported from
[`gem-geophysical-descriptor/src/descriptor/authentic.py`](https://github.com/Dcrr1717/gem-geophysical-descriptor).
It is applied back in the setting it was designed for: a matrix of neural
activations (inputs x units).

- **X1** (R^10): functional graph of the units (edges where |Pearson| > 0.40),
  summarised by ten graph invariants.
- **X2** (R^12): Vietoris-Rips persistent homology (H0, H1) of the activation
  point cloud, summarised by `sig6`.
- **X3** (R^22): the label-free fusion [X1/|X1|, X2/|X2|].

Two pre-registered Go/No-Go experiments compare it with strong baselines:
CKA to the final representation (Kornblith et al. 2019), linear-probe
accuracy, raw activation statistics with PCA effective dimension, weight
norm, and the Fourier restricted / excluded loss progress measures of Nanda
et al. (2023).

1. **Grokking detection.** Does the descriptor locate the
   memorization -> generalization transition earlier, more accurately or more
   cleanly than the baselines? Does it add signal the baselines miss?
2. **Representation discrimination.** Can it classify layer / task /
   memorized-vs-grokked from the activations? Evaluation uses grouped CV by
   training run, margins and bootstrap CIs, exactly as in the GEM benchmark.

## Results in brief: NO-GO on both experiments

The setup is 15 runs (p = 59; add / sub / div; 4 seeds each, plus 3
weight-decay-0 controls). 11 runs grokked, between steps 450 and 4800.

| | best topology | best baseline | margin [95 % CI] | verdict |
|---|---|---|---|---|
| Locate the transition (timing error, lower is better) | X3 0.068 | CKA to final 0.083 | +0.015 [-0.017, +0.052] | tie |
| Step cleanness (R^2) | X1 0.49 | probe 0.77 | -0.28 | worse |
| Early warning (R^2 of steps-to-grok) | X2 -0.44 | Fourier restricted loss +0.20 | -0.64 [-1.15, +0.20] | worse |
| Layer / task / phase accuracy | 0.89 / 0.61 / 0.85 | 0.98 / 0.84 / 0.95 | -0.09 / -0.23 / -0.10 | worse |
| Topology added to all baselines (best task) | 0.987 | 0.972 | +0.015 [+0.000, +0.034] | < +0.05 |

The descriptor clearly sees grokking, but no better than the baselines:
- X3 has the lowest timing error of all signals, but the margin is not
  significant.
- Its trajectories are noisier, and they step just as much in non-grokking
  controls.
- Its clearest effect, rising unit correlation in the embedding, is mirrored
  by a plain PCA effective dimension.
- Only the mechanism-aware Fourier restricted / excluded losses (Nanda et
  al. 2023) move *before* held-out accuracy, and only restricted loss
  predicts time-to-grok out of sample. Topology adds nothing to them.

Topology ties the baselines; that is the result.

**Details:** see [`docs/RESULTS.md`](docs/RESULTS.md) (verdicts and
interpretation) and [`results/results.md`](results/results.md) (every number).
Background, method and references: [`docs/SCIENCE.md`](docs/SCIENCE.md).
Figures: [`results/figures/`](results/figures/).

## Install and run

Everything runs offline on a CPU: no datasets, no API keys and no pretrained
models are downloaded. All models are trained in-session.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or any torch build
pip install -r requirements.txt -r requirements-dev.txt
pip install -e . --no-deps
```

```bash
python -m tsi.run --config configs/smoke.yaml   # ~2-4 min: tiny runs, every code path (CI)
python -m tsi.run --config configs/full.yaml    # the reported experiment (~1 h, 4 cores)
pytest                                          # offline unit + integration tests
python -m mypy                                  # type check (src + tests)
```

Per-run features are cached in `data/` (git-ignored), so a re-run only redoes
the analysis. Outputs: `results/results.json`, `results/results.md`,
`results/figures/*.png`. The smoke config writes to `results/smoke/`
(git-ignored); its numbers are not results.

## Layout

```
src/tsi/
  descriptor/authentic.py  X1 / X2 / X3 (verbatim port, see module docstring)
  data.py, model.py        modular-arithmetic data; 1-layer transformer (Nanda et al. 2023)
  train.py                 full-batch AdamW + weight decay; logs activations per site
  features.py              descriptor + baselines per (checkpoint, site)
  baselines.py             linear CKA, linear probe, activation / PCA statistics
  fourier.py               Fourier restricted / excluded loss (Nanda et al. 2023)
  changepoint.py           label-free change-point and onset detector
  evaluation.py            grouped CV + paired run bootstrap (ported from GEM)
  exp_detection.py         Experiment 1
  exp_discrimination.py    Experiment 2
  report.py, figures.py    results.md and figures
  run.py                   CLI pipeline
configs/  smoke.yaml (CI), full.yaml (reported)
tests/    pytest suite (port fidelity vs the original code, model, metrics, experiments)
```

## License

MIT, Copyright (c) 2026 Cesar Daniel Reinoso. See [`LICENSE`](LICENSE).

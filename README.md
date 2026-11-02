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
accuracy, raw activation statistics with PCA effective dimension, and weight
norm.

1. **Grokking detection.** Does the descriptor locate the
   memorization -> generalization transition earlier, more accurately or more
   cleanly than the baselines? Does it add signal the baselines miss?
2. **Representation discrimination.** Can it classify layer / task /
   memorized-vs-grokked from the activations? Evaluation uses grouped CV by
   training run, margins and bootstrap CIs, exactly as in the GEM benchmark.

**Results:** see [`docs/RESULTS.md`](docs/RESULTS.md) (verdicts and
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

"""Feature-set registry and phase labelling shared by both experiments.

Tiers (mirroring the GEM benchmark):
  ``topology``    the authentic descriptor views (candidates)
  ``baseline``    the baselines named in the brief
  ``union``       all baselines concatenated
  ``incremental`` baselines_union + a topological view
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .features import RunFeatures

UNION_VIEWS = ["act_stats", "pca_stats", "weight_norm", "cka_final", "probe_acc"]

FEATURE_SETS: Dict[str, Tuple[str, List[str], str]] = {
    "x1": ("topology", ["x1"], "X1 functional graph of the units (|Pearson| > 0.40), R^10"),
    "x2": ("topology", ["x2"], "X2 Vietoris-Rips PH of the activation cloud (sig6 H0/H1), R^12"),
    "x1x2": ("topology", ["x1", "x2"], "authentic descriptor X1 + X2, R^22"),
    "x3": ("topology", ["x3"], "X3 label-free fusion [X1/|X1|, X2/|X2|], R^22"),
    "act_stats": ("baseline", ["act_stats"], "raw activation statistics (mean, variance, ...)"),
    "pca_stats": ("baseline", ["pca_stats"], "PCA effective dimension + explained variance"),
    "act_pca": ("baseline", ["act_stats", "pca_stats"], "activation statistics + PCA dimension"),
    "weight_norm": ("baseline", ["weight_norm"], "parameter norms (total and per component)"),
    "cka_final": ("baseline", ["cka_final"], "linear CKA to the final-model representation"),
    "probe_acc": ("baseline", ["probe_acc"], "linear-probe accuracy for the answer c"),
    "baselines_union": ("union", UNION_VIEWS, "all baselines concatenated"),
    "baselines_union+x1x2": ("incremental", UNION_VIEWS + ["x1", "x2"],
                             "baselines_union + X1 + X2"),
    "baselines_union+x3": ("incremental", UNION_VIEWS + ["x3"], "baselines_union + X3"),
}
TOPOLOGY = [k for k, v in FEATURE_SETS.items() if v[0] == "topology"]
BASELINES = [k for k, v in FEATURE_SETS.items() if v[0] == "baseline"]
INCREMENTAL = [k for k, v in FEATURE_SETS.items() if v[0] == "incremental"]


def signal_matrix(rf: RunFeatures, views: List[str], site: str) -> np.ndarray:
    """(T, d) matrix of the given views at one site (``"all"`` concatenates
    every site). ``weight_norm`` is site-independent and enters once."""
    T = len(rf.steps)
    cols = []
    for v in views:
        arr = rf.feats[v]
        if v == "weight_norm":
            cols.append(arr[:, 0])
        elif site == "all":
            cols.append(arr.reshape(T, -1))
        else:
            cols.append(arr[:, rf.sites.index(site)])
    return np.hstack(cols)


@dataclass
class PhaseThresholds:
    """Accuracy thresholds that define the phases (config key ``phases``)."""

    mem: float = 0.99          # train accuracy: memorized
    onset: float = 0.5         # held-out accuracy: generalization has started
    grok: float = 0.9          # held-out accuracy: grokked (transition time)
    grokked_min: float = 0.95  # held-out accuracy for the "grokked" class (Exp. 2)


THRESHOLDS = PhaseThresholds()


def set_phase_thresholds(**kw: float) -> None:
    """Override the defaults (called once by ``tsi.run`` from the config)."""
    global THRESHOLDS
    THRESHOLDS = PhaseThresholds(**kw)


@dataclass(frozen=True)
class Phases:
    """Checkpoint indices of the phase boundaries of one run (None = never)."""

    i_mem: Optional[int]     # first train accuracy >= THRESHOLDS.mem
    i_grok50: Optional[int]  # first held-out accuracy >= THRESHOLDS.onset (0.5)
    i_grok90: Optional[int]  # first held-out accuracy >= THRESHOLDS.grok (0.9)

    @property
    def grokked(self) -> bool:
        """Delayed generalization: memorized first (held-out accuracy still
        below the onset threshold, 0.5, when train accuracy reaches 0.99),
        then held-out accuracy >= the grok threshold (0.9). Runs that generalize while memorizing are not
        grokking runs: their post-memorization window starts after the
        transition."""
        return (self.i_mem is not None and self.i_grok90 is not None
                and self.i_grok50 is not None and self.i_grok50 > self.i_mem)


def _first(mask: np.ndarray) -> Optional[int]:
    hit = np.flatnonzero(mask)
    return int(hit[0]) if len(hit) else None


def run_phases(rf: RunFeatures) -> Phases:
    th = THRESHOLDS
    tr, te = rf.curves["train_acc"], rf.curves["test_acc"]
    return Phases(i_mem=_first(tr >= th.mem), i_grok50=_first(te >= th.onset),
                  i_grok90=_first(te >= th.grok))


def phase_labels(rf: RunFeatures) -> np.ndarray:
    """Per checkpoint: 0 = memorized (train >= 0.99, held-out < 0.5),
    1 = grokked (train >= 0.99, held-out >= 0.95), -1 = neither (defaults)."""
    th = THRESHOLDS
    tr, te = rf.curves["train_acc"], rf.curves["test_acc"]
    lab = np.full(len(tr), -1, dtype=int)
    lab[(tr >= th.mem) & (te < th.onset)] = 0
    lab[(tr >= th.mem) & (te >= th.grokked_min)] = 1
    return lab

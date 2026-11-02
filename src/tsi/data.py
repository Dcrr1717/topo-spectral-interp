"""Modular-arithmetic datasets (the grokking setup of Power et al. 2022).

Every example is the token sequence ``[a, b, =]`` with ``a, b`` in ``[0, p)``;
the model reads the answer ``c = op(a, b) mod p`` at the ``=`` position
(Nanda et al. 2023 layout). The vocabulary is ``p + 1`` tokens, where token
``p`` is ``=``. A fixed random fraction of the pairs is the training set, the
rest is the held-out set.

Operations (all from the task list of Power et al. 2022):
``add`` a + b, ``sub`` a - b, ``mul`` a * b (all pairs), and ``div`` a / b =
a * b^(p-2) (Fermat inverse, pairs with b != 0 only, so p * (p - 1) pairs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


def _div(a: np.ndarray, b: np.ndarray, p: int) -> np.ndarray:
    inv = np.array([pow(int(x), p - 2, p) for x in range(p)], dtype=np.int64)
    return (a * inv[b]) % p


OPS: Dict[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]] = {
    "add": lambda a, b, p: (a + b) % p,
    "sub": lambda a, b, p: (a - b) % p,
    "mul": lambda a, b, p: (a * b) % p,
    "div": _div,
}


@dataclass(frozen=True)
class ModularDataset:
    """All pairs, their labels and a train / test split."""

    p: int
    op: str
    tokens: np.ndarray  # (n_pairs, 3) int64: [a, b, =]
    labels: np.ndarray  # (n_pairs,) int64
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def vocab_size(self) -> int:
        return self.p + 1

    @property
    def n_pairs(self) -> int:
        return len(self.labels)


def make_dataset(p: int, op: str, train_frac: float, seed: int) -> ModularDataset:
    """Build the full table of ``op(a, b) mod p`` and split it at random."""
    if op not in OPS:
        raise ValueError(f"unknown op {op!r}; choose from {sorted(OPS)}")
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    ga, gb = np.meshgrid(np.arange(p), np.arange(1 if op == "div" else 0, p), indexing="ij")
    a, b = ga.ravel(), gb.ravel()
    tokens = np.stack([a, b, np.full_like(a, p)], axis=1).astype(np.int64)
    labels = OPS[op](a, b, p).astype(np.int64)
    n = len(labels)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(train_frac * n))
    return ModularDataset(p=p, op=op, tokens=tokens, labels=labels,
                          train_idx=np.sort(perm[:n_train]), test_idx=np.sort(perm[n_train:]))


def probe_indices(ds: ModularDataset, n_probe: int, seed: int) -> np.ndarray:
    """A fixed random subset of ALL pairs (train and held-out) whose activations
    are logged at every checkpoint. ``n_probe >= n_pairs`` returns every pair."""
    n = ds.n_pairs
    if n_probe >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed + 104729)
    return np.sort(rng.choice(n, size=n_probe, replace=False))

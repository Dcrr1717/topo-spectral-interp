"""Fourier progress measures of Nanda et al. (2023): restricted and excluded loss.

For modular addition, a grokked network computes logits
``l(a, b, c) ~ sum_k alpha_k cos(w_k (a + b - c))`` over a few *key
frequencies* ``w_k = 2 pi k / p`` (Nanda et al. 2023, Section 4). Their two
progress measures (Section 5.1) test how far the train-set performance
already rests on that circuit:

* **restricted loss**: keep ONLY the key-frequency components of the logits
  and evaluate the loss. It falls when the generalizing circuit forms, before
  held-out loss moves.
* **excluded loss**: REMOVE the key-frequency components and evaluate the
  loss on the training set. It rises as the model's train performance comes
  to depend on the circuit, i.e. as memorization is replaced.

Implementation (logit space, exact projection)
-----------------------------------------------
Every operation is written as a cyclic group law: ``w(c) = s(a, b)`` with
``s = u(a) +- v(b) mod n``.

* ``add`` / ``sub`` use the identity coordinates with n = p.
* ``mul`` / ``div`` use discrete-log coordinates (a primitive root g, n =
  p - 1) on the nonzero tokens. The multiplicative group is cyclic, so its
  irreducible representations are again Fourier characters (Chughtai et al.
  2023). Pairs involving 0 are not in the group and are left out of these
  measures.

The "key directions" ``cos / sin(2 pi k (s - w(c)) / n)`` depend on (pair,
output) only through the offset ``d = s - w(c) mod n``. On the complete grid
of pairs, the projection of the logits onto functions of ``d`` is the mean
logit at each offset, ``M(d)``. The key-frequency part is ``M`` filtered to
the key frequencies. So the restricted logits are ``g(d)`` and the excluded
logits are ``l - g(d)``; they sum to the original logits.

* **Key frequencies.** They are read from the FINAL checkpoint, as in Nanda
  et al.: the smallest set of frequencies holding >= ``energy_frac`` of the
  non-constant energy of ``M``, capped at ``max_keys``. This makes both
  measures retrospective, like CKA to the final model.
* **Which labels are used.** Both losses are evaluated on training pairs
  only (softmax over group outputs). They use no held-out labels; the
  projection uses only the model's outputs on all inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

FOURIER_NAMES = ["log10_restricted_loss", "log10_excluded_loss"]


def primitive_root(p: int) -> int:
    """Smallest generator of the multiplicative group mod a prime p."""
    phi = p - 1
    factors, m, f = [], phi, 2
    while f * f <= m:
        if m % f == 0:
            factors.append(f)
            while m % f == 0:
                m //= f
        f += 1
    if m > 1:
        factors.append(m)
    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in factors):
            return g
    raise ValueError(f"no primitive root for p={p}")


@dataclass(frozen=True)
class GroupView:
    """The dataset pairs seen as a cyclic group law w(c) = s(a, b) mod n."""

    n: int
    pair_idx: np.ndarray  # dataset rows that belong to the group grid
    s: np.ndarray         # (N,) group coordinate of the combined input
    out_idx: np.ndarray   # (n,) logit column (token) of each group element
    target: np.ndarray    # (N,) group coordinate of the true answer


def group_view(tokens: np.ndarray, labels: np.ndarray, op: str, p: int) -> GroupView:
    a, b = tokens[:, 0].astype(np.int64), tokens[:, 1].astype(np.int64)
    if op in ("add", "sub"):
        sign = 1 if op == "add" else -1
        return GroupView(n=p, pair_idx=np.arange(len(a)), s=(a + sign * b) % p,
                         out_idx=np.arange(p), target=labels.astype(np.int64))
    if op in ("mul", "div"):
        g = primitive_root(p)
        n = p - 1
        exp = np.array([pow(g, k, p) for k in range(n)], dtype=np.int64)  # w -> token
        log = np.full(p, -1, dtype=np.int64)
        log[exp] = np.arange(n)
        keep = np.flatnonzero((a != 0) & (b != 0))
        sign = 1 if op == "mul" else -1
        s = (log[a[keep]] + sign * log[b[keep]]) % n
        return GroupView(n=n, pair_idx=keep, s=s, out_idx=exp, target=log[labels[keep]])
    raise ValueError(f"no group view for op {op!r}")


def _offsets(gv: GroupView) -> np.ndarray:
    return (gv.s[:, None] - np.arange(gv.n)[None, :]) % gv.n  # (N, n)


def offset_profile(logits: np.ndarray, gv: GroupView) -> np.ndarray:
    """M(d): mean group-output logit at each offset d = s - w(c) (length n)."""
    L = logits[gv.pair_idx][:, gv.out_idx].astype(np.float64)
    d = _offsets(gv)
    return np.bincount(d.ravel(), weights=L.ravel(), minlength=gv.n) / len(gv.pair_idx)


def frequency_energy(M: np.ndarray) -> np.ndarray:
    """Energy of each frequency k = 0..n//2 of the profile (two-sided weights)."""
    n = len(M)
    E = np.abs(np.fft.rfft(M)) ** 2
    w = np.full(len(E), 2.0)
    w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    return E * w


def key_frequencies(M: np.ndarray, energy_frac: float = 0.9, max_keys: int = 8) -> List[int]:
    """Smallest set of non-constant frequencies holding >= energy_frac of the
    non-constant energy of M (at most max_keys)."""
    E = frequency_energy(M)[1:]
    if E.sum() <= 0:
        return [1]
    order = np.argsort(E)[::-1]
    cum = np.cumsum(E[order]) / E.sum()
    k = int(np.searchsorted(cum, energy_frac) + 1)
    return sorted(int(i) + 1 for i in order[: max(1, min(k, max_keys))])


def _ce(logits: np.ndarray, target: np.ndarray) -> float:
    z = logits - logits.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1))
    return float((lse - z[np.arange(len(target)), target]).mean())


def restricted_excluded_loss(logits: np.ndarray, gv: GroupView, train_mask: np.ndarray,
                             keys: List[int]) -> tuple[float, float]:
    """(restricted loss, excluded loss) on the training pairs of the group.

    ``logits``: (n_pairs, p) model outputs on every dataset pair.
    ``train_mask``: (n_pairs,) bool, True for training pairs.
    """
    L = logits[gv.pair_idx][:, gv.out_idx].astype(np.float64)
    d = _offsets(gv)
    M = np.bincount(d.ravel(), weights=L.ravel(), minlength=gv.n) / len(gv.pair_idx)
    F = np.fft.rfft(M)
    mask = np.zeros(len(F), dtype=bool)
    mask[[k for k in keys if 0 < k < len(F)]] = True
    g = np.fft.irfft(np.where(mask, F, 0.0), n=gv.n)
    R = g[d]
    tr = train_mask[gv.pair_idx]
    return _ce(R[tr], gv.target[tr]), _ce((L - R)[tr], gv.target[tr])

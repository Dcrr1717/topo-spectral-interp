"""Fourier progress measures (restricted / excluded loss, Nanda et al. 2023)."""
import numpy as np

from tsi.data import make_dataset
from tsi.fourier import (frequency_energy, group_view, key_frequencies, offset_profile,
                         primitive_root, restricted_excluded_loss)


def _key_logits(ds_tokens: np.ndarray, op: str, p: int, amps: dict) -> np.ndarray:
    """Logits that are pure key-frequency components in the op's group coordinates."""
    gv = group_view(ds_tokens, np.zeros(len(ds_tokens), dtype=np.int64), op, p)
    logits = np.zeros((len(ds_tokens), p))
    d = (gv.s[:, None] - np.arange(gv.n)[None, :]) % gv.n
    block = sum(a * np.cos(2 * np.pi * k * d / gv.n) for k, a in amps.items())
    logits[np.ix_(gv.pair_idx, gv.out_idx)] = block
    return logits


def test_primitive_root_generates_group() -> None:
    for p in (7, 13, 59):
        g = primitive_root(p)
        assert sorted(pow(g, k, p) for k in range(p - 1)) == list(range(1, p))


def test_group_view_matches_labels() -> None:
    for op in ("add", "sub", "mul", "div"):
        ds = make_dataset(13, op, 0.5, 0)
        gv = group_view(ds.tokens, ds.labels, op, 13)
        lab = ds.labels[gv.pair_idx]
        assert np.all(gv.out_idx[gv.s] == lab) and np.all(gv.out_idx[gv.target] == lab)
        n_pairs = 13 * 13 if op in ("add", "sub") else 12 * 12
        assert len(gv.pair_idx) == n_pairs


def test_recovers_key_frequencies_and_limits() -> None:
    for op, p in (("add", 13), ("sub", 13), ("div", 13)):
        ds = make_dataset(p, op, 0.5, 0)
        gv = group_view(ds.tokens, ds.labels, op, p)
        logits = _key_logits(ds.tokens, op, p, {2: 6.0, 5: 3.0})
        M = offset_profile(logits, gv)
        assert key_frequencies(M) == [2, 5]
        E = frequency_energy(M)
        assert E[[2, 5]].sum() / E[1:].sum() > 0.999
        train = np.zeros(len(ds.labels), dtype=bool)
        train[ds.train_idx] = True
        r, e = restricted_excluded_loss(logits, gv, train, [2, 5])
        tr = train[gv.pair_idx]
        L = logits[gv.pair_idx][:, gv.out_idx][tr]
        z = L - L.max(1, keepdims=True)
        full = float((np.log(np.exp(z).sum(1)) - z[np.arange(len(z)), gv.target[tr]]).mean())
        assert abs(r - full) < 1e-9                    # nothing outside the keys
        assert abs(e - np.log(gv.n)) < 1e-9            # nothing left: uniform logits


def test_memorization_is_not_restricted() -> None:
    rng = np.random.default_rng(0)
    p = 13
    ds = make_dataset(p, "add", 0.5, 0)
    gv = group_view(ds.tokens, ds.labels, "add", p)
    train = np.zeros(len(ds.labels), dtype=bool)
    train[ds.train_idx] = True
    logits = rng.standard_normal((len(ds.labels), p))
    logits[ds.train_idx, ds.labels[ds.train_idx]] += 12.0   # lookup-table memorization
    r, e = restricted_excluded_loss(logits, gv, train, [3])
    assert e < 0.1  # memorized train answers survive removing the keys
    # a lookup-table bump at the answer (offset d = 0) has energy at EVERY
    # frequency, so one key frequency keeps only a small part of it
    assert r > 0.6 * np.log(p) and r > 10 * e

"""Fidelity and sanity tests of the authentic descriptor port."""
import networkx as nx
import numpy as np
from ripser import ripser
from sklearn.decomposition import PCA

from tsi.descriptor.authentic import (X1_NAMES, X2_NAMES, X3_NAMES, authentic_views,
                                      desc_x3, graph_invariants, sig6, x2_rips_pointcloud)


# --- verbatim copies of the ORIGINAL neural-activation descriptor -----------------
# github.com/Dcrr1717/multiview-nn-descriptor, phase16_reextract.py (Reinoso &
# Escudero 2026), used here as the reference the port must reproduce.
def orig_desc_x1(act: np.ndarray, threshold: float = 0.40, max_n: int = 80) -> np.ndarray:
    N = min(act.shape[1], max_n); corr = np.corrcoef(act[:, :N].T); corr = np.nan_to_num(corr)
    np.fill_diagonal(corr, 0)
    G = nx.from_numpy_array((np.abs(corr) > threshold).astype(float))
    deg = np.array([d for _, d in G.degree()])
    if len(deg) == 0:
        deg = np.array([0.])
    try:
        bc = np.array(list(nx.betweenness_centrality(G).values()))
    except Exception:
        bc = np.zeros(len(deg))
    return np.array([nx.density(G), nx.average_clustering(G), nx.transitivity(G), deg.mean(),
                     deg.std(), deg.max() / max(1, len(deg)),
                     float(nx.number_connected_components(G)) / max(G.number_of_nodes(), 1),
                     G.number_of_edges() / max(G.number_of_nodes(), 1), bc.mean(), bc.std()],
                    dtype=np.float64)


def orig_desc_x2(act: np.ndarray, n_pts: int = 50) -> np.ndarray:
    A = act[:n_pts]; nc = min(15, A.shape[1], A.shape[0] - 1)
    if nc < 2:
        return np.zeros(12)
    Ar = PCA(n_components=nc, random_state=42).fit_transform(A)
    Ar = (Ar - Ar.min()) / (Ar.max() - Ar.min() + 1e-8)
    dgms = ripser(Ar, maxdim=1)["dgms"]; h0 = dgms[0]; h0 = h0[h0[:, 1] < np.inf]
    h1 = dgms[1]; h1 = h1[h1[:, 1] < np.inf]

    def s6(d: np.ndarray) -> np.ndarray:
        if len(d) == 0:
            return np.zeros(6)
        p = d[:, 1] - d[:, 0]
        return np.array([p.mean(), p.std(), p.max(), p.sum(), float(len(p)), np.percentile(p, 75)])
    return np.concatenate([s6(h0), s6(h1)])


def _acts(seed: int = 0, n: int = 300, units: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 6))
    W = rng.standard_normal((6, units))
    return z @ W + 0.5 * rng.standard_normal((n, units))


def test_x1_matches_original_desc_x1() -> None:
    for seed in range(3):
        a = _acts(seed)
        np.testing.assert_allclose(graph_invariants(a), orig_desc_x1(a), rtol=0, atol=1e-12)


def test_x1_dead_units_match_original_nan_handling() -> None:
    a = _acts(1)
    a[:, 3] = 0.0  # a dead ReLU unit
    a[:, 10] = 2.5
    np.testing.assert_allclose(graph_invariants(a), orig_desc_x1(a), atol=1e-12)
    G_edges_dead = graph_invariants(a[:, [3, 10]])
    assert G_edges_dead[0] == 0.0  # no edges between dead units


def test_x2_matches_original_desc_x2() -> None:
    for seed in range(3):
        a = _acts(seed)
        np.testing.assert_allclose(x2_rips_pointcloud(a), orig_desc_x2(a), atol=1e-12)


def test_shapes_and_names() -> None:
    v = authentic_views(_acts())
    assert v["x1"].shape == (len(X1_NAMES),) == (10,)
    assert v["x2"].shape == (len(X2_NAMES),) == (12,)
    assert v["x3"].shape == (len(X3_NAMES),) == (22,)
    np.testing.assert_allclose(np.linalg.norm(v["x3"][:10]), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(v["x3"][10:]), 1.0, atol=1e-6)
    np.testing.assert_allclose(desc_x3(v["x1"], v["x2"]), v["x3"])


def test_x1_invariant_to_sample_and_unit_permutation() -> None:
    a = _acts(2, units=60)
    rng = np.random.default_rng(0)
    base = graph_invariants(a)
    np.testing.assert_allclose(graph_invariants(a[rng.permutation(len(a))]), base, atol=1e-10)
    np.testing.assert_allclose(graph_invariants(a[:, rng.permutation(a.shape[1])]), base,
                               atol=1e-10)


def test_x2_invariant_to_unit_permutation() -> None:
    a = _acts(3)
    perm = np.random.default_rng(1).permutation(a.shape[1])
    np.testing.assert_allclose(x2_rips_pointcloud(a[:, perm]), x2_rips_pointcloud(a),
                               atol=1e-8)


def test_x2_detects_a_loop() -> None:
    rng = np.random.default_rng(0)
    th = rng.uniform(0, 2 * np.pi, 50)
    circle = np.c_[np.cos(th), np.sin(th)] @ rng.standard_normal((2, 32))
    circle += 0.02 * rng.standard_normal(circle.shape)
    blob = rng.standard_normal((50, 32))
    i = X2_NAMES.index("H1_pers_max")
    assert x2_rips_pointcloud(circle)[i] > 3 * x2_rips_pointcloud(blob)[i]


def test_degenerate_inputs() -> None:
    assert np.all(x2_rips_pointcloud(np.ones((50, 8))) == 0)
    assert np.all(sig6(np.zeros(0)) == 0)
    x1 = graph_invariants(np.zeros((40, 16)))
    assert np.all(np.isfinite(x1)) and x1[0] == 0.0

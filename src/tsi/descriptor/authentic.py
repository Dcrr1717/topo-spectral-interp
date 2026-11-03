"""Authentic multiview descriptor X1 / X2 / X3 for neural activations (port).

Sources
-------
* X1 (`graph_invariants`), `sig6` and the X3 fusion (`desc_x3`): verbatim from
  https://github.com/Dcrr1717/gem-geophysical-descriptor, file
  ``src/descriptor/authentic.py``. That module is itself a verbatim port of
  ``topo-spectral-geofluids/tsg/descriptors.py`` in
  https://github.com/Dcrr1717/multiview-nn-descriptor, which ports the
  neural-activation descriptor of Reinoso & Escudero (2026) (``desc_x1``,
  ``desc_x2``, ``desc_x3`` in ``phase16_reextract.py``).
* X2 (`x2_rips_pointcloud`): verbatim from the same ``tsg/descriptors.py``.
  GEM's ``authentic.py`` documents this Vietoris-Rips variant as "the original
  Vietoris-Rips desc_x2" but does not port it. GEM applies X2 to 2D fields
  through the sublevel-set filtration of a periodic cubical complex. An
  activation matrix has no spatial grid, so that filtration does not apply.
  Here we are back in the setting where the descriptor started (an
  activation matrix, samples x units), and there X2 IS the Rips persistence
  of the activation point cloud.

The function bodies are unchanged. Only type annotations on the signatures,
the docstrings and the comments marked "[TSI port]" are new.

Views (applied to an activation matrix ``act`` of shape (samples, units))
--------------------------------------------------------------------------
* X1 (R^10): functional graph of the units. An edge joins two units whose
  activations over the samples have |Pearson| > theta = 0.40. The units are
  capped at the first ``max_n = 80``. The ten invariants: density, average
  clustering, transitivity, degree mean / std / max-fraction, fraction of
  connected components, edges per node, betweenness mean / std. GEM's
  ``var_floor`` gives dead units (std <= 1e-6, e.g. ReLU units that never
  fire on the probe set) no edges. Without it, round-off correlations would
  produce a random graph. The original ``desc_x1`` mapped their NaN
  correlations to 0, which is the same result.
* X2 (R^12): Vietoris-Rips persistent homology (``ripser``, maxdim 1) of the
  first ``n_pts = 50`` samples. The point cloud is projected on its top
  ``r = 15`` principal components and min-max scaled to [0, 1]. The output
  is the ``sig6`` summary (mean, std, max, sum, count, p75 of the finite
  persistences) of H0 and H1.
* X3 (R^22): the label-free fusion of the original ``desc_x3``. Each block
  is L2-normalised and the blocks are concatenated:
  X3 = [X1/|X1|, X2/|X2|].
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

X1_NAMES = ["density", "avg_clustering", "transitivity", "deg_mean", "deg_std",
            "deg_max_frac", "components_frac", "edges_per_node", "betw_mean", "betw_std"]
X2_NAMES = [f"{h}_{s}" for h in ("H0", "H1")
            for s in ("pers_mean", "pers_std", "pers_max", "pers_sum", "count", "pers_p75")]
X3_NAMES = [f"x1n_{n}" for n in X1_NAMES] + [f"x2n_{n}" for n in X2_NAMES]


# ── X1: functional graph ─────────────────────────────────────────────────
def graph_invariants(act: np.ndarray, theta: float = 0.40, max_n: int = 80,
                     corr: str = "pearson", var_floor: float = 1e-6) -> np.ndarray:
    """`desc_x1` of the original repository applied to an activation-like matrix.

    act: (samples, units). Units whose temporal std is below `var_floor`
    (steady regions) carry no correlation information and get no edges;
    without this floor, round-off fluctuations of a converged steady state
    would produce a random graph.

    [TSI port] Here ``act`` is a real activation matrix (probe inputs x
    hidden units), the original setting; "steady regions" are dead units.
    """
    N = min(act.shape[1], max_n)
    A = act[:, :N]
    live = A.std(axis=0) > var_floor
    C = np.zeros((N, N))
    if live.sum() >= 2:
        sub = A[:, live]
        Cs = np.corrcoef(sub.T) if corr == "pearson" else spearmanr(sub)[0]
        C[np.ix_(live, live)] = np.nan_to_num(np.atleast_2d(Cs))
    np.fill_diagonal(C, 0)
    G = nx.from_numpy_array((np.abs(C) > theta).astype(float))
    deg = np.array([d for _, d in G.degree()]) if G.number_of_nodes() else np.array([0.])
    try:
        bc = np.array(list(nx.betweenness_centrality(G).values()))
    except Exception:
        bc = np.zeros(len(deg))
    return np.array([nx.density(G), nx.average_clustering(G), nx.transitivity(G),
                     deg.mean(), deg.std(), deg.max() / max(1, len(deg)),
                     float(nx.number_connected_components(G)) / max(G.number_of_nodes(), 1),
                     G.number_of_edges() / max(G.number_of_nodes(), 1), bc.mean(), bc.std()],
                    dtype=np.float64)


# ── X2: persistent homology ──────────────────────────────────────────────
def sig6(pers: np.ndarray) -> np.ndarray:
    """Original `sig6` summary of a vector of persistences."""
    if len(pers) == 0:
        return np.zeros(6)
    p = np.asarray(pers)
    return np.array([p.mean(), p.std(), p.max(), p.sum(), float(len(p)), np.percentile(p, 75)])


def x2_rips_pointcloud(A: np.ndarray, r: int = 15, n_pts: int = 50) -> np.ndarray:
    """Variant: the original Vietoris-Rips `desc_x2` applied to an
    activation-like matrix A (samples x units), e.g. `patch_matrix(field)`
    (snapshot mode) or `region_series(window)` (window mode).

    [TSI port] Here A is the activation matrix itself (probe inputs x
    units), exactly as in the original ``desc_x2(act, n_pts=50)``.
    """
    from ripser import ripser
    A = A[:n_pts]
    nc = min(r, A.shape[1], A.shape[0] - 1)
    if nc < 2 or A.std() < 1e-12:
        return np.zeros(12)
    Ar = PCA(n_components=nc, random_state=42).fit_transform(A)
    Ar = (Ar - Ar.min()) / (Ar.max() - Ar.min() + 1e-8)
    dgms = ripser(Ar, maxdim=1)["dgms"]
    h0 = dgms[0][np.isfinite(dgms[0][:, 1])]
    h1 = dgms[1][np.isfinite(dgms[1][:, 1])]
    return np.concatenate([sig6(h0[:, 1] - h0[:, 0]), sig6(h1[:, 1] - h1[:, 0])])


# ── X3: label-free fusion (original desc_x3) ─────────────────────────────
def desc_x3(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    n1 = x1 / (np.linalg.norm(x1) + 1e-8)
    n2 = x2 / (np.linalg.norm(x2) + 1e-8)
    return np.concatenate([n1, n2])


def authentic_views(act: np.ndarray, theta: float = 0.40, max_n: int = 80,
                    corr: str = "pearson", r: int = 15, n_pts: int = 50) -> dict[str, np.ndarray]:
    """[TSI port] Convenience wrapper: X1, X2 and X3 of one activation matrix."""
    x1 = graph_invariants(act, theta=theta, max_n=max_n, corr=corr)
    x2 = x2_rips_pointcloud(act, r=r, n_pts=n_pts)
    return {"x1": x1, "x2": x2, "x3": desc_x3(x1, x2)}

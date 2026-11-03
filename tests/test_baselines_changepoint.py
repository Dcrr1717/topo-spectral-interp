"""Baselines, change-point detector and evaluation helpers."""
import numpy as np

from tsi.baselines import (activation_stats, linear_cka, linear_probe_accuracy, pca_stats)
from tsi.changepoint import detect, onset_index, single_changepoint, standardize_columns
from tsi.evaluation import (bootstrap_mean_diff, evaluate_grouped,
                            evaluate_grouped_regression, paired_bootstrap_margin,
                            paired_bootstrap_r2)


def test_linear_cka_properties() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 20))
    Q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    assert abs(linear_cka(X, X) - 1) < 1e-10
    assert abs(linear_cka(X, 3.0 * X @ Q + 5.0) - 1) < 1e-10  # orthogonal + scale + shift
    assert linear_cka(X, rng.standard_normal((200, 20))) < 0.3
    assert linear_cka(X, np.zeros((200, 5))) == 0.0


def test_linear_probe_accuracy() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    X = np.eye(5)[y] * 3 + 0.1 * rng.standard_normal((400, 5))
    assert linear_probe_accuracy(X, y, 5) > 0.98
    assert linear_probe_accuracy(rng.standard_normal((400, 5)), y, 5) < 0.4


def test_activation_and_pca_stats() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 10))
    pr = pca_stats(X)
    assert 9.0 < pr[0] <= 10.0 and 9.0 < pr[1] <= 10.0
    low = rng.standard_normal((2000, 1)) @ rng.standard_normal((1, 10))
    assert pca_stats(low)[0] < 1.01 and pca_stats(low)[2] > 0.99
    st = activation_stats(np.maximum(X, 0))
    assert 0.4 < st[4] < 0.6  # half of ReLU activations are exactly zero
    assert np.all(pca_stats(np.ones((5, 3))) == 0)


def test_changepoint_finds_step() -> None:
    rng = np.random.default_rng(0)
    z = np.r_[np.zeros(40), np.ones(60)][:, None] + 0.05 * rng.standard_normal((100, 3))
    tau, r2 = single_changepoint(standardize_columns(z))
    assert tau == 40 and r2 > 0.95
    cp = detect(z)
    assert cp.index == 40 and 38 <= cp.onset10 <= 41 and 38 <= cp.onset50 <= 41
    noise = rng.standard_normal((100, 3))
    assert detect(noise).r2_step < 0.2
    assert detect(np.ones((30, 2))).r2_step == 0.0


def test_onset_of_ramp() -> None:
    u = np.r_[np.zeros(20), np.linspace(0, 1, 21), np.ones(20)]
    assert 21 <= onset_index(u, 0.1) <= 24
    assert 29 <= onset_index(u, 0.5) <= 32


def test_grouped_cv_and_bootstrap() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 20)
    y = np.tile(np.r_[np.zeros(10, int), np.ones(10, int)], 8)
    good = y[:, None] + 0.1 * rng.standard_normal((160, 2))
    bad = rng.standard_normal((160, 2))
    rg = evaluate_grouped(good, y, groups, n_repeats=2, n_estimators=20)
    rb = evaluate_grouped(bad, y, groups, n_repeats=2, n_estimators=20)
    assert rg["best_accuracy"] > 0.95 and rb["best_accuracy"] < 0.75
    m = paired_bootstrap_margin(rg, rb, y, groups, n_boot=300)
    assert m["margin"] > 0.2 and m["ci95"][0] > 0
    same = paired_bootstrap_margin(rg, rg, y, groups, n_boot=100)
    assert same["margin"] == 0 and same["ci95"] == [0.0, 0.0]


def test_grouped_regression_and_bootstrap() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 25)
    x = rng.standard_normal((150, 1))
    y = 3 * x[:, 0] + 0.1 * rng.standard_normal(150)
    ra = evaluate_grouped_regression(x, y, groups, n_estimators=20)
    rb = evaluate_grouped_regression(rng.standard_normal((150, 1)), y, groups, n_estimators=20)
    assert ra["best_r2"] > 0.95 and rb["best_r2"] < 0.2
    c = paired_bootstrap_r2(ra, rb, y, groups, n_boot=200)
    assert c["ci95"][0] > 0.5
    d = bootstrap_mean_diff(np.ones(10), np.zeros(10), n_boot=50)
    assert d["mean"] == 1.0 and d["ci95"] == [1.0, 1.0]

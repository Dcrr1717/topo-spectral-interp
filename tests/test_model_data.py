"""Model, data and training-loop tests (tiny, CPU, offline)."""
from pathlib import Path
import numpy as np
import torch

from tsi.data import make_dataset, probe_indices
from tsi.features import DescriptorConfig, load_run_features, run_features, save_run_features
from tsi.model import ModelConfig, Transformer, site_names, weight_norms
from tsi.train import RunConfig, train_run


def test_dataset_labels_and_split() -> None:
    ds = make_dataset(7, "sub", 0.5, seed=0)
    assert ds.tokens.shape == (49, 3) and np.all(ds.tokens[:, 2] == 7)
    a, b = ds.tokens[:, 0], ds.tokens[:, 1]
    assert np.all(ds.labels == (a - b) % 7)
    assert len(ds.train_idx) + len(ds.test_idx) == 49
    assert not set(ds.train_idx) & set(ds.test_idx)
    mul = make_dataset(5, "mul", 0.5, 0)
    assert np.all(mul.labels == (mul.tokens[:, 0] * mul.tokens[:, 1]) % 5)
    p = probe_indices(ds, 20, seed=0)
    assert len(p) == 20 and len(np.unique(p)) == 20
    assert len(probe_indices(ds, 1000, 0)) == 49


def test_fast_forward_matches_reference() -> None:
    torch.manual_seed(0)
    for n_layers in (1, 2):
        m = Transformer(ModelConfig(vocab_size=12, n_out=11, d_model=32, n_heads=4, d_mlp=64,
                                    n_layers=n_layers))
        t = torch.randint(0, 11, (64, 3))
        t[:, 2] = 11
        torch.testing.assert_close(m(t), m.forward_reference(t), atol=1e-5, rtol=1e-5)


def test_cache_sites() -> None:
    m = Transformer(ModelConfig(vocab_size=12, n_out=11, d_model=32, n_heads=4, d_mlp=64,
                                n_layers=2))
    cache: dict = {}
    t = torch.randint(0, 11, (10, 3))
    m(t, cache=cache)
    assert list(cache) == site_names(2)
    assert cache["mlp_1"].shape == (10, 64) and cache["resid_post_0"].shape == (10, 32)
    wn = weight_norms(m)
    assert set(wn) == {"embed", "attn", "mlp", "unembed", "total"} and wn["total"] > 0


def test_train_run_and_features_roundtrip(tmp_path: Path) -> None:
    rc = RunConfig(op="add", seed=0, p=7, train_frac=0.6, d_model=16, n_heads=2, d_mlp=32,
                   n_steps=40, log_every=10, n_probe=40)
    rec = train_run(rc)
    assert list(rec.steps) == [0, 10, 20, 30, 40]
    assert rec.acts["mlp_0"].shape == (5, 40, 32)
    assert rec.train_acc[-1] >= rec.train_acc[0]
    rf = run_features(rec, DescriptorConfig(n_pts=30, n_snapshots=3))
    assert rf.feats["x1"].shape == (5, 4, 10)
    assert rf.feats["x2"].shape == (5, 4, 12)
    assert rf.feats["weight_norm"].shape == (5, 1, 5)
    assert rf.feats["restricted_loss"].shape == (5, 1, 1)
    assert rf.feats["excluded_loss"].shape == (5, 1, 1)
    assert rec.logits.shape == (5, 49, 7) and rec.is_train.sum() == round(0.6 * 49)
    assert 1 <= len(rf.meta["fourier_keys"]) <= 8
    np.testing.assert_allclose(rf.feats["cka_final"][-1, :, 0], 1.0, atol=1e-6)
    path = tmp_path / "run.npz"
    save_run_features(rf, str(path))
    back = load_run_features(str(path))
    assert back.meta["run_id"] == "grok_add_s0" and back.sites == rf.sites
    for k in rf.feats:
        np.testing.assert_allclose(back.feats[k], rf.feats[k])
    assert back.snapshots["resid_post_0"].shape[1] == 30

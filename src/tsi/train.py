"""Train one small transformer on modular arithmetic and log its activations.

Full-batch AdamW with strong weight decay (the grokking recipe of Power et al.
2022 / Nanda et al. 2023). Every ``log_every`` steps the run records train /
held-out loss and accuracy, the weight norms and the activations of every
site (``tsi.model.site_names``) on a fixed probe set of input pairs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .data import ModularDataset, make_dataset, probe_indices
from .model import ModelConfig, Transformer, site_names, weight_norms

WEIGHT_NORM_NAMES: List[str] = ["embed", "attn", "mlp", "unembed", "total"]


@dataclass(frozen=True)
class RunConfig:
    """Everything that defines one training run."""

    op: str
    seed: int
    p: int = 59
    train_frac: float = 0.4
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 1
    lr: float = 1e-3
    weight_decay: float = 1.0
    betas: tuple[float, float] = (0.9, 0.98)
    warmup_steps: int = 10
    n_steps: int = 6000
    log_every: int = 50
    n_probe: int = 1024
    kind: str = "grok"  # "grok" or "control" (a run not expected to grok)

    @property
    def run_id(self) -> str:
        return f"{self.kind}_{self.op}_s{self.seed}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["betas"] = list(self.betas)
        d["run_id"] = self.run_id
        return d


@dataclass
class RunRecord:
    """Logged trajectory of one run (activations are float32, kept in memory)."""

    config: RunConfig
    steps: np.ndarray
    train_loss: np.ndarray
    test_loss: np.ndarray
    train_acc: np.ndarray
    test_acc: np.ndarray
    weight_norm: np.ndarray  # (T, len(WEIGHT_NORM_NAMES))
    acts: Dict[str, np.ndarray]  # site -> (T, n_probe, units)
    probe_idx: np.ndarray
    probe_labels: np.ndarray
    probe_is_train: np.ndarray
    sites: List[str] = field(default_factory=list)
    logits: np.ndarray = field(default_factory=lambda: np.zeros(0))  # (T, n_pairs, p)
    is_train: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def _metrics(logits: torch.Tensor, Y: torch.Tensor) -> tuple[float, float]:
    return (float(F.cross_entropy(logits, Y)),
            float((logits.argmax(dim=1) == Y).float().mean()))


def train_run(cfg: RunConfig, verbose: bool = False) -> RunRecord:
    torch.manual_seed(cfg.seed)
    ds: ModularDataset = make_dataset(cfg.p, cfg.op, cfg.train_frac, cfg.seed)
    mcfg = ModelConfig(vocab_size=ds.vocab_size, n_out=cfg.p, d_model=cfg.d_model,
                       n_heads=cfg.n_heads, d_mlp=cfg.d_mlp, n_layers=cfg.n_layers)
    model = Transformer(mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=cfg.betas)
    X = torch.from_numpy(ds.tokens)
    Y = torch.from_numpy(ds.labels)
    tr, te = torch.from_numpy(ds.train_idx), torch.from_numpy(ds.test_idx)
    Xtr, Ytr, Yte = X[tr], Y[tr], Y[te]
    pidx = probe_indices(ds, cfg.n_probe, cfg.seed)
    Xp = X[torch.from_numpy(pidx)]
    sites = site_names(cfg.n_layers)

    log: Dict[str, List[Any]] = {k: [] for k in
                                 ("steps", "train_loss", "test_loss", "train_acc", "test_acc",
                                  "weight_norm")}
    acts: Dict[str, List[np.ndarray]] = {s: [] for s in sites}
    all_logits: List[np.ndarray] = []  # logits on every pair (Fourier progress measures)

    def checkpoint(step: int) -> None:
        model.eval()
        with torch.no_grad():
            logits = model(X)
            trl, tra = _metrics(logits[tr], Ytr)
            tel, tea = _metrics(logits[te], Yte)
            cache: Dict[str, torch.Tensor] = {}
            model(Xp, cache=cache)
        model.train()
        all_logits.append(logits.numpy().astype(np.float32))
        wn = weight_norms(model)
        log["steps"].append(step)
        log["train_loss"].append(trl)
        log["test_loss"].append(tel)
        log["train_acc"].append(tra)
        log["test_acc"].append(tea)
        log["weight_norm"].append([wn[k] for k in WEIGHT_NORM_NAMES])
        for s in sites:
            acts[s].append(cache[s].numpy().astype(np.float32))
        if verbose:
            print(f"  [{cfg.run_id}] step {step:6d} train {tra:.3f} test {tea:.3f} "
                  f"|W| {wn['total']:.1f}", flush=True)

    for step in range(cfg.n_steps + 1):
        if step % cfg.log_every == 0 or step == cfg.n_steps:
            checkpoint(step)
        if step == cfg.n_steps:
            break
        for g in opt.param_groups:
            g["lr"] = cfg.lr * min(1.0, (step + 1) / max(1, cfg.warmup_steps))
        loss = F.cross_entropy(model(Xtr), Ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    is_train = np.zeros(ds.n_pairs, dtype=bool)
    is_train[ds.train_idx] = True
    return RunRecord(
        config=cfg,
        steps=np.asarray(log["steps"], dtype=np.int64),
        train_loss=np.asarray(log["train_loss"]), test_loss=np.asarray(log["test_loss"]),
        train_acc=np.asarray(log["train_acc"]), test_acc=np.asarray(log["test_acc"]),
        weight_norm=np.asarray(log["weight_norm"]),
        acts={s: np.stack(v) for s, v in acts.items()},
        probe_idx=pidx, probe_labels=ds.labels[pidx], probe_is_train=is_train[pidx],
        sites=sites, logits=np.stack(all_logits), is_train=is_train)

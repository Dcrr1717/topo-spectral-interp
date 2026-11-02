"""A small decoder-only transformer for modular arithmetic.

The architecture follows Nanda et al. (2023): learned token and positional
embeddings, ``n_layers`` pre-residual blocks of causal multi-head attention and
a ReLU MLP, no LayerNorm, an untied unembedding read at the last (``=``)
position. Power et al. (2022) used the same family with two layers.

Activation sites (all at the ``=`` position unless stated otherwise) that
``forward(..., cache=...)`` records:

``embed``           token + positional embedding of the first operand ``a``
                    (position 0), i.e. the learned representation of ``a``
``resid_mid_{l}``   residual stream after the attention of block ``l``
``mlp_{l}``         post-ReLU hidden units of the MLP of block ``l``
``resid_post_{l}``  residual stream after block ``l``
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_out: int
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 1
    n_ctx: int = 3


def site_names(n_layers: int) -> List[str]:
    """Activation sites in forward order."""
    names = ["embed"]
    for layer in range(n_layers):
        names += [f"resid_mid_{layer}", f"mlp_{layer}", f"resid_post_{layer}"]
    return names


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, n_ctx: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)
        mask = torch.tril(torch.ones(n_ctx, n_ctx, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]] = None,
                layer: int = 0, qkv: Optional[torch.Tensor] = None,
                last_only: bool = False) -> torch.Tensor:
        """One block. ``qkv`` may be passed in precomputed (exact for layer 0,
        where it is linear in the embeddings). With ``last_only`` only the
        last position is propagated past the attention (exact when nothing
        downstream reads the other positions)."""
        B, T, D = x.shape
        if qkv is None:
            qkv = self.qkv(x)
        q, k, v = qkv.split(D, dim=-1)
        if last_only:
            q = q[:, -1:]
        Tq = q.shape[1]
        q = q.reshape(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if not last_only:  # the last query sees every position: no mask needed
            mask = self.mask
            assert isinstance(mask, torch.Tensor)
            att = att.masked_fill(~mask[:T, :T], float("-inf"))
        z = (att.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, Tq, D)
        x = (x[:, -1:] if last_only else x) + self.o(z)
        if cache is not None:
            cache[f"resid_mid_{layer}"] = x[:, -1].detach()
        h = torch.relu(self.mlp_in(x))
        if cache is not None:
            cache[f"mlp_{layer}"] = h[:, -1].detach()
        x = x + self.mlp_out(h)
        if cache is not None:
            cache[f"resid_post_{layer}"] = x[:, -1].detach()
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) / math.sqrt(cfg.d_model))
        nn.init.normal_(self.embed.weight, std=1.0 / math.sqrt(cfg.d_model))
        self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.d_mlp, cfg.n_ctx)
                                     for _ in range(cfg.n_layers)])
        self.unembed = nn.Linear(cfg.d_model, cfg.n_out, bias=False)

    def forward(self, tokens: torch.Tensor,
                cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Logits at the last position, shape (batch, n_out).

        Exact but cheap: the layer-0 projections are linear in the embeddings,
        so they are computed once per vocabulary token and gathered, and the
        last block only propagates the ``=`` position (the only one read)."""
        T = tokens.shape[1]
        E, P = self.embed.weight, self.pos[:T]
        x = E[tokens] + P
        if cache is not None:
            cache["embed"] = x[:, 0].detach()
        b0 = self.blocks[0]
        assert isinstance(b0, Block)
        qkv0 = b0.qkv(E)[tokens] + b0.qkv(P)
        n = len(self.blocks)
        for layer, block in enumerate(self.blocks):
            x = block(x, cache, layer, qkv=qkv0 if layer == 0 else None,
                      last_only=layer == n - 1)
        out: torch.Tensor = self.unembed(x[:, -1])
        return out

    def forward_reference(self, tokens: torch.Tensor) -> torch.Tensor:
        """Plain full-sequence forward pass (for testing the fast path)."""
        x = self.embed(tokens) + self.pos[: tokens.shape[1]]
        for layer, block in enumerate(self.blocks):
            x = block(x, None, layer)
        out: torch.Tensor = self.unembed(x[:, -1])
        return out


def weight_norms(model: Transformer) -> Dict[str, float]:
    """L2 norms of the parameters: total and per component."""
    groups: Dict[str, List[torch.Tensor]] = {"embed": [], "attn": [], "mlp": [], "unembed": []}
    for name, prm in model.named_parameters():
        if name.startswith("embed") or name == "pos":
            groups["embed"].append(prm)
        elif ".qkv." in name or ".o." in name:
            groups["attn"].append(prm)
        elif ".mlp_" in name:
            groups["mlp"].append(prm)
        else:
            groups["unembed"].append(prm)
    out = {k: float(torch.sqrt(sum((p.detach() ** 2).sum() for p in v)))  # type: ignore[arg-type]
           for k, v in groups.items() if v}
    out["total"] = float(math.sqrt(sum(v ** 2 for v in out.values())))
    return out

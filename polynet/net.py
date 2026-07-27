"""PolyNet — fully-convolutional policy/value network with auto-regressive heads.

Trunk: N residual blocks of C channels with squeeze-excitation, fully
convolutional so one set of weights serves every map size (batches are
size-bucketed and padded, with an off-board plane telling the net what's real).
The non-spatial global vector is injected by FiLM (per-channel scale/shift) at
several depths rather than being concatenated as constant planes.

Heads follow the AlphaStar factorisation, each masked by `polyengine.masks`:

    type -> source tile -> target tile -> sub-type

Source and target are *pointer* heads: a 1x1 conv produces one logit per tile at
native resolution, so they transfer across map sizes. Target is conditioned on
the chosen type and on the source tile's own feature vector; sub-type on the
type and source.

Value head is global-average-pool + MLP with PopArt normalisation, so the value
scale can drift (score shaping annealing, changing opponents) without
destabilising the trunk. Two auxiliary heads — final score and opponent tech —
shape the representation for free and double as a fog-inference probe.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9


@dataclass
class NetConfig:
    n_planes: int
    global_dim: int
    n_action_types: int
    subtype_dim: int
    n_techs: int
    channels: int = 160
    blocks: int = 12
    film_at: tuple[int, ...] = (1, 5, 9)
    type_embed: int = 64
    head_hidden: int = 256

    @classmethod
    def small(cls, **kw):
        """A fast config for smoke tests and tiny maps."""
        base = dict(channels=48, blocks=4, film_at=(1, 3), type_embed=32,
                    head_hidden=96)
        base.update(kw)
        return cls(**base)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(8, channels // ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x, valid=None):
        s = masked_mean(x, valid)                 # (B, C)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s[:, :, None, None]


class FiLM(nn.Module):
    """Per-channel scale/shift conditioned on the global vector."""

    def __init__(self, global_dim: int, channels: int):
        super().__init__()
        self.to_scale = nn.Linear(global_dim, channels)
        self.to_shift = nn.Linear(global_dim, channels)
        nn.init.zeros_(self.to_scale.weight); nn.init.ones_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight); nn.init.zeros_(self.to_shift.bias)

    def forward(self, x, g):
        return x * self.to_scale(g)[:, :, None, None] + self.to_shift(g)[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, channels: int, global_dim: int, use_film: bool):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, channels)
        self.se = SqueezeExcite(channels)
        self.film = FiLM(global_dim, channels) if use_film else None

    def forward(self, x, g, valid):
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        if self.film is not None:
            h = self.film(h, g)
        h = self.se(h, valid)
        return F.relu(x + h)


def masked_mean(x, valid):
    """Global average pool that ignores off-board padding. valid: (B,1,H,W)."""
    if valid is None:
        return x.mean(dim=(2, 3))
    denom = valid.sum(dim=(2, 3)).clamp(min=1.0)
    return (x * valid).sum(dim=(2, 3)) / denom


class PopArt(nn.Module):
    """Value head whose output scale tracks a running target normalisation.

    The linear layer predicts a *normalised* value; `update_stats` rescales its
    weight and bias so the de-normalised output is unchanged when the statistics
    move. That decouples "the reward scale changed" from "the network must
    relearn its value function".
    """

    def __init__(self, in_features: int, beta: float = 3e-4, eps: float = 1e-4):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.beta = beta
        self.eps = eps
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))
        self.register_buffer("nu", torch.ones(1))       # second moment

    def forward(self, x):
        """Normalised prediction (what the loss is computed against)."""
        return self.linear(x).squeeze(-1)

    def denormalise(self, normalised):
        return normalised * self.sigma + self.mu

    @torch.no_grad()
    def update_stats(self, targets):
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        self.mu.mul_(1 - self.beta).add_(self.beta * targets.mean())
        self.nu.mul_(1 - self.beta).add_(self.beta * (targets ** 2).mean())
        self.sigma.copy_((self.nu - self.mu ** 2).clamp(min=self.eps).sqrt())
        # Preserve outputs: w' = w * s_old/s_new, b' = (s_old*b + mu_old - mu_new)/s_new
        self.linear.weight.mul_(old_sigma / self.sigma)
        self.linear.bias.mul_(old_sigma / self.sigma).add_(
            (old_mu - self.mu) / self.sigma)

    @torch.no_grad()
    def normalise(self, targets):
        return (targets - self.mu) / self.sigma


class PolyNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        c, gd = cfg.channels, cfg.global_dim

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.n_planes, c, 3, padding=1, bias=False),
            nn.GroupNorm(8, c), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [ResBlock(c, gd, use_film=(i in cfg.film_at)) for i in range(cfg.blocks)])

        self.type_embed = nn.Embedding(cfg.n_action_types, cfg.type_embed)

        # Pointer heads: 1x1 convs at native resolution.
        self.source_head = nn.Sequential(
            nn.Conv2d(c + cfg.type_embed, cfg.head_hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(cfg.head_hidden, 1, 1))
        self.target_head = nn.Sequential(
            nn.Conv2d(c + cfg.type_embed + c, cfg.head_hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(cfg.head_hidden, 1, 1))

        self.type_head = nn.Sequential(
            nn.Linear(c + gd, cfg.head_hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.head_hidden, cfg.n_action_types))
        self.subtype_head = nn.Sequential(
            nn.Linear(c + cfg.type_embed + c, cfg.head_hidden), nn.ReLU(inplace=True),
            nn.Linear(cfg.head_hidden, cfg.subtype_dim))

        self.value = PopArt(c + gd)
        self.aux_score = nn.Linear(c + gd, 1)
        self.aux_opp_tech = nn.Linear(c + gd, cfg.n_techs)

    # --- trunk ---------------------------------------------------------------
    def trunk(self, spatial, glob, valid=None):
        x = self.stem(spatial)
        for block in self.blocks:
            x = block(x, glob, valid)
        pooled = masked_mean(x, valid)
        return x, torch.cat([pooled, glob], dim=-1)

    # --- heads ---------------------------------------------------------------
    def type_logits(self, ctx):
        return self.type_head(ctx)

    def source_logits(self, feat, atype):
        e = self.type_embed(atype)[:, :, None, None].expand(-1, -1, *feat.shape[2:])
        return self.source_head(torch.cat([feat, e], dim=1)).flatten(1)

    def target_logits(self, feat, atype, src):
        b, c, h, w = feat.shape
        src = src.clamp(0, h * w - 1)          # see _score: keep gathers in range
        src_feat = feat.flatten(2).gather(
            2, src.view(b, 1, 1).expand(b, c, 1)).squeeze(-1)       # (B, C)
        e = self.type_embed(atype)
        cond = torch.cat([e, src_feat], dim=-1)[:, :, None, None].expand(-1, -1, h, w)
        return self.target_head(torch.cat([feat, cond], dim=1)).flatten(1)

    def subtype_logits(self, feat, ctx, atype, src):
        b, c = feat.shape[0], feat.shape[1]
        n_tiles = feat.shape[2] * feat.shape[3]
        safe_src = src.clamp(0, n_tiles - 1)
        src_feat = feat.flatten(2).gather(
            2, safe_src.view(b, 1, 1).expand(b, c, 1)).squeeze(-1)
        src_feat = torch.where((src < 0)[:, None], torch.zeros_like(src_feat), src_feat)
        pooled = ctx[:, :c]
        return self.subtype_head(torch.cat([pooled, self.type_embed(atype), src_feat],
                                           dim=-1))

    def value_and_aux(self, ctx):
        return self.value(ctx), self.aux_score(ctx).squeeze(-1), self.aux_opp_tech(ctx)


def masked_log_softmax(logits, mask):
    """Log-softmax restricted to `mask`; fully-masked rows fall back to uniform.

    Non-finite logits are scrubbed first. A single NaN would otherwise poison the
    whole row, and `torch.multinomial` on a NaN row returns an *out-of-range*
    index — which only surfaces later, as an async CUDA assert inside an
    unrelated gather. Sanitising here keeps the failure local and legible.
    """
    logits = torch.nan_to_num(logits, nan=NEG_INF, posinf=0.0, neginf=NEG_INF)
    empty = ~mask.any(dim=-1, keepdim=True)
    safe = mask | empty
    return torch.log_softmax(logits.masked_fill(~safe, NEG_INF), dim=-1)


def sample_masked(logits, mask, generator=None, greedy=False):
    """Sample an index under `mask`; returns (index, log_prob, entropy)."""
    logp = masked_log_softmax(logits, mask)
    probs = logp.exp()
    if greedy:
        idx = probs.argmax(dim=-1)
    else:
        # Guard the sampler itself: a degenerate row (all-zero after exp) makes
        # multinomial's result undefined, so fall back to the mask's first legal
        # entry rather than emitting an index that indexes out of bounds.
        total = probs.sum(dim=-1, keepdim=True)
        ok = torch.isfinite(total) & (total > 0)
        safe_probs = torch.where(ok, probs, torch.ones_like(probs))
        idx = torch.multinomial(safe_probs, 1, generator=generator).squeeze(-1)
    idx = idx.clamp(0, logits.shape[-1] - 1)
    chosen = logp.gather(-1, idx[:, None]).squeeze(-1)
    entropy = -(probs * logp.clamp(min=NEG_INF)).sum(-1)
    return idx, chosen, entropy

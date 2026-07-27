"""Batched auto-regressive policy: engine states in, actions + log-probs out.

This is the bridge between `polyengine` (NumPy states, boolean masks) and
`PolyNet` (torch tensors). It walks the four heads in order, applying each
head's mask, and carries the per-head choices forward as conditioning.

Two entry points, sharing one implementation so training and acting can never
disagree about the factorisation:

* `act(states)` — sample actions for a batch of live games (rollout).
* `evaluate(batch, actions)` — recompute log-probs/entropy/value for stored
  actions (the PPO update).

Heads that don't apply to the sampled action type (e.g. no source for RESEARCH)
contribute exactly zero log-prob and zero entropy, so the joint log-prob is a
proper distribution over the legal action set.
"""
from __future__ import annotations

import numpy as np
import torch

from polyengine import masks as M
from polyengine import obs as O
from polyengine.core import constants as C
from polyengine.core.actions import Action
from polynet.net import masked_log_softmax, sample_masked

HEADS = ("type", "source", "target", "subtype")


def _t(x, device, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=device)


class BatchEncoder:
    """Encodes a list of (state, player) into padded tensors + masks."""

    def __init__(self, rules, device, pad_to: int | None = None):
        self.rules = rules
        self.device = device
        self.pad_to = pad_to
        self.n_planes = O.n_planes(rules)
        self.subtype_dim = M.subtype_dim(rules)

    def encode(self, pairs: list[tuple[object, int]]) -> dict:
        rules = self.rules
        sizes = [s.size for s, _ in pairs]
        pad = self.pad_to or max(sizes)
        spatial = np.stack([O.encode_spatial(s, p, rules, pad_to=pad)
                            for s, p in pairs])
        glob = np.stack([O.encode_global(s, p, rules) for s, p in pairs])

        n_tiles_pad = pad * pad
        valid = np.zeros((len(pairs), 1, pad, pad), dtype=np.float32)
        type_mask = np.zeros((len(pairs), C.N_ACTION_TYPES), dtype=bool)
        all_masks = []
        for i, (state, player) in enumerate(pairs):
            valid[i, 0, :state.size, :state.size] = 1.0
            am = M.ActionMasks(state, player, rules)
            type_mask[i] = am.types()
            all_masks.append(am)
        return {
            "spatial": _t(spatial, self.device),
            "global": _t(glob, self.device),
            "valid": _t(valid, self.device),
            "type_mask": _t(type_mask, self.device, torch.bool),
            "masks": all_masks,
            "pad": pad,
            "sizes": sizes,
            "uniform": all(sz == pad for sz in sizes),
            "n_tiles_pad": n_tiles_pad,
        }

    def write_tile_mask(self, out_row: np.ndarray, flat_mask: np.ndarray,
                        size: int, pad: int, uniform: bool) -> None:
        """Write an engine tile mask (size*size) into a padded row in place.

        When every game in the batch is the map size we padded to, padded and
        engine indices coincide and the re-indexing is a straight copy.
        """
        if uniform:
            out_row[:flat_mask.size] = flat_mask
        else:
            out_row.reshape(pad, pad)[:size, :size] = flat_mask.reshape(size, size)

    def to_engine_tile(self, padded_idx: int, size: int, pad: int,
                       uniform: bool = False) -> int:
        if uniform:
            return int(padded_idx)
        y, x = divmod(int(padded_idx), pad)
        return y * size + x


class ARPolicy:
    """Auto-regressive policy over the four masked heads."""

    def __init__(self, net, rules, device="cpu", pad_to: int | None = None):
        self.net = net
        self.rules = rules
        self.device = torch.device(device)
        self.enc = BatchEncoder(rules, self.device, pad_to)
        # Sorted arrays for vectorised head-applicability tests.
        self._src_types = np.array(sorted(M.NEEDS_SOURCE), dtype=np.int64)
        self._tgt_types = np.array(sorted(M.NEEDS_TARGET), dtype=np.int64)
        self._sub_types = np.array(sorted(M.NEEDS_SUBTYPE), dtype=np.int64)

    # --- acting ---------------------------------------------------------------
    @torch.no_grad()
    def act(self, pairs: list[tuple[object, int]], greedy: bool = False,
            generator=None) -> tuple[list[Action], dict]:
        batch = self.enc.encode(pairs)
        feat, ctx = self.net.trunk(batch["spatial"], batch["global"], batch["valid"])
        b = len(pairs)
        pad = batch["pad"]

        atype, lp_type, _ = sample_masked(self.net.type_logits(ctx),
                                          batch["type_mask"], generator, greedy)
        # Pull the sampled types to host ONCE. Indexing a CUDA tensor per game
        # (`int(atype[i])`) forces a device synchronisation each time, which
        # dominated rollout cost far more than any of the actual computation.
        atype_np = atype.cpu().numpy()
        sizes = batch["sizes"]
        uniform = batch["uniform"]

        # source
        src_mask = np.zeros((b, batch["n_tiles_pad"]), dtype=bool)
        for i in range(b):
            at = int(atype_np[i])
            if at in M.NEEDS_SOURCE:
                self.enc.write_tile_mask(src_mask[i], batch["masks"][i].sources(at),
                                         sizes[i], pad, uniform)
        src_mask_t = _t(src_mask, self.device, torch.bool)
        src_idx, lp_src, _ = sample_masked(self.net.source_logits(feat, atype),
                                           src_mask_t, generator, greedy)
        src_applies_np = np.isin(atype_np, self._src_types)
        src_applies = _t(src_applies_np, self.device, torch.bool)
        src_idx = torch.where(src_applies, src_idx, torch.zeros_like(src_idx))
        src_np = src_idx.cpu().numpy()

        # target
        tgt_mask = np.zeros((b, batch["n_tiles_pad"]), dtype=bool)
        for i in range(b):
            at = int(atype_np[i])
            if at in M.NEEDS_TARGET:
                eng_src = self.enc.to_engine_tile(int(src_np[i]), sizes[i], pad, uniform)
                self.enc.write_tile_mask(
                    tgt_mask[i], batch["masks"][i].targets(at, eng_src),
                    sizes[i], pad, uniform)
        tgt_mask_t = _t(tgt_mask, self.device, torch.bool)
        tgt_idx, lp_tgt, _ = sample_masked(
            self.net.target_logits(feat, atype, src_idx), tgt_mask_t, generator, greedy)

        # subtype
        sub_mask = np.zeros((b, self.enc.subtype_dim), dtype=bool)
        for i in range(b):
            at = int(atype_np[i])
            if at in M.NEEDS_SUBTYPE:
                # BUILD/REWARD sub-types depend on which source tile was picked.
                eng_src = (self.enc.to_engine_tile(int(src_np[i]), sizes[i], pad,
                                                   uniform)
                           if at in M.NEEDS_SOURCE else -1)
                sub_mask[i] = batch["masks"][i].subtypes(at, eng_src)
        sub_mask_t = _t(sub_mask, self.device, torch.bool)
        src_for_sub = torch.where(src_applies, src_idx, torch.full_like(src_idx, -1))
        sub_idx, lp_sub, _ = sample_masked(
            self.net.subtype_logits(feat, ctx, atype, src_for_sub),
            sub_mask_t, generator, greedy)

        value, _, _ = self.net.value_and_aux(ctx)

        tgt_np = tgt_idx.cpu().numpy()
        sub_np = sub_idx.cpu().numpy()
        actions: list[Action] = []
        for i in range(b):
            at = int(atype_np[i])
            size = sizes[i]
            src = dst = sub = -1
            if at in M.NEEDS_SOURCE:
                src = self.enc.to_engine_tile(int(src_np[i]), size, pad, uniform)
            if at in M.NEEDS_TARGET:
                dst = self.enc.to_engine_tile(int(tgt_np[i]), size, pad, uniform)
            elif at == C.A_CAPTURE:
                dst = src
            if at in M.NEEDS_SUBTYPE:
                sub = int(sub_np[i])
            actions.append(Action(at, src, dst, sub))

        applies = {
            "source": src_applies,
            "target": _t(np.isin(atype_np, self._tgt_types), self.device, torch.bool),
            "subtype": _t(np.isin(atype_np, self._sub_types), self.device, torch.bool),
        }
        logp = (lp_type
                + lp_src * applies["source"]
                + lp_tgt * applies["target"]
                + lp_sub * applies["subtype"])
        info = {
            "logp": logp,
            "value": value,
            "atype": atype,
            "src_idx": src_idx,
            "tgt_idx": tgt_idx,
            "sub_idx": sub_idx,
            "type_mask": batch["type_mask"],
            "src_mask": src_mask_t,
            "tgt_mask": tgt_mask_t,
            "sub_mask": sub_mask_t,
            "spatial": batch["spatial"],
            "global": batch["global"],
            "valid": batch["valid"],
        }
        return actions, info

    # --- training -------------------------------------------------------------
    def evaluate(self, stored: dict) -> dict:
        """Recompute log-prob / entropy / value for stored transitions."""
        feat, ctx = self.net.trunk(stored["spatial"], stored["global"],
                                   stored["valid"])
        atype = stored["atype"]
        lp_type, ent_type = _score(self.net.type_logits(ctx),
                                   stored["type_mask"], atype)
        lp_src, ent_src = _score(self.net.source_logits(feat, atype),
                                 stored["src_mask"], stored["src_idx"])
        lp_tgt, ent_tgt = _score(self.net.target_logits(feat, atype,
                                                        stored["src_idx"]),
                                 stored["tgt_mask"], stored["tgt_idx"])
        src_for_sub = torch.where(stored["applies_source"], stored["src_idx"],
                                  torch.full_like(stored["src_idx"], -1))
        lp_sub, ent_sub = _score(
            self.net.subtype_logits(feat, ctx, atype, src_for_sub),
            stored["sub_mask"], stored["sub_idx"])

        a_src = stored["applies_source"].float()
        a_tgt = stored["applies_target"].float()
        a_sub = stored["applies_subtype"].float()
        logp = lp_type + lp_src * a_src + lp_tgt * a_tgt + lp_sub * a_sub
        entropy = ent_type + ent_src * a_src + ent_tgt * a_tgt + ent_sub * a_sub

        value, aux_score, aux_tech = self.net.value_and_aux(ctx)
        return {"logp": logp, "entropy": entropy, "value": value,
                "aux_score": aux_score, "aux_tech": aux_tech}


def _score(logits, mask, index):
    logp = masked_log_softmax(logits, mask)
    probs = logp.exp()
    chosen = logp.gather(-1, index.clamp(min=0)[:, None]).squeeze(-1)
    entropy = -(probs * logp).sum(-1)
    return chosen, entropy

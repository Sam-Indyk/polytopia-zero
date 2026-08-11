"""PPO learner for the auto-regressive masked policy.

Standard clipped-surrogate PPO with a few things this problem needs:

* **PopArt value targets.** Returns are normalised by a running estimate before
  the value loss, and the head's weights are rescaled so its de-normalised
  output is preserved. Reward scale changes (shaping annealing, tougher
  opponents) then don't destabilise the trunk.
* **Summed head entropy.** Entropy is the sum over the heads that actually
  applied to each action, so END_TURN (one head) and MOVE (three heads) are not
  penalised on different scales.
* **Auxiliary losses.** Final-score regression and opponent-tech prediction at
  0.25 weight — free representation shaping, and the tech head doubles as a
  probe of what the net infers through fog.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 3
    minibatch: int = 1024
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    # Minimum entropy each head must keep, as a fraction of the maximum it
    # could have. A single global entropy bonus cannot prevent one head
    # collapsing while the others stay busy: measured on run 3 at iteration
    # 1987, summed entropy was a healthy-looking 0.79 while the action-TYPE
    # head sat at 0.070 of a possible 2.30 -- P(research) was 0.000005, so the
    # agent could not have discovered the tech tree in a thousand lifetimes.
    #
    # Below the floor a head gets an extra, sharply rising penalty; above it,
    # nothing. So this costs nothing on a healthy run and only bites on a head
    # that is dying.
    # Raised 0.22 -> 0.40 after run 4. The old value was picked as a "this head
    # is dying" tripwire and became the operating point instead: the type head
    # fell from 0.566 to 0.242 within 100 iterations and then sat between 0.23
    # and 0.29 for the next thousand, never recovering. A floor the policy will
    # park on is a choice of operating point, so it should be chosen as one.
    entropy_floor: float = 0.40
    entropy_floor_coef: float = 0.5
    # Minimum probability every *legal* action type must keep. The entropy floor
    # constrains the spread of the distribution; this constrains its support,
    # which is a different property and the one that was actually failing.
    action_floor: float = 0.02
    action_floor_coef: float = 0.05
    aux_coef: float = 0.25
    max_grad_norm: float = 0.5
    normalise_adv: bool = True
    precision: str = "fp32"           # "fp32" | "bf16" (autocast, CUDA only)


class PPOLearner:
    def __init__(self, policy, cfg: PPOConfig | None = None):
        self.policy = policy
        self.net = policy.net
        self.cfg = cfg or PPOConfig()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)

    def update(self, batch: dict, advantages: np.ndarray, returns: np.ndarray,
               aux_targets: dict | None = None) -> dict:
        cfg = self.cfg
        device = next(self.net.parameters()).device
        # bf16 needs no GradScaler (unlike fp16): its exponent range matches
        # fp32, so gradients do not underflow and there is nothing to scale.
        amp = cfg.precision == "bf16" and device.type == "cuda"
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=device)

        self.net.value.update_stats(ret)
        ret_norm = self.net.value.normalise(ret)

        n = adv.shape[0]
        idx_all = np.arange(n)
        logs = {"policy_loss": [], "value_loss": [], "entropy": [],
                "clip_frac": [], "approx_kl": [], "action_floor_pen": []}

        for _ in range(cfg.epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, cfg.minibatch):
                mb = idx_all[start:start + cfg.minibatch]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb, device=device)
                sub = {k: (v[mb_t] if torch.is_tensor(v) else v)
                       for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    out = self.policy.evaluate(sub)

                # Back to fp32 before any loss arithmetic. The trunk tolerates
                # bf16 fine, but PPO's ratio is exp() of a *difference* of
                # log-probs: bf16 carries ~3 significant decimal digits, so a
                # difference near zero loses most of its precision and the
                # clipped surrogate stops meaning what it should.
                logp = out["logp"].float()
                value = out["value"].float()
                entropy_t = out["entropy"].float()

                mb_adv = adv[mb_t]
                if cfg.normalise_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = (logp - sub["old_logp"].float()).exp()
                unclipped = ratio * mb_adv
                clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.mse_loss(value, ret_norm[mb_t])
                entropy = entropy_t.mean()

                loss = (policy_loss
                        + cfg.value_coef * value_loss
                        - cfg.entropy_coef * entropy)

                # Per-head entropy floor. Each head is measured against the
                # entropy of a uniform choice over the options that were
                # actually legal for it, so "collapsed" means collapsed
                # relative to the freedom it had, not to a fixed constant.
                floor_pen = 0.0
                head_ent = {}
                for name, key, mask_key in (
                        ("type", "ent_type", "type_mask"),
                        ("source", "ent_source", "src_mask"),
                        ("target", "ent_target", "tgt_mask"),
                        ("subtype", "ent_subtype", "sub_mask")):
                    ent_h = out[key].float()
                    n_legal = sub[mask_key].float().sum(-1).clamp(min=1.0)
                    ceiling = torch.log(n_legal)
                    if name == "type":
                        applies = torch.ones_like(ent_h)
                    else:
                        applies = out["applies_" + name].float()
                    live = applies > 0
                    if not bool(live.any()):
                        continue
                    frac = (ent_h[live] / ceiling[live].clamp(min=1e-6)).mean()
                    head_ent[name] = float(frac.detach())
                    floor_pen = floor_pen + torch.relu(cfg.entropy_floor - frac) ** 2
                loss = loss + cfg.entropy_floor_coef * floor_pen

                # Per-ACTION floor on the type head, which the aggregate floor
                # above provably cannot supply. Measured at iteration 1000 of
                # run 4: the type head sat at frac 0.27 -- comfortably clear of
                # its 0.22 floor -- while RESEARCH held a *median* probability
                # of 0.0007 and EXAMINE 0.007. The head was satisfying the
                # floor by hedging between MOVE (0.55), ATTACK (0.55) and
                # TRAIN (0.59) while two whole verbs went to zero, and a scalar
                # measure of spread cannot tell those apart.
                #
                # Stated on log-probability rather than probability on purpose:
                # `relu(eps - p)` saturates at eps and its gradient dies exactly
                # where the action is most extinct, whereas `log eps - log p`
                # keeps growing as p -> 0. At p = 0.0007 against eps = 0.02 it
                # is 3.4, so a dead verb is pulled back hard and a merely
                # unfashionable one is left alone.
                if cfg.action_floor > 0.0:
                    legal = sub["type_mask"].float()
                    deficit = torch.relu(math.log(cfg.action_floor)
                                         - out["logp_type"].float()) * legal
                    # Mean over the legal options, so the penalty does not scale
                    # with how many action types happen to be available.
                    act_pen = (deficit.sum(-1) / legal.sum(-1).clamp(min=1.0)).mean()
                    loss = loss + cfg.action_floor_coef * act_pen
                    logs["action_floor_pen"].append(float(act_pen.detach()))

                if aux_targets is not None:
                    score_t = aux_targets["score"][mb_t]
                    tech_t = aux_targets["tech"][mb_t]
                    loss = loss + cfg.aux_coef * (
                        F.mse_loss(out["aux_score"].float(), score_t)
                        + F.binary_cross_entropy_with_logits(
                            out["aux_tech"].float(), tech_t))

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                               cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    logs["policy_loss"].append(float(policy_loss))
                    logs["value_loss"].append(float(value_loss))
                    logs["entropy"].append(float(entropy))
                    for name, frac in head_ent.items():
                        logs.setdefault("ent_" + name, []).append(frac)
                    logs["clip_frac"].append(
                        float(((ratio - 1).abs() > cfg.clip).float().mean()))
                    logs["approx_kl"].append(float((sub["old_logp"] - out["logp"]).mean()))

        return {k: float(np.mean(v)) if v else 0.0 for k, v in logs.items()}

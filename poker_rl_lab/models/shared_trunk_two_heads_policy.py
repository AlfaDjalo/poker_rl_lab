"""
models/shared_trunk_two_heads_policy.py — DualSeatActorCritic variant 1.

One RepresentationModel ("trunk") shared by both seats, feeding two
SEPARATE PolicyValueNetwork instances ("heads") — one for SB (seat 0),
one for BB (seat 1). See models/dual_seat_policy.py's module docstring
for how this compares to the single-head variant and why
seat_parameters() returns what it does.

Why share the trunk at all
------------------------------
SB and BB are looking at genuinely different strategic problems (push
range vs. calling range), so giving each its own policy/value HEAD lets
them specialise. But both problems still start from the same raw
ingredients — hole cards, stack depth, position — so the trunk
(RepresentationModel: hand/board/point/game/position/history encoders +
combination MLP) can reasonably be shared: whatever it learns about how
stack depth interacts with hand strength while training SB's head is
also useful signal for BB's head, and vice versa. This is the standard
"shared representation, separate heads" multi-task pattern.

Consequence for training
----------------------------
Because self.representation_model is one nn.Module instance referenced
by BOTH act(0, ...) and act(1, ...), its parameters show up in both
seat_parameters(0) and seat_parameters(1). A training loop that
ALTERNATES (freeze seat 1, train seat 0, swap — see
trainers/duo_training_loop.py) will therefore keep updating the trunk on
every single step regardless of which seat is nominally "frozen"; only
the head that seat's optimizer doesn't cover stays frozen. This is
intentional — the trunk is meant to benefit from both seats' data — but
callers doing careful ablations (e.g. "does freezing help?") should be
aware the trunk is never actually frozen by seat_parameters() alone; a
true full-freeze requires excluding the trunk from BOTH optimizers by
construction, which this class does not do automatically.

act_batch() (NEW)
------------------
Batched sibling of act() — see dual_seat_policy.DualSeatActorCritic's
own act_batch() docstring for the exact "throughput change, not a
behaviour change" contract this fulfils. Mirrors
PokerActorCritic.act_batch() (models/policy_value_network.py) almost
exactly, just routed through the seat-specific head instead of a single
shared head: one representation_model.forward_batch() call, one
head(latent) call, one batched Categorical, instead of `len(obs_batch)`
separate act() calls each paying full per-encoder dispatch overhead
(dominated by the TF-backed hand encoder — see representation_model.py's
own "batched forward_batch()" docstring). This is what
trainers.generic_tree_training_loop.collect_generic_tree_rollouts()
now calls once per seat per synchronized tick, instead of once per
episode per decision.

Dependencies
------------
    PyTorch. models.dual_seat_policy.DualSeatActorCritic,
    models.policy_value_network (PolicyValueNetwork, ActionResult,
    apply_legal_mask), representation.representation_model
    (RepresentationModel, RepresentationInput).
"""

from __future__ import annotations

import itertools
from typing import Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic, _check_seat
from poker_rl_lab.models.policy_value_network import ActionResult, PolicyValueNetwork, apply_legal_mask
from poker_rl_lab.representation.representation_model import RepresentationInput, RepresentationModel


class SharedTrunkTwoHeadsActorCritic(DualSeatActorCritic):
    """
    Shared RepresentationModel trunk, two independent PolicyValueNetwork
    heads (one per seat).

    Parameters
    ----------
    representation_model : RepresentationModel
        Shared trunk. Its output_dim must match both heads' latent_dim
        (enforced below, mirroring PokerActorCritic's own check).
    head_sb : PolicyValueNetwork
        Policy/value head used for seat 0 (SB).
    head_bb : PolicyValueNetwork
        Policy/value head used for seat 1 (BB).
    """

    def __init__(
        self,
        representation_model: RepresentationModel,
        head_sb: PolicyValueNetwork,
        head_bb: PolicyValueNetwork,
    ) -> None:
        super().__init__()

        for name, head in (("head_sb", head_sb), ("head_bb", head_bb)):
            if head.latent_dim != representation_model.output_dim:
                raise ValueError(
                    f"{name}.latent_dim ({head.latent_dim}) must match "
                    f"representation_model.output_dim "
                    f"({representation_model.output_dim})"
                )

        self.representation_model = representation_model
        self.head_sb = head_sb
        self.head_bb = head_bb

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _head_for(self, seat: int) -> PolicyValueNetwork:
        _check_seat(seat)
        return self.head_sb if seat == 0 else self.head_bb

    # ------------------------------------------------------------------
    # Rollout-time API (single example)
    # ------------------------------------------------------------------

    def act(
        self,
        seat: int,
        obs: RepresentationInput,
        legal_mask: Sequence[bool],
        deterministic: bool = False,
    ) -> ActionResult:
        return self.act_batch(seat, [obs], [legal_mask], deterministic)[0]

    # ------------------------------------------------------------------
    # Rollout-time API (batched) — NEW
    # ------------------------------------------------------------------

    def act_batch(
        self,
        seat: int,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        deterministic: bool = False,
    ) -> List[ActionResult]:
        """
        Choose one action per decision point, for a batch of
        simultaneous decision points ALL belonging to `seat`.

        Numerically identical, example-by-example, to calling
        act(seat, ...) once per (obs, legal_mask) pair — pure
        throughput optimisation, exactly like
        PokerActorCritic.act_batch()'s own guarantee.

        Parameters
        ----------
        seat : int
            0 or 1. Selects which head processes the WHOLE batch — every
            example here must actually be this seat's decision (a caller
            mixing seats in one call is a caller bug; see
            trainers.generic_tree_training_loop.collect_generic_tree_rollouts
            for how callers group decisions by seat before calling this).
        obs_batch : Sequence[RepresentationInput]
        legal_masks : Sequence[Sequence[bool]]
            Same length as obs_batch, each of length NUM_ACTIONS.
        deterministic : bool
            If True, every row takes its argmax action. If False
            (default), every row samples independently from its own
            Categorical distribution.

        Returns
        -------
        List[ActionResult]
            Length len(obs_batch), same order as obs_batch.
        """
        head = self._head_for(seat)

        n = len(obs_batch)
        if n == 0:
            raise ValueError("act_batch() requires a non-empty obs_batch")
        if len(legal_masks) != n:
            raise ValueError(
                f"obs_batch has {n} examples but legal_masks has "
                f"{len(legal_masks)} — they must match 1:1."
            )

        latents = self.representation_model.forward_batch(obs_batch)  # (n, latent_dim)
        logits, values = head(latents)                                 # (n, action_dim), (n,)

        mask_t = torch.as_tensor(legal_masks, dtype=torch.bool, device=logits.device)
        masked_logits = apply_legal_mask(logits, mask_t)

        dist = Categorical(logits=masked_logits)

        if deterministic:
            actions = torch.argmax(masked_logits, dim=-1)
        else:
            actions = dist.sample()

        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        return [
            ActionResult(
                action_type=int(actions[i].item()),
                log_prob=log_probs[i],
                value=values[i],
                entropy=entropies[i],
            )
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # PPO update-time API
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        seat: int,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        actions: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not (len(obs_batch) == len(legal_masks) == len(actions)):
            raise ValueError(
                f"obs_batch ({len(obs_batch)}), legal_masks "
                f"({len(legal_masks)}), and actions ({len(actions)}) "
                f"must all have the same length"
            )

        head = self._head_for(seat)

        latents = self.representation_model.forward_batch(obs_batch)
        logits, values = head(latents)

        mask_t = torch.as_tensor(legal_masks, dtype=torch.bool, device=logits.device)
        masked_logits = apply_legal_mask(logits, mask_t)

        dist = Categorical(logits=masked_logits)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=logits.device)

        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy()

        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Parameter groups
    # ------------------------------------------------------------------

    def seat_parameters(self, seat: int) -> Iterator[nn.Parameter]:
        """Trunk + this seat's own head. See module docstring for the
        alternating-training implication of the trunk being shared."""
        head = self._head_for(seat)
        return itertools.chain(self.representation_model.parameters(), head.parameters())

    def action_probabilities(self, seat, obs, legal_mask):
        head = self._head_for(seat)
        latent = self.representation_model(obs)
        logits, _ = head(latent)
        mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
        masked_logits = apply_legal_mask(logits, mask_t)
        return torch.softmax(masked_logits, dim=-1)
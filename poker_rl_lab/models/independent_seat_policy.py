"""
models/independent_seat_policy.py — DualSeatActorCritic variant 3:
fully independent networks, one per seat, NO shared trunk.

Why this variant exists
--------------------------
models/shared_trunk_two_heads_policy.py's own docstring already flags
the consequence of a shared trunk: "a training loop that ALTERNATES...
will therefore keep updating the trunk on every single step regardless
of which seat is nominally frozen." What that docstring doesn't cover
is the SIMULTANEOUS-mode failure mode observed training push_fold_duo:

    BB only ever receives training data on hands where SB happened to
    shove. Once SB's shove rate drops for any reason (including
    ordinary early-training sampling noise), BB's RolloutBuffer that
    iteration shrinks to a handful of transitions. Advantage
    normalization (RolloutBuffer.get()) divides by that small batch's
    own std, which is a noisy estimate at low n — producing an
    occasional oversized PPO gradient step for BB's update. With
    SharedTrunkTwoHeadsActorCritic, that gradient flows through the
    SAME representation_model both seats depend on, so an oversized,
    noisy BB-driven step can wreck whatever hand-strength-dependent
    signal SB's head was starting to exploit. SB's next batch then
    works from a degraded trunk, folding becomes the locally-safest
    action (zero variance, bounded loss), and once SB folds uniformly,
    BB's data starves further -- a self-reinforcing collapse to a
    degenerate all-fold / all-call equilibrium, independent of
    entropy_coef or epoch count.

IndependentSeatActorCritic removes the shared-trunk half of that
feedback loop entirely: seat 0 and seat 1 each get their OWN
RepresentationModel + PolicyValueNetwork (a full, separate
PokerActorCritic each), so an oversized/noisy gradient on one seat's
network has zero path into the other's parameters. This trades away
the cross-seat representation sharing SharedTrunkTwoHeadsActorCritic
was designed to exploit (see that module's own "Why share the trunk at
all" docstring section) in exchange for training stability — the right
trade when one seat's data is much sparser/burstier than the other's,
as push-fold's SB-gates-BB structure guarantees.

The two networks MAY still share the same frozen pretrained
hand_encoder base instance (the `hand_encoder` argument, forwarded
identically into both build_actor_critic() calls) — that's safe and
free: the base encoder's own weights are never trained (see
representation_model.py's "NOT an nn.Module (TF-backed); stored, not
registered" note, and composite_hand_encoder.py's "Frozen base +
trainable adapter" pattern). Only each network's OWN trainable adapter
layers (CompositeHandEncoder/CompositeBoardEncoder's attn_scorer/
project, the combination MLP, the policy/value heads) are duplicated
and genuinely independent per seat.

act_batch()
------------
Delegates straight to each seat's own PokerActorCritic.act_batch() —
same "throughput change, not a behaviour change" guarantee as
SharedSingleHeadActorCritic's identical delegation.

Dependencies
------------
    PyTorch. models.dual_seat_policy.DualSeatActorCritic,
    models.policy_value_network.PokerActorCritic,
    representation.representation_model.RepresentationInput.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import torch.nn as nn

from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic, _check_seat
from poker_rl_lab.models.policy_value_network import ActionResult, PokerActorCritic
from poker_rl_lab.representation.representation_model import RepresentationInput


class IndependentSeatActorCritic(DualSeatActorCritic):
    """
    Two fully independent PokerActorCritic networks, one per seat — no
    shared trunk, no shared head. See module docstring for why this
    exists (training-stability isolation, not representation sharing).

    Parameters
    ----------
    actor_critic_sb : PokerActorCritic
        Network used exclusively for seat 0.
    actor_critic_bb : PokerActorCritic
        Network used exclusively for seat 1.

    Notes
    -----
    seat_parameters(0) and seat_parameters(1) return two COMPLETELY
    DISJOINT parameter sets (unlike SharedTrunkTwoHeadsActorCritic,
    where the trunk appears in both). An ALTERNATING/LEAGUE training
    loop's "freeze the passive seat" therefore does exactly what it
    says here, with no shared-trunk leakage to reason about.
    """

    def __init__(
        self,
        actor_critic_sb: PokerActorCritic,
        actor_critic_bb: PokerActorCritic,
    ) -> None:
        super().__init__()
        self.actor_critic_sb = actor_critic_sb
        self.actor_critic_bb = actor_critic_bb

    def _net_for(self, seat: int) -> PokerActorCritic:
        _check_seat(seat)
        return self.actor_critic_sb if seat == 0 else self.actor_critic_bb

    def act(
        self,
        seat: int,
        obs: RepresentationInput,
        legal_mask: Sequence[bool],
        deterministic: bool = False,
    ) -> ActionResult:
        return self._net_for(seat).act(obs, legal_mask, deterministic=deterministic)

    def act_batch(
        self,
        seat: int,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        deterministic: bool = False,
    ) -> List[ActionResult]:
        return self._net_for(seat).act_batch(obs_batch, legal_masks, deterministic=deterministic)

    def evaluate_actions(
        self,
        seat: int,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        actions: Sequence[int],
    ) -> Tuple:
        return self._net_for(seat).evaluate_actions(obs_batch, legal_masks, actions)

    def seat_parameters(self, seat: int) -> Iterator[nn.Parameter]:
        return self._net_for(seat).parameters()

    def action_probabilities(self, seat, obs, legal_mask):
        return self._net_for(seat).action_probabilities(obs, legal_mask)
"""
models/shared_single_head_policy.py — DualSeatActorCritic variant 2.

One RepresentationModel + one PolicyValueNetwork (i.e. a single ordinary
models.policy_value_network.PokerActorCritic), used for BOTH seats.
"Seat" is not a network-level concept here at all — there is exactly one
set of weights, and it plays both SB and BB.

Why this is a valid design (not just "forgetting" to specialise)
----------------------------------------------------------------------
The network already receives seat-identifying information as part of
the observation itself: representation/position_encoder.py's
PositionFeatures carries is_button / is_small_blind / is_big_blind /
is_next_to_act, all derived from the StructuredObservation the
Environment builds (see envs/push_fold_duo_env.py's `_build_obs` —
hero_seat, dealer_position, and each PlayerObs are all seat-accurate).
So a single shared network CAN in principle learn two different
behaviours conditioned on those position features — this is exactly the
"seat is an input feature, fully shared" option from the design
discussion, not an accidental collapse to one policy.

What you're trading away vs. the two-head variant
-------------------------------------------------------
Capacity for specialisation is now entirely inside the shared trunk +
single head, rather than split trunk/head — the network must route
"am I SB or BB" through the same weights it uses for "what are my hole
cards", rather than getting a dedicated head per role. Whether that
hurts in practice (vs.
models.shared_trunk_two_heads_policy.SharedTrunkTwoHeadsActorCritic) is
exactly the kind of comparison this lab is meant to make easy — hence
both classes implementing the identical DualSeatActorCritic interface.

act_batch() (NEW)
------------------
Delegates straight to the wrapped PokerActorCritic.act_batch()
(models/policy_value_network.py) — that method already IS the batched,
seat-agnostic implementation this class needs, since "seat" plays no
role in which weights are used here (see class docstring). `seat` is
still validated (mirrors act()/evaluate_actions()'s existing behaviour)
so a caller passing an invalid seat index fails loudly here too, rather
than only inside a 2-way dispatch that doesn't exist for this variant.

Dependencies
------------
    PyTorch. models.dual_seat_policy.DualSeatActorCritic,
    models.policy_value_network.PokerActorCritic (the class this wraps),
    representation.representation_model.RepresentationInput.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence, Tuple

import torch.nn as nn

from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic, _check_seat
from poker_rl_lab.models.policy_value_network import ActionResult, PokerActorCritic
from poker_rl_lab.representation.representation_model import RepresentationInput


class SharedSingleHeadActorCritic(DualSeatActorCritic):
    """
    Wraps one ordinary PokerActorCritic so it satisfies the
    DualSeatActorCritic interface, using the exact same weights for both
    seats.

    Parameters
    ----------
    actor_critic : PokerActorCritic
        The single shared network. Build it exactly as for any other
        curriculum stage (models.actor_critic_factory.build_actor_critic()).

    Notes
    -----
    `seat` is validated but otherwise unused by act()/act_batch()/
    evaluate_actions() — see module docstring for why that's correct
    here (the observation already encodes which seat is acting).
    seat_parameters(0) and seat_parameters(1) both return the SAME full
    parameter set: see models/dual_seat_policy.py's module docstring for
    what this implies for alternating training with this variant (it
    degenerates to always training everything).
    """

    def __init__(self, actor_critic: PokerActorCritic) -> None:
        super().__init__()
        self.actor_critic = actor_critic

    def act(
        self,
        seat: int,
        obs: RepresentationInput,
        legal_mask: Sequence[bool],
        deterministic: bool = False,
    ) -> ActionResult:
        _check_seat(seat)
        return self.actor_critic.act(obs, legal_mask, deterministic=deterministic)

    def act_batch(
        self,
        seat: int,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        deterministic: bool = False,
    ) -> List[ActionResult]:
        """Delegates to PokerActorCritic.act_batch() — see that method's
        docstring (models/policy_value_network.py) for the exact
        batching contract. `seat` is validated only; it plays no role
        in which weights process the batch (see class docstring)."""
        _check_seat(seat)
        return self.actor_critic.act_batch(
            obs_batch, legal_masks, deterministic=deterministic
        )

    def evaluate_actions(
        self,
        seat: int,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        actions: Sequence[int],
    ) -> Tuple:
        _check_seat(seat)
        return self.actor_critic.evaluate_actions(obs_batch, legal_masks, actions)

    def seat_parameters(self, seat: int) -> Iterator[nn.Parameter]:
        _check_seat(seat)
        return self.parameters()

    def action_probabilities(self, seat, obs, legal_mask):
        _check_seat(seat)
        return self.actor_critic.action_probabilities(obs, legal_mask)
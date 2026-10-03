"""
models/dual_seat_policy.py — Shared interface for two-seat policies.

Push-fold-duo (and, per the user's note, every future scenario built on
envs.push_fold_duo_env-style two-agent envs) needs a policy per SEAT, not
a single hero policy. This module defines the interface that ANY
seat-aware policy must satisfy, so that:

    - envs/push_fold_duo_env.py-driving code (a training loop, an
      evaluator, a human-vs-agent demo) never has to know or care
      WHICH concrete network architecture is behind the policy.
    - trainers/duo_training_loop.py (alternating / simultaneous /
      league) can be written once against this interface and reused
      unchanged across every concrete implementation.

Two concrete implementations exist today (see their own files):

    SharedTrunkTwoHeadsActorCritic  (models/shared_trunk_two_heads_policy.py)
        One RepresentationModel (trunk), two separate PolicyValueNetwork
        heads — one "owned" by seat 0 (SB), one by seat 1 (BB). The
        trunk's weights are shared and updated by BOTH seats' losses;
        each head's weights are updated only by its own seat's loss.

    SharedSingleHeadActorCritic  (models/shared_single_head_policy.py)
        One RepresentationModel + one PolicyValueNetwork, used
        identically for both seats. "Which seat is acting" is not a
        network-level concept at all here — it's already present in the
        observation itself, via PositionFeatures.is_small_blind /
        is_big_blind / is_next_to_act (representation/position_encoder.py)
        — the SAME set of weights plays both roles, exactly like a
        single symmetric self-play agent.

Why `seat_parameters(seat)` exists
--------------------------------------
A training loop that wants to ALTERNATE (freeze seat 1, train seat 0,
swap) needs to know which parameters belong to "seat 0's optimization
step" for a given architecture:

    - SharedTrunkTwoHeads: seat 0's step should update the trunk + head 0
      (not head 1). seat 1's step should update the trunk + head 1.
      The trunk therefore appears in BOTH seats' parameter sets — see
      that class's own docstring for why this is the intended behaviour
      (not a bug) and what it implies for alternating training.
    - SharedSingleHead: there is only one parameter set. Both seats'
      "own" parameters are literally the same objects — alternating
      between "seat 0" and "seat 1" for this architecture is equivalent
      to just always training the one network, every step, since a
      freeze on one grants no different a state than a freeze on the
      other (both freeze everything). Callers are expected to notice
      this degenerate case rather than the architecture forbidding it.

act_batch() (NEW)
------------------
Previously ONLY models.policy_value_network.PokerActorCritic (the
single-hero actor-critic) exposed a batched act_batch(). Neither
DualSeatActorCritic nor either concrete duo implementation did, so
trainers.generic_tree_training_loop.collect_generic_tree_rollouts()
(and its predecessor, trainers.duo_training_loop.collect_duo_rollouts())
called .act() once per hero decision, per episode, in a plain Python
loop — every neural network forward pass during duo rollout collection
was a batch-of-1, even though every sub-encoder in
representation/representation_model.py already supports real batching
via forward_batch(). This method is the missing piece: same contract as
act() (same seat, same per-example semantics), but takes a LIST of
(RepresentationInput, legal_mask) pairs sharing the SAME seat and
returns one ActionResult per example from a single forward pass.
Concrete subclasses implement this by batching their trunk/head calls
exactly like PokerActorCritic.act_batch() already does.

Dependencies
------------
    PyTorch. models.policy_value_network.ActionResult (return shape for
    act()/act_batch()), representation.representation_model.RepresentationInput.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn

from poker_rl_lab.models.policy_value_network import ActionResult
from poker_rl_lab.representation.representation_model import RepresentationInput

#: The only two valid seat indices push-fold-duo (and any other
#: heads-up two-agent env built the same way) will ever pass in.
VALID_SEATS = (0, 1)


def _check_seat(seat: int) -> None:
    if seat not in VALID_SEATS:
        raise ValueError(f"seat must be one of {VALID_SEATS}, got {seat}")


class DualSeatActorCritic(nn.Module, ABC):
    """
    Common surface every two-seat policy architecture must implement.

    Contract
    --------
    act(seat, obs, legal_mask, deterministic=False) -> ActionResult
        Choose one action for `seat` at one decision point. Same
        per-field contract as models.policy_value_network.PokerActorCritic
        .act() (see that class's docstring), with an added `seat`
        argument selecting which seat's policy/value function is used.

    act_batch(seat, obs_batch, legal_masks, deterministic=False)
            -> List[ActionResult]
        Choose one action per decision point, for a BATCH of
        simultaneous decision points that all belong to the SAME seat
        (e.g. one per in-flight episode currently waiting on this
        seat). Numerically identical, example-by-example, to calling
        act(seat, ...) once per pair — see
        models.policy_value_network.PokerActorCritic.act_batch()'s
        docstring for the exact "throughput change, not a behaviour
        change" guarantee this mirrors.

    evaluate_actions(seat, obs_batch, legal_masks, actions)
            -> (log_probs, values, entropy)
        Re-evaluate a batch of (state, action) pairs FOR ONE SEAT under
        the current policy — the PPO update-time counterpart to act().
        Same contract as PokerActorCritic.evaluate_actions().

    seat_parameters(seat) -> Iterator[nn.Parameter]
        The parameters an optimizer training `seat` in isolation should
        cover. See module docstring for what this means per
        architecture. Used by trainers.duo_training_loop for the
        "alternating" training mode; simultaneous/joint training modes
        can ignore this and just use self.parameters() directly.
    """

    @abstractmethod
    def act(
        self,
        seat: int,
        obs: RepresentationInput,
        legal_mask: Sequence[bool],
        deterministic: bool = False,
    ) -> ActionResult:
        ...

    @abstractmethod
    def act_batch(
        self,
        seat: int,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        deterministic: bool = False,
    ) -> List[ActionResult]:
        ...

    @abstractmethod
    def evaluate_actions(
        self,
        seat: int,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        actions: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...

    @abstractmethod
    def seat_parameters(self, seat: int) -> Iterator[nn.Parameter]:
        ...

    @abstractmethod
    def action_probabilities(
        self,
        seat: int,
        obs: RepresentationInput,
        legal_mask: Sequence[bool],
    ) -> torch.Tensor:
        ...        
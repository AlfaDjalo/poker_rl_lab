"""
models/policy_value_network.py — Policy and value heads on top of the
Representation Model's latent vector, per the RL Lab implementation
guide (§7, "Policy Learning").

CHANGES IN THIS VERSION
------------------------
Added PokerActorCritic.act_batch(): a batched counterpart to act() that
takes a LIST of RepresentationInput + legal_masks (one per hand) and
returns one ActionResult per hand from a SINGLE forward pass, instead of
`len(batch)` separate .act() calls.

Why this was missing
---------------------
act() already only does single-example work (RepresentationModel.forward,
which itself is just forward_batch([x])[0]) — every rollout-collection
call paid full per-encoder dispatch overhead for a batch of one. Nothing
about the math changes: apply_legal_mask, Categorical, and
policy_value_network.forward() are already fully batch-shaped (see their
own docstrings/impls below) — act() alone never took advantage of that.
act_batch() is the missing plumbing, used by
envs.vectorized_env.VectorizedPokerEnv + trainers.ppo_trainer's
vectorized rollout collection to process many independent hands' hero
decisions in one pass.

act_batch() changes nothing about WHAT is computed per example — each
row's logits/value/sampled action/log_prob/entropy are numerically
identical to what act() would produce for that same
(RepresentationInput, legal_mask) pair called alone (batched
BatchNorm-free MLPs and a batched Categorical are exact, not
approximate). This is a throughput change, not a behavior change, and
in particular introduces no cross-example leakage: RepresentationModel's
own forward_batch() docstring already establishes that boards/points/
position/history/stack are pooled and projected per-example, never
mixed across the batch dimension.

[... rest of original module docstring unchanged ...]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from poker_rl_lab.actions.abstract_action import NUM_ACTIONS
from poker_rl_lab.representation.representation_model import RepresentationModel, RepresentationInput

# Additive mask value for illegal actions. Large enough that
# exp(logit - max_logit) underflows to exactly 0.0 in float32 for any
# realistic logit range, without producing -inf/NaN in the backward pass
# the way literal float('-inf') can when an entire row is masked.
_MASK_NEG = -1.0e9


# ---------------------------------------------------------------------------
# Policy / value network
# ---------------------------------------------------------------------------

class PolicyValueNetwork(nn.Module):
    """
    Maps a latent representation to policy logits and a state value.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the input latent vector (must match
        RepresentationModel.output_dim when used with PokerActorCritic).
    action_dim : int
        Number of discrete actions. Default NUM_ACTIONS
        (actions/abstract_action.py).
    hidden_dim : int
        Hidden width of the shared trunk. Default 128.
    num_shared_layers : int
        Number of hidden layers in the shared trunk before the two
        heads split off. Default 1.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int = NUM_ACTIONS,
        hidden_dim: int = 128,
        num_shared_layers: int = 1,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if num_shared_layers < 1:
            raise ValueError(f"num_shared_layers must be >= 1, got {num_shared_layers}")

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        layers = []
        in_dim = latent_dim
        for _ in range(num_shared_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        latent : torch.Tensor
            Shape (batch, latent_dim) or (latent_dim,) for a single example.

        Returns
        -------
        logits : torch.Tensor
            Shape (batch, action_dim) — raw, UNMASKED policy logits.
        value : torch.Tensor
            Shape (batch,) — state value estimate.
        """
        squeeze = False
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
            squeeze = True

        hidden = self.trunk(latent)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)

        if squeeze:
            logits = logits.squeeze(0)
            value = value.squeeze(0)

        return logits, value


# ---------------------------------------------------------------------------
# Masking helper
# ---------------------------------------------------------------------------

def apply_legal_mask(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """
    Additively mask illegal actions to -inf-equivalent before softmax.

    Parameters
    ----------
    logits : torch.Tensor
        Shape (..., action_dim).
    legal_mask : torch.Tensor
        Same leading shape as logits' batch dims, dtype bool.
        True = legal.

    Returns
    -------
    torch.Tensor
        Same shape as logits, with illegal entries set to _MASK_NEG.

    Raises
    ------
    ValueError
        If a row has no legal actions at all (Categorical would be
        undefined) — this indicates an upstream bug (a terminal
        observation should never reach the policy).
    """
    if not torch.any(legal_mask, dim=-1).all():
        raise ValueError(
            "apply_legal_mask received a row with no legal actions. "
            "A terminal observation should never be passed to the policy."
        )
    return logits.masked_fill(~legal_mask, _MASK_NEG)


# ---------------------------------------------------------------------------
# Actor-critic: RepresentationModel + PolicyValueNetwork + sampling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActionResult:
    """One sampled action and everything a PPO rollout buffer needs to store."""
    action_type: int          # int value of actions.abstract_action.ActionType
    log_prob: torch.Tensor    # scalar, log pi(action | state)
    value: torch.Tensor       # scalar, V(state)
    entropy: torch.Tensor     # scalar, policy entropy at this state


class PokerActorCritic(nn.Module):
    """
    End-to-end actor-critic: RepresentationInput -> chosen action.

    See original module docstring for full architectural rationale.
    act_batch() (new) is the batched sibling of act() — same math,
    processes many independent decision points in one forward pass.

    Parameters
    ----------
    representation_model : RepresentationModel
    policy_value_network : PolicyValueNetwork
        Must have latent_dim == representation_model.output_dim.
    """

    def __init__(
        self,
        representation_model: RepresentationModel,
        policy_value_network: PolicyValueNetwork,
    ) -> None:
        super().__init__()

        if policy_value_network.latent_dim != representation_model.output_dim:
            raise ValueError(
                f"policy_value_network.latent_dim "
                f"({policy_value_network.latent_dim}) must match "
                f"representation_model.output_dim "
                f"({representation_model.output_dim})"
            )

        self.representation_model = representation_model
        self.policy_value_network = policy_value_network

    # ------------------------------------------------------------------
    # Rollout-time API (single example, no grad tracking required by caller)
    # ------------------------------------------------------------------

    def act(
        self,
        obs: RepresentationInput,
        legal_mask: Sequence[bool],
        deterministic: bool = False,
    ) -> ActionResult:
        """
        Choose one action for one decision point.

        Kept for single-hand callers (evaluation demos, HumanController
        play-testing, etc.) where batching isn't applicable. Prefer
        act_batch() whenever more than one decision point is available
        at once (i.e. anywhere driven by
        envs.vectorized_env.VectorizedPokerEnv) — see that method's
        docstring for why it's dramatically cheaper per-example.
        """
        latent = self.representation_model(obs)               # (latent_dim,)
        logits, value = self.policy_value_network(latent)      # (action_dim,), scalar

        mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
        masked_logits = apply_legal_mask(logits, mask_t)

        dist = Categorical(logits=masked_logits)

        if deterministic:
            action = torch.argmax(masked_logits)
        else:
            action = dist.sample()

        return ActionResult(
            action_type=int(action.item()),
            log_prob=dist.log_prob(action),
            value=value,
            entropy=dist.entropy(),
        )

    # ------------------------------------------------------------------
    # Batched rollout-time API (NEW)
    # ------------------------------------------------------------------

    def act_batch(
        self,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        deterministic: bool = False,
    ) -> List[ActionResult]:
        """
        Choose one action per decision point, for a BATCH of
        simultaneous decision points (e.g. one per sub-env of a
        VectorizedPokerEnv).

        Numerically identical, example-by-example, to calling act() once
        per (obs, legal_mask) pair — this is purely a throughput
        optimisation (one forward_batch() call instead of
        len(obs_batch) individual forward() calls; one batched
        Categorical instead of len(obs_batch) individual ones).

        Parameters
        ----------
        obs_batch : Sequence[RepresentationInput]
            One decision point per hand. All hands may be at different
            points in different hands (different games entirely, if
            RepresentationModel.forward_batch()'s uniform board/point
            count requirement is satisfied — see that method's
            docstring for the one constraint this inherits).
        legal_masks : Sequence[Sequence[bool]]
            One legal_mask per example, same order as obs_batch. Each
            of length NUM_ACTIONS.
        deterministic : bool
            If True, every row takes its argmax action (evaluation
            mode). If False (default), every row samples independently
            from its own Categorical distribution — i.e. this remains a
            genuine independent sample per hand, not one shared sample
            broadcast across the batch.

        Returns
        -------
        List[ActionResult]
            Length len(obs_batch), in the same order as obs_batch. Each
            entry's log_prob/value/entropy is a 0-dim (scalar) tensor,
            matching act()'s own per-call return shape, so callers
            (RolloutBuffer.add(), etc.) don't need to special-case the
            batched path.

        Raises
        ------
        ValueError
            If obs_batch and legal_masks have different lengths, or
            either is empty.
        """
        n = len(obs_batch)
        if n == 0:
            raise ValueError("act_batch() requires a non-empty obs_batch")
        if len(legal_masks) != n:
            raise ValueError(
                f"obs_batch has {n} examples but legal_masks has "
                f"{len(legal_masks)} — they must match 1:1."
            )

        latents = self.representation_model.forward_batch(obs_batch)   # (n, latent_dim)
        logits, values = self.policy_value_network(latents)             # (n, action_dim), (n,)

        mask_t = torch.as_tensor(legal_masks, dtype=torch.bool, device=logits.device)
        masked_logits = apply_legal_mask(logits, mask_t)                # (n, action_dim)

        dist = Categorical(logits=masked_logits)

        if deterministic:
            actions = torch.argmax(masked_logits, dim=-1)                # (n,)
        else:
            actions = dist.sample()                                      # (n,) — independent per row

        log_probs = dist.log_prob(actions)   # (n,)
        entropies = dist.entropy()           # (n,)

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
    # PPO update-time API (batched, gradients required)
    # ------------------------------------------------------------------

    def evaluate_actions(
        self,
        obs_batch: Sequence[RepresentationInput],
        legal_masks: Sequence[Sequence[bool]],
        actions: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate a batch of (state, action) pairs from a rollout buffer.

        Unchanged from the original implementation — already batched.
        See original docstring for the full PPO-update rationale.
        """
        if not (len(obs_batch) == len(legal_masks) == len(actions)):
            raise ValueError(
                f"obs_batch ({len(obs_batch)}), legal_masks "
                f"({len(legal_masks)}), and actions ({len(actions)}) "
                f"must all have the same length"
            )

        latents = self.representation_model.forward_batch(obs_batch)  # (batch, latent_dim)
        logits, values = self.policy_value_network(latents)            # (batch, action_dim), (batch,)

        mask_t = torch.as_tensor(legal_masks, dtype=torch.bool, device=logits.device)
        masked_logits = apply_legal_mask(logits, mask_t)

        dist = Categorical(logits=masked_logits)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=logits.device)

        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy()

        return log_probs, values, entropy
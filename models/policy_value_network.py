"""
models/policy_value_network.py — Policy and value heads on top of the
Representation Model's latent vector, per the RL Lab implementation
guide (§7, "Policy Learning"):

    The initial implementation should use PPO. PPO should remain
    isolated from: observations, representations, environments.

    Policy Outputs: The network should initially produce policy logits
    and state value. Additional auxiliary prediction heads may be added
    without changing the public interfaces.

Where this sits in the pipeline (§2, "Overall Architecture")
------------------------------------------------------------
    Scenario Generator -> Environment -> Structured Observation
        -> Representation Model -> Latent Representation
        -> Policy Network -> Chosen Action -> Environment -> CAP Engine

representation/representation_model.py produces the latent vector. This
module is everything from there onward through "Chosen Action": policy
logits, state value, legal-action masking, and sampling. It is
deliberately split into two pieces to match the guide's isolation
requirement:

    PolicyValueNetwork  — pure nn.Module: latent (batch, latent_dim) ->
                           (logits (batch, NUM_ACTIONS), value (batch,)).
                           Knows nothing about legal actions, masking,
                           sampling, or PPO. This is the reusable
                           "Neural Architecture" piece (§6) — swapping
                           MLP for a transformer body later only touches
                           this class.

    PokerActorCritic    — composes a RepresentationModel + a
                           PolicyValueNetwork and adds legal-action
                           masking, action sampling, and the
                           act()/evaluate_actions() surface a PPO loop
                           needs. This is the "isolation boundary": a
                           future Deep CFR / NFSP / ReBeL implementation
                           (§7) can share PolicyValueNetwork and
                           RepresentationModel but replace this class
                           with something that doesn't sample a single
                           action from a Categorical distribution.

Legal-action masking
---------------------
actions/abstract_action.py's docstring describes masking as "the Policy
multiplies its logits by this mask before softmax." This module instead
sets illegal logits to a large negative constant (-1e9) before softmax,
not zero-multiplication: zeroing a logit does not zero its
post-softmax probability (softmax(0) is not 0), and a legal action can
legitimately have a raw logit near zero. Additive masking with -inf is
the standard, numerically-stable way to guarantee illegal actions get
exactly zero probability regardless of raw logit magnitude.

Dependencies
------------
    PyTorch. actions.abstract_action for NUM_ACTIONS.
    representation.representation_model.RepresentationModel /
    RepresentationInput for PokerActorCritic (not needed by
    PolicyValueNetwork alone).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from actions.abstract_action import NUM_ACTIONS
from representation.representation_model import RepresentationModel, RepresentationInput

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

    Composes a RepresentationModel (structured observation -> latent) and
    a PolicyValueNetwork (latent -> logits, value), and adds legal-action
    masking + Categorical sampling. This is the concrete "Policy Network"
    box from the guide's §2 architecture diagram; downstream code
    (run_episode() in envs/base_env.py, a future PPO trainer) should
    depend on this class's act()/evaluate_actions() methods rather than
    reaching into RepresentationModel or PolicyValueNetwork directly.

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

        Parameters
        ----------
        obs : RepresentationInput
            Adapted from the Environment's StructuredObservation (see
            representation_model.py's RepresentationInput docstring —
            that adapter is env-specific and lives outside this module,
            per the guide's "Environment must remain independent of the
            neural network architecture").
        legal_mask : Sequence[bool]
            Length NUM_ACTIONS. Directly from
            StructuredObservation.legal_mask.
        deterministic : bool
            If True, take the argmax action (evaluation mode) instead of
            sampling (training mode). Default False.

        Returns
        -------
        ActionResult
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

        This is the method a PPO update step calls: given states and the
        actions actually taken during rollout (under the OLD policy),
        return the CURRENT policy's log-probability of those same
        actions, the current value estimates, and the current entropy —
        the three quantities the clipped surrogate objective and value
        loss need.

        Parameters
        ----------
        obs_batch : Sequence[RepresentationInput]
        legal_masks : Sequence[Sequence[bool]]
            One legal_mask per example, same order as obs_batch.
        actions : Sequence[int]
            Action (ActionType int value) actually taken at each state.

        Returns
        -------
        log_probs : torch.Tensor
            Shape (batch,).
        values : torch.Tensor
            Shape (batch,).
        entropy : torch.Tensor
            Shape (batch,).
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
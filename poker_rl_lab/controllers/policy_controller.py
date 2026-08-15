"""
controllers/policy_controller.py — Controllers backed by a
PokerActorCritic (models/policy_value_network.py), per
RL_lab_implementation_guide.md (§9, "Self Play"):

    Controllers may include:
    ...
    - FrozenPolicyController
    - LearningPolicyController

Both wrap the exact same underlying pieces a hero uses at inference time
(evaluation.policy_evaluator.policy_from_actor_critic's own composition:
adapter.adapt(obs) -> actor_critic.act(...)) — a Controller playing a
non-hero seat is architecturally identical to the hero, just not the one
being optimised this iteration (see base_controller.py's module
docstring). Neither class duplicates that adaptation logic; both funnel
through the same ObservationAdapter.adapt() -> PokerActorCritic.act()
path so a bug fix or representation change to either never has to be
made twice.

FrozenPolicyController vs. LearningPolicyController — the actual
difference
--------------------------------------------------------------------------
Both hold a PokerActorCritic and act via .act(..., deterministic=...).
The distinction the guide's naming implies is about GRADIENT FLOW and
TRAINING PARTICIPATION, not architecture:

    FrozenPolicyController
        A fixed opponent — e.g. a past checkpoint loaded for league play
        (guide §10: "Previous Checkpoints") or simply "the best policy
        so far" held fixed while a challenger trains. Always acts under
        torch.no_grad() regardless of the wrapped network's own
        requires_grad state, and never contributes transitions to a
        RolloutBuffer itself. Use this for every seat that is NOT the
        one currently being trained.

    LearningPolicyController
        The seat actually being optimised this training run. Structurally
        it's the same act() call, but it exposes the extra hooks a PPO
        loop needs to actually learn from this seat's decisions: it can
        optionally record each transition into a caller-supplied
        RolloutBuffer as it acts (mirroring exactly what
        PPOTrainer.collect_rollouts() already does for the hero in
        envs/base_env.py-driven single-seat training — see
        trainers/ppo_trainer.py), so a future multi-seat self-play loop
        can reuse this Controller for the learning seat(s) and get
        PPO-ready transitions out without re-deriving that bookkeeping.

Today only one learning seat is required (guide §9: "Initially only one
learning seat is required."), and PPOTrainer.collect_rollouts() already
implements that single-seat case directly against BasePokerEnv, without
going through a Controller at all. LearningPolicyController therefore
isn't wired into the training loop yet — it exists so the Controller
abstraction is COMPLETE per the guide's list today, ready for the
multi-seat self-play trainer §9 describes as a "future implementation",
without requiring a redesign when that trainer is built. Its optional
recording path is written and unit-testable now rather than deferred.

KNOWN GAP inherited from base_controller.py
----------------------------------------------
Both Controllers here expect a real StructuredObservation (hole cards,
board, betting, history) to make a meaningful decision — that's the
whole point of a policy-backed Controller. See
base_controller.py's module docstring: today's RiverEnv/FullHoldemEnv
villain slots only pass legal_mask for non-hero seats, so wrapping
either of these in LegalMaskControllerAdapter to plug into an existing
env's villain_policy slot would feed them a synthesized, card-blind
observation — explicitly NOT a supported use of that adapter (see its
own docstring). These two Controllers are correct and ready for a
future env that hands every seat its own real observation; they are not
a workaround for today's mask-only villain interface.

Dependencies
------------
    PyTorch (torch.no_grad only). actions.abstract_action,
    envs.observations, controllers.base_controller.Controller.
    models.policy_value_network.PokerActorCritic and
    representation.observation_adapter's AdapterLike surface are
    duck-typed (constructor parameters), matching the same pattern
    trainers/ppo_trainer.py already uses for ActorCriticLike/AdapterLike
    — no hard import of either concrete class is required to type-check
    or unit-test this module with stubs.
"""

from __future__ import annotations

from typing import Optional, Protocol

import torch

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.controllers.base_controller import Controller
from poker_rl_lab.envs.observations import StructuredObservation


# ---------------------------------------------------------------------------
# Duck-typed interfaces (documentation + isolation, not enforced at runtime)
# ---------------------------------------------------------------------------

class ActorCriticLike(Protocol):
    """The subset of models.policy_value_network.PokerActorCritic these
    Controllers call. See trainers/ppo_trainer.py's identical Protocol
    for the same rationale (testable with a lightweight stub, no TF-backed
    hand encoder or poker_eval extension required)."""
    def act(self, obs, legal_mask, deterministic: bool = False): ...


class AdapterLike(Protocol):
    """representation.observation_adapter.ObservationAdapter's surface."""
    def adapt(self, obs): ...


class RolloutBufferLike(Protocol):
    """trainers.rollout_buffer.RolloutBuffer's recording surface — only
    the one method LearningPolicyController optionally calls."""
    def add(
        self, obs, legal_mask, action, log_prob, value, reward, done
    ) -> None: ...


# ---------------------------------------------------------------------------
# FrozenPolicyController
# ---------------------------------------------------------------------------

class FrozenPolicyController(Controller):
    """
    A fixed, non-training opponent backed by a (possibly previously
    trained) PokerActorCritic.

    Always acts deterministically under torch.no_grad() — appropriate
    for both "Previous Checkpoints" and "Current Best" evaluation
    baselines (guide §10) and for holding an opponent fixed during
    another seat's training. Use evaluation.policy_evaluator's
    save_checkpoint/load_checkpoint (via the owning Experiment) to
    produce the actor_critic this wraps from a past run.

    Parameters
    ----------
    actor_critic : ActorCriticLike
    adapter : AdapterLike
        Converts each StructuredObservation into the RepresentationInput
        actor_critic.act() expects — same role as
        evaluation.policy_evaluator.policy_from_actor_critic's `adapter`
        parameter.
    deterministic : bool
        If True (default), always take the argmax action — matching
        "Evaluation environments should remain deterministic" (guide
        §10) and the sensible default for a fixed opponent (no reason
        to add sampling noise to a policy that isn't learning from it).
        Set False for a frozen-but-stochastic opponent, e.g. to give a
        training seat a wider distribution of counterplay to learn
        against.
    """

    def __init__(
        self,
        actor_critic: ActorCriticLike,
        adapter: AdapterLike,
        deterministic: bool = True,
    ) -> None:
        self._actor_critic = actor_critic
        self._adapter = adapter
        self._deterministic = deterministic

    def act(self, obs: StructuredObservation) -> AbstractAction:
        rep_input = self._adapter.adapt(obs)
        with torch.no_grad():
            result = self._actor_critic.act(
                rep_input, obs.legal_mask, deterministic=self._deterministic
            )
        return AbstractAction(ActionType(result.action_type))


# ---------------------------------------------------------------------------
# LearningPolicyController
# ---------------------------------------------------------------------------

class LearningPolicyController(Controller):
    """
    The seat currently being optimised.

    Structurally identical to FrozenPolicyController's act() call
    (adapter.adapt() -> actor_critic.act()), but:
      - defaults to deterministic=False (sampling), matching PPO's own
        training-time action selection
        (trainers/ppo_trainer.py's collect_rollouts()) rather than
        evaluation's argmax default;
      - action selection still happens under torch.no_grad() (this is
        rollout COLLECTION, not a gradient step — identical rationale to
        PPOTrainer.collect_rollouts()'s own docstring: "no gradient
        should be tracked here"; the actual PPO update later
        recomputes log-probs with gradients via
        PokerActorCritic.evaluate_actions(), exactly as
        trainers/ppo_trainer.py's update() does);
      - can optionally push each decision into a caller-supplied
        RolloutBuffer-like object as it acts, via the `buffer` parameter
        below — so a future multi-seat self-play loop built around
        Controllers can collect this seat's transitions the same way
        PPOTrainer.collect_rollouts() already does for a single-hero env,
        without re-deriving that bookkeeping per §9's future work.

    Parameters
    ----------
    actor_critic : ActorCriticLike
    adapter : AdapterLike
    deterministic : bool
        Default False (sample) — see class docstring.
    buffer : RolloutBufferLike, optional
        If given, every act() call additionally records a Transition via
        buffer.add(...), with reward=0.0 and done=False — matching
        every OTHER intermediate transition PPOTrainer.collect_rollouts()
        records (see envs/base_env.py's BasePokerEnv contract: reward is
        only nonzero, and done only True, on the terminal step). The
        caller (a future self-play driver) is responsible for calling
        buffer.start_episode()/finish_episode() around the episode and
        for patching in the true terminal reward/done on the seat's last
        transition once the hand ends — this Controller has no way to
        know, from inside one act() call, whether the action it just
        chose will end the hand. Omit `buffer` (default None) for
        FrozenPolicyController-equivalent inference-only use (e.g.
        quick manual play-testing) without any recording side effect.
    """

    def __init__(
        self,
        actor_critic: ActorCriticLike,
        adapter: AdapterLike,
        deterministic: bool = False,
        buffer: Optional[RolloutBufferLike] = None,
    ) -> None:
        self._actor_critic = actor_critic
        self._adapter = adapter
        self._deterministic = deterministic
        self._buffer = buffer

    def act(self, obs: StructuredObservation) -> AbstractAction:
        rep_input = self._adapter.adapt(obs)
        with torch.no_grad():
            result = self._actor_critic.act(
                rep_input, obs.legal_mask, deterministic=self._deterministic
            )

        if self._buffer is not None:
            self._buffer.add(
                obs=rep_input,
                legal_mask=obs.legal_mask,
                action=result.action_type,
                log_prob=float(result.log_prob.item()),
                value=float(result.value.item()),
                reward=0.0,
                done=False,
            )

        return AbstractAction(ActionType(result.action_type))
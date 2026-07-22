"""
evaluation/policy_evaluator.py — Benchmark a policy against fixed baselines.

Per RL_lab_implementation_guide.md (§10, "Evaluation"):

    Evaluation is independent from training.
    Policies should be benchmarked against:
        Random, Rule Based, Previous Checkpoints, Current Best
    Metrics should include:
        BB/100, Win Rate, Average Reward, Action Frequencies,
        Training Stability
    Evaluation environments should remain deterministic under fixed
    random seeds.

Independence from training
------------------------------
This module never touches PPOTrainer, RolloutBuffer, or an optimizer —
only envs.base_env.run_episode (the already-existing, training-agnostic
episode runner) plus any HeroPolicy callable. A trained actor_critic is
turned into that same callable shape via policy_from_actor_critic()
below, so it plugs into exactly the same evaluate_policy() /
compare_policies() functions as every baseline in
evaluation.baseline_policies and envs.base_env.RandomHeroPolicy — the
evaluation harness genuinely cannot tell a neural policy from a
hand-written one, which is the point (§10 wants training and evaluation
kept structurally separate, not just separate function names).

Determinism
------------
evaluate_policy() reseeds `scenario_gen` with the given seed immediately
before generating its episodes (not once at construction time), so:
  - two calls with the same seed reproduce an identical scenario
    sequence, and
  - compare_policies() gives every candidate policy the SAME sequence
    of hands (reseeding before each one), for a fair head-to-head
    rather than each policy facing different variance in the deal —
    the standard "paired samples" variance-reduction trick.
This only controls which hands are dealt. It does NOT make a
stochastic policy's own action sampling deterministic — use
policy_from_actor_critic(..., deterministic=True) for that, which is
also the default (matching "evaluation should be deterministic").

BB/100 convention
--------------------
bb_per_100 = mean_reward_per_hand / big_blind * 100 — the standard
poker convention for reporting win rate independent of stake size.
big_blind is supplied explicitly rather than read off the env, since
neither PushFoldEnv nor RiverEnv exposes one fixed big_blind of their
own (it's scenario-dependent, to allow variable-stakes generators).

Dependencies
------------
    Standard library, PyTorch (only for policy_from_actor_critic's
    torch.no_grad()), envs.base_env (BasePokerEnv, run_episode),
    actions.abstract_action. No poker_eval, no TensorFlow — those only
    become relevant transitively if the actor_critic / adapter passed
    into policy_from_actor_critic need them.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from actions.abstract_action import AbstractAction, ActionType
from envs.base_env import BasePokerEnv, run_episode
from envs.observations import StructuredObservation

HeroPolicy = Callable[[StructuredObservation], AbstractAction]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Summary statistics for one policy's evaluation run."""
    policy_name: str
    num_episodes: int
    mean_reward: float
    std_error: float                # standard error of the mean reward
    reward_std: float                # raw std across episode rewards ("stability")
    bb_per_100: float
    win_rate: float                  # fraction of episodes with reward > 0
    action_counts: Dict[str, int] = field(default_factory=dict)
    action_frequencies: Dict[str, float] = field(default_factory=dict)

    def confidence_interval_95(self) -> Tuple[float, float]:
        """(low, high) 95% CI on mean_reward, normal approximation."""
        margin = 1.96 * self.std_error
        return (self.mean_reward - margin, self.mean_reward + margin)

    def summary(self) -> str:
        lo, hi = self.confidence_interval_95()
        return (
            f"{self.policy_name}: "
            f"reward={self.mean_reward:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]), "
            f"bb/100={self.bb_per_100:+.2f}, "
            f"win_rate={self.win_rate:.1%}, "
            f"n={self.num_episodes}"
        )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    env: BasePokerEnv,
    scenario_gen,
    policy: HeroPolicy,
    num_episodes: int,
    big_blind: int,
    seed: Optional[int] = None,
    policy_name: str = "policy",
) -> EvaluationResult:
    """
    Run `policy` for `num_episodes` hands and summarise the results.

    Parameters
    ----------
    env : envs.base_env.BasePokerEnv
        Any concrete environment (PushFoldEnv, RiverEnv, ...).
    scenario_gen : scenarios.scenario_generator.ScenarioGenerator
    policy : HeroPolicy
        Callable(obs) -> AbstractAction — see run_episode()'s
        hero_policy parameter (envs/base_env.py) for the exact
        contract this must satisfy.
    num_episodes : int
        Number of hands to play. Larger reduces std_error but does not
        change bias — this is variance reduction, not correction for a
        biased evaluator (e.g. mock_poker_eval).
    big_blind : int
        Big blind size in chips, for BB/100 normalisation. Must be > 0.
    seed : int, optional
        If given, scenario_gen.seed(seed) is called immediately before
        generating episodes — see module docstring on determinism.
        Omit for an unseeded (fresh-RNG-state) run.
    policy_name : str
        Label attached to the returned EvaluationResult, for logging.

    Returns
    -------
    EvaluationResult
    """
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    if big_blind <= 0:
        raise ValueError(f"big_blind must be > 0, got {big_blind}")

    if seed is not None:
        scenario_gen.seed(seed)

    rewards: List[float] = []
    action_counts: Counter = Counter()

    for _ in range(num_episodes):
        scenario = scenario_gen.generate()
        total_reward, trajectory = run_episode(env, scenario, policy)
        rewards.append(total_reward)
        for step in trajectory:
            action_counts[step.action.action_type.name] += 1

    mean_reward = statistics.mean(rewards)
    reward_std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    std_error = reward_std / math.sqrt(num_episodes) if num_episodes > 1 else 0.0
    win_rate = sum(1 for r in rewards if r > 0) / num_episodes

    total_actions = sum(action_counts.values())
    action_frequencies = (
        {name: count / total_actions for name, count in action_counts.items()}
        if total_actions > 0
        else {}
    )

    return EvaluationResult(
        policy_name=policy_name,
        num_episodes=num_episodes,
        mean_reward=mean_reward,
        std_error=std_error,
        reward_std=reward_std,
        bb_per_100=mean_reward / big_blind * 100.0,
        win_rate=win_rate,
        action_counts=dict(action_counts),
        action_frequencies=action_frequencies,
    )


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------

def compare_policies(
    env: BasePokerEnv,
    scenario_gen,
    policies: Dict[str, HeroPolicy],
    num_episodes: int,
    big_blind: int,
    seed: Optional[int] = None,
) -> Dict[str, EvaluationResult]:
    """
    Evaluate several policies, optionally against the identical
    sequence of dealt hands.

    Parameters
    ----------
    policies : Dict[str, HeroPolicy]
        Name -> policy callable, e.g.::

            {
                "random":       RandomHeroPolicy(seed=0),
                "always_fold":  always_fold_policy,
                "calling_station": calling_station_policy,
                "trained":      policy_from_actor_critic(actor_critic, adapter),
            }
    seed : int, optional
        Reseeded before EACH policy's episodes (not once overall), so
        every policy faces the identical scenario sequence — paired
        samples, reducing variance in the comparison versus each policy
        getting independently-dealt hands. Omit for unpaired samples.

    Returns
    -------
    Dict[str, EvaluationResult]
        Same keys as `policies`.
    """
    results: Dict[str, EvaluationResult] = {}
    for name, policy in policies.items():
        results[name] = evaluate_policy(
            env, scenario_gen, policy, num_episodes, big_blind,
            seed=seed, policy_name=name,
        )
    return results


# ---------------------------------------------------------------------------
# Actor-critic adapter
# ---------------------------------------------------------------------------

def policy_from_actor_critic(
    actor_critic,
    adapter,
    deterministic: bool = True,
) -> HeroPolicy:
    """
    Adapt a trained actor-critic into a HeroPolicy callable for use with
    evaluate_policy() / compare_policies() / run_episode().

    Parameters
    ----------
    actor_critic
        models.policy_value_network.PokerActorCritic, or any object
        duck-typing .act(obs, legal_mask, deterministic) ->
        ActionResult-shaped result (see that module).
    adapter
        representation.observation_adapter.ObservationAdapter, or any
        object duck-typing .adapt(obs) -> RepresentationInput.
    deterministic : bool
        If True (default — the standard choice for evaluation, per the
        guide's "Evaluation environments should remain deterministic"):
        always take the argmax action. If False, sample from the
        policy's distribution, matching training-time behaviour —
        useful for measuring exploration/variance rather than
        best-play skill.

    Returns
    -------
    HeroPolicy
    """
    def policy(obs: StructuredObservation) -> AbstractAction:
        rep_input = adapter.adapt(obs)
        with torch.no_grad():
            result = actor_critic.act(
                rep_input, obs.legal_mask, deterministic=deterministic
            )
        return AbstractAction(ActionType(result.action_type))

    return policy
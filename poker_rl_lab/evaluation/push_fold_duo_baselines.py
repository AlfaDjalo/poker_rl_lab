"""
evaluation/push_fold_duo_baselines.py — Fixed, non-learning baselines and
a two-seat evaluation harness for the push-fold-duo curriculum stage.

Per RL_lab_implementation_guide.md (§10, "Evaluation"):

    Policies should be benchmarked against:
        Random, Rule Based, Previous Checkpoints, Current Best

evaluation.baseline_policies / evaluation.policy_evaluator already cover
this for every single-hero BasePokerEnv stage. Push-fold-duo has no
single hero (see envs/push_fold_duo_env.py's own docstring on why it
does NOT implement BasePokerEnv) — a "policy" here is really a PAIR, one
per seat, and env.step()'s reward is per-seat. So this module is the
duo-shaped analog: fixed rule-based policies for EACH seat individually
(any combination can be evaluated against any other), plus a harness
that plays a (sb_policy, bb_policy) pair through PushFoldDuoEnv and
reports both seats' results.

Seat-shaped policies
--------------------
A DuoSeatPolicy has the exact same shape as
evaluation.baseline_policies.HeroPolicy
(``Callable[[StructuredObservation], AbstractAction]``) — it is not a
new type, just a per-seat usage of the same callable contract. This
means:
  - envs.push_fold_duo_env.PushFoldDuoEnv's step() is driven by calling
    sb_policy(obs_sb) then, if SB pushed, bb_policy(obs_bb) — see
    evaluate_duo_matchup() below.
  - a trained models.dual_seat_policy.DualSeatActorCritic seat can be
    wrapped into this exact same shape via
    duo_seat_policy_from_actor_critic() (the duo analog of
    evaluation.policy_evaluator.policy_from_actor_critic), so baseline
    and trained seats plug into evaluate_duo_matchup() interchangeably.

Rule-based baselines
---------------------
SB only ever sees {FOLD, ALL_IN}; BB only ever sees {FOLD, CALL} (see
envs/push_fold_duo_env.py's own legal-mask helpers) — so "rule based"
here reduces to the two maximally-simple extremes per seat:

    sb_always_push  — pushes every hand (the widest possible SB range;
                       loses to a well-tuned BB calling range but is the
                       correct floor/ceiling probe for "does BB call
                       correctly against literally anything").
    sb_always_fold  — folds every hand (SB's maximally-passive floor).
    bb_always_call  — calls every push (the "calling station" analog
                       for this game; useful for checking that a trained
                       SB's push range shows a profit against a BB that
                       never gives up the fight).
    bb_always_fold  — folds to every push (BB's maximally-passive floor;
                       a trained SB should trivially beat this, since it
                       banks the whole pot on any push).

Dependencies
------------
    PyTorch (torch.no_grad only, inside duo_seat_policy_from_actor_critic).
    actions.abstract_action, envs.observations, envs.push_fold_duo_env,
    scenarios.push_fold_duo_generator (SB_SEAT / BB_SEAT constants only).
    No poker_eval, no TensorFlow — matches
    evaluation.baseline_policies' own "no heavy dependency" scope.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.envs.observations import StructuredObservation
from poker_rl_lab.envs.push_fold_duo_env import PushFoldDuoEnv
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.scenarios.push_fold_duo_generator import BB_SEAT, SB_SEAT

DuoSeatPolicy = Callable[[StructuredObservation], AbstractAction]


# ---------------------------------------------------------------------------
# Rule-based baselines
# ---------------------------------------------------------------------------

def _preference_policy(preferred_order) -> DuoSeatPolicy:
    """Same pattern as evaluation.baseline_policies.preference_policy —
    duplicated (not imported) since this file has no dependency on that
    module's env-agnostic-but-single-hero framing, and the fallback
    branch below is dead code for SB/BB (FOLD is always legal for both),
    kept only for defensive symmetry with the original."""
    def policy(obs: StructuredObservation) -> AbstractAction:
        for action_type in preferred_order:
            if obs.legal_mask[action_type]:
                return AbstractAction(action_type)
        for i, legal in enumerate(obs.legal_mask):
            if legal:
                return AbstractAction(ActionType(i))
        raise RuntimeError(
            "No legal actions available — this policy should never be "
            "called on a terminal observation."
        )
    return policy


#: SB pushes every hand it's dealt.
sb_always_push: DuoSeatPolicy = _preference_policy([ActionType.ALL_IN, ActionType.FOLD])

#: SB folds every hand it's dealt.
sb_always_fold: DuoSeatPolicy = _preference_policy([ActionType.FOLD, ActionType.ALL_IN])

#: BB calls every push it faces.
bb_always_call: DuoSeatPolicy = _preference_policy([ActionType.CALL, ActionType.FOLD])

#: BB folds to every push it faces.
bb_always_fold: DuoSeatPolicy = _preference_policy([ActionType.FOLD, ActionType.CALL])


# ---------------------------------------------------------------------------
# Trained-policy adapter
# ---------------------------------------------------------------------------

def duo_seat_policy_from_actor_critic(
    actor_critic: DualSeatActorCritic,
    seat: int,
    adapter,
    deterministic: bool = True,
) -> DuoSeatPolicy:
    """
    Adapt one seat of a trained DualSeatActorCritic into a DuoSeatPolicy
    callable — the duo analog of
    evaluation.policy_evaluator.policy_from_actor_critic.

    Parameters
    ----------
    actor_critic : DualSeatActorCritic
    seat : int
        SB_SEAT or BB_SEAT — which seat's policy/value function to use.
    adapter
        representation.observation_adapter.ObservationAdapter, or
        anything duck-typing .adapt(obs) -> RepresentationInput.
    deterministic : bool
        If True (default, matching "Evaluation environments should
        remain deterministic" — guide §10), always take the argmax
        action. If False, sample.

    Returns
    -------
    DuoSeatPolicy
    """
    def policy(obs: StructuredObservation) -> AbstractAction:
        rep_input = adapter.adapt(obs)
        with torch.no_grad():
            result = actor_critic.act(
                seat, rep_input, obs.legal_mask, deterministic=deterministic
            )
        return AbstractAction(ActionType(result.action_type))

    return policy


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DuoEvaluationResult:
    """Summary statistics for one (sb_policy, bb_policy) matchup."""
    sb_name: str
    bb_name: str
    num_episodes: int

    mean_reward_sb: float
    mean_reward_bb: float
    std_error_sb: float
    std_error_bb: float
    bb_per_100_sb: float
    bb_per_100_bb: float

    sb_push_rate: float
    """Fraction of hands SB pushed (vs. folded)."""

    bb_call_rate: float
    """Fraction of hands BB CALLED, out of hands where BB actually got
    to act (i.e. SB pushed). NaN if SB never pushed in this sample."""

    def confidence_interval_95_sb(self) -> Tuple[float, float]:
        margin = 1.96 * self.std_error_sb
        return (self.mean_reward_sb - margin, self.mean_reward_sb + margin)

    def confidence_interval_95_bb(self) -> Tuple[float, float]:
        margin = 1.96 * self.std_error_bb
        return (self.mean_reward_bb - margin, self.mean_reward_bb + margin)

    def summary(self) -> str:
        lo_sb, hi_sb = self.confidence_interval_95_sb()
        lo_bb, hi_bb = self.confidence_interval_95_bb()
        return (
            f"{self.sb_name} (SB) vs {self.bb_name} (BB), n={self.num_episodes}\n"
            f"  SB: reward={self.mean_reward_sb:+.3f} "
            f"(95% CI [{lo_sb:+.3f}, {hi_sb:+.3f}]), "
            f"bb/100={self.bb_per_100_sb:+.2f}, push_rate={self.sb_push_rate:.1%}\n"
            f"  BB: reward={self.mean_reward_bb:+.3f} "
            f"(95% CI [{lo_bb:+.3f}, {hi_bb:+.3f}]), "
            f"bb/100={self.bb_per_100_bb:+.2f}, call_rate={self.bb_call_rate:.1%}"
        )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_duo_matchup(
    env: PushFoldDuoEnv,
    scenario_gen,
    sb_policy: DuoSeatPolicy,
    bb_policy: DuoSeatPolicy,
    num_episodes: int,
    big_blind: int,
    seed: Optional[int] = None,
    sb_name: str = "sb_policy",
    bb_name: str = "bb_policy",
) -> DuoEvaluationResult:
    """
    Play `sb_policy` against `bb_policy` for `num_episodes` hands and
    summarise both seats' results.

    Parameters
    ----------
    env : PushFoldDuoEnv
    scenario_gen : scenarios.push_fold_duo_generator.PushFoldDuoScenarioGenerator
    sb_policy, bb_policy : DuoSeatPolicy
        Any rule-based baseline above, or
        duo_seat_policy_from_actor_critic(...).
    num_episodes : int
    big_blind : int
        For BB/100 normalisation. Must be > 0.
    seed : int, optional
        If given, scenario_gen.seed(seed) is called immediately before
        generating episodes — same determinism convention as
        evaluation.policy_evaluator.evaluate_policy.
    sb_name, bb_name : str
        Labels attached to the returned result, for logging.

    Returns
    -------
    DuoEvaluationResult
    """
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
    if big_blind <= 0:
        raise ValueError(f"big_blind must be > 0, got {big_blind}")

    if seed is not None:
        scenario_gen.seed(seed)

    sb_rewards = []
    bb_rewards = []
    sb_push_count = 0
    bb_acted_count = 0
    bb_call_count = 0

    for _ in range(num_episodes):
        scenario = scenario_gen.generate()
        obs_sb = env.reset(scenario)

        action_sb = sb_policy(obs_sb)
        if action_sb.action_type == ActionType.ALL_IN:
            sb_push_count += 1

        step1 = env.step(action_sb)

        if step1.done:
            rewards = step1.rewards
        else:
            obs_bb = step1.obs
            action_bb = bb_policy(obs_bb)
            bb_acted_count += 1
            if action_bb.action_type == ActionType.CALL:
                bb_call_count += 1

            step2 = env.step(action_bb)
            rewards = step2.rewards

        sb_rewards.append(rewards[SB_SEAT])
        bb_rewards.append(rewards[BB_SEAT])

    mean_sb = statistics.mean(sb_rewards)
    mean_bb = statistics.mean(bb_rewards)
    std_sb = statistics.pstdev(sb_rewards) if len(sb_rewards) > 1 else 0.0
    std_bb = statistics.pstdev(bb_rewards) if len(bb_rewards) > 1 else 0.0
    stderr_sb = std_sb / math.sqrt(num_episodes) if num_episodes > 1 else 0.0
    stderr_bb = std_bb / math.sqrt(num_episodes) if num_episodes > 1 else 0.0

    return DuoEvaluationResult(
        sb_name=sb_name,
        bb_name=bb_name,
        num_episodes=num_episodes,
        mean_reward_sb=mean_sb,
        mean_reward_bb=mean_bb,
        std_error_sb=stderr_sb,
        std_error_bb=stderr_bb,
        bb_per_100_sb=mean_sb / big_blind * 100.0,
        bb_per_100_bb=mean_bb / big_blind * 100.0,
        sb_push_rate=sb_push_count / num_episodes,
        bb_call_rate=(bb_call_count / bb_acted_count) if bb_acted_count > 0 else float("nan"),
    )


# ---------------------------------------------------------------------------
# Convenience: trained policy vs. every baseline
# ---------------------------------------------------------------------------

def evaluate_against_baselines(
    env: PushFoldDuoEnv,
    scenario_gen,
    actor_critic: DualSeatActorCritic,
    adapter,
    num_episodes: int,
    big_blind: int,
    seed: Optional[int] = None,
) -> Dict[str, DuoEvaluationResult]:
    """
    Benchmark a trained DualSeatActorCritic's SB seat and BB seat each
    against the opposite seat's fixed baselines, plus one self-play
    matchup (trained vs. trained).

    Mirrors evaluation.policy_evaluator.compare_policies' spirit for the
    duo case: rather than one hero facing several opponent policies,
    each TRAINED seat here faces every fixed baseline for the OTHER
    seat, since there is no single "opponent" role in this game.

    Parameters
    ----------
    env : PushFoldDuoEnv
    scenario_gen : scenarios.push_fold_duo_generator.PushFoldDuoScenarioGenerator
    actor_critic : DualSeatActorCritic
    adapter
        representation.observation_adapter.ObservationAdapter (or
        duck-typed equivalent).
    num_episodes : int
        Hands per matchup (each of the 5 matchups below gets its own
        `num_episodes` hands).
    big_blind : int
    seed : int, optional
        Reseeded before EACH matchup (not once overall) — every matchup
        faces the identical scenario sequence, mirroring
        evaluation.policy_evaluator.compare_policies' paired-samples
        variance reduction.

    Returns
    -------
    Dict[str, DuoEvaluationResult]
        Keys: "trained_sb_vs_always_call_bb", "trained_sb_vs_always_fold_bb",
        "always_push_sb_vs_trained_bb", "always_fold_sb_vs_trained_bb",
        "trained_sb_vs_trained_bb".
    """
    trained_sb = duo_seat_policy_from_actor_critic(actor_critic, SB_SEAT, adapter)
    trained_bb = duo_seat_policy_from_actor_critic(actor_critic, BB_SEAT, adapter)

    matchups: Dict[str, Tuple[DuoSeatPolicy, DuoSeatPolicy, str, str]] = {
        "trained_sb_vs_always_call_bb": (trained_sb, bb_always_call, "trained", "always_call"),
        "trained_sb_vs_always_fold_bb": (trained_sb, bb_always_fold, "trained", "always_fold"),
        "always_push_sb_vs_trained_bb": (sb_always_push, trained_bb, "always_push", "trained"),
        "always_fold_sb_vs_trained_bb": (sb_always_fold, trained_bb, "always_fold", "trained"),
        "trained_sb_vs_trained_bb": (trained_sb, trained_bb, "trained", "trained"),
    }

    results: Dict[str, DuoEvaluationResult] = {}
    for key, (sb_pol, bb_pol, sb_name, bb_name) in matchups.items():
        results[key] = evaluate_duo_matchup(
            env, scenario_gen, sb_pol, bb_pol, num_episodes, big_blind,
            seed=seed, sb_name=sb_name, bb_name=bb_name,
        )
    return results
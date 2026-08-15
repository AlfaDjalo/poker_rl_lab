"""
evaluation/river_duo_baselines.py — Fixed, non-learning baselines and a
two-seat evaluation harness for the river-duo curriculum stage.

Mirrors evaluation/push_fold_duo_baselines.py exactly in spirit — see
that module's docstring for the full "why a duo-shaped harness, not
evaluation.policy_evaluator" rationale (river-duo has no single hero,
same as push-fold-duo; env.step()'s reward is per-seat).

Why one policy per seat drives BOTH of that seat's possible decision
points
--------------------------------------------------------------------------
Unlike push-fold-duo (exactly one decision per seat, ever), river-duo's
OOP can face two structurally different decisions in the same hand:
the initial CHECK/BET_100 choice, and — only if OOP checked and IP then
bet — a later CALL/FOLD choice. A DuoSeatPolicy is still just
``Callable[[StructuredObservation], AbstractAction]`` (same shape as
push_fold_duo_baselines.DuoSeatPolicy): it reads whatever the CURRENT
obs.legal_mask actually offers and picks the first legal action in a
fixed preference order, so ONE `oop_*` policy callable correctly
handles both of OOP's decision shapes without needing to know which one
it's being asked about — this is exactly
evaluation.baseline_policies.preference_policy's own design, reused
here rather than reinvented.

Rule-based baselines
---------------------
Six baselines total (three shapes x two seats), matching the maximal-
simple extremes push_fold_duo_baselines.py uses for SB/BB:

    {oop,ip}_always_bet        — bets/raises at every opportunity,
                                   calls when facing a bet. The widest
                                   possible aggression range.
    {oop,ip}_always_check_fold — checks when possible, folds to any bet.
                                   The maximally passive floor.
    {oop,ip}_never_folds       — calls any bet, checks/bets otherwise
                                   never folds. "Calling station" analog.

Dependencies
------------
    Standard library only. actions.abstract_action, envs.observations,
    envs.river_duo_env, scenarios.river_duo_generator (OOP_SEAT/IP_SEAT
    constants only). PyTorch only inside
    duo_seat_policy_from_actor_critic (torch.no_grad). No poker_eval, no
    TensorFlow — matches push_fold_duo_baselines.py's own scope.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from actions.abstract_action import AbstractAction, ActionType
from envs.observations import StructuredObservation
from envs.river_duo_env import RiverDuoEnv
from models.dual_seat_policy import DualSeatActorCritic
from scenarios.river_duo_generator import IP_SEAT, OOP_SEAT

DuoSeatPolicy = Callable[[StructuredObservation], AbstractAction]


# ---------------------------------------------------------------------------
# Rule-based baselines
# ---------------------------------------------------------------------------

def _preference_policy(preferred_order) -> DuoSeatPolicy:
    """Same pattern as evaluation.baseline_policies.preference_policy /
    push_fold_duo_baselines._preference_policy — duplicated (not
    imported) since this file has no dependency on either module's
    framing. Falls back to the first legal action if nothing in
    preferred_order is legal (defensive; every river-duo decision point
    offers exactly 2 of {CHECK, BET_100, CALL, FOLD}, all of which are
    covered by every preference order below)."""
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


#: Bets/raises whenever legal, calls whenever facing a bet.
oop_always_bet: DuoSeatPolicy = _preference_policy(
    [ActionType.BET_100, ActionType.CALL, ActionType.CHECK, ActionType.FOLD]
)
ip_always_bet: DuoSeatPolicy = _preference_policy(
    [ActionType.BET_100, ActionType.CALL, ActionType.CHECK, ActionType.FOLD]
)

#: Checks whenever legal, folds to any bet.
oop_always_check_fold: DuoSeatPolicy = _preference_policy(
    [ActionType.CHECK, ActionType.FOLD, ActionType.BET_100, ActionType.CALL]
)
ip_always_check_fold: DuoSeatPolicy = _preference_policy(
    [ActionType.CHECK, ActionType.FOLD, ActionType.BET_100, ActionType.CALL]
)

#: Never folds: calls any bet, checks when free (never bets first itself).
oop_never_folds: DuoSeatPolicy = _preference_policy(
    [ActionType.CALL, ActionType.CHECK, ActionType.BET_100, ActionType.FOLD]
)
ip_never_folds: DuoSeatPolicy = _preference_policy(
    [ActionType.CALL, ActionType.CHECK, ActionType.BET_100, ActionType.FOLD]
)


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
    callable — identical shape to
    push_fold_duo_baselines.duo_seat_policy_from_actor_critic (see that
    function's docstring); duplicated rather than imported to keep this
    module's dependency footprint self-contained, matching the rest of
    the RL Lab's per-stage evaluation modules.

    Parameters
    ----------
    actor_critic : DualSeatActorCritic
    seat : int
        OOP_SEAT or IP_SEAT.
    adapter
        representation.observation_adapter.ObservationAdapter, or
        anything duck-typing .adapt(obs) -> RepresentationInput.
    deterministic : bool
        If True (default), always take the argmax action. If False,
        sample.

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
    """Summary statistics for one (oop_policy, ip_policy) matchup."""
    oop_name: str
    ip_name: str
    num_episodes: int

    mean_reward_oop: float
    mean_reward_ip: float
    std_error_oop: float
    std_error_ip: float
    bb_per_100_oop: float
    bb_per_100_ip: float

    outcome_counts: Dict[str, int]
    """Raw counts of RiverDuoEnv's info["outcome"] values across the
    matchup, e.g. {"check_check": 412, "oop_bet_ip_call": 88, ...} — a
    cheap sanity check that both seats' action mix looks non-degenerate
    (e.g. an all-check-fold baseline should show ~0 *_bet_* outcomes)."""

    def confidence_interval_95_oop(self) -> Tuple[float, float]:
        margin = 1.96 * self.std_error_oop
        return (self.mean_reward_oop - margin, self.mean_reward_oop + margin)

    def confidence_interval_95_ip(self) -> Tuple[float, float]:
        margin = 1.96 * self.std_error_ip
        return (self.mean_reward_ip - margin, self.mean_reward_ip + margin)

    def summary(self) -> str:
        lo_o, hi_o = self.confidence_interval_95_oop()
        lo_i, hi_i = self.confidence_interval_95_ip()
        outcomes = ", ".join(f"{k}={v}" for k, v in sorted(self.outcome_counts.items()))
        return (
            f"{self.oop_name} (OOP) vs {self.ip_name} (IP), n={self.num_episodes}\n"
            f"  OOP: reward={self.mean_reward_oop:+.3f} "
            f"(95% CI [{lo_o:+.3f}, {hi_o:+.3f}]), bb/100={self.bb_per_100_oop:+.2f}\n"
            f"  IP:  reward={self.mean_reward_ip:+.3f} "
            f"(95% CI [{lo_i:+.3f}, {hi_i:+.3f}]), bb/100={self.bb_per_100_ip:+.2f}\n"
            f"  outcomes: {outcomes}"
        )


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_duo_matchup(
    env: RiverDuoEnv,
    scenario_gen,
    oop_policy: DuoSeatPolicy,
    ip_policy: DuoSeatPolicy,
    num_episodes: int,
    big_blind: int,
    seed: Optional[int] = None,
    oop_name: str = "oop_policy",
    ip_name: str = "ip_policy",
) -> DuoEvaluationResult:
    """
    Play `oop_policy` against `ip_policy` for `num_episodes` hands and
    summarise both seats' results.

    Drives the env by reading env.current_seat before every step() and
    querying whichever policy owns that seat — the same "seat-owning
    Controller" pattern controllers/base_controller.py's Controller.act
    already establishes, just without a Controller object (matching
    river_duo_env.py's / push_fold_duo_env.py's own plain-callable
    convention). Handles all five terminal outcomes river-duo can reach
    (check-check, oop-bet-ip-call/fold, ip-bet-oop-call/fold) uniformly:
    the loop just keeps stepping with whichever seat is current until
    done, regardless of how many decisions that took.

    Parameters
    ----------
    env : RiverDuoEnv
    scenario_gen : scenarios.river_duo_generator.RiverDuoScenarioGenerator
    oop_policy, ip_policy : DuoSeatPolicy
        Any rule-based baseline above, or
        duo_seat_policy_from_actor_critic(...).
    num_episodes : int
    big_blind : int
        For BB/100 normalisation. Must be > 0.
    seed : int, optional
        If given, scenario_gen.seed(seed) is called immediately before
        generating episodes — same determinism convention as
        evaluation.policy_evaluator.evaluate_policy /
        push_fold_duo_baselines.evaluate_duo_matchup.
    oop_name, ip_name : str
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

    policies = {OOP_SEAT: oop_policy, IP_SEAT: ip_policy}

    oop_rewards = []
    ip_rewards = []
    outcome_counts: Dict[str, int] = {}

    for _ in range(num_episodes):
        scenario = scenario_gen.generate()
        obs = env.reset(scenario)

        result = None
        while True:
            seat = env.current_seat
            action = policies[seat](obs)
            step = env.step(action)
            if step.done:
                result = step
                break
            obs = step.obs

        outcome = result.info.get("outcome", "unknown")
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        oop_rewards.append(result.rewards[OOP_SEAT])
        ip_rewards.append(result.rewards[IP_SEAT])

    mean_oop = statistics.mean(oop_rewards)
    mean_ip = statistics.mean(ip_rewards)
    std_oop = statistics.pstdev(oop_rewards) if len(oop_rewards) > 1 else 0.0
    std_ip = statistics.pstdev(ip_rewards) if len(ip_rewards) > 1 else 0.0
    stderr_oop = std_oop / math.sqrt(num_episodes) if num_episodes > 1 else 0.0
    stderr_ip = std_ip / math.sqrt(num_episodes) if num_episodes > 1 else 0.0

    return DuoEvaluationResult(
        oop_name=oop_name,
        ip_name=ip_name,
        num_episodes=num_episodes,
        mean_reward_oop=mean_oop,
        mean_reward_ip=mean_ip,
        std_error_oop=stderr_oop,
        std_error_ip=stderr_ip,
        bb_per_100_oop=mean_oop / big_blind * 100.0,
        bb_per_100_ip=mean_ip / big_blind * 100.0,
        outcome_counts=outcome_counts,
    )


# ---------------------------------------------------------------------------
# Convenience: trained policy vs. every baseline
# ---------------------------------------------------------------------------

def evaluate_against_baselines(
    env: RiverDuoEnv,
    scenario_gen,
    actor_critic: DualSeatActorCritic,
    adapter,
    num_episodes: int,
    big_blind: int,
    seed: Optional[int] = None,
) -> Dict[str, DuoEvaluationResult]:
    """
    Benchmark a trained DualSeatActorCritic's OOP seat and IP seat each
    against the opposite seat's fixed baselines, plus one self-play
    matchup — mirrors
    push_fold_duo_baselines.evaluate_against_baselines exactly (see that
    function's docstring for the "no single hero" rationale).

    Parameters
    ----------
    env : RiverDuoEnv
    scenario_gen : scenarios.river_duo_generator.RiverDuoScenarioGenerator
    actor_critic : DualSeatActorCritic
    adapter
        representation.observation_adapter.ObservationAdapter (or
        duck-typed equivalent).
    num_episodes : int
        Hands per matchup (each of the 7 matchups below gets its own
        `num_episodes` hands).
    big_blind : int
    seed : int, optional
        Reseeded before EACH matchup — paired-samples variance
        reduction, same convention as every other evaluate_* function
        in this codebase.

    Returns
    -------
    Dict[str, DuoEvaluationResult]
    """
    trained_oop = duo_seat_policy_from_actor_critic(actor_critic, OOP_SEAT, adapter)
    trained_ip = duo_seat_policy_from_actor_critic(actor_critic, IP_SEAT, adapter)

    matchups: Dict[str, Tuple[DuoSeatPolicy, DuoSeatPolicy, str, str]] = {
        "trained_oop_vs_always_bet_ip": (trained_oop, ip_always_bet, "trained", "always_bet"),
        "trained_oop_vs_check_fold_ip": (trained_oop, ip_always_check_fold, "trained", "check_fold"),
        "trained_oop_vs_never_folds_ip": (trained_oop, ip_never_folds, "trained", "never_folds"),
        "always_bet_oop_vs_trained_ip": (oop_always_bet, trained_ip, "always_bet", "trained"),
        "check_fold_oop_vs_trained_ip": (oop_always_check_fold, trained_ip, "check_fold", "trained"),
        "never_folds_oop_vs_trained_ip": (oop_never_folds, trained_ip, "never_folds", "trained"),
        "trained_oop_vs_trained_ip": (trained_oop, trained_ip, "trained", "trained"),
    }

    results: Dict[str, DuoEvaluationResult] = {}
    for key, (oop_pol, ip_pol, oop_name, ip_name) in matchups.items():
        results[key] = evaluate_duo_matchup(
            env, scenario_gen, oop_pol, ip_pol, num_episodes, big_blind,
            seed=seed, oop_name=oop_name, ip_name=ip_name,
        )
    return results
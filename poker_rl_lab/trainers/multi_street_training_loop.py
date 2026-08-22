"""
trainers/multi_street_training_loop.py — Self-play PPO training loop for
envs.multi_street_env.MultiStreetEnv (Phase D).

Why not reuse duo_training_loop.py's collect_duo_rollouts / DuoTrainingLoop
directly
--------------------------------------------------------------------------
collect_duo_rollouts hardcodes push-fold-duo's exact shape: SB acts
at most once, BB acts at most once, in that fixed order (see its own
docstring: "A push-fold-duo hand has at most two decisions"). A
MultiStreetEnv hand can have an arbitrary number of decisions per seat
(bet/raise/re-raise across several streets), so the per-hand splitting
logic has to iterate MultiStreetEnv.run_hand()'s actual per-seat
decision lists instead of assuming exactly one SB step then exactly one
BB step. Everything else duo_training_loop.py already got right for
the two-seat case — _ppo_update_seat (the clipped-surrogate PPO math)
and _infer_device — is imported and reused verbatim, not duplicated,
same pattern trainers/generic_tree_training_loop.py already established
for exactly this reason (see that module's own docstring).

KNOWN LIMITATION — 2 seats only
------------------------------------
Same limitation generic_tree_training_loop.py and generic_tree_experiment.py
already flag for their own reasons: models.dual_seat_policy.DualSeatActorCritic
(and both concrete implementations) validate `seat in (0, 1)` via
_check_seat. This loop is therefore written for exactly 2 seats (SB/BB)
— PreflopScenarioGenerator itself is N-seat-general, but
MultiStreetTrainingLoop is not, until a wider actor-critic exists.

Not yet batched
-----------------
collect_multi_street_rollouts() plays hands one at a time, calling
actor_critic.act() (batch-of-1) per decision — the same starting point
trainers/duo_training_loop.py's own collect_duo_rollouts had before
trainers/generic_tree_training_loop.py's lockstep-batched rewrite (see
that module's own "BATCHED rollout collection" docstring section).
MultiStreetEnv isn't cheaply cloneable the way GenericTreeEnv is (it
wraps a live PokerState/GameState rather than a handful of plain
Python lists), so the same lockstep-multiple-hands-per-tick batching
trick would need its own design pass — left as a follow-up performance
improvement, not required for this loop to train correctly.

Dependencies
------------
    PyTorch. actions.abstract_action, envs.multi_street_env,
    scenarios.preflop_scenario_generator.PreflopScenarioGenerator (or
    any object exposing the same seed()/generate() surface),
    models.dual_seat_policy.DualSeatActorCritic,
    trainers.rollout_buffer.RolloutBuffer,
    trainers.duo_training_loop (_ppo_update_seat, _infer_device — reused,
    not duplicated).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.envs.multi_street_env import MultiStreetEnv, PolicyDecision
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.trainers.duo_training_loop import _infer_device, _ppo_update_seat
from poker_rl_lab.trainers.rollout_buffer import RolloutBuffer

SB_SEAT: int = 0
BB_SEAT: int = 1
_SEAT_INDICES = (SB_SEAT, BB_SEAT)


# ---------------------------------------------------------------------------
# Per-iteration diagnostics
# ---------------------------------------------------------------------------

@dataclass
class MultiStreetIterationStats:
    """Diagnostics for one training-loop iteration. SIMULTANEOUS mode
    only (see module docstring) — both seats are always updated, so
    every dict below always has both SB_SEAT and BB_SEAT keys."""
    iteration: int
    mean_reward_by_seat: Dict[int, float] = field(default_factory=dict)
    mean_decisions_by_seat: Dict[int, float] = field(default_factory=dict)
    policy_loss: Dict[int, float] = field(default_factory=dict)
    value_loss: Dict[int, float] = field(default_factory=dict)
    entropy: Dict[int, float] = field(default_factory=dict)
    clip_fraction: Dict[int, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def _mean_reward(buffer: RolloutBuffer) -> float:
    if len(buffer) == 0:
        return 0.0
    # Only the terminal transition of each episode carries a nonzero
    # reward (see RolloutBuffer's own module docstring) — sum ALL
    # transitions (nonterminal ones contribute 0.0) and divide by the
    # number of EPISODES, not the number of transitions, so this is a
    # true mean reward per hand rather than diluted by hands with more
    # decisions.
    total = sum(t.reward for t in buffer._transitions)
    num_episodes = sum(1 for t in buffer._transitions if t.done)
    return total / num_episodes if num_episodes else 0.0


def _mean_decisions(buffer: RolloutBuffer) -> float:
    num_episodes = sum(1 for t in buffer._transitions if t.done)
    return len(buffer) / num_episodes if num_episodes else 0.0


# ---------------------------------------------------------------------------
# Training-time policy wrapper
# ---------------------------------------------------------------------------

def _make_training_policy(actor_critic: DualSeatActorCritic, seat: int, adapter):
    """
    Wrap actor_critic into the TrainingPolicy shape
    MultiStreetEnv.run_hand() expects: Callable[[StructuredObservation],
    PolicyDecision]. Sampling (deterministic=False) — this is rollout
    collection under the current ("old") policy, identical rationale to
    every other collect_*_rollouts() in this codebase.
    """
    def policy(obs) -> PolicyDecision:
        rep_input = adapter.adapt(obs)
        with torch.no_grad():
            result = actor_critic.act(seat, rep_input, obs.legal_mask, deterministic=False)
        return PolicyDecision(
            action=AbstractAction(ActionType(result.action_type)),
            log_prob=float(result.log_prob.item()),
            value=float(result.value.item()),
        )

    return policy


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_multi_street_rollouts(
    env: MultiStreetEnv,
    scenario_gen,
    actor_critic: DualSeatActorCritic,
    adapter,
    num_episodes: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Dict[int, RolloutBuffer]:
    """
    Play `num_episodes` hands via MultiStreetEnv.run_hand(), splitting
    each hand's per-seat decision list into a per-seat RolloutBuffer.

    Parameters
    ----------
    env : MultiStreetEnv
    scenario_gen
        Any object exposing seed()/generate() -> BaseScenario for
        exactly 2 players (e.g. scenarios.preflop_scenario_generator.
        PreflopScenarioGenerator(player_count=2, ...)).
    actor_critic : DualSeatActorCritic
    adapter : representation.observation_adapter.ObservationAdapter
    num_episodes : int
    gamma, gae_lambda : float
        Forwarded to both per-seat RolloutBuffers. Unlike push-fold-duo
        (where every episode is a single decision and GAE degenerates
        to advantage = reward - value — see duo_training_loop.py's own
        docstring), a MultiStreetEnv hand can have several decisions
        per seat across streets, so GAE does real work here.

    Returns
    -------
    Dict[int, RolloutBuffer]
        Keys SB_SEAT, BB_SEAT. Buffers may have very different lengths
        across hands (a hand where SB folds preflop gives BB zero
        decisions that hand; a multi-street re-raising war gives both
        seats several).
    """
    buffers: Dict[int, RolloutBuffer] = {
        s: RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda) for s in _SEAT_INDICES
    }
    policies = {
        s: _make_training_policy(actor_critic, s, adapter) for s in _SEAT_INDICES
    }

    for _ in range(num_episodes):
        scenario = scenario_gen.generate()
        decisions, rewards = env.run_hand(scenario, policies)

        for seat in _SEAT_INDICES:
            seat_decisions = decisions.get(seat, [])
            if not seat_decisions:
                # Cannot happen preflop (both seats always act at least
                # once — see PreflopScenarioGenerator), but guarded the
                # same way generic_tree_training_loop.py's own bugfix
                # guards against a seat with zero decisions this hand.
                continue

            buffers[seat].start_episode()
            last_idx = len(seat_decisions) - 1
            for idx, d in enumerate(seat_decisions):
                rep_input = adapter.adapt(d.obs)
                is_terminal = idx == last_idx
                buffers[seat].add(
                    obs=rep_input,
                    legal_mask=d.obs.legal_mask,
                    action=int(d.action.action_type),
                    log_prob=d.log_prob,
                    value=d.value,
                    reward=(rewards[seat] if is_terminal else 0.0),
                    done=is_terminal,
                )
            buffers[seat].finish_episode()

    return buffers


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class MultiStreetTrainingLoop:
    """
    Orchestrates collect -> update cycles for a DualSeatActorCritic over
    a MultiStreetEnv, SIMULTANEOUS mode only (see module docstring's
    KNOWN LIMITATION).

    Parameters
    ----------
    env : MultiStreetEnv
    scenario_gen
        See collect_multi_street_rollouts()'s own docstring.
    actor_critic : DualSeatActorCritic
    adapter : representation.observation_adapter.ObservationAdapter
    optimizer_sb, optimizer_bb : torch.optim.Optimizer
        Pre-constructed, scoped to actor_critic.seat_parameters(SB_SEAT)
        / actor_critic.seat_parameters(BB_SEAT) respectively — same
        convention as DuoTrainingLoop.
    """

    def __init__(
        self,
        env: MultiStreetEnv,
        scenario_gen,
        actor_critic: DualSeatActorCritic,
        adapter,
        optimizer_sb: torch.optim.Optimizer,
        optimizer_bb: torch.optim.Optimizer,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ) -> None:
        self.env = env
        self.scenario_gen = scenario_gen
        self.actor_critic = actor_critic
        self.adapter = adapter
        self.optimizers: Dict[int, torch.optim.Optimizer] = {
            SB_SEAT: optimizer_sb, BB_SEAT: optimizer_bb,
        }
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def train(
        self,
        num_iterations: int,
        episodes_per_iteration: int = 256,
        epochs: int = 4,
        minibatch_size: int = 64,
        log_every: int = 10,
        verbose: bool = True,
    ) -> List[MultiStreetIterationStats]:
        history: List[MultiStreetIterationStats] = []
        start_time = time.time()

        for i in range(num_iterations):
            buffers = collect_multi_street_rollouts(
                self.env, self.scenario_gen, self.actor_critic, self.adapter,
                episodes_per_iteration, gamma=self.gamma, gae_lambda=self.gae_lambda,
            )

            policy_loss, value_loss, entropy, clip_fraction = {}, {}, {}, {}
            for seat in _SEAT_INDICES:
                loss = _ppo_update_seat(
                    self.actor_critic, seat, buffers[seat], self.optimizers[seat],
                    epochs, minibatch_size, self.clip_epsilon, self.value_coef,
                    self.entropy_coef, self.max_grad_norm,
                )
                policy_loss[seat] = loss["policy_loss"]
                value_loss[seat] = loss["value_loss"]
                entropy[seat] = loss["entropy"]
                clip_fraction[seat] = loss["clip_fraction"]

            stats = MultiStreetIterationStats(
                iteration=i,
                mean_reward_by_seat={s: _mean_reward(buffers[s]) for s in _SEAT_INDICES},
                mean_decisions_by_seat={s: _mean_decisions(buffers[s]) for s in _SEAT_INDICES},
                policy_loss=policy_loss, value_loss=value_loss,
                entropy=entropy, clip_fraction=clip_fraction,
            )

            elapsed = time.time() - start_time
            stats.elapsed_seconds = elapsed
            history.append(stats)

            iteration_num = i + 1
            if verbose and log_every > 0 and (
                iteration_num % log_every == 0 or iteration_num == num_iterations
            ):
                print(
                    f"[MultiStreetTrainingLoop] {iteration_num}/{num_iterations} "
                    f"iterations completed ({elapsed:.1f}s elapsed, "
                    f"{elapsed / iteration_num:.2f}s/iter)"
                )

        if verbose:
            total_time = time.time() - start_time
            avg = total_time / num_iterations if num_iterations else 0.0
            print(
                f"[MultiStreetTrainingLoop] Training complete: "
                f"{num_iterations} iterations in {total_time:.1f}s ({avg:.2f}s/iter avg)"
            )

        return history
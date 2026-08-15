"""
trainers/generic_tree_training_loop.py — Pluggable training loop for
ANY envs.generic_tree_env.GenericTreeEnv / scenarios.duo_game_config.DuoGameConfig,
generalizing trainers/duo_training_loop.py (SB/BB) and
trainers/river_duo_training_loop.py (OOP/IP) into one seat-count-agnostic
loop driven entirely by the config.

Why this doesn't duplicate PPO math or the mode enum
------------------------------------------------------
`_ppo_update_seat` / `_infer_device` (trainers/duo_training_loop.py) are
already fully generic — that module's own docstring says so explicitly:
"this function is already fully generic (any DualSeatActorCritic +
per-seat RolloutBuffer + optimizer; it has no push-fold-specific
assumptions baked in)." They're imported, not copied, here. `DuoTrainingMode`
is reused for the same reason — SIMULTANEOUS/ALTERNATING/LEAGUE are the
same three semantics regardless of seat count, so a third copy of that
enum (river_duo_training_loop.py made a second) would just be more files
saying the same thing.

What actually generalizes here vs. duo_training_loop.py / river_duo_training_loop.py
------------------------------------------------------------------------------------
Both existing loops hardcode two seat constants (SB_SEAT/BB_SEAT or
OOP_SEAT/IP_SEAT) and know the shape of their own game's decision tree
(SB has exactly one decision; OOP has one-or-two). This module knows
NEITHER: it walks env.current_seat wherever GenericTreeEnv.step() sends
it, for however many seats config.seats declares, and records a fully
chronological per-hand decision list.

BATCHED rollout collection (CHANGED)
--------------------------------------
Previously `collect_generic_tree_rollouts` played `num_episodes` hands
STRICTLY SEQUENTIALLY, one at a time, calling `actor_critic.act(seat,
...)` once per decision — a batch-of-1 neural network forward pass per
decision, even though every sub-encoder in
representation/representation_model.py and both concrete
DualSeatActorCritic implementations (models/shared_trunk_two_heads_policy.py,
models/shared_single_head_policy.py) now expose a real `act_batch()`
(see those modules' own docstrings for the "throughput change, not a
behaviour change" guarantee this relies on).

This version instead runs ALL `num_episodes` hands in LOCKSTEP: one
independent `GenericTreeEnv` clone per hand, all reset up front. At each
synchronized "tick", every still-in-flight hand's current decision is
grouped by which seat is acting, and ONE `act_batch(seat, ...)` call
processes every hand currently waiting on that seat — typically 1-2
ticks total for push-fold-duo/river-duo-shaped trees, so this collapses
what used to be `episodes_per_iteration` sequential forward passes into
a small constant number of large batched ones.

Poker semantics are completely unchanged: each cloned GenericTreeEnv
runs its own independent hand (own dealt cards, own RNG for any board
runout) exactly as the sequential version did — batching only changes
scheduling of the neural network calls, mirroring
envs/vectorized_env.py's own "no cross-episode information leakage"
guarantee for the single-hero stages.

BUGFIX: finish_episode() now only called for acted_seats
-------------------------------------------------------------
The previous version called `buffers[s].finish_episode()` for every
seat in `seat_indices`, regardless of whether that seat actually acted
in a given hand. For any hand where a seat didn't act (e.g. BB whenever
SB folds in push-fold-duo — the common case), that seat's
`start_episode()` was never called this round, so `finish_episode()`
would silently re-run GAE over that seat's transitions from its LAST
active hand and append a duplicate advantage/return pair —
`len(buffer.advantages)` would drift ahead of `len(buffer._transitions)`,
which RolloutBuffer.get() is supposed to catch (and would eventually
raise on), but corrupts training in the meantime. Fixed: `finish_episode()`
(and `start_episode()`) are now only ever called for the seats that
actually acted in that specific hand.

KNOWN LIMITATION — actor-critic side is still 2-seat-only
--------------------------------------------------------------
models.dual_seat_policy.DualSeatActorCritic (and both its concrete
implementations, SharedTrunkTwoHeadsActorCritic / SharedSingleHeadActorCritic)
validate `seat in (0, 1)` via `_check_seat`. This loop is written to be
correct for any seat count the CONFIG declares, but will raise from
inside the actor_critic itself the moment a 3+ seat config is used,
until a wider actor-critic exists. Out of scope here.

Dependencies
------------
    PyTorch. actions.abstract_action, envs.generic_tree_env,
    scenarios.duo_game_config.DuoGameConfig,
    scenarios.generic_tree_scenario_generator.GenericTreeScenarioGenerator,
    models.dual_seat_policy.DualSeatActorCritic,
    trainers.rollout_buffer.RolloutBuffer,
    trainers.duo_training_loop (_ppo_update_seat, _infer_device,
    DuoTrainingMode — reused, not duplicated).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.envs.generic_tree_env import GenericTreeEnv
from poker_rl_lab.scenarios.duo_game_config import DuoGameConfig
from poker_rl_lab.scenarios.generic_tree_scenario_generator import GenericTreeScenarioGenerator
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.trainers.duo_training_loop import DuoTrainingMode, _infer_device, _ppo_update_seat
from poker_rl_lab.trainers.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Per-iteration diagnostics
# ---------------------------------------------------------------------------

@dataclass
class GenericTreeIterationStats:
    """
    Diagnostics for one training-loop iteration.

    Keyed by seat INDEX throughout (config.seat_index(seat_id)), not by
    per-game seat-id constants — that's the whole point of this being
    generic. `active_seat` is None for SIMULTANEOUS (every seat updated
    this iteration); an int for ALTERNATING/LEAGUE.
    """
    iteration: int
    active_seat: Optional[int]
    mean_reward_by_seat: Dict[int, float] = field(default_factory=dict)
    policy_loss: Dict[int, float] = field(default_factory=dict)
    value_loss: Dict[int, float] = field(default_factory=dict)
    entropy: Dict[int, float] = field(default_factory=dict)
    clip_fraction: Dict[int, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def _mean_reward(buffer: RolloutBuffer) -> float:
    if len(buffer) == 0:
        return 0.0
    return sum(t.reward for t in buffer._transitions) / len(buffer)


# ---------------------------------------------------------------------------
# Env cloning (one independent GenericTreeEnv instance per in-flight hand)
# ---------------------------------------------------------------------------

def _clone_env(template: GenericTreeEnv) -> GenericTreeEnv:
    """
    Build a fresh, independent GenericTreeEnv sharing `template`'s
    config/scoring configuration but owning its own per-hand mutable
    state (self._committed, self._stack, etc. — see GenericTreeEnv.__init__).

    Sharing the same `scoring_engine` INSTANCE across clones is safe:
    CppScoringEngine.evaluate() / MockScoringEngine.evaluate() are pure
    functions of their arguments, holding no per-hand state of their
    own (see poker_engine/scoring/scoring_engine.py,
    mock_scoring_engine.py) — exactly the same sharing
    envs/vectorized_env.py's own env_factory pattern would produce if
    scoring_engine were threaded through it, just done directly here
    since GenericTreeEnv doesn't take an env_factory today.
    """
    return GenericTreeEnv(
        template.config,
        scoring_engine=template._scoring_engine,
        score_type=template._score_type,
        showdown_type=template._showdown_type,
        num_showdown_samples=getattr(template, "_num_showdown_samples", 30),
    )


# ---------------------------------------------------------------------------
# Rollout collection (BATCHED)
# ---------------------------------------------------------------------------

def collect_generic_tree_rollouts(
    env: GenericTreeEnv,
    scenario_gen: GenericTreeScenarioGenerator,
    actor_critic: DualSeatActorCritic,
    adapter,
    num_episodes: int,
    opponent_override: Optional[Dict[int, DualSeatActorCritic]] = None,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Dict[int, RolloutBuffer]:
    """
    Play `num_episodes` hands through env.config's tree IN LOCKSTEP
    (one GenericTreeEnv clone per hand, all running simultaneously),
    splitting each hand's decision(s) into one RolloutBuffer per seat
    index — see module docstring's "BATCHED rollout collection" section
    for why this replaces the old one-hand-at-a-time loop.

    Parameters
    ----------
    env : GenericTreeEnv
        Used as a TEMPLATE only (its config + scoring configuration are
        copied into `num_episodes` independent clones via _clone_env —
        `env` itself is never reset or stepped by this function).
    opponent_override : Dict[int, DualSeatActorCritic], optional
        Same LEAGUE-mode contract as
        duo_training_loop.collect_duo_rollouts / river_duo_training_loop's
        equivalent — if seat `s` has an entry here, that seat's actions
        this call are chosen from the override policy instead of
        `actor_critic`. Seats absent here use `actor_critic`.
    gamma, gae_lambda : float
        Forwarded to every per-seat RolloutBuffer.

    Returns
    -------
    Dict[int, RolloutBuffer]
        One entry per seat in env.config.seats, keyed by seat INDEX.
        Buffers may have unequal lengths — some seats may act more
        often than others depending on the tree.

    Notes
    -----
    Every act_batch() call happens under torch.no_grad() — data
    collection under the current ("old") policy, identical rationale to
    every other collect_*_rollouts() in this codebase.
    """
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")

    opponent_override = opponent_override or {}
    seat_indices = [s.index for s in env.config.seats]

    def _policy_for(seat: int) -> DualSeatActorCritic:
        return opponent_override.get(seat, actor_critic)

    buffers: Dict[int, RolloutBuffer] = {
        s: RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda) for s in seat_indices
    }

    # --- set up all hands in lockstep -----------------------------------
    envs: List[GenericTreeEnv] = [_clone_env(env) for _ in range(num_episodes)]
    decisions: List[List[dict]] = [[] for _ in range(num_episodes)]
    obs_list: List = [None] * num_episodes
    final_rewards: List[Optional[Dict[int, float]]] = [None] * num_episodes

    for i in range(num_episodes):
        obs_list[i] = envs[i].reset(scenario_gen.generate())

    active = list(range(num_episodes))

    # --- run every hand to completion, batching act_batch() per tick ----
    while active:
        groups: Dict[int, List[int]] = {}
        for i in active:
            groups.setdefault(envs[i].current_seat, []).append(i)

        newly_done: List[int] = []

        for seat, idxs in groups.items():
            policy = _policy_for(seat)
            rep_inputs = [adapter.adapt(obs_list[i]) for i in idxs]
            masks = [obs_list[i].legal_mask for i in idxs]

            with torch.no_grad():
                results = policy.act_batch(seat, rep_inputs, masks, deterministic=False)

            for rep_input, mask, i, result in zip(rep_inputs, masks, idxs, results):
                decisions[i].append({
                    "seat": seat,
                    "rep": rep_input,
                    "mask": mask,
                    "action": result.action_type,
                    "log_prob": float(result.log_prob.item()),
                    "value": float(result.value.item()),
                })

                action = AbstractAction(ActionType(result.action_type))
                step = envs[i].step(action)

                if step.done:
                    final_rewards[i] = step.rewards
                    newly_done.append(i)
                else:
                    obs_list[i] = step.obs

        for i in newly_done:
            active.remove(i)

    # --- flush each hand's decisions into the per-seat buffers ----------
    for i in range(num_episodes):
        rewards = final_rewards[i]
        assert rewards is not None, f"hand {i} never reached a terminal step"

        last_index_for_seat: Dict[int, int] = {}
        for idx, d in enumerate(decisions[i]):
            last_index_for_seat[d["seat"]] = idx

        # BUGFIX: only seats that actually acted this hand get
        # start_episode()/finish_episode() calls — see module docstring.
        acted_seats = set(last_index_for_seat.keys())

        for s in acted_seats:
            buffers[s].start_episode()

        for idx, d in enumerate(decisions[i]):
            seat = d["seat"]
            is_terminal_for_seat = (idx == last_index_for_seat[seat])
            buffers[seat].add(
                obs=d["rep"], legal_mask=d["mask"], action=d["action"],
                log_prob=d["log_prob"], value=d["value"],
                reward=(rewards[seat] if is_terminal_for_seat else 0.0),
                done=is_terminal_for_seat,
            )

        for s in acted_seats:
            buffers[s].finish_episode()

    return buffers


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class GenericTreeTrainingLoop:
    """
    Orchestrates collect -> update cycles for a DualSeatActorCritic over
    ANY GenericTreeEnv, in one of three interchangeable modes (see
    trainers.duo_training_loop.DuoTrainingMode — reused directly).

    Parameters
    ----------
    env : GenericTreeEnv
    scenario_gen : GenericTreeScenarioGenerator
    actor_critic : DualSeatActorCritic
        See module docstring's KNOWN LIMITATION — only 2-seat configs
        work with today's concrete implementations.
    adapter : representation.observation_adapter.ObservationAdapter
    optimizers : Dict[int, torch.optim.Optimizer]
        One entry per seat index in env.config.seats, each pre-scoped to
        actor_critic.seat_parameters(that seat).
    mode : DuoTrainingMode
        Default SIMULTANEOUS.
    iters_per_seat : int
        ALTERNATING/LEAGUE only: how many consecutive iterations one
        seat stays active before switching to the NEXT seat index
        (wrapping around), not just toggling between two. Default 5.
    snapshot_pool_size, snapshot_every : int
        LEAGUE only — see trainers.duo_training_loop.DuoTrainingLoop's
        identical parameters.
    """

    def __init__(
        self,
        env: GenericTreeEnv,
        scenario_gen: GenericTreeScenarioGenerator,
        actor_critic: DualSeatActorCritic,
        adapter,
        optimizers: Dict[int, torch.optim.Optimizer],
        mode: DuoTrainingMode = DuoTrainingMode.SIMULTANEOUS,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        iters_per_seat: int = 5,
        snapshot_pool_size: int = 8,
        snapshot_every: int = 5,
    ) -> None:
        self.env = env
        self.scenario_gen = scenario_gen
        self.actor_critic = actor_critic
        self.adapter = adapter

        self.seat_indices: List[int] = [s.index for s in env.config.seats]
        missing = set(self.seat_indices) - set(optimizers)
        if missing:
            raise ValueError(
                f"optimizers is missing entries for seat indices {sorted(missing)} "
                f"(config {env.config.name!r} declares seats {self.seat_indices})"
            )
        self.optimizers = optimizers

        self.mode = mode
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.iters_per_seat = iters_per_seat
        self.snapshot_pool_size = snapshot_pool_size
        self.snapshot_every = snapshot_every

        self._active_seat = self.seat_indices[0]
        self._iters_since_switch = 0
        self._snapshot_pools: Dict[int, List[dict]] = {s: [] for s in self.seat_indices}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        num_iterations: int,
        episodes_per_iteration: int = 256,
        epochs: int = 4,
        minibatch_size: int = 64,
        log_every: int = 10,
        verbose: bool = True,
    ) -> List[GenericTreeIterationStats]:
        history: List[GenericTreeIterationStats] = []
        start_time = time.time()

        for i in range(num_iterations):
            if self.mode == DuoTrainingMode.SIMULTANEOUS:
                stats = self._iteration_simultaneous(i, episodes_per_iteration, epochs, minibatch_size)
            elif self.mode == DuoTrainingMode.ALTERNATING:
                stats = self._iteration_alternating(i, episodes_per_iteration, epochs, minibatch_size)
            elif self.mode == DuoTrainingMode.LEAGUE:
                stats = self._iteration_league(i, episodes_per_iteration, epochs, minibatch_size)
            else:
                raise ValueError(f"Unknown mode {self.mode}")

            elapsed = time.time() - start_time
            stats.elapsed_seconds = elapsed
            history.append(stats)

            iteration_num = i + 1
            if verbose and log_every > 0 and (
                iteration_num % log_every == 0 or iteration_num == num_iterations
            ):
                print(
                    f"[GenericTreeTrainingLoop:{self.env.config.name}] "
                    f"{iteration_num}/{num_iterations} iterations "
                    f"completed ({elapsed:.1f}s elapsed, "
                    f"{elapsed / iteration_num:.2f}s/iter)"
                )

        if verbose:
            total_time = time.time() - start_time
            avg = total_time / num_iterations if num_iterations else 0.0
            print(
                f"[GenericTreeTrainingLoop:{self.env.config.name}] Training "
                f"complete: {num_iterations} iterations in {total_time:.1f}s "
                f"({avg:.2f}s/iter avg)"
            )

        return history

    # ------------------------------------------------------------------
    # SIMULTANEOUS
    # ------------------------------------------------------------------

    def _iteration_simultaneous(
        self, i: int, episodes_per_iteration: int, epochs: int, minibatch_size: int
    ) -> GenericTreeIterationStats:
        buffers = collect_generic_tree_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        policy_loss, value_loss, entropy, clip_fraction = {}, {}, {}, {}
        for seat in self.seat_indices:
            loss = _ppo_update_seat(
                self.actor_critic, seat, buffers[seat], self.optimizers[seat],
                epochs, minibatch_size, self.clip_epsilon, self.value_coef,
                self.entropy_coef, self.max_grad_norm,
            )
            policy_loss[seat] = loss["policy_loss"]
            value_loss[seat] = loss["value_loss"]
            entropy[seat] = loss["entropy"]
            clip_fraction[seat] = loss["clip_fraction"]

        return GenericTreeIterationStats(
            iteration=i, active_seat=None,
            mean_reward_by_seat={s: _mean_reward(buffers[s]) for s in self.seat_indices},
            policy_loss=policy_loss, value_loss=value_loss,
            entropy=entropy, clip_fraction=clip_fraction,
        )

    # ------------------------------------------------------------------
    # ALTERNATING
    # ------------------------------------------------------------------

    def _iteration_alternating(
        self, i: int, episodes_per_iteration: int, epochs: int, minibatch_size: int
    ) -> GenericTreeIterationStats:
        buffers = collect_generic_tree_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        active = self._active_seat
        loss = _ppo_update_seat(
            self.actor_critic, active, buffers[active], self.optimizers[active],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )

        self._advance_active_seat()

        return GenericTreeIterationStats(
            iteration=i, active_seat=active,
            mean_reward_by_seat={s: _mean_reward(buffers[s]) for s in self.seat_indices},
            policy_loss={active: loss["policy_loss"]},
            value_loss={active: loss["value_loss"]},
            entropy={active: loss["entropy"]},
            clip_fraction={active: loss["clip_fraction"]},
        )

    # ------------------------------------------------------------------
    # LEAGUE
    # ------------------------------------------------------------------

    def _iteration_league(
        self, i: int, episodes_per_iteration: int, epochs: int, minibatch_size: int
    ) -> GenericTreeIterationStats:
        active = self._active_seat

        opponent_override: Dict[int, DualSeatActorCritic] = {}
        for seat in self.seat_indices:
            if seat == active:
                continue
            pool = self._snapshot_pools[seat]
            if pool:
                opponent_override[seat] = self._load_snapshot(random.choice(pool))

        buffers = collect_generic_tree_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, opponent_override=opponent_override,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        loss = _ppo_update_seat(
            self.actor_critic, active, buffers[active], self.optimizers[active],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )

        if i % self.snapshot_every == 0:
            self._snapshot_current()

        self._advance_active_seat(snapshot_on_switch=True)

        return GenericTreeIterationStats(
            iteration=i, active_seat=active,
            mean_reward_by_seat={s: _mean_reward(buffers[s]) for s in self.seat_indices},
            policy_loss={active: loss["policy_loss"]},
            value_loss={active: loss["value_loss"]},
            entropy={active: loss["entropy"]},
            clip_fraction={active: loss["clip_fraction"]},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _advance_active_seat(self, snapshot_on_switch: bool = False) -> None:
        self._iters_since_switch += 1
        if self._iters_since_switch >= self.iters_per_seat:
            if snapshot_on_switch:
                self._snapshot_current()
            idx = self.seat_indices.index(self._active_seat)
            self._active_seat = self.seat_indices[(idx + 1) % len(self.seat_indices)]
            self._iters_since_switch = 0

    def _snapshot_current(self) -> None:
        pool = self._snapshot_pools[self._active_seat]
        pool.append(copy.deepcopy(self.actor_critic.state_dict()))
        if len(pool) > self.snapshot_pool_size:
            pool.pop(0)

    def _load_snapshot(self, state_dict: dict) -> DualSeatActorCritic:
        snapshot = copy.deepcopy(self.actor_critic)
        snapshot.load_state_dict(state_dict)
        snapshot.eval()
        return snapshot
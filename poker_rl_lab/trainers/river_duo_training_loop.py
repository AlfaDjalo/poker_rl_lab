"""
trainers/river_duo_training_loop.py — Pluggable training loop for
envs.river_duo_env.RiverDuoEnv, sibling of trainers/duo_training_loop.py
(which drives envs.push_fold_duo_env.PushFoldDuoEnv).

Why this is a separate file from trainers/duo_training_loop.py
------------------------------------------------------------------
duo_training_loop.py's collect_duo_rollouts() is hardcoded to
push-fold-duo's fixed decision shape: SB always has exactly one
decision, BB has zero or one. River-duo's decision tree is richer (see
scenarios/river_duo_generator.py / envs/river_duo_env.py's own
docstrings):

    OOP acts first (CHECK or BET_100).
        OOP CHECK -> IP acts (CHECK or BET_100).
            IP CHECK -> done.
            IP BET   -> OOP acts AGAIN (CALL or FOLD) -> done.
        OOP BET   -> IP acts (CALL or FOLD) -> done.

So per hand: IP always has EXACTLY one decision; OOP has ONE decision
(if OOP bet first, or IP checked back) or TWO (if OOP checked and IP
bet). A single push-fold-duo-shaped "SB decision, maybe BB decision"
rollout collector cannot express OOP's second decision, hence a
dedicated collect_river_duo_rollouts() below.

Everything else — the PPO update math, the three training modes
(SIMULTANEOUS/ALTERNATING/LEAGUE), snapshot pooling for LEAGUE — is
IDENTICAL to duo_training_loop.py and is not duplicated: this module
imports `_ppo_update_seat` directly from that file, since that function
is already fully generic (any DualSeatActorCritic + per-seat
RolloutBuffer + optimizer; it has no push-fold-specific assumptions
baked in).

Backfilling terminal reward/done
------------------------------------
Only the terminal env.step() call returns non-empty `rewards` (see
DuoStepResult's own docstring: "rewards ... Empty ({}) for any
intermediate (non-terminal) step"). So collect_river_duo_rollouts()
first plays a full hand while recording every (seat, rep_input,
legal_mask, action, log_prob, value) decision in chronological order
with reward=0.0/done=False, THEN — once the hand's true per-seat
rewards are known — patches EACH seat's OWN LAST recorded decision in
that hand to carry that seat's real reward and done=True. For IP
(always exactly one decision) this is always its only transition; for
OOP this is its second decision when one exists, else its only one.
Any earlier OOP decision within the same hand correctly keeps
reward=0.0/done=False, matching every other multi-step environment's
convention (see push_fold_env.py / river_env.py / envs/base_env.py's
BasePokerEnv contract).

Dependencies
------------
    PyTorch. actions.abstract_action, envs.river_duo_env,
    scenarios.river_duo_generator (OOP_SEAT/IP_SEAT), models.dual_seat_policy,
    trainers.rollout_buffer.RolloutBuffer, trainers.duo_training_loop
    (_ppo_update_seat, _infer_device — reused, not duplicated).
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch

from actions.abstract_action import AbstractAction, ActionType
from envs.river_duo_env import RiverDuoEnv
from models.dual_seat_policy import DualSeatActorCritic
from scenarios.river_duo_generator import OOP_SEAT, IP_SEAT
from trainers.duo_training_loop import _ppo_update_seat
from trainers.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Mode selection (identical semantics to duo_training_loop.DuoTrainingMode)
# ---------------------------------------------------------------------------

class RiverDuoTrainingMode(Enum):
    SIMULTANEOUS = "simultaneous"
    ALTERNATING = "alternating"
    LEAGUE = "league"


# ---------------------------------------------------------------------------
# Per-iteration diagnostics
# ---------------------------------------------------------------------------

@dataclass
class RiverDuoIterationStats:
    """Diagnostics for one training-loop iteration. See
    trainers.duo_training_loop.DuoIterationStats — identical shape,
    keyed by OOP_SEAT/IP_SEAT instead of SB_SEAT/BB_SEAT."""
    iteration: int
    active_seat: Optional[int]
    mean_reward_oop: float
    mean_reward_ip: float
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
# Rollout collection
# ---------------------------------------------------------------------------

def collect_river_duo_rollouts(
    env: RiverDuoEnv,
    scenario_gen,
    actor_critic: DualSeatActorCritic,
    adapter,
    num_episodes: int,
    opponent_override: Optional[Dict[int, DualSeatActorCritic]] = None,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> "tuple[RolloutBuffer, RolloutBuffer]":
    """
    Play `num_episodes` hands, splitting each hand's decision(s) into a
    per-seat RolloutBuffer. See module docstring for the backfilling
    rationale and OOP's variable (one-or-two) decision count.

    Parameters
    ----------
    opponent_override : Dict[int, DualSeatActorCritic], optional
        Same LEAGUE-mode contract as
        duo_training_loop.collect_duo_rollouts — if seat `s` has an
        entry here, that seat's actions this call come from the
        override policy instead of `actor_critic`.
    gamma, gae_lambda : float
        Forwarded to both per-seat RolloutBuffers.

    Returns
    -------
    (buffer_oop, buffer_ip) : Tuple[RolloutBuffer, RolloutBuffer]
        buffer_oop may contain MORE transitions than buffer_ip (OOP has
        two decisions on check-then-facing-a-bet hands; IP always has
        exactly one).

    Notes
    -----
    Every act() call happens under torch.no_grad() — data collection
    under the current ("old") policy, identical rationale to
    duo_training_loop.collect_duo_rollouts / PPOTrainer.collect_rollouts.
    """
    opponent_override = opponent_override or {}

    def _policy_for(seat: int) -> DualSeatActorCritic:
        return opponent_override.get(seat, actor_critic)

    buf_oop = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
    buf_ip = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)

    for _ in range(num_episodes):
        scenario = scenario_gen.generate()
        obs = env.reset(scenario)

        # Chronological record of this hand's decisions, one dict per
        # decision: {"seat", "rep", "mask", "action", "log_prob", "value"}.
        decisions: List[dict] = []

        while True:
            seat = env.current_seat
            rep_input = adapter.adapt(obs)

            with torch.no_grad():
                result = _policy_for(seat).act(
                    seat, rep_input, obs.legal_mask, deterministic=False
                )

            decisions.append({
                "seat": seat,
                "rep": rep_input,
                "mask": obs.legal_mask,
                "action": result.action_type,
                "log_prob": float(result.log_prob.item()),
                "value": float(result.value.item()),
            })

            action = AbstractAction(ActionType(result.action_type))
            step = env.step(action)

            if step.done:
                rewards = step.rewards
                break
            obs = step.obs

        # Last decision index per seat -> that decision carries the
        # seat's true terminal reward/done; earlier same-seat decisions
        # (only possible for OOP) stay reward=0.0/done=False.
        last_index_for_seat: Dict[int, int] = {}
        for i, d in enumerate(decisions):
            last_index_for_seat[d["seat"]] = i

        buffers = {OOP_SEAT: buf_oop, IP_SEAT: buf_ip}
        for seat in (OOP_SEAT, IP_SEAT):
            buffers[seat].start_episode()

        for i, d in enumerate(decisions):
            seat = d["seat"]
            is_terminal_for_seat = (i == last_index_for_seat[seat])
            buffers[seat].add(
                obs=d["rep"], legal_mask=d["mask"], action=d["action"],
                log_prob=d["log_prob"], value=d["value"],
                reward=(rewards[seat] if is_terminal_for_seat else 0.0),
                done=is_terminal_for_seat,
            )

        for seat in (OOP_SEAT, IP_SEAT):
            buffers[seat].finish_episode()

    return buf_oop, buf_ip


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class RiverDuoTrainingLoop:
    """
    Orchestrates collect -> update cycles for a DualSeatActorCritic over
    a RiverDuoEnv, in one of three interchangeable modes — identical
    semantics to trainers.duo_training_loop.DuoTrainingLoop, just built
    on collect_river_duo_rollouts() instead of collect_duo_rollouts().

    Parameters
    ----------
    env : RiverDuoEnv
    scenario_gen : scenarios.river_duo_generator.RiverDuoScenarioGenerator
    actor_critic : DualSeatActorCritic
    adapter : representation.observation_adapter.ObservationAdapter
    optimizer_oop, optimizer_ip : torch.optim.Optimizer
        Pre-constructed, each scoped to
        actor_critic.seat_parameters(OOP_SEAT) /
        actor_critic.seat_parameters(IP_SEAT) respectively.
    mode : RiverDuoTrainingMode
        Default SIMULTANEOUS.
    iters_per_seat : int
        ALTERNATING/LEAGUE only. Default 5.
    snapshot_pool_size : int
        LEAGUE only. Default 8.
    snapshot_every : int
        LEAGUE only. Default 5.
    """

    def __init__(
        self,
        env: RiverDuoEnv,
        scenario_gen,
        actor_critic: DualSeatActorCritic,
        adapter,
        optimizer_oop: torch.optim.Optimizer,
        optimizer_ip: torch.optim.Optimizer,
        mode: RiverDuoTrainingMode = RiverDuoTrainingMode.SIMULTANEOUS,
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
        self.optimizers: Dict[int, torch.optim.Optimizer] = {
            OOP_SEAT: optimizer_oop, IP_SEAT: optimizer_ip,
        }
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

        self._active_seat = OOP_SEAT
        self._iters_since_switch = 0
        self._snapshot_pools: Dict[int, List[dict]] = {OOP_SEAT: [], IP_SEAT: []}

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
    ) -> List[RiverDuoIterationStats]:
        history: List[RiverDuoIterationStats] = []
        start_time = time.time()

        for i in range(num_iterations):
            if self.mode == RiverDuoTrainingMode.SIMULTANEOUS:
                stats = self._iteration_simultaneous(i, episodes_per_iteration, epochs, minibatch_size)
            elif self.mode == RiverDuoTrainingMode.ALTERNATING:
                stats = self._iteration_alternating(i, episodes_per_iteration, epochs, minibatch_size)
            elif self.mode == RiverDuoTrainingMode.LEAGUE:
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
                    f"[RiverDuoTrainingLoop] {iteration_num}/{num_iterations} iterations "
                    f"completed ({elapsed:.1f}s elapsed, "
                    f"{elapsed / iteration_num:.2f}s/iter)"
                )

        if verbose:
            total_time = time.time() - start_time
            avg = total_time / num_iterations if num_iterations else 0.0
            print(
                f"[RiverDuoTrainingLoop] Training complete: {num_iterations} iterations "
                f"in {total_time:.1f}s ({avg:.2f}s/iter avg)"
            )

        return history

    # ------------------------------------------------------------------
    # SIMULTANEOUS
    # ------------------------------------------------------------------

    def _iteration_simultaneous(
        self, i: int, episodes_per_iteration: int, epochs: int, minibatch_size: int
    ) -> RiverDuoIterationStats:
        buf_oop, buf_ip = collect_river_duo_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        loss_oop = _ppo_update_seat(
            self.actor_critic, OOP_SEAT, buf_oop, self.optimizers[OOP_SEAT],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )
        loss_ip = _ppo_update_seat(
            self.actor_critic, IP_SEAT, buf_ip, self.optimizers[IP_SEAT],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )

        return RiverDuoIterationStats(
            iteration=i, active_seat=None,
            mean_reward_oop=_mean_reward(buf_oop), mean_reward_ip=_mean_reward(buf_ip),
            policy_loss={OOP_SEAT: loss_oop["policy_loss"], IP_SEAT: loss_ip["policy_loss"]},
            value_loss={OOP_SEAT: loss_oop["value_loss"], IP_SEAT: loss_ip["value_loss"]},
            entropy={OOP_SEAT: loss_oop["entropy"], IP_SEAT: loss_ip["entropy"]},
            clip_fraction={OOP_SEAT: loss_oop["clip_fraction"], IP_SEAT: loss_ip["clip_fraction"]},
        )

    # ------------------------------------------------------------------
    # ALTERNATING
    # ------------------------------------------------------------------

    def _iteration_alternating(
        self, i: int, episodes_per_iteration: int, epochs: int, minibatch_size: int
    ) -> RiverDuoIterationStats:
        buf_oop, buf_ip = collect_river_duo_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        active = self._active_seat
        active_buffer = buf_oop if active == OOP_SEAT else buf_ip

        loss = _ppo_update_seat(
            self.actor_critic, active, active_buffer, self.optimizers[active],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )

        self._advance_active_seat()

        return RiverDuoIterationStats(
            iteration=i, active_seat=active,
            mean_reward_oop=_mean_reward(buf_oop), mean_reward_ip=_mean_reward(buf_ip),
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
    ) -> RiverDuoIterationStats:
        active = self._active_seat
        passive = IP_SEAT if active == OOP_SEAT else OOP_SEAT

        opponent_override: Dict[int, DualSeatActorCritic] = {}
        pool = self._snapshot_pools[passive]
        if pool:
            opponent_override[passive] = self._load_snapshot(random.choice(pool))

        buf_oop, buf_ip = collect_river_duo_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, opponent_override=opponent_override,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        active_buffer = buf_oop if active == OOP_SEAT else buf_ip
        loss = _ppo_update_seat(
            self.actor_critic, active, active_buffer, self.optimizers[active],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )

        if i % self.snapshot_every == 0:
            self._snapshot_current()

        self._advance_active_seat(snapshot_on_switch=True)

        return RiverDuoIterationStats(
            iteration=i, active_seat=active,
            mean_reward_oop=_mean_reward(buf_oop), mean_reward_ip=_mean_reward(buf_ip),
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
            self._active_seat = IP_SEAT if self._active_seat == OOP_SEAT else OOP_SEAT
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
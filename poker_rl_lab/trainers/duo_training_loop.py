"""
trainers/duo_training_loop.py — Pluggable training loop for two-seat
(models.dual_seat_policy.DualSeatActorCritic) environments, built first
for envs.push_fold_duo_env.PushFoldDuoEnv.

Why this is a separate file from trainers/ppo_trainer.py
------------------------------------------------------------
PPOTrainer (trainers/ppo_trainer.py) assumes one hero seat and a
BasePokerEnv. PushFoldDuoEnv has no fixed hero — both seats are being
optimised, and the env's step() returns per-seat rewards instead of a
single scalar (see that module's own docstring). Rather than bend
PPOTrainer's single-hero contract to fit, this file duplicates its
clipped-surrogate PPO math (see `_ppo_update_seat` — this is
deliberately the same algorithm as PPOTrainer.update(), just
parameterised by `seat` and by which DualSeatActorCritic.evaluate_actions
call to make) and adds the layer PPOTrainer doesn't need: deciding WHICH
seat(s) get trained on a given iteration.

Per-hand rollout splitting
------------------------------
A push-fold-duo hand has at most two decisions: SB always acts once;
BB acts once more only if SB pushed. `collect_duo_rollouts` plays each
hand to completion and files SB's decision into one RolloutBuffer and
(if it happened) BB's decision into a second RolloutBuffer. Both are
single-step "episodes" from RolloutBuffer's point of view — SB's true
reward isn't known until AFTER BB decides (if SB pushed), but that's
fine: SB's transition is recorded with reward=0.0 nowhere in between,
it is simply recorded once, after the hand fully resolves, with the
hand's final reward and done=True. See trainers/rollout_buffer.py's own
docstring on why single-step episodes reduce GAE to
advantage = reward - value, which is exactly right here.

Three training modes (DuoTrainingMode)
-------------------------------------------
    SIMULTANEOUS
        Every iteration, collect rollouts with the CURRENT policy for
        both seats, then run a PPO update for BOTH seats (each seat's
        own optimizer, each covering that seat's `seat_parameters`).
        Both seats always see an up-to-date opponent.

    ALTERNATING
        Only one seat is "active" at a time. Rollouts are still
        collected for both seats every iteration (the passive seat must
        still act to generate an opponent move for the active seat's
        hands), but only the active seat's optimizer actually steps.
        After `iters_per_seat` iterations, the active seat flips.

    LEAGUE
        Same alternating schedule as ALTERNATING, but the PASSIVE seat's
        moves are generated from a pool of past frozen snapshots of the
        whole network (sampled uniformly), not just its current live
        weights — so the active seat has to keep beating a spread of
        past opponents, not just whatever the passive seat looks like
        right now (mitigates cyclic strategies / catastrophic
        forgetting in 2-player self-play). Snapshots are taken every
        `snapshot_every` iterations and whenever a seat's active block
        ends.

Because all three modes share `collect_duo_rollouts` and
`_ppo_update_seat`, switching modes is a one-line change
(`DuoTrainingLoop(..., mode=DuoTrainingMode.ALTERNATING)`) — nothing
about the env, policy, or PPO math changes between them.

Works unchanged with EITHER DualSeatActorCritic implementation
--------------------------------------------------------------------
models.shared_trunk_two_heads_policy.SharedTrunkTwoHeadsActorCritic or
models.shared_single_head_policy.SharedSingleHeadActorCritic — this
file only ever calls the models.dual_seat_policy.DualSeatActorCritic
interface (act / evaluate_actions / seat_parameters), never anything
architecture-specific. See dual_seat_policy.py's own docstring for what
`seat_parameters()` — and therefore ALTERNATING/LEAGUE mode — degenerates
to for the single-head variant (every seat's optimizer covers the same
full parameter set).

Dependencies
------------
    PyTorch. actions.abstract_action, envs.push_fold_duo_env,
    models.dual_seat_policy, trainers.rollout_buffer.RolloutBuffer.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
# from poker_rl_lab.envs.push_fold_duo_env import PushFoldDuoEnv
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.trainers.rollout_buffer import RolloutBuffer


def _infer_device(model) -> torch.device:
    """Best-effort device inference — see trainers.ppo_trainer._infer_device
    for the full rationale (duplicated here rather than imported, since
    this module already avoids importing trainers.ppo_trainer to keep
    the single-hero and duo training stacks independent)."""
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")

SB_SEAT: int = 0
BB_SEAT: int = 1


# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------

class DuoTrainingMode(Enum):
    SIMULTANEOUS = "simultaneous"
    ALTERNATING = "alternating"
    LEAGUE = "league"


# ---------------------------------------------------------------------------
# Per-iteration diagnostics
# ---------------------------------------------------------------------------

@dataclass
class DuoIterationStats:
    """
    Diagnostics for one training-loop iteration.

    `active_seat` is None for SIMULTANEOUS (both seats trained this
    iteration); an int (0 or 1) for ALTERNATING/LEAGUE. policy_loss /
    value_loss / entropy / clip_fraction are keyed by whichever seat(s)
    were actually updated this iteration (both keys for SIMULTANEOUS,
    one key otherwise) — reward means are always reported for both
    seats regardless of mode, since rollouts are always collected for
    both.
    """
    iteration: int
    active_seat: Optional[int]
    mean_reward_sb: float
    mean_reward_bb: float
    policy_loss: Dict[int, float] = field(default_factory=dict)
    value_loss: Dict[int, float] = field(default_factory=dict)
    entropy: Dict[int, float] = field(default_factory=dict)
    clip_fraction: Dict[int, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    """Wall-clock seconds elapsed since train() started, through the end
    of this iteration. 0.0 if this DuoIterationStats wasn't produced by
    train() (e.g. constructed directly in a test)."""


def _mean_reward(buffer: RolloutBuffer) -> float:
    if len(buffer) == 0:
        return 0.0
    return sum(t.reward for t in buffer._transitions) / len(buffer)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_duo_rollouts(
    env: PushFoldDuoEnv,
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
    per-seat RolloutBuffer.

    Parameters
    ----------
    opponent_override : Dict[int, DualSeatActorCritic], optional
        If seat `s` has an entry here, that seat's action this call is
        chosen from the override policy instead of `actor_critic` — how
        LEAGUE mode plays the active seat against a frozen snapshot
        without this function needing to know about snapshots at all.
        Seats not present here use `actor_critic` (the normal case).
    gamma, gae_lambda : float
        Forwarded to both per-seat RolloutBuffers.

    Returns
    -------
    (buffer_sb, buffer_bb) : Tuple[RolloutBuffer, RolloutBuffer]
        buffer_bb may contain FEWER transitions than buffer_sb — every
        hand has an SB decision, only hands where SB pushed have a BB
        decision.

    Notes
    -----
    Every act() call happens under torch.no_grad() — this is data
    collection under the current ("old") policy, not a training step,
    identical rationale to trainers/ppo_trainer.py's
    PPOTrainer.collect_rollouts().
    """
    opponent_override = opponent_override or {}

    def _policy_for(seat: int) -> DualSeatActorCritic:
        return opponent_override.get(seat, actor_critic)

    buf_sb = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
    buf_bb = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)

    for _ in range(num_episodes):
        scenario = scenario_gen.generate()
        obs_sb = env.reset(scenario)
        rep_sb = adapter.adapt(obs_sb)

        with torch.no_grad():
            result_sb = _policy_for(SB_SEAT).act(
                SB_SEAT, rep_sb, obs_sb.legal_mask, deterministic=False
            )
        action_sb = AbstractAction(ActionType(result_sb.action_type))
        step1 = env.step(action_sb)

        if step1.done:
            # SB folded — hand over, BB never got an observation.
            buf_sb.start_episode()
            buf_sb.add(
                obs=rep_sb, legal_mask=obs_sb.legal_mask,
                action=result_sb.action_type,
                log_prob=float(result_sb.log_prob.item()),
                value=float(result_sb.value.item()),
                reward=step1.rewards[SB_SEAT], done=True,
            )
            buf_sb.finish_episode()
            continue

        # SB pushed — BB now decides.
        obs_bb = step1.obs
        rep_bb = adapter.adapt(obs_bb)

        with torch.no_grad():
            result_bb = _policy_for(BB_SEAT).act(
                BB_SEAT, rep_bb, obs_bb.legal_mask, deterministic=False
            )
        action_bb = AbstractAction(ActionType(result_bb.action_type))
        step2 = env.step(action_bb)
        assert step2.done, "BB's decision always ends a push-fold-duo hand."

        buf_sb.start_episode()
        buf_sb.add(
            obs=rep_sb, legal_mask=obs_sb.legal_mask,
            action=result_sb.action_type,
            log_prob=float(result_sb.log_prob.item()),
            value=float(result_sb.value.item()),
            reward=step2.rewards[SB_SEAT], done=True,
        )
        buf_sb.finish_episode()

        buf_bb.start_episode()
        buf_bb.add(
            obs=rep_bb, legal_mask=obs_bb.legal_mask,
            action=result_bb.action_type,
            log_prob=float(result_bb.log_prob.item()),
            value=float(result_bb.value.item()),
            reward=step2.rewards[BB_SEAT], done=True,
        )
        buf_bb.finish_episode()

    return buf_sb, buf_bb


# ---------------------------------------------------------------------------
# Per-seat PPO update (mirrors trainers/ppo_trainer.py's PPOTrainer.update)
# ---------------------------------------------------------------------------

def _ppo_update_seat(
    actor_critic: DualSeatActorCritic,
    seat: int,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: Optional[float],
) -> Dict[str, float]:
    """
    Clipped-surrogate PPO update for one seat over everything currently
    in `buffer`. Identical algorithm to PPOTrainer.update() — see that
    method's docstring for the loss formula — generalised only in that
    it calls actor_critic.evaluate_actions(seat, ...) and steps an
    optimizer scoped to actor_critic.seat_parameters(seat).
    """
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}

    batch = buffer.get()
    obs: List = batch["obs"]
    legal_masks: List = batch["legal_masks"]
    actions: torch.Tensor = batch["actions"]

    # See trainers/ppo_trainer.py's update() for why this move is
    # necessary once the model lives on CUDA — RolloutBuffer.get()
    # always builds its tensors on CPU.
    device = _infer_device(actor_critic)
    old_log_probs: torch.Tensor = batch["old_log_probs"].to(device)
    advantages: torch.Tensor = batch["advantages"].to(device)
    returns: torch.Tensor = batch["returns"].to(device)

    n = len(obs)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
    num_minibatches = 0
    indices = list(range(n))

    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]

            mb_obs = [obs[i] for i in mb_idx]
            mb_masks = [legal_masks[i] for i in mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            new_log_probs, values, entropy = actor_critic.evaluate_actions(
                seat, mb_obs, mb_masks, mb_actions.tolist()
            )

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, mb_returns)
            entropy_mean = entropy.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    actor_critic.seat_parameters(seat), max_grad_norm
                )
            optimizer.step()

            with torch.no_grad():
                clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()

            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy_mean.item()
            totals["clip_fraction"] += clip_fraction.item()
            num_minibatches += 1

    return {k: v / num_minibatches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class DuoTrainingLoop:
    """
    Orchestrates collect -> update cycles for a DualSeatActorCritic
    over a PushFoldDuoEnv, in one of three interchangeable modes — see
    module docstring for what each mode does.

    Parameters
    ----------
    env : PushFoldDuoEnv
    scenario_gen : scenarios.push_fold_duo_generator.PushFoldDuoScenarioGenerator
    actor_critic : DualSeatActorCritic
        Either SharedTrunkTwoHeadsActorCritic or SharedSingleHeadActorCritic
        — this class never branches on which.
    adapter : representation.observation_adapter.ObservationAdapter
    optimizer_sb, optimizer_bb : torch.optim.Optimizer
        Pre-constructed, each scoped to
        actor_critic.seat_parameters(SB_SEAT) /
        actor_critic.seat_parameters(BB_SEAT) respectively. For
        SharedSingleHeadActorCritic these cover the identical full
        parameter set — see dual_seat_policy.py's docstring on what
        that implies for ALTERNATING/LEAGUE.
    mode : DuoTrainingMode
        Default SIMULTANEOUS.
    iters_per_seat : int
        ALTERNATING/LEAGUE only: how many consecutive iterations one
        seat stays active before switching. Default 5.
    snapshot_pool_size : int
        LEAGUE only: max number of past snapshots kept per seat (oldest
        dropped first). Default 8.
    snapshot_every : int
        LEAGUE only: also force a snapshot every N iterations,
        independent of the active-seat switch schedule. Default 5.
    """

    def __init__(
        self,
        env: PushFoldDuoEnv,
        scenario_gen,
        actor_critic: DualSeatActorCritic,
        adapter,
        optimizer_sb: torch.optim.Optimizer,
        optimizer_bb: torch.optim.Optimizer,
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
        self.optimizers: Dict[int, torch.optim.Optimizer] = {
            SB_SEAT: optimizer_sb, BB_SEAT: optimizer_bb,
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

        self._active_seat = SB_SEAT
        self._iters_since_switch = 0
        self._snapshot_pools: Dict[int, List[dict]] = {SB_SEAT: [], BB_SEAT: []}

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
    ) -> List[DuoIterationStats]:
        """
        Parameters
        ----------
        num_iterations, episodes_per_iteration, epochs, minibatch_size
            See the per-mode `_iteration_*` methods.
        log_every : int
            Print a one-line progress update every `log_every`
            iterations (and always on the final iteration), e.g.
            "[DuoTrainingLoop] 10/50 iterations completed (42.3s
            elapsed, 4.23s/iter)". Pass 0 to disable periodic progress
            lines (the final training-complete summary still prints
            unless verbose=False).
        verbose : bool
            If False, suppresses all progress/timing output from this
            method entirely. Default True.

        Returns
        -------
        List[DuoIterationStats]
            One entry per iteration. Each entry's elapsed_seconds gives
            cumulative wall-clock time through that iteration.
        """
        history: List[DuoIterationStats] = []
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
                    f"[DuoTrainingLoop] {iteration_num}/{num_iterations} iterations "
                    f"completed ({elapsed:.1f}s elapsed, "
                    f"{elapsed / iteration_num:.2f}s/iter)"
                )

        if verbose:
            total_time = time.time() - start_time
            avg = total_time / num_iterations if num_iterations else 0.0
            print(
                f"[DuoTrainingLoop] Training complete: {num_iterations} iterations "
                f"in {total_time:.1f}s ({avg:.2f}s/iter avg)"
            )

        return history

    # ------------------------------------------------------------------
    # SIMULTANEOUS
    # ------------------------------------------------------------------

    def _iteration_simultaneous(
        self, i: int, episodes_per_iteration: int, epochs: int, minibatch_size: int
    ) -> DuoIterationStats:
        buf_sb, buf_bb = collect_duo_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        loss_sb = _ppo_update_seat(
            self.actor_critic, SB_SEAT, buf_sb, self.optimizers[SB_SEAT],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )
        loss_bb = _ppo_update_seat(
            self.actor_critic, BB_SEAT, buf_bb, self.optimizers[BB_SEAT],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )

        return DuoIterationStats(
            iteration=i, active_seat=None,
            mean_reward_sb=_mean_reward(buf_sb), mean_reward_bb=_mean_reward(buf_bb),
            policy_loss={SB_SEAT: loss_sb["policy_loss"], BB_SEAT: loss_bb["policy_loss"]},
            value_loss={SB_SEAT: loss_sb["value_loss"], BB_SEAT: loss_bb["value_loss"]},
            entropy={SB_SEAT: loss_sb["entropy"], BB_SEAT: loss_bb["entropy"]},
            clip_fraction={SB_SEAT: loss_sb["clip_fraction"], BB_SEAT: loss_bb["clip_fraction"]},
        )

    # ------------------------------------------------------------------
    # ALTERNATING
    # ------------------------------------------------------------------

    def _iteration_alternating(
        self, i: int, episodes_per_iteration: int, epochs: int, minibatch_size: int
    ) -> DuoIterationStats:
        buf_sb, buf_bb = collect_duo_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        active = self._active_seat
        active_buffer = buf_sb if active == SB_SEAT else buf_bb

        loss = _ppo_update_seat(
            self.actor_critic, active, active_buffer, self.optimizers[active],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )

        self._advance_active_seat()

        return DuoIterationStats(
            iteration=i, active_seat=active,
            mean_reward_sb=_mean_reward(buf_sb), mean_reward_bb=_mean_reward(buf_bb),
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
    ) -> DuoIterationStats:
        active = self._active_seat
        passive = BB_SEAT if active == SB_SEAT else SB_SEAT

        opponent_override: Dict[int, DualSeatActorCritic] = {}
        pool = self._snapshot_pools[passive]
        if pool:
            opponent_override[passive] = self._load_snapshot(random.choice(pool))
        # else: no snapshot exists yet for the passive seat — fall back
        # to the live actor_critic (equivalent to ALTERNATING for the
        # first few iterations of a fresh run).

        buf_sb, buf_bb = collect_duo_rollouts(
            self.env, self.scenario_gen, self.actor_critic, self.adapter,
            episodes_per_iteration, opponent_override=opponent_override,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        active_buffer = buf_sb if active == SB_SEAT else buf_bb
        loss = _ppo_update_seat(
            self.actor_critic, active, active_buffer, self.optimizers[active],
            epochs, minibatch_size, self.clip_epsilon, self.value_coef,
            self.entropy_coef, self.max_grad_norm,
        )

        if i % self.snapshot_every == 0:
            self._snapshot_current()

        self._advance_active_seat(snapshot_on_switch=True)

        return DuoIterationStats(
            iteration=i, active_seat=active,
            mean_reward_sb=_mean_reward(buf_sb), mean_reward_bb=_mean_reward(buf_bb),
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
            self._active_seat = BB_SEAT if self._active_seat == SB_SEAT else SB_SEAT
            self._iters_since_switch = 0

    def _snapshot_current(self) -> None:
        """
        Store a frozen copy of the WHOLE network's current weights under
        the currently-active seat's pool. A "snapshot for seat X" means
        "what the network looked like the last time seat X was being
        trained" — for SharedTrunkTwoHeads this is meaningfully
        different per seat (different head weights); for
        SharedSingleHead every seat's snapshot is just a copy of the one
        shared network at that point in time.
        """
        pool = self._snapshot_pools[self._active_seat]
        pool.append(copy.deepcopy(self.actor_critic.state_dict()))
        if len(pool) > self.snapshot_pool_size:
            pool.pop(0)

    def _load_snapshot(self, state_dict: dict) -> DualSeatActorCritic:
        """Deep-copy the current actor_critic and load a past
        state_dict into it, for use as a frozen opponent this call
        only. Never touched by an optimizer."""
        snapshot = copy.deepcopy(self.actor_critic)
        snapshot.load_state_dict(state_dict)
        snapshot.eval()
        return snapshot
"""
trainers/ppo_trainer.py — PPO training loop.

Per the RL Lab implementation guide (§7, "Policy Learning"):

    The initial implementation should use PPO. PPO should remain
    isolated from: observations, representations, environments.
    Changing the reinforcement learning algorithm should not require
    changes to other modules.

This module is the top of the pipeline described in
models/policy_value_network.py's docstring:

    Scenario Generator -> Environment -> Structured Observation
        -> Representation Model -> Latent Representation
        -> Policy Network -> Chosen Action -> Environment -> CAP Engine

PPOTrainer only ever calls through three narrow interfaces — it has no
knowledge of poker, card representations, or neural architecture:

    env            : envs.base_env.BasePokerEnv        (reset/step)
    scenario_gen   : scenarios.scenario_generator.ScenarioGenerator (generate)
    actor_critic   : models.policy_value_network.PokerActorCritic
                     (act / evaluate_actions / parameters — any object
                     duck-typing this surface works, which is what makes
                     the unit tests in test_ppo_trainer.py possible
                     without instantiating the real (TF-backed) hand
                     encoder or the poker_eval C++ extension)
    adapter        : an object with .adapt(StructuredObservation) ->
                     RepresentationInput. In production this is
                     representation.observation_adapter.ObservationAdapter;
                     tests can pass a trivial identity-like stub.

Algorithm
---------
Standard clipped-surrogate PPO (Schulman et al., 2017):

    ratio          = exp(new_log_prob - old_log_prob)
    surrogate      = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
    policy_loss    = -mean(surrogate)
    value_loss     = mse(V(s), returns)
    entropy_loss   = -mean(entropy)              # bonus, encourages exploration
    loss           = policy_loss
                     + value_coef   * value_loss
                     + entropy_coef * entropy_loss

Advantages (A) and returns come from rollout_buffer.RolloutBuffer's
per-episode GAE — see that module's docstring for why GAE is computed
per-hand rather than over one flat cross-episode buffer.

One rollout/update cycle
--------------------------
    1. collect_rollouts(): run `episodes_per_iteration` full hands with
       the CURRENT policy (torch.no_grad — this is data collection, not
       a training step), filling a RolloutBuffer.
    2. update(): run `epochs` passes over the buffer in `minibatch_size`
       chunks, each pass computing fresh log-probs/values/entropy under
       the (now-changing) current policy via evaluate_actions(), and
       taking one optimizer step per minibatch.
    3. buffer.clear(): PPO is on-policy — data collected under the old
       policy must not be reused after update() has moved the policy
       away from it.

train() wraps this cycle N times and returns per-iteration stats for
logging (mean reward, mean losses, clip fraction).

Dependencies
------------
    PyTorch only, plus actions.abstract_action for AbstractAction /
    ActionType (pure Python, no C++ extension — see that module's own
    docstring) and trainers.rollout_buffer.RolloutBuffer.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from actions.abstract_action import AbstractAction, ActionType
from trainers.rollout_buffer import RolloutBuffer


def _infer_device(model) -> torch.device:
    """
    Best-effort device inference for an ActorCriticLike model.

    Real training code passes a PokerActorCritic (nn.Module) — this
    returns wherever its parameters actually live (CPU or CUDA). Unit
    tests may pass a lightweight duck-typed stub without real
    torch.nn.Parameter state; for those, fall back to CPU rather than
    raising, since a stub has no "device" to move anything to anyway.
    """
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Duck-typed interfaces (documentation + isolation, not enforced at runtime)
# ---------------------------------------------------------------------------

class ActorCriticLike(Protocol):
    """
    The subset of models.policy_value_network.PokerActorCritic that
    PPOTrainer actually calls. Any object implementing this surface
    works — real training code passes a PokerActorCritic; unit tests
    can pass a lightweight stub (see test_ppo_trainer.py) to exercise
    the PPO math without instantiating the TF-backed hand encoder or
    the poker_eval C++ extension.
    """
    def act(self, obs, legal_mask, deterministic: bool = False): ...
    def evaluate_actions(self, obs_batch, legal_masks, actions): ...
    def parameters(self): ...


class AdapterLike(Protocol):
    """representation.observation_adapter.ObservationAdapter's surface."""
    def adapt(self, obs): ...


# ---------------------------------------------------------------------------
# Per-iteration stats
# ---------------------------------------------------------------------------

@dataclass
class IterationStats:
    """Diagnostics for one collect+update cycle, for logging/plotting."""
    mean_episode_reward: float
    num_episodes: int
    num_transitions: int
    policy_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float
    elapsed_seconds: float = 0.0
    """Wall-clock seconds elapsed since train() started, through the end
    of this iteration. 0.0 if this IterationStats wasn't produced by
    train() (e.g. constructed directly in a test)."""


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Clipped-surrogate PPO trainer.

    Parameters
    ----------
    actor_critic : ActorCriticLike
        Policy + value network. Must be an nn.Module (or duck-type
        .parameters()) so `optimizer` can update it.
    adapter : AdapterLike
        Converts each hero StructuredObservation into the
        RepresentationInput the actor_critic's representation model
        expects.
    optimizer : torch.optim.Optimizer
        Pre-constructed optimizer over actor_critic.parameters(), e.g.
        torch.optim.Adam(actor_critic.parameters(), lr=3e-4).
    clip_epsilon : float
        PPO clip range. Default 0.2 (standard).
    value_coef : float
        Weight on the value loss term. Default 0.5.
    entropy_coef : float
        Weight on the entropy bonus. Default 0.01.
    max_grad_norm : Optional[float]
        If set, gradients are clipped to this global norm before each
        optimizer step. Default 0.5. Pass None to disable.
    gamma : float
        Discount factor, forwarded to RolloutBuffer. Default 1.0 — a
        poker hand is short (at most a few streets) and the guide's
        reward convention places the entire chip outcome on the
        terminal step, so there is no strong reason to discount future
        streets within a single hand.
    gae_lambda : float
        GAE lambda, forwarded to RolloutBuffer. Default 0.95.
    """

    def __init__(
        self,
        actor_critic: ActorCriticLike,
        adapter: AdapterLike,
        optimizer: torch.optim.Optimizer,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ) -> None:
        if clip_epsilon <= 0:
            raise ValueError(f"clip_epsilon must be > 0, got {clip_epsilon}")
        if value_coef < 0:
            raise ValueError(f"value_coef must be >= 0, got {value_coef}")
        if entropy_coef < 0:
            raise ValueError(f"entropy_coef must be >= 0, got {entropy_coef}")

        self.actor_critic = actor_critic
        self.adapter = adapter
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        env,
        scenario_gen,
        num_episodes: int,
        buffer: Optional[RolloutBuffer] = None,
    ) -> RolloutBuffer:
        """
        Play `num_episodes` full hands with the current policy and
        record every hero decision into a RolloutBuffer.

        Parameters
        ----------
        env : envs.base_env.BasePokerEnv
            Any concrete environment (PushFoldEnv, RiverEnv, ...).
        scenario_gen : scenarios.scenario_generator.ScenarioGenerator
            Supplies one BaseScenario per episode via .generate().
        num_episodes : int
            Number of hands to play.
        buffer : RolloutBuffer, optional
            Append into an existing buffer instead of creating a new
            one (e.g. to mix scenario generators / environments into
            one PPO update). Defaults to a fresh buffer using this
            trainer's gamma/gae_lambda.

        Returns
        -------
        RolloutBuffer
            Populated with `num_episodes` completed episodes — every
            episode already has finish_episode() called, so buffer.get()
            can be called directly.

        Notes
        -----
        Action selection uses torch.no_grad(): this is data collection
        under the current (soon-to-be "old") policy, not a training
        step — no gradient should be tracked here. The recorded
        log_prob/value are exactly what PPO's ratio computation in
        update() needs as "old_log_probs" / baseline values.
        """
        if num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")

        if buffer is None:
            buffer = RolloutBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda)

        for _ in range(num_episodes):
            scenario = scenario_gen.generate()
            obs = env.reset(scenario)

            buffer.start_episode()

            while not obs.is_terminal:
                rep_input = self.adapter.adapt(obs)

                with torch.no_grad():
                    result = self.actor_critic.act(
                        rep_input, obs.legal_mask, deterministic=False
                    )

                action = AbstractAction(ActionType(result.action_type))
                next_obs, reward, done, _info = env.step(action)

                buffer.add(
                    obs=rep_input,
                    legal_mask=obs.legal_mask,
                    action=result.action_type,
                    log_prob=float(result.log_prob.item()),
                    value=float(result.value.item()),
                    reward=reward,
                    done=done,
                )

                obs = next_obs

            buffer.finish_episode()

        return buffer

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        buffer: RolloutBuffer,
        epochs: int = 4,
        minibatch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Run `epochs` passes of minibatch clipped-surrogate PPO updates
        over everything currently in `buffer`.

        Parameters
        ----------
        buffer : RolloutBuffer
            Must have finish_episode() already called for every episode
            it contains (buffer.get() enforces this).
        epochs : int
            Number of full passes over the buffer. Default 4.
        minibatch_size : int
            Transitions per gradient step. Default 64. If larger than
            the buffer, one minibatch covering everything is used.

        Returns
        -------
        dict
            Mean policy_loss / value_loss / entropy / clip_fraction
            across every minibatch in every epoch.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if minibatch_size < 1:
            raise ValueError(f"minibatch_size must be >= 1, got {minibatch_size}")

        batch = buffer.get()
        obs: List = batch["obs"]
        legal_masks: List = batch["legal_masks"]
        actions: torch.Tensor = batch["actions"]

        # RolloutBuffer.get() always builds its tensors on CPU (rollout
        # collection stores plain python floats — see rollout_buffer.py).
        # evaluate_actions() below runs on whatever device the model's
        # own parameters live on, so these must be moved there ONCE,
        # up front, rather than per-minibatch — otherwise `ratio =
        # torch.exp(new_log_probs - mb_old_log_probs)` and the value
        # loss below would raise a device-mismatch error the moment the
        # model is moved to CUDA.
        device = _infer_device(self.actor_critic)
        old_log_probs: torch.Tensor = batch["old_log_probs"].to(device)
        advantages: torch.Tensor = batch["advantages"].to(device)
        returns: torch.Tensor = batch["returns"].to(device)

        n = len(obs)
        if n == 0:
            raise ValueError("Cannot update() on an empty buffer.")

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
        num_minibatches = 0

        indices = list(range(n))

        for _epoch in range(epochs):
            random.shuffle(indices)

            for start in range(0, n, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]

                mb_obs = [obs[i] for i in mb_idx]
                mb_masks = [legal_masks[i] for i in mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                new_log_probs, values, entropy = self.actor_critic.evaluate_actions(
                    mb_obs, mb_masks, mb_actions.tolist()
                )

                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)

                entropy_mean = entropy.mean()
                entropy_loss = -entropy_mean

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                    )

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy_mean.item()
                totals["clip_fraction"] += clip_fraction.item()
                num_minibatches += 1

        return {k: v / num_minibatches for k, v in totals.items()}

    # ------------------------------------------------------------------
    # Full train loop
    # ------------------------------------------------------------------

    def train(
        self,
        env,
        scenario_gen,
        num_iterations: int,
        episodes_per_iteration: int = 32,
        epochs: int = 4,
        minibatch_size: int = 64,
        log_every: int = 10,
        verbose: bool = True,
    ) -> List[IterationStats]:
        """
        Run `num_iterations` full collect -> update -> clear cycles.

        Parameters
        ----------
        env : envs.base_env.BasePokerEnv
        scenario_gen : scenarios.scenario_generator.ScenarioGenerator
        num_iterations : int
            Number of PPO iterations (collect+update cycles).
        episodes_per_iteration : int
            Hands played per iteration. Default 32.
        epochs : int
            PPO epochs per iteration. Default 4.
        minibatch_size : int
            Minibatch size per PPO epoch. Default 64.
        log_every : int
            Print a one-line progress update every `log_every`
            iterations (and always on the final iteration), e.g.
            "[PPOTrainer] 10/50 iterations completed (42.3s elapsed,
            4.23s/iter)". Pass 0 to disable periodic progress lines
            (the final training-complete summary still prints unless
            verbose=False).
        verbose : bool
            If False, suppresses all progress/timing output from this
            method entirely. Default True.

        Returns
        -------
        List[IterationStats]
            One entry per iteration, in order. Callers own logging /
            checkpointing / early stopping based on this — this method
            deliberately does neither (per the guide's §7 isolation
            requirement: PPO stays isolated from experiment tracking
            just as it does from observations/representations/envs).
            Each entry's elapsed_seconds gives cumulative wall-clock
            time through that iteration, for callers that want their
            own plotting/logging beyond the printed progress lines.
        """
        history: List[IterationStats] = []
        start_time = time.time()

        for i in range(num_iterations):
            buffer = self.collect_rollouts(env, scenario_gen, episodes_per_iteration)

            num_transitions = len(buffer)
            mean_reward = self._iteration_reward(buffer)

            update_stats = self.update(buffer, epochs=epochs, minibatch_size=minibatch_size)

            elapsed = time.time() - start_time

            history.append(
                IterationStats(
                    mean_episode_reward=mean_reward,
                    num_episodes=episodes_per_iteration,
                    num_transitions=num_transitions,
                    policy_loss=update_stats["policy_loss"],
                    value_loss=update_stats["value_loss"],
                    entropy=update_stats["entropy"],
                    clip_fraction=update_stats["clip_fraction"],
                    elapsed_seconds=elapsed,
                )
            )

            buffer.clear()

            iteration_num = i + 1
            if verbose and log_every > 0 and (
                iteration_num % log_every == 0 or iteration_num == num_iterations
            ):
                print(
                    f"[PPOTrainer] {iteration_num}/{num_iterations} iterations "
                    f"completed ({elapsed:.1f}s elapsed, "
                    f"{elapsed / iteration_num:.2f}s/iter)"
                )

        if verbose:
            total_time = time.time() - start_time
            avg = total_time / num_iterations if num_iterations else 0.0
            print(
                f"[PPOTrainer] Training complete: {num_iterations} iterations "
                f"in {total_time:.1f}s ({avg:.2f}s/iter avg)"
            )

        return history

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_end_indices(buffer: RolloutBuffer) -> List[int]:
        """Indices of terminal (done=True) transitions in the buffer."""
        return [i for i, t in enumerate(buffer._transitions) if t.done]

    def _iteration_reward(self, buffer: RolloutBuffer) -> float:
        """
        Mean per-episode total reward for this iteration, computed by
        summing each transition's raw (un-normalised) reward within an
        episode. Uses buffer._transitions directly rather than
        buffer.get()'s normalised advantages/returns, since this is a
        human-readable diagnostic (chips won/lost per hand), not a
        training signal.
        """
        end_indices = self._episode_end_indices(buffer)
        if not end_indices:
            return 0.0

        totals = []
        start = 0
        for end in end_indices:
            episode_reward = sum(t.reward for t in buffer._transitions[start:end + 1])
            totals.append(episode_reward)
            start = end + 1

        return sum(totals) / len(totals)
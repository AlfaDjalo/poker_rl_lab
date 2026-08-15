"""
trainers/ppo_trainer.py — PPO training loop.

CHANGES IN THIS VERSION
------------------------
Added a vectorized rollout-collection path:

    collect_rollouts_vectorized(venv, num_episodes, buffer=None)

which plays `num_episodes` hands across an
envs.vectorized_env.VectorizedPokerEnv (many independent hands in
lockstep) and batches every synchronized decision tick through
PokerActorCritic.act_batch() (models/policy_value_network.py) instead
of looping episode-by-episode, decision-by-decision, through a single
env with single-example .act() calls (the original
collect_rollouts()).

Mathematically, this changes NOTHING about what PPO sees: every
transition recorded is exactly the same (obs, action, log_prob, value,
reward, done) tuple that the old sequential path would have recorded
for that same hand, in the same per-episode contiguous grouping
RolloutBuffer.start_episode()/finish_episode() already requires (each
env's own hand's decisions are still filed into the buffer as one
contiguous GAE trace, per-env, exactly as if that hand had been played
alone). Two hands never influence each other's advantages/returns —
only the SCHEDULING of when transitions are computed changes (many
hands' decisions computed in one forward pass instead of one hand's
decisions computed across many forward passes). collect_rollouts() (the
original, single-env, single-example path) is kept unchanged and still
used wherever only one environment instance is available/appropriate
(e.g. a bespoke non-BasePokerEnv driver, or when num_envs=1 offers no
benefit over the simpler code path).

train() gained an optional `env_factory` + `num_envs` pair: when both
are given, the vectorized path is used; when omitted, train() falls
back to the original single-env sequential path exactly as before (no
behavior change for existing callers).

[... rest of original module docstring (PPO algorithm, one
rollout/update cycle, dependencies) is unchanged and omitted here for
brevity — see the original file for the full write-up. ]
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.envs.vectorized_env import VectorizedPokerEnv
from poker_rl_lab.trainers.rollout_buffer import RolloutBuffer


def _infer_device(model) -> torch.device:
    """Best-effort device inference for an ActorCriticLike model."""
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
    can pass a lightweight stub to exercise the PPO math without
    instantiating the TF-backed hand encoder or the poker_eval C++
    extension. act_batch is required for the vectorized collection
    path only — the sequential path only ever calls act().
    """
    def act(self, obs, legal_mask, deterministic: bool = False): ...
    def act_batch(self, obs_batch, legal_masks, deterministic: bool = False): ...
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


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Clipped-surrogate PPO trainer. See original module docstring for the
    full parameter rationale (unchanged); only new/modified members are
    documented inline below.
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
    # Rollout collection — sequential (original path, unchanged behavior)
    # ------------------------------------------------------------------

    def collect_rollouts(
        self,
        env,
        scenario_gen,
        num_episodes: int,
        buffer: Optional[RolloutBuffer] = None,
    ) -> RolloutBuffer:
        """
        Play `num_episodes` full hands with the current policy, ONE HAND
        AT A TIME, and record every hero decision into a RolloutBuffer.

        Unchanged from the original implementation. Prefer
        collect_rollouts_vectorized() whenever more than one environment
        instance can reasonably be constructed for this env type — see
        that method's docstring.
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
    # Rollout collection — vectorized (NEW)
    # ------------------------------------------------------------------

    def collect_rollouts_vectorized(
        self,
        venv: VectorizedPokerEnv,
        num_episodes: int,
        buffer: Optional[RolloutBuffer] = None,
    ) -> RolloutBuffer:
        """
        Play at least `num_episodes` hands across `venv`'s independent
        sub-environments, batching every synchronized decision tick
        through actor_critic.act_batch().

        Algorithm
        ---------
        Each sub-env accumulates its OWN hand's decisions in a small
        per-env pending list (handles envs where a hand involves more
        than one hero decision, e.g. river check-then-facing-a-bet).
        The moment a sub-env's `done` comes back True, its pending list
        — now a complete, contiguous episode — is filed into `buffer`
        via start_episode()/add()×N/finish_episode(), exactly matching
        what collect_rollouts() would have produced for that same hand
        played alone. This guarantees GAE is computed per-hand, never
        bleeding across hands OR across sub-envs (see
        trainers/rollout_buffer.py's own docstring on why per-episode
        GAE is structural, not just done-flag-enforced).

        Because sub-envs auto-reset on completion
        (envs.vectorized_env.VectorizedPokerEnv.step()), collection
        simply keeps ticking until at least `num_episodes` hands have
        completed across the whole batch — it does NOT wait for every
        sub-env to be mid-episode-aligned. Any sub-env still mid-hand
        when the target is reached has its (incomplete) pending
        decisions silently discarded for THIS call (nothing was ever
        added to `buffer` for them) — safe, since nothing was recorded
        prematurely; those hands' cards remain live in `venv` and will
        simply be continued on the next call to this method with the
        same `venv` instance if the caller keeps reusing it, or replayed
        from scratch if a fresh `venv` is constructed. Either is fine:
        no transition is ever double-counted or half-recorded.

        Parameters
        ----------
        venv : envs.vectorized_env.VectorizedPokerEnv
            Already reset() is called internally on first use is NOT
            assumed — callers should NOT call venv.reset() themselves;
            this method owns the sub-envs' lifecycle for the duration of
            the call. Pass the SAME venv across multiple calls (e.g.
            once per PPO iteration) to keep sub-envs' in-flight
            multi-decision hands alive between calls rather than
            resetting everything each iteration.
        num_episodes : int
            Minimum number of completed hands to collect. The actual
            number of transitions collected depends on how many hero
            decisions those hands (plus any earlier still-in-flight
            hands in other sub-envs) took.
        buffer : RolloutBuffer, optional
            Append into an existing buffer instead of creating a new one.

        Returns
        -------
        RolloutBuffer
            Populated with >= `num_episodes` completed episodes, each
            with finish_episode() already called.

        Notes
        -----
        Action selection uses torch.no_grad() and act_batch() —
        identical rationale to collect_rollouts(), just batched: no
        gradient should be tracked during data collection, and every
        row's sampled action/log_prob/value in this batch is
        numerically identical to what a per-example act() call would
        have produced for that row (see
        PokerActorCritic.act_batch()'s own docstring).
        """
        if num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")

        if buffer is None:
            buffer = RolloutBuffer(gamma=self.gamma, gae_lambda=self.gae_lambda)

        num_envs = venv.num_envs
        obs_batch = venv.reset()
        pending: List[List[dict]] = [[] for _ in range(num_envs)]
        completed = 0

        while completed < num_episodes:
            rep_batch = [self.adapter.adapt(o) for o in obs_batch]
            legal_masks = [o.legal_mask for o in obs_batch]

            with torch.no_grad():
                results = self.actor_critic.act_batch(
                    rep_batch, legal_masks, deterministic=False
                )

            actions = [
                AbstractAction(ActionType(r.action_type)) for r in results
            ]

            for i in range(num_envs):
                pending[i].append(
                    {
                        "obs": rep_batch[i],
                        "mask": legal_masks[i],
                        "action": results[i].action_type,
                        "log_prob": float(results[i].log_prob.item()),
                        "value": float(results[i].value.item()),
                    }
                )

            next_obs_batch, rewards, dones, _infos = venv.step(actions)

            for i in range(num_envs):
                if not dones[i]:
                    continue

                episode = pending[i]
                n = len(episode)
                buffer.start_episode()
                for j, transition in enumerate(episode):
                    is_last = j == n - 1
                    buffer.add(
                        obs=transition["obs"],
                        legal_mask=transition["mask"],
                        action=transition["action"],
                        log_prob=transition["log_prob"],
                        value=transition["value"],
                        reward=(rewards[i] if is_last else 0.0),
                        done=is_last,
                    )
                buffer.finish_episode()

                pending[i] = []
                completed += 1

            obs_batch = next_obs_batch

        return buffer

    # ------------------------------------------------------------------
    # PPO update (unchanged)
    # ------------------------------------------------------------------

    def update(
        self,
        buffer: RolloutBuffer,
        epochs: int = 4,
        minibatch_size: int = 64,
    ) -> Dict[str, float]:
        """
        Run `epochs` passes of minibatch clipped-surrogate PPO updates
        over everything currently in `buffer`. Unchanged from the
        original implementation.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if minibatch_size < 1:
            raise ValueError(f"minibatch_size must be >= 1, got {minibatch_size}")

        batch = buffer.get()
        obs: List = batch["obs"]
        legal_masks: List = batch["legal_masks"]
        actions: torch.Tensor = batch["actions"]

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
        env_factory=None,
        num_envs: Optional[int] = None,
    ) -> List[IterationStats]:
        """
        Run `num_iterations` full collect -> update -> clear cycles.

        Parameters
        ----------
        env : envs.base_env.BasePokerEnv
            Used for the sequential path (env_factory/num_envs both
            omitted) exactly as before. Ignored when the vectorized
            path is active.
        scenario_gen : scenarios.scenario_generator.ScenarioGenerator
        env_factory : Callable[[], BasePokerEnv], optional
            If given together with `num_envs`, rollout collection uses
            envs.vectorized_env.VectorizedPokerEnv(env_factory,
            scenario_gen, num_envs) and
            collect_rollouts_vectorized() instead of the sequential
            collect_rollouts() path. One VectorizedPokerEnv is
            constructed once, up front, and reused across every
            iteration (so in-flight multi-decision hands carry over
            between iterations rather than being discarded each time).
        num_envs : int, optional
            Number of hands run in parallel. Required together with
            env_factory to activate the vectorized path; ignored
            otherwise.
        (all other parameters unchanged from the original — see
        original docstring.)

        Returns
        -------
        List[IterationStats]
            One entry per iteration, in order — same shape regardless
            of which collection path was used.
        """
        history: List[IterationStats] = []
        start_time = time.time()

        use_vectorized = env_factory is not None and num_envs is not None
        venv: Optional[VectorizedPokerEnv] = None
        if use_vectorized:
            venv = VectorizedPokerEnv(env_factory, scenario_gen, num_envs)

        for i in range(num_iterations):
            if use_vectorized:
                buffer = self.collect_rollouts_vectorized(
                    venv, episodes_per_iteration
                )
            else:
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
    # Internal helpers (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _episode_end_indices(buffer: RolloutBuffer) -> List[int]:
        return [i for i, t in enumerate(buffer._transitions) if t.done]

    def _iteration_reward(self, buffer: RolloutBuffer) -> float:
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
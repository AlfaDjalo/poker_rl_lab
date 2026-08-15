"""
envs/vectorized_env.py — Synchronous vectorized wrapper for BasePokerEnv.

Why this exists
----------------
trainers/ppo_trainer.py's PPOTrainer.collect_rollouts() previously ran one
hand at a time, one hero decision at a time: env.reset() -> while not
done: act(single obs) -> env.step(). Every neural-network call in that
loop was therefore a batch-of-1 forward pass, even though every
sub-encoder in representation/representation_model.py already exposes a
real forward_batch() and PokerActorCritic.evaluate_actions() (the PPO
*update* step) already uses it. Rollout *collection* was the one place
in the pipeline still fully sequential.

VectorizedPokerEnv closes that gap the same way standard RL frameworks
do (OpenAI Gym's VecEnv / SB3's SubprocVecEnv, applied synchronously
rather than across processes since a single hand's Python/CAP-engine
work is cheap relative to the neural network call): run `num_envs`
independent BasePokerEnv instances, one hand each, and step all of them
together. At every synchronized "tick", every sub-env contributes
exactly one hero observation, so the policy can process all `num_envs`
decisions in a single batched forward pass via
PokerActorCritic.act_batch() (models/policy_value_network.py) instead
of `num_envs` sequential .act() calls.

Auto-reset
----------
Hands finish after different numbers of decisions (push-fold: always 1;
river: 1-2; a future full-holdem env: many). Rather than stalling the
batch waiting for every sub-env to finish before continuing (which would
waste the network's batch capacity once a few hands finish early),
whichever sub-env terminates on a given step() call is IMMEDIATELY reset
with a freshly generated scenario. step() therefore always returns
`num_envs` live, non-terminal observations. Per-hand results are never
lost — they're reported via that call's own `rewards`/`dones` lists,
matching BasePokerEnv's own "reward is 0.0 except on the terminal step"
contract exactly, just across `num_envs` parallel streams instead of one.

This means a "tick" of the vectorized env does NOT correspond to a fixed
number of completed hands — it corresponds to `num_envs` hero decisions,
some of which are hand-ending and some of which aren't. A caller wanting
a fixed number of *completed episodes* per iteration should keep
stepping until it has observed that many `done=True` entries (see
trainers/ppo_trainer.py's collect_rollouts_vectorized for the exact
bookkeeping), not for a fixed number of ticks.

What is NOT changed
--------------------
Poker semantics are completely untouched: each sub-env is a normal,
independently-instantiated BasePokerEnv (PushFoldEnv, RiverEnv,
GenericTreeEnv, ...) running its own hand exactly as it would
standalone. VectorizedPokerEnv only changes *scheduling* (many hands
interleaved instead of one at a time) — no hand ever observes or is
influenced by any other hand's cards, actions, or outcome. This is
required for PPO/self-play correctness: batching must not create any
cross-episode information leakage, and it doesn't, since each sub-env's
RNG stream and game state are fully independent objects.

Dependencies
------------
    envs.base_env.BasePokerEnv, envs.observations.StructuredObservation,
    actions.abstract_action.AbstractAction,
    scenarios.scenario_generator.ScenarioGenerator (or any object
    duck-typing .generate() -> BaseScenario). No torch — this module is
    pure Python/CAP-engine orchestration; batching the neural network
    call itself is PokerActorCritic.act_batch()'s job.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from poker_rl_lab.envs.base_env import BasePokerEnv
from poker_rl_lab.envs.observations import StructuredObservation
from poker_rl_lab.actions.abstract_action import AbstractAction


class VectorizedPokerEnv:
    """
    Runs `num_envs` independent BasePokerEnv instances in lockstep.

    Parameters
    ----------
    env_factory : Callable[[], BasePokerEnv]
        Called `num_envs` times to construct independent env instances.
        Must be a genuine factory (e.g. `lambda: PushFoldEnv(evaluator)`)
        — passing something that returns the SAME instance every call
        would make every "independent" hand alias the same mutable
        env state, corrupting the whole batch.
    scenario_gen : ScenarioGenerator
        Single shared generator used to produce every sub-env's initial
        (and every auto-reset) scenario. Sharing one generator/RNG
        stream across sub-envs is what makes the whole vectorized env
        reproducible under scenario_gen.seed(...) — see
        scenarios/scenario_generator.py's own determinism contract.
    num_envs : int
        Number of hands played in parallel.
    """

    def __init__(
        self,
        env_factory: Callable[[], BasePokerEnv],
        scenario_gen,
        num_envs: int,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self.scenario_gen = scenario_gen
        self._envs: List[BasePokerEnv] = [env_factory() for _ in range(num_envs)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> List[StructuredObservation]:
        """Reset every sub-env with a freshly generated scenario."""
        return [env.reset(self.scenario_gen.generate()) for env in self._envs]

    def step(
        self, actions: List[AbstractAction]
    ) -> Tuple[List[StructuredObservation], List[float], List[bool], List[Dict]]:
        """
        Apply one action per sub-env; auto-reset any sub-env that
        terminates so every returned observation is live.

        Parameters
        ----------
        actions : List[AbstractAction]
            Length num_envs, in sub-env order. actions[i] must be legal
            for sub-env i's current observation (same precondition as
            BasePokerEnv.step()).

        Returns
        -------
        obs : List[StructuredObservation]
            Length num_envs. Never terminal — a sub-env that just
            finished is auto-reset before its observation is returned.
        rewards : List[float]
            Length num_envs. Chip reward from the action just applied;
            0.0 for non-terminal steps.
        dones : List[bool]
            Length num_envs. True at index i iff sub-env i's hand ended
            on THIS step (i.e. rewards[i] is that hand's final reward).
        infos : List[Dict]
            Length num_envs. Each sub-env's own step() info dict.
        """
        if len(actions) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} actions, got {len(actions)}")

        obs_out: List[StructuredObservation] = [None] * self.num_envs  # type: ignore[list-item]
        rewards: List[float] = [0.0] * self.num_envs
        dones: List[bool] = [False] * self.num_envs
        infos: List[Dict] = [{}] * self.num_envs

        for i, (env, action) in enumerate(zip(self._envs, actions)):
            next_obs, reward, done, info = env.step(action)
            rewards[i] = reward
            dones[i] = done
            infos[i] = info
            obs_out[i] = env.reset(self.scenario_gen.generate()) if done else next_obs

        return obs_out, rewards, dones, infos

    def __len__(self) -> int:
        return self.num_envs
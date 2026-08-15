"""
envs/base_env.py — Abstract base class for all poker environments.

Training code must only depend on this interface. It must never import
from a concrete environment class (PushFoldEnv, RiverEnv, etc.).

This indirection allows:
  - Swapping environments without changing the trainer.
  - Curriculum progression (push-fold → river → holdem → …) by
    passing a different env to the same training loop.
  - Testing any env with the shared conformance suite.

Concrete environments
---------------------
  Inherit from BasePokerEnv and implement reset() and step():

      class PushFoldEnv(BasePokerEnv):
          def reset(self, scenario): ...
          def step(self, action):   ...

Utilities included here
-----------------------
  run_episode     Run one complete episode given a hero-policy callable.
  RandomHeroPolicy  Uniform-random policy over legal actions; for tests and
                    baseline evaluation.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple

from poker_rl_lab.scenarios.base_scenario import BaseScenario
from poker_rl_lab.envs.observations import StructuredObservation
from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS


# ── Abstract interface ────────────────────────────────────────────────────────

class BasePokerEnv(ABC):
    """
    Abstract base class for all RL poker environments.

    The training loop interacts exclusively through this interface.
    Concrete subclasses handle game-specific logic and villain behaviour.

    Contract
    --------
    reset(scenario) -> StructuredObservation
        Begin a new episode. The returned observation must be non-terminal
        and must have at least one True entry in legal_mask.

    step(action) -> (obs, reward, done, info)
        Apply *action* and advance the episode.
        - obs.is_terminal is True iff done is True.
        - When done is True, legal_mask is all-False.
        - reward is a finite float (chip change for the hero).
        - Calling step() when done=True must raise RuntimeError.
        - Calling step() before reset() must raise RuntimeError.
    """

    @abstractmethod
    def reset(self, scenario: BaseScenario) -> StructuredObservation:
        """
        Start a new episode from *scenario*.

        Parameters
        ----------
        scenario : BaseScenario
            Initial conditions produced by a ScenarioGenerator.

        Returns
        -------
        StructuredObservation
            Non-terminal observation at the hero's first decision point.
        """
        ...

    @abstractmethod
    def step(
        self,
        action: AbstractAction,
    ) -> Tuple[StructuredObservation, float, bool, Dict]:
        """
        Apply *action* and advance the episode.

        Parameters
        ----------
        action : AbstractAction
            Must be legal (True in the current legal_mask).

        Returns
        -------
        obs : StructuredObservation
        reward : float
            Chip change for the hero this episode.
            Intermediate (non-terminal) steps return 0.0.
        done : bool
            True when the episode has ended.
        info : dict
            Diagnostic information. Keys vary by environment.
        """
        ...


# ── Episode runner ────────────────────────────────────────────────────────────

class StepRecord:
    """One hero decision and its outcome."""
    __slots__ = ("obs", "action", "reward", "next_obs", "done")

    def __init__(
        self,
        obs:      StructuredObservation,
        action:   AbstractAction,
        reward:   float,
        next_obs: StructuredObservation,
        done:     bool,
    ) -> None:
        self.obs      = obs
        self.action   = action
        self.reward   = reward
        self.next_obs = next_obs
        self.done     = done

    def __repr__(self) -> str:
        return (
            f"StepRecord(action={self.action.action_type.name}, "
            f"reward={self.reward:.2f}, done={self.done})"
        )


def run_episode(
    env:          BasePokerEnv,
    scenario:     BaseScenario,
    hero_policy:  Callable[[StructuredObservation], AbstractAction],
) -> Tuple[float, List[StepRecord]]:
    """
    Run one complete episode and return the result.

    Parameters
    ----------
    env : BasePokerEnv
        Any concrete environment.
    scenario : BaseScenario
        Initial conditions for this episode.
    hero_policy : callable
        ``policy(obs) -> AbstractAction``. Called at every hero decision.

    Returns
    -------
    total_reward : float
        Sum of all rewards across the episode (equal to the final
        step's reward since intermediate rewards are 0).
    trajectory : List[StepRecord]
        One record per hero decision, in order.
    """
    obs        = env.reset(scenario)
    trajectory = []
    total      = 0.0

    while not obs.is_terminal:
        action                       = hero_policy(obs)
        next_obs, reward, done, info = env.step(action)
        trajectory.append(StepRecord(obs, action, reward, next_obs, done))
        total += reward
        obs    = next_obs

    return total, trajectory


# ── Random hero policy ────────────────────────────────────────────────────────

class RandomHeroPolicy:
    """
    Uniform-random policy over legal actions.

    Use for:
    - Baseline evaluation (expected win-rate = ~50% heads-up).
    - Smoke-testing that any env can complete full episodes.
    - Conformance checks in the test suite.

    Parameters
    ----------
    seed : int, optional
        Seed for reproducible action selection.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def __call__(self, obs: StructuredObservation) -> AbstractAction:
        """Sample uniformly from legal actions."""
        legal = [i for i, m in enumerate(obs.legal_mask) if m]
        if not legal:
            raise ValueError("No legal actions available — is the observation terminal?")
        return AbstractAction(ActionType(self._rng.choice(legal)))
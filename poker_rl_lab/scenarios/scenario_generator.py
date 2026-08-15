"""
scenario_generator.py — Abstract base class for all scenario generators.

Training code must never construct Scenarios directly. It always requests
them from a ScenarioGenerator. This indirection allows:
- Swapping random generation for database-loaded scenarios.
- Constraining to specific board textures or stack depths.
- Deterministic reproduction of any training run via a fixed seed.

Usage
-----
    from rl_lab.scenarios.push_fold_generator import PushFoldScenarioGenerator

    gen = PushFoldScenarioGenerator(player_count=2)
    gen.seed(42)

    for _ in range(1000):
        scenario = gen.generate()
        # pass scenario to Environment.reset()
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from poker_rl_lab.scenarios.base_scenario import BaseScenario


class ScenarioGenerator(ABC):
    """
    Abstract interface for all scenario generators.

    All concrete generators must implement generate(). They must be
    deterministic when seeded with the same value.

    Parameters
    ----------
    player_count : int
        Number of seats to include in generated scenarios.
    """

    def __init__(self, player_count: int) -> None:
        if player_count < 2:
            raise ValueError(
                f"player_count must be >= 2, got {player_count}"
            )
        self._player_count = player_count
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def player_count(self) -> int:
        """Number of seats in scenarios produced by this generator."""
        return self._player_count

    def seed(self, value: int) -> None:
        """
        Seed the internal random number generator.

        Calling seed() with the same value before generate() always
        produces an identical sequence of Scenarios.

        Parameters
        ----------
        value : int
            Any integer seed value.
        """
        self._rng.seed(value)

    @abstractmethod
    def generate(self) -> BaseScenario:
        """
        Generate one Scenario.

        Returns
        -------
        BaseScenario
            A legal, fully initialised scenario ready for Environment.reset().
        """
        ...

    # ------------------------------------------------------------------
    # Card helpers shared across generators
    # ------------------------------------------------------------------

    def _deal_unique_cards(self, count: int, excluded_mask: int = 0) -> list[int]:
        """
        Draw `count` distinct card ids, none of which appear in excluded_mask.

        Parameters
        ----------
        count : int
            Number of cards to draw.
        excluded_mask : int
            Bitmask of cards already in use; none of these will be returned.

        Returns
        -------
        list[int]
            Unique card ids in [0, 51], in draw order.
        """
        available = [c for c in range(52) if not (excluded_mask >> c & 1)]
        if len(available) < count:
            raise ValueError(
                f"Cannot deal {count} cards; only {len(available)} available "
                f"after excluding {bin(excluded_mask).count('1')} cards."
            )
        return self._rng.sample(available, count)

    @staticmethod
    def _cards_to_mask(card_ids: list[int]) -> int:
        """Convert a list of card ids to a bitmask."""
        mask = 0
        for c in card_ids:
            mask |= 1 << c
        return mask
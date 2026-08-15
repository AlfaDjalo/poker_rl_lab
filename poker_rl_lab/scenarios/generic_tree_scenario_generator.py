"""
scenarios/generic_tree_scenario_generator.py — Produces
GenericTreeScenario instances (envs/generic_tree_env.py) from a
DuoGameConfig (scenarios/duo_game_config.py).

This is the "GenericTreeScenarioGenerator" both
scenarios/generic_tree_env.py's and duo_game_config.py's own docstrings
name as the next step: the piece that actually deals cards and sizes
stacks/pot per a config's StackConfig/PotConfig/DealConfig, so a brand
new fixed-tree YAML under scenarios/duo_games/ needs zero new Python —
not even a new generator — the way push_fold_duo_generator.py /
river_duo_generator.py currently require one bespoke generator class
each.

Not a scenarios.scenario_generator.ScenarioGenerator subclass
------------------------------------------------------------------
That base class's generate() -> BaseScenario contract doesn't match
this generator's GenericTreeScenario output type (a deliberately
narrower shape than BaseScenario — see generic_tree_env.py's module
docstring). This class duplicates ScenarioGenerator's small private
card-dealing helpers (_deal_unique_cards / _cards_to_mask) rather than
inheriting, and instead duck-types the same seed()/generate() surface
every generator in this codebase already exposes, so it drops into
evaluation-style callers (anything that calls
scenario_gen.seed(seed); scenario_gen.generate() in a loop) unchanged.

Stack / pot sizing
--------------------
Delegates entirely to the config's own StackConfig/PotConfig — see
_size_stack()/_size_pot()'s docstrings below for the exact semantics of
each mode, and duo_game_config.py's PotConfig docstring for the
posted-vs-pot distinction (`posted` = a seat's own chips, returnable as
excess; `pot` = dead money from earlier streets, belongs to no seat).

Dependencies
------------
    Standard library only. scenarios.duo_game_config,
    envs.generic_tree_env.GenericTreeScenario. No poker_eval, no torch
    — matches every other *_generator.py in this codebase.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from poker_rl_lab.scenarios.duo_game_config import DuoGameConfig
from poker_rl_lab.envs.generic_tree_env import GenericTreeScenario


class GenericTreeScenarioGenerator:
    """
    Deals cards and sizes stacks/pot for one DuoGameConfig.

    Parameters
    ----------
    config : DuoGameConfig
        Produced by scenarios.duo_game_config.load_duo_game(name).
    """

    def __init__(self, config: DuoGameConfig) -> None:
        self.config = config
        self._rng = random.Random()

    def seed(self, value: int) -> None:
        """Seed the internal RNG. Same determinism contract as
        scenarios.scenario_generator.ScenarioGenerator.seed()."""
        self._rng.seed(value)

    # ------------------------------------------------------------------
    # Card helpers (duplicated from ScenarioGenerator — see class
    # docstring for why this isn't a subclass)
    # ------------------------------------------------------------------

    def _deal_unique_cards(self, count: int, excluded_mask: int = 0) -> List[int]:
        available = [c for c in range(52) if not (excluded_mask >> c & 1)]
        if len(available) < count:
            raise ValueError(
                f"Cannot deal {count} cards; only {len(available)} available "
                f"after excluding {bin(excluded_mask).count('1')} cards."
            )
        return self._rng.sample(available, count)

    @staticmethod
    def _cards_to_mask(card_ids: List[int]) -> int:
        mask = 0
        for c in card_ids:
            mask |= 1 << c
        return mask

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self) -> GenericTreeScenario:
        """
        Generate one scenario for self.config: size stacks/pot per
        config, deal the board (if config.deal.deal_board) then every
        seat's hole cards, with no overlap.

        Returns
        -------
        GenericTreeScenario
            Ready for GenericTreeEnv.reset().
        """
        cfg = self.config
        n = len(cfg.seats)

        stack = self._size_stack()
        starting_stacks = tuple(stack for _ in range(n))
        posted, pot = self._size_pot(stack)

        used_mask = 0
        node_cards: Tuple = ()
        board_mask = 0
        if cfg.deal.deal_board:
            board_cards = self._deal_unique_cards(cfg.deal.node_count, excluded_mask=used_mask)
            used_mask |= self._cards_to_mask(board_cards)
            node_cards = tuple(board_cards)
            board_mask = used_mask

        hole_masks: List[int] = []
        for _ in range(n):
            cards = self._deal_unique_cards(
                cfg.deal.hole_cards_per_player, excluded_mask=used_mask
            )
            mask = self._cards_to_mask(cards)
            used_mask |= mask
            hole_masks.append(mask)

        dealer_seat_id = cfg.dealer_seat or cfg.first_seat_id

        return GenericTreeScenario(
            config_name=cfg.name,
            hole_masks=tuple(hole_masks),
            board_mask=board_mask,
            node_cards=node_cards,
            starting_stacks=starting_stacks,
            posted=posted,
            pot=pot,
            dealer_position=cfg.seat_index(dealer_seat_id),
        )

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _size_stack(self) -> int:
        """
        Every seat gets the SAME stack — see StackConfig's own
        docstring (duo_game_config.py) for why this schema never sizes
        seats asymmetrically.
        """
        cfg = self.config
        sc = cfg.stack
        if sc.mode == "fixed_bb":
            return max(1, int(round(sc.stack_bb * cfg.big_blind)))
        # "randomized_bb" — the only other mode StackConfig.__post_init__ allows
        min_chips = max(1, int(sc.min_stack_bb * cfg.big_blind))
        max_chips = max(min_chips, int(sc.max_stack_bb * cfg.big_blind))
        return self._rng.randint(min_chips, max_chips)

    def _size_pot(self, stack: int) -> Tuple[Tuple[int, ...], int]:
        """
        Returns (posted, pot).

        "blinds" mode: seat index 0 posts small_blind, seat index 1
        posts big_blind (both clamped to `stack`, matching the real
        Engine's all-in-for-less-than-a-blind handling —
        poker_state.py's _post_blind), every other seat posts 0.
        Matches push_fold_duo.yaml's 2-seat game exactly; this is the
        documented N-seat reading of "blinds" mode from PotConfig's own
        docstring ("only SB/BB post before the tree starts").
        `pot` is 0 — every chip is attributed to a seat via `posted`.

        "randomized_bb" mode: nothing is any seat's own commitment —
        the entire sized amount is dead-money `pot` (see
        GenericTreeScenario.pot's docstring), `posted` is all zero.
        """
        cfg = self.config
        n = len(cfg.seats)

        if cfg.pot.mode == "blinds":
            posted = [0] * n
            if n >= 1:
                posted[0] = min(cfg.small_blind, stack)
            if n >= 2:
                posted[1] = min(cfg.big_blind, stack)
            return tuple(posted), 0

        # "randomized_bb"
        min_chips = max(1, int(cfg.pot.min_pot_bb * cfg.big_blind))
        max_chips = max(min_chips, int(cfg.pot.max_pot_bb * cfg.big_blind))
        pot = self._rng.randint(min_chips, max_chips)
        return tuple(0 for _ in range(n)), pot
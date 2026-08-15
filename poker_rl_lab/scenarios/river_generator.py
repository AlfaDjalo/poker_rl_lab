"""
river_generator.py — Generator for heads-up river scenarios.

A river scenario begins at the final street with all five board cards
dealt. Two players remain with random stacks and a random pot. The
out-of-position player (seat 1) acts first.

Scenario structure
------------------
    street_index = 3  (river; preflop=0, flop=1, turn=2, river=3)
    node_cards   = (c0, c1, c2, c3, c4)  — 5 board cards (single board)
    hole_cards   = 2 per active player
    bet_to_call  = 0  (check is available; no bet has been placed yet)

Each player holds 2 hole cards (Omaha-compatible; for holdem only 2 are
used). Stack depths and pot sizes are drawn uniformly within configured
ranges to expose the agent to a wide variety of stack-to-pot ratios.
"""

from __future__ import annotations

from poker_rl_lab.scenarios.base_scenario import BaseScenario, PlayerScenario
from poker_rl_lab.scenarios.scenario_generator import ScenarioGenerator

# Number of board nodes for a single hold'em board (flop=3, turn=1, river=1)
_HOLDEM_NODE_COUNT = 5

# Street index of the river in a standard hold'em game
_RIVER_STREET_INDEX = 3


class RiverScenarioGenerator(ScenarioGenerator):
    """
    Generates heads-up river scenarios on a single hold'em board.

    Both players check to open each generated scenario (bet_to_call=0).
    Stack and pot sizes are randomised within configured ranges so that
    the agent learns across different SPR (stack-to-pot ratios).

    Parameters
    ----------
    player_count : int
        Total number of seats. Minimum 2. Seats beyond the first two are
        dealt hole cards but are treated as having already folded.
    small_blind : int
        Small blind size in chips.
    big_blind : int
        Big blind size in chips.
    hole_cards_per_player : int
        Number of hole cards dealt to each active player (2 for hold'em,
        4 for Omaha). Defaults to 2.
    min_pot_bb : float
        Minimum pot size in big blinds.
    max_pot_bb : float
        Maximum pot size in big blinds.
    min_stack_bb : float
        Minimum remaining stack per player in big blinds.
    max_stack_bb : float
        Maximum remaining stack per player in big blinds.
    """

    def __init__(
        self,
        player_count: int = 2,
        small_blind: int = 1,
        big_blind: int = 2,
        hole_cards_per_player: int = 2,
        min_pot_bb: float = 2.0,
        max_pot_bb: float = 20.0,
        min_stack_bb: float = 2.0,
        max_stack_bb: float = 50.0,
    ) -> None:
        super().__init__(player_count)

        if hole_cards_per_player < 2:
            raise ValueError(
                f"hole_cards_per_player must be >= 2, got {hole_cards_per_player}"
            )
        if min_pot_bb <= 0:
            raise ValueError(f"min_pot_bb must be > 0, got {min_pot_bb}")
        if max_pot_bb < min_pot_bb:
            raise ValueError(
                f"max_pot_bb ({max_pot_bb}) must be >= min_pot_bb ({min_pot_bb})"
            )
        if min_stack_bb <= 0:
            raise ValueError(f"min_stack_bb must be > 0, got {min_stack_bb}")
        if max_stack_bb < min_stack_bb:
            raise ValueError(
                f"max_stack_bb ({max_stack_bb}) must be >= min_stack_bb ({min_stack_bb})"
            )

        self._small_blind = small_blind
        self._big_blind = big_blind
        self._hole_cards_per_player = hole_cards_per_player
        self._min_pot_bb = min_pot_bb
        self._max_pot_bb = max_pot_bb
        self._min_stack_bb = min_stack_bb
        self._max_stack_bb = max_stack_bb

    def generate(self) -> BaseScenario:
        """
        Generate one river scenario.

        Returns
        -------
        BaseScenario
            River scenario; both active players face a check-or-bet decision.
        """
        # --- pot and stacks (in chips) ---
        pot = self._rng.randint(
            max(1, int(self._min_pot_bb * self._big_blind)),
            int(self._max_pot_bb * self._big_blind),
        )
        p0_stack = self._rng.randint(
            max(1, int(self._min_stack_bb * self._big_blind)),
            int(self._max_stack_bb * self._big_blind),
        )
        p1_stack = self._rng.randint(
            max(1, int(self._min_stack_bb * self._big_blind)),
            int(self._max_stack_bb * self._big_blind),
        )

        # --- deal board (5 cards for single hold'em board) ---
        board_cards = self._deal_unique_cards(_HOLDEM_NODE_COUNT)
        used_mask = self._cards_to_mask(board_cards)

        # --- deal hole cards for every seat ---
        player_masks: list[int] = []
        for _ in range(self._player_count):
            cards = self._deal_unique_cards(
                self._hole_cards_per_player, excluded_mask=used_mask
            )
            mask = self._cards_to_mask(cards)
            player_masks.append(mask)
            used_mask |= mask

        # --- build player states ---
        # Seat 0: in-position (acts second on the river)
        p0 = PlayerScenario(
            stack=p0_stack,
            hand_mask=player_masks[0],
            current_bet=0,
            total_contribution=pot // 2,  # approximate; not used by evaluator
            has_folded=False,
            is_all_in=False,
        )
        # Seat 1: out-of-position (acts first on the river)
        p1 = PlayerScenario(
            stack=p1_stack,
            hand_mask=player_masks[1],
            current_bet=0,
            total_contribution=pot - pot // 2,
            has_folded=False,
            is_all_in=False,
        )
        # Extra seats have already folded
        extras = [
            PlayerScenario(
                stack=0,
                hand_mask=player_masks[i + 2],
                current_bet=0,
                total_contribution=0,
                has_folded=True,
                is_all_in=False,
            )
            for i in range(self._player_count - 2)
        ]

        players = tuple([p0, p1] + extras)

        # Dealer is seat 0; OOP player (seat 1) acts first postflop.
        # No bet placed yet: bet_to_call = 0, check available.
        return BaseScenario(
            game_name="holdem",
            street_index=_RIVER_STREET_INDEX,
            pot=pot,
            dealer_position=0,
            current_player=1,
            bet_to_call=0,
            min_raise=self._big_blind,
            players=players,
            node_cards=tuple(board_cards),
            small_blind=self._small_blind,
            big_blind=self._big_blind,
            discard_pile=tuple(),
            raises_this_street=0,
            last_aggressor=None,
        )
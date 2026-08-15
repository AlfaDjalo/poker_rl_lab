"""
double_board_river_generator.py — Generator for double-board river scenarios.

A double-board river scenario is identical to a river scenario except that
two independent 5-card boards are dealt rather than one. This is the
standard CAP "double board" configuration used in many CAP games.

Board layout (10 nodes)
-----------------------
    Nodes 0-4  : Board 1 (flop0, flop1, flop2, turn, river)
    Nodes 5-9  : Board 2 (flop0, flop1, flop2, turn, river)

Scoring (from the CAP game guide / technical document)
-------------------------------------------------------
    Typical point structures include high/high/hand (3 points) and
    high/low/high/low/hand (5 points). This generator produces the raw
    poker state; point definitions live in the Engine's YAML config.

Scenario structure
------------------
    game_name    = "double_board"  (matches a CAP Engine YAML)
    street_index = 3  (river — both boards fully dealt)
    node_cards   = 10-tuple; indices 0-4 = board 1, 5-9 = board 2
    hole_cards   = 4 per active player (Omaha-style, as most CAP games use)
    bet_to_call  = 0  (check available; no bet placed yet)
"""

from __future__ import annotations

from poker_rl_lab.scenarios.base_scenario import BaseScenario, PlayerScenario
from poker_rl_lab.scenarios.scenario_generator import ScenarioGenerator

_DOUBLE_BOARD_NODE_COUNT = 10   # 5 nodes per board × 2 boards
_RIVER_STREET_INDEX = 3
_DEFAULT_HOLE_CARDS = 4         # Omaha-style


class DoubleBoardRiverScenarioGenerator(ScenarioGenerator):
    """
    Generates heads-up river scenarios on two independent hold'em boards.

    Both boards are fully dealt (10 node cards total). Active players
    hold 4 hole cards each (Omaha-style). The scenario opens on the river
    with no bet placed.

    Parameters
    ----------
    player_count : int
        Total number of seats. Minimum 2.
    small_blind : int
        Small blind size in chips.
    big_blind : int
        Big blind size in chips.
    hole_cards_per_player : int
        Number of hole cards per active player. Defaults to 4 (Omaha).
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
        hole_cards_per_player: int = _DEFAULT_HOLE_CARDS,
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
        Generate one double-board river scenario.

        Returns
        -------
        BaseScenario
            River scenario with two fully dealt boards. Active players
            face a check-or-bet decision with no board bet placed.
        """
        # --- pot and stacks ---
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

        # --- deal both boards (10 cards) ---
        board_cards = self._deal_unique_cards(_DOUBLE_BOARD_NODE_COUNT)
        used_mask = self._cards_to_mask(board_cards)

        # --- deal hole cards ---
        player_masks: list[int] = []
        for _ in range(self._player_count):
            cards = self._deal_unique_cards(
                self._hole_cards_per_player, excluded_mask=used_mask
            )
            mask = self._cards_to_mask(cards)
            player_masks.append(mask)
            used_mask |= mask

        # --- build player states ---
        p0 = PlayerScenario(
            stack=p0_stack,
            hand_mask=player_masks[0],
            current_bet=0,
            total_contribution=pot // 2,
            has_folded=False,
            is_all_in=False,
        )
        p1 = PlayerScenario(
            stack=p1_stack,
            hand_mask=player_masks[1],
            current_bet=0,
            total_contribution=pot - pot // 2,
            has_folded=False,
            is_all_in=False,
        )
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

        return BaseScenario(
            game_name="double_board",
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
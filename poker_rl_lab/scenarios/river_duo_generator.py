"""
scenarios/river_duo_generator.py — Scenario generator for the TWO-DECISION
(up to three-action) heads-up RIVER game.

Sibling of scenarios/push_fold_duo_generator.py — same "start before either
seat has acted" philosophy, just one street later and with a richer (but
still fixed, non-reraising) action tree:

    OOP (seat 1) acts first — CHECK or BET_100 (pot-sized bet only).
        OOP CHECK  -> IP (seat 0) acts — CHECK or BET_100.
            IP CHECK -> showdown (check-check).
            IP BET   -> OOP acts again — CALL or FOLD.
        OOP BET    -> IP (seat 0) acts — CALL or FOLD.

No re-raises anywhere in this tree — see envs/river_duo_env.py for the
full state machine. This generator only produces the INITIAL scenario,
at OOP's very first decision (mirrors river_generator.py's own
"street_index=3, bet_to_call=0" starting point).

Why stack size is standardized
-------------------------------
Per the user's spec: stack depth isn't the thing being trained here (bet
sizing is fixed at pot, so there's no continuous sizing decision that
stack depth would otherwise modulate) — both players start the street
with the SAME fixed stack (`stack_bb`), same rationale
push_fold_duo_generator.py uses for its short-stack push/fold game. Pot
size (SPR) is still randomised across a configured range so the agents
see a variety of stack-to-pot ratios, matching river_generator.py's own
min/max_pot_bb knobs.

Scenario structure
-------------------
    street_index = 3            (river; preflop=0, flop=1, turn=2, river=3)
    node_cards   = 5 board cards, all dealt
    hole_cards   = 2 per player (hold'em)
    pot          = randomised within [min_pot_bb, max_pot_bb] * big_blind
    bet_to_call  = 0             (OOP faces a check-or-bet decision)
    dealer_position = 0          (IP is the button, matching
                                   river_generator.py's own convention)
    current_player  = OOP_SEAT (1)
"""

from __future__ import annotations

from poker_rl_lab.scenarios.base_scenario import BaseScenario, PlayerScenario
from poker_rl_lab.scenarios.scenario_generator import ScenarioGenerator

OOP_SEAT: int = 1   # out-of-position; acts first; river_env.py's HERO_SEAT
IP_SEAT: int = 0    # in-position; acts second; river_env.py's VILLAIN_SEAT

_HOLDEM_NODE_COUNT = 5
_RIVER_STREET_INDEX = 3


class RiverDuoScenarioGenerator(ScenarioGenerator):
    """
    Generates heads-up river scenarios at OOP's decision point, with a
    standardized (fixed) stack depth for both seats.

    Parameters
    ----------
    small_blind : int
    big_blind : int
    hole_cards_per_player : int
        Defaults to 2 (hold'em).
    min_pot_bb : float
        Minimum pot size in big blinds (inclusive).
    max_pot_bb : float
        Maximum pot size in big blinds (inclusive).
    stack_bb : float
        Fixed remaining stack for BOTH players, in big blinds. Not
        randomised — see module docstring.
    """

    def __init__(
        self,
        small_blind: int = 1,
        big_blind: int = 2,
        hole_cards_per_player: int = 2,
        min_pot_bb: float = 2.0,
        max_pot_bb: float = 20.0,
        stack_bb: float = 50.0,
    ) -> None:
        super().__init__(player_count=2)

        if small_blind <= 0:
            raise ValueError(f"small_blind must be > 0, got {small_blind}")
        if big_blind <= 0:
            raise ValueError(f"big_blind must be > 0, got {big_blind}")
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
        if stack_bb <= 0:
            raise ValueError(f"stack_bb must be > 0, got {stack_bb}")

        self._small_blind = small_blind
        self._big_blind = big_blind
        self._hole_cards_per_player = hole_cards_per_player
        self._min_pot_bb = min_pot_bb
        self._max_pot_bb = max_pot_bb
        self._stack_bb = stack_bb

    def generate(self) -> BaseScenario:
        """
        Generate one river-duo scenario, OOP to act.

        Returns
        -------
        BaseScenario
            street_index=3, five board cards dealt, both players holding
            the same fixed stack, OOP (seat 1) facing a CHECK/BET_100
            decision.
        """
        pot = self._rng.randint(
            max(1, int(self._min_pot_bb * self._big_blind)),
            int(self._max_pot_bb * self._big_blind),
        )
        stack = max(1, int(round(self._stack_bb * self._big_blind)))

        # --- deal board (5 cards) ---
        board_cards = self._deal_unique_cards(_HOLDEM_NODE_COUNT)
        used_mask = self._cards_to_mask(board_cards)

        # --- deal hole cards ---
        ip_cards = self._deal_unique_cards(
            self._hole_cards_per_player, excluded_mask=used_mask
        )
        ip_mask = self._cards_to_mask(ip_cards)
        used_mask |= ip_mask

        oop_cards = self._deal_unique_cards(
            self._hole_cards_per_player, excluded_mask=used_mask
        )
        oop_mask = self._cards_to_mask(oop_cards)

        # --- build player states ---
        ip = PlayerScenario(
            stack=stack,
            hand_mask=ip_mask,
            current_bet=0,
            total_contribution=pot // 2,   # approximate; unused by evaluator
            has_folded=False,
            is_all_in=False,
        )
        oop = PlayerScenario(
            stack=stack,
            hand_mask=oop_mask,
            current_bet=0,
            total_contribution=pot - pot // 2,
            has_folded=False,
            is_all_in=False,
        )

        players = tuple(
            oop if s == OOP_SEAT else ip for s in range(2)
        )

        return BaseScenario(
            game_name="holdem",
            street_index=_RIVER_STREET_INDEX,
            pot=pot,
            dealer_position=IP_SEAT,
            current_player=OOP_SEAT,
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
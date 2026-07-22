"""
full_holdem_generator.py — Generator for full-table preflop hold'em
scenarios.

Per RL_lab_implementation_guide.md's curriculum (§1):

    1. Push/Fold
    2. River Hold'em
    3. Full Hold'em      <-- this generator feeds that stage

Unlike PushFoldScenarioGenerator (already-shoved, 2 legal actions) and
RiverScenarioGenerator (single street, self-simulated betting), scenarios
from this generator are intended to be played through the REAL Engine
(poker_engine.state.poker_state.PokerState), via
scenarios.hand_snapshot_adapter.scenario_to_snapshot() ->
PokerState.load_state(). This is the first curriculum stage where the RL
Lab must not duplicate betting/turn-order logic itself (per the guide's
core rule) — the Environment wrapping this generator (envs.full_holdem_env,
follow-up file) hands the resulting scenario straight to the Engine and
lets it drive every subsequent street.

Because the Engine is the consumer, antes and blinds here are posted
using the EXACT same formula as PokerState.start_hand() /
GameState._post_antes() / GameState._post_blinds()
(poker_engine/state/poker_state.py), not an independent reimplementation:

    antes            : every seat, in seat order, min(ante, stack)
    small blind seat : (dealer_position + 1) % n
    big blind seat   : (dealer_position + 2) % n
    first to act     : (dealer_position + 3) % n   (UTG)
    bet_to_call      : big_blind   (NOT big_blind + ante)
    min_raise        : big_blind

Note this engine does NOT special-case heads-up (n == 2): the button is
seat (dealer_position + 1) % 2, not the dealer itself, which differs from
the conventional heads-up rule "button = small blind, acts first
preflop". This mirrors a discrepancy already flagged in
representation/position_encoder.py's module docstring (which contrasts
this same engine formula against push_fold_generator.py's own, different,
heads-up convention). This generator intentionally matches the Engine
exactly, since its whole purpose is to seed real Engine hands — any
scenario it produces must be one PokerState.start_hand() could plausibly
have produced itself.

Scenario structure
-------------------
    street_index = 0            (preflop)
    node_cards   = all None     (node_count slots, nothing dealt)
    hole_cards   = 2 per player (hold'em)
    bet_to_call  = big_blind
    min_raise    = big_blind
    current_player = first-to-act seat (UTG, or the BB itself if
                     player_count <= 3 puts UTG at/behind the blinds —
                     the formula is applied uniformly regardless of table
                     size, exactly as the Engine does)

Short stacks (stack < ante, or remaining stack < blind after ante) are
posted for min(amount, stack) and marked is_all_in, exactly matching
GameState._post_antes / _post_blind's own clamping — this generator does
not avoid generating short-stacked / already-all-in preflop scenarios,
since the Engine itself can produce them (e.g. a sub-blind stack) and the
agent needs to learn them too.
"""

from __future__ import annotations

from typing import List

from .base_scenario import BaseScenario, PlayerScenario
from .scenario_generator import ScenarioGenerator

_PREFLOP_STREET_INDEX = 0
_DEFAULT_HOLE_CARDS = 2          # hold'em
_DEFAULT_NODE_COUNT = 5          # holdem.yaml: 5 board nodes, none dealt yet


class FullHoldemScenarioGenerator(ScenarioGenerator):
    """
    Generates full-table preflop hold'em scenarios (antes + blinds already
    posted, no board cards dealt), matching the real Engine's own
    start-of-hand state exactly.

    Parameters
    ----------
    player_count : int
        Total number of seats. Minimum 2.
    small_blind : int
        Small blind size in chips.
    big_blind : int
        Big blind size in chips.
    ante : int
        Per-player ante in chips. 0 (default) matches holdem.yaml.
    hole_cards_per_player : int
        Hole cards dealt to each player. Defaults to 2 (hold'em).
    node_count : int
        Number of board node slots (all left undealt / None). Must match
        the target game's GameDefinition.node_count (5 for holdem.yaml).
    min_stack_bb : float
        Minimum starting stack per player, in big blinds.
    max_stack_bb : float
        Maximum starting stack per player, in big blinds.
    game_name : str
        Engine game/variant key. Defaults to "holdem" (holdem.yaml).
    dealer_position : int
        Fixed button seat used for every generated scenario. Defaults to
        0. (Randomising the button adds no strategic variety once
        position_encoder.py's features are seat-relative — every seat
        already sees every relative position across enough hands with a
        fixed button — so this is a constructor knob, not re-rolled
        per generate() call.)
    """

    def __init__(
        self,
        player_count: int = 6,
        small_blind: int = 1,
        big_blind: int = 2,
        ante: int = 0,
        hole_cards_per_player: int = _DEFAULT_HOLE_CARDS,
        node_count: int = _DEFAULT_NODE_COUNT,
        min_stack_bb: float = 20.0,
        max_stack_bb: float = 100.0,
        game_name: str = "holdem",
        dealer_position: int = 0,
    ) -> None:
        super().__init__(player_count)

        if small_blind <= 0:
            raise ValueError(f"small_blind must be > 0, got {small_blind}")
        if big_blind <= 0:
            raise ValueError(f"big_blind must be > 0, got {big_blind}")
        if ante < 0:
            raise ValueError(f"ante must be >= 0, got {ante}")
        if hole_cards_per_player < 1:
            raise ValueError(
                f"hole_cards_per_player must be >= 1, got {hole_cards_per_player}"
            )
        if node_count < 0:
            raise ValueError(f"node_count must be >= 0, got {node_count}")
        if min_stack_bb <= 0:
            raise ValueError(f"min_stack_bb must be > 0, got {min_stack_bb}")
        if max_stack_bb < min_stack_bb:
            raise ValueError(
                f"max_stack_bb ({max_stack_bb}) must be >= min_stack_bb ({min_stack_bb})"
            )
        if not (0 <= dealer_position < player_count):
            raise ValueError(
                f"dealer_position {dealer_position} out of range for "
                f"{player_count} players"
            )

        self._small_blind = small_blind
        self._big_blind = big_blind
        self._ante = ante
        self._hole_cards_per_player = hole_cards_per_player
        self._node_count = node_count
        self._min_stack_bb = min_stack_bb
        self._max_stack_bb = max_stack_bb
        self._game_name = game_name
        self._dealer_position = dealer_position

    def generate(self) -> BaseScenario:
        """
        Generate one preflop full-table scenario.

        Returns
        -------
        BaseScenario
            street_index=0, antes + blinds posted, no board cards dealt.
        """
        n = self._player_count
        dealer = self._dealer_position

        # --- starting stacks (chips, before antes/blinds) ---
        min_chips = max(1, int(self._min_stack_bb * self._big_blind))
        max_chips = max(min_chips, int(self._max_stack_bb * self._big_blind))
        stacks = [
            self._rng.randint(min_chips, max_chips) for _ in range(n)
        ]

        # --- deal hole cards ---
        used_mask = 0
        hand_masks: List[int] = []
        for _ in range(n):
            cards = self._deal_unique_cards(
                self._hole_cards_per_player, excluded_mask=used_mask
            )
            mask = self._cards_to_mask(cards)
            hand_masks.append(mask)
            used_mask |= mask

        # --- post antes (seat order), mirroring GameState._post_antes ---
        current_bets = [0] * n
        total_contrib = [0] * n
        is_all_in = [False] * n
        pot = 0

        if self._ante > 0:
            for i in range(n):
                amount = min(self._ante, stacks[i])
                stacks[i] -= amount
                current_bets[i] += amount
                total_contrib[i] += amount
                pot += amount
                if stacks[i] == 0:
                    is_all_in[i] = True

        # --- post blinds, mirroring GameState._post_blind exactly ---
        sb_seat = (dealer + 1) % n
        bb_seat = (dealer + 2) % n

        for seat, amount in ((sb_seat, self._small_blind), (bb_seat, self._big_blind)):
            blind = min(amount, stacks[seat])
            stacks[seat] -= blind
            current_bets[seat] += blind
            total_contrib[seat] += blind
            pot += blind
            if stacks[seat] == 0:
                is_all_in[seat] = True

        # --- build players ---
        players = tuple(
            PlayerScenario(
                stack=stacks[i],
                hand_mask=hand_masks[i],
                current_bet=current_bets[i],
                total_contribution=total_contrib[i],
                has_folded=False,
                is_all_in=is_all_in[i],
            )
            for i in range(n)
        )

        first_to_act = (dealer + 3) % n

        return BaseScenario(
            game_name=self._game_name,
            street_index=_PREFLOP_STREET_INDEX,
            pot=pot,
            dealer_position=dealer,
            current_player=first_to_act,
            bet_to_call=self._big_blind,
            min_raise=self._big_blind,
            players=players,
            node_cards=tuple(None for _ in range(self._node_count)),
            small_blind=self._small_blind,
            big_blind=self._big_blind,
            discard_pile=tuple(),
            raises_this_street=0,
            last_aggressor=None,
        )
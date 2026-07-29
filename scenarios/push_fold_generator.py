"""
push_fold_generator.py — Generator for preflop push-or-fold scenarios.

A push-fold scenario is the simplest complete poker decision:
- Preflop only; no board cards.
- One player (the "shover") has already moved all-in.
- One player (the "caller") must decide: call or fold.
- Short effective stacks (configurable range in big blinds).

This is the first training milestone because the strategy space is small
and well-understood, allowing easy verification that learning is occurring.

Scenario structure (2-player example)
--------------------------------------
    Seat 0: dealer / small blind / shover — already all-in
    Seat 1: big blind / caller          — must act (current_player)

    street_index = 0  (preflop)
    node_cards   = ()  (no board)
    bet_to_call  = shover stack (total amount to match)
"""

from __future__ import annotations

from scenarios.base_scenario import BaseScenario, PlayerScenario
from scenarios.scenario_generator import ScenarioGenerator


class PushFoldScenarioGenerator(ScenarioGenerator):
    """
    Generates preflop push-or-fold scenarios.

    The shover is always seat 0 and has already committed their entire
    stack. The caller is always seat 1 and faces a call-or-fold decision.
    All other seats (if player_count > 2) have already folded.

    Parameters
    ----------
    player_count : int
        Total number of seats. Minimum 2.
    small_blind : int
        Small blind size in chips.
    big_blind : int
        Big blind size in chips.
    min_stack_bb : float
        Minimum effective stack in big blinds (inclusive).
    max_stack_bb : float
        Maximum effective stack in big blinds (inclusive).
    starting_stack : int
        Starting stack for the caller in chips. The shover's stack is
        drawn uniformly from [min_stack_bb * bb, max_stack_bb * bb].
    """

    def __init__(
        self,
        player_count: int = 2,
        small_blind: int = 1,
        big_blind: int = 2,
        min_stack_bb: float = 2.0,
        max_stack_bb: float = 20.0,
        starting_stack: int = 200,
    ) -> None:
        super().__init__(player_count)

        if min_stack_bb <= 0:
            raise ValueError(f"min_stack_bb must be > 0, got {min_stack_bb}")
        if max_stack_bb < min_stack_bb:
            raise ValueError(
                f"max_stack_bb ({max_stack_bb}) must be >= min_stack_bb ({min_stack_bb})"
            )

        self._small_blind = small_blind
        self._big_blind = big_blind
        self._min_stack_bb = min_stack_bb
        self._max_stack_bb = max_stack_bb
        self._starting_stack = starting_stack

    def generate(self) -> BaseScenario:
        """
        Generate one push-fold scenario.

        Returns
        -------
        BaseScenario
            Preflop scenario where seat 0 has shoved and seat 1 must act.
        """
        # --- choose shover stack ---
        min_chips = max(1, int(self._min_stack_bb * self._big_blind))
        max_chips = int(self._max_stack_bb * self._big_blind)
        shove_amount = self._rng.randint(min_chips, max_chips)

        # Cap shove at caller's effective stack
        shove_amount = min(shove_amount, self._starting_stack)

        # --- deal hole cards ---
        shover_cards = self._deal_unique_cards(2, excluded_mask=0)
        shover_mask = self._cards_to_mask(shover_cards)

        caller_cards = self._deal_unique_cards(2, excluded_mask=shover_mask)
        caller_mask = self._cards_to_mask(caller_cards)

        # --- deal folded players' cards (not used strategically) ---
        used_mask = shover_mask | caller_mask
        folded_masks: list[int] = []
        for _ in range(self._player_count - 2):
            cards = self._deal_unique_cards(2, excluded_mask=used_mask)
            mask = self._cards_to_mask(cards)
            folded_masks.append(mask)
            used_mask |= mask

        # --- build player states ---
        # Seat 0: shover (already all-in, posted SB + shoved)
        shover = PlayerScenario(
            stack=0,
            hand_mask=shover_mask,
            current_bet=shove_amount,
            total_contribution=shove_amount,
            has_folded=False,
            is_all_in=True,
        )

        # Seat 1: caller (posted BB, faces decision)
        caller_bet_posted = min(self._big_blind, self._starting_stack)
        caller_remaining = self._starting_stack - caller_bet_posted
        caller = PlayerScenario(
            stack=caller_remaining,
            hand_mask=caller_mask,
            current_bet=caller_bet_posted,
            total_contribution=caller_bet_posted,
            has_folded=False,
            is_all_in=False,
        )

        # Extra seats (already folded preflop before the shove)
        folded_players = [
            PlayerScenario(
                stack=self._starting_stack,
                hand_mask=folded_masks[i],
                current_bet=0,
                total_contribution=0,
                has_folded=True,
                is_all_in=False,
            )
            for i in range(self._player_count - 2)
        ]

        players = tuple([shover, caller] + folded_players)

        # --- pot: everything already committed (shover's full push +
        # caller's posted big blind) — see base_scenario.py's pot
        # docstring: pot INCLUDES current_bet, matching the real
        # Engine's convention. The shover's current_bet (shove_amount)
        # already represents their entire commitment including their
        # own posted blind, so it is not added again here.
        pot = shove_amount + caller_bet_posted
        # In push-fold we treat the shove as the bet_to_call
        bet_to_call = shove_amount

        return BaseScenario(
            game_name="holdem",
            street_index=0,
            pot=pot,
            dealer_position=0,
            current_player=1,
            bet_to_call=bet_to_call,
            min_raise=shove_amount * 2,  # irrelevant; caller can only call/fold
            players=players,
            node_cards=tuple(),
            small_blind=self._small_blind,
            big_blind=self._big_blind,
            discard_pile=tuple(),
            raises_this_street=1,
            last_aggressor=0,
        )
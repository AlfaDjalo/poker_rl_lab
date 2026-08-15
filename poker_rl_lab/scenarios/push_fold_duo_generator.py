"""
scenarios/push_fold_duo_generator.py — Scenario generator for the
TWO-DECISION heads-up push-fold game.

This is a different scenario shape from scenarios/push_fold_generator.py.
That generator starts the episode *after* the shove has already happened
(seat 0 already all-in) because envs/push_fold_env.py only ever gives the
hero (seat 1) a single call/fold decision.

This generator instead starts BEFORE either decision has been made:

    Seat 0 (SB / button) — acts first. Legal actions: FOLD or ALL_IN
                            (push). No calling/betting sizes — push-fold
                            by definition only has two options here.
    Seat 1 (BB)           — acts only if seat 0 pushes. Legal actions:
                            FOLD or CALL.

Both seats start with the SAME short effective stack (the defining
feature of a push-fold spot — if stacks differed, the shorter stack's
shove size would need to be a genuine bet-sizing decision, which is out
of scope for this two-action game). Blinds are posted exactly like the
Engine's own heads-up convention noted elsewhere in this codebase (see
representation/position_encoder.py's docstring): in heads-up, the
button posts the small blind and acts first preflop.

Scenario structure
-------------------
    street_index = 0            (preflop)
    node_cards   = ()           (no board; push-fold never sees one)
    pot          = sb + bb      (both blinds are already posted before
                                  SB's decision, so pot must include
                                  them — see base_scenario.py's own
                                  docstring: "pot: total chips already
                                  committed... INCLUDING current_bet")
    dealer_position = 0         (seat 0 is the button)
    current_player  = 0         (SB acts first)
    bet_to_call     = big_blind - small_blind
                                 (irrelevant to SB's actual decision set
                                 — SB only ever sees {FOLD, ALL_IN} — but
                                 kept accurate/consistent for anything
                                 downstream that inspects BettingObs)
    min_raise       = big_blind (unused; SB has no sizing decision)

Consumer
--------
scenarios.push_fold_duo_generator -> envs.push_fold_duo_env.PushFoldDuoEnv
See that module's docstring for how the two sequential decisions are
actually played out.
"""

from __future__ import annotations

from poker_rl_lab.scenarios.base_scenario import BaseScenario, PlayerScenario
from poker_rl_lab.scenarios.scenario_generator import ScenarioGenerator

SB_SEAT: int = 0   # button / small blind; acts first; push-or-fold
BB_SEAT: int = 1   # big blind; acts second, only if SB pushes; call-or-fold


class PushFoldDuoScenarioGenerator(ScenarioGenerator):
    """
    Generates heads-up push-fold scenarios at the SB's decision point,
    before any shove has happened.

    Parameters
    ----------
    small_blind : int
        Small blind size in chips.
    big_blind : int
        Big blind size in chips.
    min_stack_bb : float
        Minimum effective stack in big blinds (inclusive). Both players
        always start with the SAME stack — see module docstring.
    max_stack_bb : float
        Maximum effective stack in big blinds (inclusive).
    """

    def __init__(
        self,
        small_blind: int = 1,
        big_blind: int = 2,
        min_stack_bb: float = 4.0,
        max_stack_bb: float = 20.0,
    ) -> None:
        super().__init__(player_count=2)

        if small_blind <= 0:
            raise ValueError(f"small_blind must be > 0, got {small_blind}")
        if big_blind <= small_blind:
            raise ValueError(
                f"big_blind ({big_blind}) must be > small_blind ({small_blind})"
            )
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

    def generate(self) -> BaseScenario:
        """
        Generate one heads-up push-fold scenario, SB to act.

        Returns
        -------
        BaseScenario
            street_index=0, both blinds posted, no shove yet, seat 0
            (SB) is current_player with a FOLD/ALL_IN decision pending.
        """
        min_chips = max(
            self._big_blind + 1, int(self._min_stack_bb * self._big_blind)
        )
        max_chips = max(min_chips, int(self._max_stack_bb * self._big_blind))
        starting_stack = self._rng.randint(min_chips, max_chips)

        used_mask = 0
        sb_cards = self._deal_unique_cards(2, excluded_mask=used_mask)
        sb_mask = self._cards_to_mask(sb_cards)
        used_mask |= sb_mask

        bb_cards = self._deal_unique_cards(2, excluded_mask=used_mask)
        bb_mask = self._cards_to_mask(bb_cards)

        sb_posted = min(self._small_blind, starting_stack)
        sb = PlayerScenario(
            stack=starting_stack - sb_posted,
            hand_mask=sb_mask,
            current_bet=sb_posted,
            total_contribution=sb_posted,
            has_folded=False,
            is_all_in=(starting_stack - sb_posted == 0),
        )

        bb_posted = min(self._big_blind, starting_stack)
        bb = PlayerScenario(
            stack=starting_stack - bb_posted,
            hand_mask=bb_mask,
            current_bet=bb_posted,
            total_contribution=bb_posted,
            has_folded=False,
            is_all_in=(starting_stack - bb_posted == 0),
        )

        return BaseScenario(
            game_name="holdem",
            street_index=0,
            # Both blinds are already posted (live in each player's
            # current_bet) before SB's decision — pot must include them,
            # matching base_scenario.py's inclusive pot convention.
            pot=sb_posted + bb_posted,
            dealer_position=SB_SEAT,
            current_player=SB_SEAT,
            bet_to_call=bb_posted - sb_posted,
            min_raise=self._big_blind,
            players=(sb, bb),
            node_cards=tuple(),
            small_blind=self._small_blind,
            big_blind=self._big_blind,
            discard_pile=tuple(),
            raises_this_street=0,
            last_aggressor=None,
        )
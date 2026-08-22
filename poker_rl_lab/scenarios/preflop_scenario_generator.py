"""
scenarios/preflop_scenario_generator.py — Produces preflop BaseScenario
instances (street_index=0, blinds posted, hole cards dealt, no board
cards known yet) for envs.multi_street_env.MultiStreetEnv.reset().

Why this is the missing piece for "earlier street betting"
------------------------------------------------------------
envs.multi_street_env.MultiStreetEnv (Phase D) drives the real Engine
GameState across arbitrary streets/re-raises, but it only ever
consumes whatever BaseScenario it's handed — it has no scenario
generator of its own, the same relationship
envs.generic_tree_env.GenericTreeEnv has to
scenarios.generic_tree_scenario_generator.GenericTreeScenarioGenerator.
Every existing scenario generator in this codebase (push_fold_generator,
GenericTreeScenarioGenerator) either deals no board at all or deals the
FULL board up front — none of them produce a genuine "action starts
before any board card is known" position. This module does exactly
that: street_index=0, node_cards all None, blinds posted per real
dealer rotation.

Heads-up first-to-act discrepancy (see module header note above; not
introduced here)
------------------------------------------------------------------------
poker_engine/state/poker_state.py's _first_to_act_preflop() is
`(dealer_position + 3) % n` — correct for n >= 3 (three seats past the
button lands on UTG), but for n == 2 this evaluates to
`(dealer_position + 1) % 2`, i.e. the BIG BLIND, which is backwards:
standard heads-up hold'em has the button/small-blind act FIRST
preflop, not last. representation/position_encoder.py's own docstring
already documents this exact discrepancy for _post_blinds' heads-up
handling. Since MultiStreetEnv takes `current_player` directly from the
BaseScenario it's given (never recomputes it via poker_state.py), this
generator computes first-to-act itself, WITH the n==2 special case,
rather than reproducing poker_state.py's bug. If poker_state.py is
ever fixed to special-case n==2 correctly, this generator's
_first_to_act_preflop() should be updated to match (or better, both
should delegate to one shared helper — out of scope for this file).

Blind seats
-----------
Same n==2 convention already established by
representation/position_encoder.py's _compute_position_features (button
posts the small blind heads-up) and mirrored by
scenarios/generic_tree_scenario_generator.py's own _size_pot() "blinds"
mode note. For n >= 3: SB = dealer+1, BB = dealer+2, standard.

Dependencies
------------
    poker_engine.games.loader.load_game (for hole_cards / node_count —
    this generator is game-agnostic; pass any game_name whose
    GameDefinition it should deal for). scenarios.scenario_generator.ScenarioGenerator
    (base class — reuses its seeded card-dealing helpers).
    scenarios.base_scenario.BaseScenario, PlayerScenario.
"""

from __future__ import annotations

from typing import List, Tuple

from poker_rl_lab.scenarios.base_scenario import BaseScenario, PlayerScenario
from poker_rl_lab.scenarios.scenario_generator import ScenarioGenerator


def _sb_bb_seats(dealer: int, n: int) -> Tuple[int, int]:
    """(small_blind_seat, big_blind_seat) for a table of size n with the
    button at `dealer`. See module docstring's "Blind seats" section."""
    if n == 2:
        return dealer, (dealer + 1) % n
    return (dealer + 1) % n, (dealer + 2) % n


def _first_to_act_preflop(dealer: int, n: int) -> int:
    """Seat index that acts first preflop. See module docstring's
    "Heads-up first-to-act discrepancy" section for why n==2 is
    special-cased here rather than reusing poker_state.py's formula."""
    if n == 2:
        return dealer  # button/small blind acts first heads-up preflop
    return (dealer + 3) % n  # UTG: three seats past the button


class PreflopScenarioGenerator(ScenarioGenerator):
    """
    Deals a fresh preflop position: blinds posted per real dealer
    rotation, every seat's hole cards dealt, no board cards known.

    Parameters
    ----------
    player_count : int
        Number of seats. >= 2.
    game_name : str
        Loaded via poker_engine.games.loader.load_game() at construction
        for `hole_cards` (how many cards to deal each seat) and
        `node_count` (how many board node slots the returned scenario's
        node_cards tuple should have, all None). The GameDefinition's
        own small_blind/big_blind are NOT used — pass them explicitly
        below, matching every other generator in this codebase (e.g.
        scenarios.generic_tree_scenario_generator.py takes its own
        DuoGameConfig.small_blind/big_blind rather than reading the
        Engine game's).
    small_blind, big_blind : int
        Blind sizes for every generated scenario.
    stack_mode : str
        "fixed_bb" (every seat gets `stack_bb` big blinds, every
        episode) or "randomized_bb" (every seat gets the SAME randomly
        drawn stack in [min_stack_bb, max_stack_bb] big blinds that
        episode) — same two modes and "every seat gets the SAME stack"
        convention as scenarios.duo_game_config.StackConfig, for
        consistency across the codebase's generators.
    stack_bb : float
        Used when stack_mode == "fixed_bb". Default 100.0.
    min_stack_bb, max_stack_bb : float
        Used when stack_mode == "randomized_bb".
    randomize_dealer : bool
        If True (default), the button is drawn uniformly at random each
        episode. If False, the button rotates by one seat each call to
        generate() (starting at seat 0) — useful for evaluation runs
        that want deterministic button coverage across a fixed number
        of hands rather than random sampling.
    """

    def __init__(
        self,
        player_count: int,
        game_name: str,
        small_blind: int = 1,
        big_blind: int = 2,
        stack_mode: str = "fixed_bb",
        stack_bb: float = 100.0,
        min_stack_bb: float = 20.0,
        max_stack_bb: float = 100.0,
        randomize_dealer: bool = True,
    ) -> None:
        super().__init__(player_count)

        if stack_mode not in ("fixed_bb", "randomized_bb"):
            raise ValueError(
                f"stack_mode must be 'fixed_bb' or 'randomized_bb', got {stack_mode!r}"
            )
        if small_blind <= 0:
            raise ValueError(f"small_blind must be > 0, got {small_blind}")
        if big_blind <= small_blind:
            raise ValueError(
                f"big_blind ({big_blind}) must be > small_blind ({small_blind})"
            )

        from poker_engine.games.loader import load_game

        self.game_def, self.rules = load_game(game_name)
        self.game_name = game_name
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.stack_mode = stack_mode
        self.stack_bb = stack_bb
        self.min_stack_bb = min_stack_bb
        self.max_stack_bb = max_stack_bb
        self.randomize_dealer = randomize_dealer

        self._next_dealer = 0  # only advanced when randomize_dealer is False

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------

    def _size_stack(self) -> int:
        if self.stack_mode == "fixed_bb":
            return max(1, int(round(self.stack_bb * self.big_blind)))
        min_chips = max(1, int(self.min_stack_bb * self.big_blind))
        max_chips = max(min_chips, int(self.max_stack_bb * self.big_blind))
        return self._rng.randint(min_chips, max_chips)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self) -> BaseScenario:
        n = self.player_count

        if self.randomize_dealer:
            dealer = self._rng.randrange(n)
        else:
            dealer = self._next_dealer
            self._next_dealer = (self._next_dealer + 1) % n

        stack = self._size_stack()
        starting_stacks = [stack] * n

        sb_seat, bb_seat = _sb_bb_seats(dealer, n)
        sb_amount = min(self.small_blind, starting_stacks[sb_seat])
        bb_amount = min(self.big_blind, starting_stacks[bb_seat])

        current_bets = [0] * n
        current_bets[sb_seat] = sb_amount
        current_bets[bb_seat] = bb_amount

        stacks_after_blinds = list(starting_stacks)
        stacks_after_blinds[sb_seat] -= sb_amount
        stacks_after_blinds[bb_seat] -= bb_amount

        pot = sb_amount + bb_amount  # see BaseScenario.pot's own docstring:
        # includes every player's current-street current_bet already.

        used_mask = 0
        players: List[PlayerScenario] = []
        for i in range(n):
            cards = self._deal_unique_cards(
                self.game_def.hole_cards, excluded_mask=used_mask
            )
            hand_mask = self._cards_to_mask(cards)
            used_mask |= hand_mask

            players.append(
                PlayerScenario(
                    stack=stacks_after_blinds[i],
                    hand_mask=hand_mask,
                    current_bet=current_bets[i],
                    total_contribution=current_bets[i],
                    has_folded=False,
                    is_all_in=(stacks_after_blinds[i] == 0),
                )
            )

        node_cards = tuple(None for _ in range(self.game_def.node_count))

        return BaseScenario(
            game_name=self.game_name,
            street_index=0,
            pot=pot,
            dealer_position=dealer,
            current_player=_first_to_act_preflop(dealer, n),
            bet_to_call=bb_amount,
            min_raise=self.big_blind,
            players=tuple(players),
            node_cards=node_cards,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            discard_pile=(),
            raises_this_street=0,
            last_aggressor=None,
        )
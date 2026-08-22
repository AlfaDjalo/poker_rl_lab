"""
envs/generic_tree_showdown_adapter.py — Delegates GenericTreeEnv's
showdown resolution to the REAL poker_engine.showdown.ShowdownResolver,
instead of GenericTreeEnv re-implementing point/split-pot/hi-lo logic
itself.

Why this exists
----------------
GenericTreeEnv._resolve_showdown() (envs/generic_tree_env.py) currently
only ever compares a single ScoreType.HIGH / ShowdownType.HOLDEM score
per seat — correct for river_duo / push_fold_duo, but WRONG for any
config whose game_name points at a multi-point ruleset (a 3-point
high/high/hand game, triple board, a hi/lo game, Drawmaha's Omaha
showdown, ...). Getting those right means honouring rules.points
(score_type per point, node_sets, showdown_type_override,
payout_type: split_pot vs points, scoop_from) — exactly what
poker_engine.showdown.ShowdownResolver already does correctly for live
play. Reimplementing that here would violate this codebase's own rule
that the RL Lab must never duplicate poker/betting logic (see
representation/position_encoder.py's module docstring).

This module is the bridge: it builds a minimal, throwaway
GameState-shaped object from GenericTreeEnv's own per-seat arrays
(hole masks, committed chips, dealt board) and hands it to the real
ShowdownResolver, then reads back `.payouts`. GenericTreeEnv never
constructs pot layers or compares scores itself for these configs —
ShowdownResolver.resolve() already builds side pots via
game_state.build_side_pots() and does the real point-by-point payout
math.

Scope
-----
Only for DEAL_BOARD configs (river_duo-shaped and beyond). Boardless
configs (push_fold_duo: deal.deal_board=False, node_count=0) keep using
GenericTreeEnv's existing single ScoreType.HIGH/ShowdownType.HOLDEM
fast path — see that module's _resolve_showdown for why: push-fold's
`game_name="holdem"` Engine YAML has its own board node_count (5) that
does NOT match push_fold_duo.yaml's own deal.node_count (0); the two
configs are only using `game_name` for label purposes, not board
layout, so routing that case through real rules.points/node_sets would
be indexing into a board that was never sized to match. Every
deal_board=True config authored so far (river_duo, double_board_plo_river,
and any new config built on this pattern) deliberately sizes
deal.node_count to match its game_name's own board_layout.nodes, so no
such mismatch exists there.

Monte Carlo runout
-------------------
Kept for future-proofing (a deal_board=True config with node slots
still None at showdown time), but is a no-op in practice today: every
existing DEAL_BOARD config's scenario generator deals ALL
deal.node_count cards up front at reset() (see
generic_tree_scenario_generator.py's generate() — `reveal_at_start`
only controls what's OBSERVABLE to the policy progressively, not what
is actually dealt), so self._node_cards_full has no None entries by
the time a showdown terminal is reached.

Dependencies
------------
    poker_engine.games.loader.load_game, poker_engine.showdown.showdown_resolver.ShowdownResolver.
    No poker_eval import here directly — score_type/showdown_type
    values flow through opaquely from `rules`, exactly like
    ShowdownResolver itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Minimal GameState-shaped proxy (duck-typed against what
# ShowdownResolver.resolve() actually reads/writes)
# ---------------------------------------------------------------------------

@dataclass
class _ProxyPlayer:
    """Duck-types poker_engine.state.player_state.PlayerState for exactly
    the fields ShowdownResolver touches: has_folded, hand_mask,
    total_contribution (read by build_side_pots), stack (written by
    resolve() — result discarded, this object is throwaway)."""
    stack: int
    hand_mask: int
    total_contribution: int
    has_folded: bool = False


class _ProxyGameState:
    """
    Duck-types poker_engine.state.game_state.GameState for exactly the
    surface ShowdownResolver.resolve() uses: .players, .node_cards,
    .build_side_pots().

    build_side_pots() is copied verbatim from GameState (see
    poker_engine/state/game_state.py) rather than imported, since
    GameState.build_side_pots() is an instance method with no
    dependency on any OTHER GameState field — safe to duplicate this
    one pure function rather than pull in the full GameState machinery
    (turn order, apply_action, betting_type) this adapter never needs.
    """

    __slots__ = ("players", "node_cards", "pot_carry")

    def __init__(
        self,
        players: List[_ProxyPlayer],
        node_cards: List[Optional[int]],
        pot_carry: int = 0,
    ) -> None:
        self.players = players
        self.node_cards = node_cards
        # Dead money with no per-seat attribution (GenericTreeEnv's own
        # self._pot_carry) — injected directly into build_side_pots()'s
        # own layering below, rather than patched onto resolver.resolve()'s
        # payouts afterward. This is the SAME convention
        # GenericTreeEnv._resolve_showdown already uses for its own
        # (now-superseded) side-pot construction: fold pot_carry into
        # the TOP (largest) contribution layer, since that's the layer
        # every seat still holding chips at showdown is eligible for.
        self.pot_carry = pot_carry

    def build_side_pots(self) -> List[Tuple[int, set]]:
        contributions = [
            (i, p.total_contribution)
            for i, p in enumerate(self.players)
            if p.total_contribution > 0
        ]

        if not contributions:
            # Every seat committed 0 this hand (e.g. a "randomized_bb"
            # pot config where the whole pot is dead money) — the
            # entire pot is pot_carry, eligible to every seat.
            if self.pot_carry:
                return [(self.pot_carry, set(range(len(self.players))))]
            return []

        contributions.sort(key=lambda x: x[1])

        side_pots: List[Tuple[int, set]] = []
        prev = 0
        remaining_players = set(i for i, _ in contributions)

        for i, amount in contributions:
            if amount > prev:
                layer = amount - prev
                pot_amount = layer * len(remaining_players)
                side_pots.append((pot_amount, remaining_players.copy()))
                prev = amount
            remaining_players.remove(i)

        if self.pot_carry and side_pots:
            top_amount, top_eligible = side_pots[-1]
            side_pots[-1] = (top_amount + self.pot_carry, top_eligible)

        return side_pots


# ---------------------------------------------------------------------------
# Game-rules cache
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _load_game_rules(game_name: str):
    """Cached poker_engine.games.loader.load_game(game_name) -> (game_def, rules).
    Cached process-wide since a config's game_name never changes across
    an env's lifetime and load_game() re-parses YAML on every call."""
    from poker_engine.games.loader import load_game

    return load_game(game_name)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def resolve_generic_tree_showdown(
    game_name: str,
    hole_masks: Sequence[int],
    committed: Sequence[int],
    pot_carry: int,
    node_cards_full: Sequence[Optional[int]],
    node_count: int,
    scoring_engine,
    num_showdown_samples: int,
    runout_rng: random.Random,
) -> Dict[int, int]:
    """
    Resolve one N-seat showdown using the REAL rules.points for
    `game_name`, via poker_engine.showdown.ShowdownResolver.

    Parameters
    ----------
    game_name : str
        Same as DuoGameConfig.game_name — must reference an Engine game
        YAML whose board_layout.nodes matches `node_count` exactly (see
        module docstring's "Scope" section).
    hole_masks : Sequence[int]
        One 52-bit hand bitmask per seat.
    committed : Sequence[int]
        Total chips each seat has committed this hand (GenericTreeEnv's
        self._committed) — used for side-pot eligibility exactly like
        PlayerState.total_contribution.
    pot_carry : int
        Dead-money pot with no per-seat attribution (GenericTreeEnv's
        self._pot_carry) — folded into the largest side-pot layer
        (mirrors GenericTreeEnv._resolve_showdown's own existing
        pot_carry handling).
    node_cards_full : Sequence[Optional[int]]
        The FULL board known so far, length node_count. None entries
        are completed via `runout_rng` (see module docstring — a no-op
        for every config authored so far).
    node_count : int
        Must equal len(node_cards_full) and the referenced game's
        board_layout.nodes.
    scoring_engine
        Duck-typed CppScoringEngine / MockScoringEngine — same object
        GenericTreeEnv already owns.
    num_showdown_samples : int
        Runouts to average when node_cards_full has any None entry.
        Degenerates to exactly 1 pass when the board is already
        complete (see module docstring).
    runout_rng : random.Random
        GenericTreeEnv's own seeded RNG (self._runout_rng) — reused so
        this adapter introduces no new source of nondeterminism.

    Returns
    -------
    Dict[int, int]
        Seat index -> chips won this showdown (0 for seats that won
        nothing). Same contract GenericTreeEnv._resolve_showdown's
        internal payout accounting already expects.
    """
    from poker_engine.showdown.showdown_resolver import ShowdownResolver

    if len(node_cards_full) != node_count:
        raise ValueError(
            f"node_cards_full has length {len(node_cards_full)}, expected "
            f"node_count {node_count}"
        )

    game_def, rules = _load_game_rules(game_name)
    resolver = ShowdownResolver(scoring_engine, rules)

    n = len(hole_masks)
    missing = [i for i, c in enumerate(node_cards_full) if c is None]
    num_samples = num_showdown_samples if missing else 1

    accumulated: List[float] = [0.0] * n

    for _ in range(num_samples):
        board = list(node_cards_full)
        if missing:
            used_mask = 0
            for m in hole_masks:
                used_mask |= m
            for c in board:
                if c is not None:
                    used_mask |= 1 << c
            available = [c for c in range(52) if not (used_mask >> c) & 1]
            if len(available) < len(missing):
                raise RuntimeError(
                    f"Not enough cards left to complete the board: "
                    f"need {len(missing)}, have {len(available)}."
                )
            runout = runout_rng.sample(available, len(missing))
            for idx, card in zip(missing, runout):
                board[idx] = card

        proxy_players = [
            _ProxyPlayer(
                stack=0,
                hand_mask=hole_masks[i],
                total_contribution=committed[i],
                has_folded=False,
            )
            for i in range(n)
        ]

        proxy_state = _ProxyGameState(
            players=proxy_players, node_cards=board, pot_carry=pot_carry
        )
        result = resolver.resolve(proxy_state)

        for seat, amount in result.payouts.items():
            accumulated[seat] += amount

    return {i: accumulated[i] / num_samples for i in range(n)}


# ---------------------------------------------------------------------------
# Integration note (envs/generic_tree_env.py) — not executed here.
# ---------------------------------------------------------------------------
#
# GenericTreeEnv._resolve_terminal()'s showdown branch becomes:
#
#     else:
#         if self._use_real_rules_showdown:
#             payouts = resolve_generic_tree_showdown(
#                 game_name=self.config.game_name,
#                 hole_masks=self._hole_masks,
#                 committed=self._committed,
#                 pot_carry=self._pot_carry,
#                 node_cards_full=self._node_cards_full,
#                 node_count=self.config.deal.node_count,
#                 scoring_engine=self._scoring_engine,
#                 num_showdown_samples=self._num_showdown_samples,
#                 runout_rng=self._runout_rng,
#             )
#             final_stack = [
#                 self._stack[i] + payouts.get(i, 0) for i in range(len(self.config.seats))
#             ]
#             rewards = {
#                 seat: float(final_stack[seat] - self._starting_stack[seat])
#                 for seat in range(len(self.config.seats))
#             }
#         else:
#             rewards = self._resolve_showdown()   # existing single-point fast path
#         info = {"outcome": "showdown", "terminal": terminal_id}
#
# self._resolve_showdown() (the old single-ScoreType.HIGH/ShowdownType.HOLDEM
# method), _complete_board_for_showdown(), and the score_type/showdown_type
# constructor args are all UNCHANGED and kept exactly as-is for the
# deal_board=False (push-fold-duo) path — see this module's "Scope" section.
# ---------------------------------------------------------------------------
"""
envs/generic_tree_env.py — N-seat, config-driven executor for
DuoGameConfig (scenarios/duo_game_config.py) fixed-action-tree
mini-games.

This is the "GenericTreeEnv" scenarios/duo_game_config.py's own
docstring already refers to as scenarios/generic_tree_env.py's future
home. It interprets ANY validated DuoGameConfig — push-fold-duo,
river-duo, double-board-plo-river, or a new N-seat YAML under
scenarios/duo_games/ — without per-game Python code.

Showdown resolution — POINTS-AWARE (PATCHED)
--------------------------------------------------------------------------
Earlier versions of this file resolved every showdown by flattening
every dealt board node card into ONE board mask and calling
scoring_engine.evaluate() exactly once, with a single game-wide
score_type/showdown_type. That is correct for a single-board,
single-point game (push_fold_duo, river_duo — holdem's rules declare
exactly one HIGH point over one node_set) but silently WRONG for any
split_pot/multi-point game such as double_board_plo: it merged both
independent 5-card boards' cards into one pseudo-10-card board and
evaluated a single Omaha hand off it, instead of splitting the pot
evenly across the two independent boards the way
poker_engine.showdown.showdown_resolver.ShowdownResolver._resolve_split_pot
does (`num_components = number of points, not point x board pairs`).

This version loads the real `rules` object for `config.game_name` (via
the same poker_engine.games.loader.load_game() every other real-Engine
entry point in this codebase uses) and resolves each side pot PER
POINT, mirroring ShowdownResolver's own split_pot algorithm:

    - Each point's board mask is built from ITS OWN node_sets (not the
      env's flattened self._board_mask).
    - A point spanning multiple node_sets uses the BEST score across
      them per player (best-of), matching
      ShowdownResolver._best_scores_for_point.
    - Each side pot's chip amount is divided evenly across
      len(rules.points) components (remainder to the first components),
      matching ShowdownResolver._resolve_split_pot's own remainder
      handling.
    - Each point uses ITS OWN score_type and
      (showdown_type_override or rules.showdown_type) — not a single
      env-wide constant — matching
      ShowdownResolver._evaluate_all_points.

Both rules.payout_type values are supported:
    "split_pot"  each point independently wins an even share of the
                 pot (double_board_plo: two boards, half the pot each).
                 Mirrors ShowdownResolver._resolve_split_pot.
    "points"     every point is tallied (1 / len(winners) per point a
                 seat wins), and whoever has the most points takes the
                 ENTIRE pot layer (tie = chop). Mirrors
                 ShowdownResolver._resolve_points_game. This is what
                 "holdem" (push_fold_duo, river_duo) actually declares
                 — a single-point tally degenerates to exactly the same
                 winner(s) split_pot would have produced with one
                 component, so this was silently correct before this
                 patch only because holdem has just one point; it must
                 be implemented for real once any point is added.
Any other payout_type raises NotImplementedError loudly rather than
silently mis-paying.

Monte Carlo showdown averaging (num_showdown_samples) — UNCHANGED IN SPIRIT
-----------------------------------------------------------------------------
Still averages `num_showdown_samples` independent runouts when ANY
point's board is incomplete (see _showdown_boards_incomplete()), and
degenerates to exactly 1 sample when every point's board is already
fully dealt (river_duo, double_board_plo_river). Runouts across
DIFFERENT points within the same sample never share a card — each
point's completion reserves its drawn cards before the next point's
completion runs (see _complete_point_board_for_showdown's
`reserved_mask` parameter) — since two independent physical boards can
never legitimately share a card.

Why side-pot layering even for a "no re-raise" tree
--------------------------------------------------------
Every DuoGameConfig built so far uses equal starting stacks, so in
practice every showdown has equal commitments and side pots degenerate
to one pot. But nothing in the schema *forces* equal stacks, so
_resolve_showdown() still builds real side-pot layers.

Why a showdown terminal never has a folded seat to exclude
------------------------------------------------------------
In this schema every branch is authored explicitly: a path that
includes a FOLD action leads to a "fold" terminal, never a "showdown"
terminal. So _resolve_showdown() can safely assume every seat is still
live.

Dependencies
------------
    scenarios.duo_game_config, envs.observations, actions.abstract_action
    (pure Python — no poker_eval needed to import this file). poker_eval
    / mock_poker_eval and poker_engine.games.loader are only touched at
    construction time (loading rules, picking default score/showdown
    types) — same shim pattern as every other entry point in this
    codebase.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

try:
    import poker_rl_lab.poker_eval as poker_eval  # noqa: F401  (prefer the real extension if present)
except ImportError:
    import poker_rl_lab.mock_poker_eval as mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)
    import poker_eval  # noqa: F401  (now resolves to the shim)

from poker_rl_lab.scenarios.duo_game_config import DuoGameConfig
from poker_rl_lab.envs.observations import (
    BettingObs,
    BoardObs,
    HistoryEventObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)
from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS
from poker_rl_lab.scenarios.duo_game_config import DuoGameConfig, DealNodeConfig
from poker_rl_lab.envs.generic_tree_showdown_adapter import resolve_generic_tree_showdown


def _mask_to_cards(mask: int) -> Tuple[int, ...]:
    return tuple(i for i in range(52) if (mask >> i) & 1)


def _default_scoring_engine():
    """Prefer the real Engine scoring path; fall back to the documented
    mock — identical pattern to engine_scoring_evaluator.py's own
    _default_scoring_engine()."""
    try:
        from poker_engine.scoring.scoring_engine import CppScoringEngine

        return CppScoringEngine()
    except ImportError:
        from poker_engine.scoring.mock_scoring_engine import MockScoringEngine

        return MockScoringEngine()


def _load_rules(game_name: str):
    """
    Load the real GameRules for `game_name` — same loader
    observation_adapter.ObservationAdapter._get_game() uses. This is
    what makes _resolve_showdown() points-aware (see module docstring):
    GenericTreeEnv previously never had access to rules.points at all.
    """
    from poker_engine.games.loader import load_game

    _, rules = load_game(game_name)
    return rules


# ---------------------------------------------------------------------------
# Scenario input shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenericTreeScenario:
    """
    Initial conditions for one GenericTreeEnv episode.

    Attributes
    ----------
    config_name : str
        Must equal the DuoGameConfig.name this scenario targets.
    hole_masks : Tuple[int, ...]
        One 52-bit hand bitmask per seat, indexed by seat index.
    board_mask : int
        0 if config.deal.deal_board is False (no board this game).
    node_cards : Tuple[Optional[int], ...]
        Board cards by node index, for StructuredObservation.board —
        empty tuple if config.deal.deal_board is False.
    starting_stacks : Tuple[int, ...]
        Each seat's stack BEFORE any of `posted` is subtracted, one per
        seat, indexed by seat index.
    posted : Tuple[int, ...]
        Chips each seat has already committed before the first decision.
    dealer_position : int
        Kept for parity with BaseScenario; GenericTreeEnv actually
        derives dealer_position from config.dealer_seat at every
        observation instead (see _build_obs).
    """
    config_name: str
    hole_masks: Tuple[int, ...]
    board_mask: int = 0
    node_cards: Tuple[Optional[int], ...] = ()
    starting_stacks: Tuple[int, ...] = ()
    posted: Tuple[int, ...] = ()
    pot: int = 0
    dealer_position: int = 0

    def __post_init__(self) -> None:
        n = len(self.hole_masks)
        if len(self.starting_stacks) != n:
            raise ValueError(
                f"starting_stacks length ({len(self.starting_stacks)}) must "
                f"match hole_masks length ({n})"
            )
        if len(self.posted) != n:
            raise ValueError(
                f"posted length ({len(self.posted)}) must match "
                f"hole_masks length ({n})"
            )
        for i in range(n):
            if self.posted[i] > self.starting_stacks[i]:
                raise ValueError(
                    f"seat {i}: posted ({self.posted[i]}) exceeds "
                    f"starting_stacks ({self.starting_stacks[i]})"
                )


@dataclass(frozen=True)
class TreeStepResult:
    """
    Result of one GenericTreeEnv.step() call.

    Attributes
    ----------
    obs : StructuredObservation, optional
        Observation for the NEXT acting seat. None iff done is True.
    rewards : Dict[int, float]
        Chip change per seat index. Empty ({}) on any intermediate
        (non-terminal) step; has one entry per seat on the terminal
        step.
    done : bool
    info : dict
        Diagnostic keys. Intermediate steps: {"node": <next node id>}.
        Terminal steps: {"outcome": "fold"|"showdown", "terminal": id,
        ...}.
    """
    obs: Optional[StructuredObservation]
    rewards: Dict[int, float]
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GenericTreeEnv:
    """
    Executes any validated DuoGameConfig's fixed action tree for N seats.

    Parameters
    ----------
    config : DuoGameConfig
    scoring_engine : optional
        Defaults to CppScoringEngine, falling back to MockScoringEngine.
    score_type, showdown_type : optional
        FALLBACK values only — see module docstring's "Showdown
        resolution" section. Each point now supplies its own
        score_type; showdown_type is used only when a point doesn't
        declare showdown_type_override, and itself defaults to
        rules.showdown_type (loaded from config.game_name) rather than
        a hardcoded HOLDEM constant, unless explicitly overridden here.
    num_showdown_samples : int
        Number of independent board runouts averaged per showdown when
        ANY point's board is incomplete at terminal time. Default 30.
        Degenerates to exactly 1 sample when every point's board is
        already fully dealt, and has no effect on outcome (only on
        wasted repeated calls) when scoring_engine is MockScoringEngine.
    """

    #: Bet-type actions this schema's fixed (non-continuous-sizing)
    #: trees actually use.
    _SUPPORTED_BET_TYPES = (ActionType.ALL_IN, ActionType.BET_100)

    def __init__(
        self,
        config: DuoGameConfig,
        scoring_engine=None,
        score_type=None,
        showdown_type=None,
        runout_seed=None,
        num_showdown_samples: int = 30,
    ) -> None:
        self.config = config
        self._scoring_engine = scoring_engine or _default_scoring_engine()

        # -- Points-aware showdown setup (NEW) --------------------------
        self._rules = _load_rules(config.game_name)
        self._points = self._rules.points
        self._payout_type = self._rules.payout_type
        if not self._points:
            raise ValueError(
                f"game {config.game_name!r} declares zero showdown points "
                f"— GenericTreeEnv cannot resolve a showdown for it."
            )

        self._score_type = (
            score_type if score_type is not None else poker_eval.ScoreType.HIGH
        )
        self._showdown_type = (
            showdown_type if showdown_type is not None else self._rules.showdown_type
        )
        if num_showdown_samples < 1:
            raise ValueError(
                f"num_showdown_samples must be >= 1, got {num_showdown_samples}"
            )
        self._num_showdown_samples = int(num_showdown_samples)

        n = len(config.seats)
        self._scenario: Optional[GenericTreeScenario] = None
        self._done: bool = True
        self.current_seat: int = -1
        self._current_node: str = ""

        self._hole_masks: List[int] = [0] * n
        self._board_mask: int = 0
        self._node_cards: Tuple[Optional[int], ...] = ()
        self._starting_stack: List[int] = [0] * n
        self._committed: List[int] = [0] * n   # total chips committed so far
        self._pot_carry: int = 0
        self._stack: List[int] = [0] * n        # remaining (uncommitted) stack
        self._is_all_in: List[bool] = [False] * n
        self._history: List[HistoryEventObs] = []
        self._node_cards_full: Tuple[Optional[int], ...] = ()
        self._revealed_nodes: set = set()

        self._runout_rng = random.Random(runout_seed)
        self._use_real_rules_showdown = self.config.deal.deal_board

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, scenario: GenericTreeScenario) -> StructuredObservation:
        n = len(self.config.seats)
        if scenario.config_name != self.config.name:
            raise ValueError(
                f"scenario.config_name {scenario.config_name!r} does not "
                f"match this env's config {self.config.name!r}"
            )
        if len(scenario.hole_masks) != n:
            raise ValueError(
                f"scenario has {len(scenario.hole_masks)} seats' hole "
                f"cards, config {self.config.name!r} has {n} seats"
            )

        self._scenario = scenario
        self._done = False
        self._current_node = self.config.start_node
        self.current_seat = self.config.seat_index(self.config.first_seat_id)
        self._history = []

        self._hole_masks = list(scenario.hole_masks)
        self._board_mask = scenario.board_mask
        self._node_cards = scenario.node_cards

        self._node_cards_full = scenario.node_cards
        self._revealed_nodes = (
            set(range(len(scenario.node_cards)))
            if self.config.deal.reveal_at_start
            else set()
        )
        self._starting_stack = list(scenario.starting_stacks)
        self._committed = list(scenario.posted)
        self._pot_carry = scenario.pot
        self._stack = [
            scenario.starting_stacks[i] - scenario.posted[i] for i in range(n)
        ]
        self._is_all_in = [self._stack[i] == 0 for i in range(n)]

        resolved = self._advance_through_deal_nodes(self.config.start_node)
        if self.config.is_terminal(resolved):
            raise RuntimeError(
                f"config {self.config.name!r}: start_node "
                f"{self.config.start_node!r} leads directly to terminal "
                f"{resolved!r} via deal nodes with no decision in between "
                f"— degenerate tree."
            )
        self._current_node = resolved
        decision_node = self.config.nodes[resolved]
        self.current_seat = self.config.seat_index(decision_node.seat)
        return self._build_obs()

    def step(self, action: AbstractAction) -> TreeStepResult:
        if self._done or self._scenario is None:
            raise RuntimeError(
                "Call reset() before step(), or the episode has already ended."
            )

        node = self.config.nodes[self._current_node]
        atype = action.action_type

        if atype not in node.actions:
            raise ValueError(
                f"Illegal action {atype.name} at node {node.id!r} "
                f"(legal: {[a.name for a in node.actions]})"
            )

        target = node.actions[atype]
        acting_seat = self.current_seat
        pot_before = sum(self._committed)

        if atype == ActionType.FOLD:
            self._record_history(acting_seat, atype, pot_before, amount=0, is_aggressor=False)
        elif atype == ActionType.CALL:
            amount = self._apply_call(acting_seat)
            self._record_history(acting_seat, atype, pot_before, amount=amount, is_aggressor=False)
        elif atype == ActionType.CHECK:
            self._record_history(acting_seat, atype, pot_before, amount=0, is_aggressor=False)
        elif atype.is_bet:
            amount = self._apply_bet(acting_seat, atype)
            self._record_history(acting_seat, atype, pot_before, amount=amount, is_aggressor=True)
        else:
            raise ValueError(f"Unsupported action type in generic tree: {atype}")

        resolved = self._advance_through_deal_nodes(target)

        if self.config.is_terminal(resolved):
            return self._resolve_terminal(resolved)

        next_node = self.config.nodes[resolved]
        self._current_node = resolved
        self.current_seat = self.config.seat_index(next_node.seat)
        return TreeStepResult(
            obs=self._build_obs(), rewards={}, done=False,
            info={"node": resolved},
        )

    # ------------------------------------------------------------------
    # Chip application
    # ------------------------------------------------------------------

    def _apply_call(self, seat: int) -> int:
        to_match = max(
            (self._committed[i] for i in range(len(self._committed)) if i != seat),
            default=0,
        )
        to_call = max(to_match - self._committed[seat], 0)
        amount = min(to_call, self._stack[seat])
        self._committed[seat] += amount
        self._stack[seat] -= amount
        if self._stack[seat] == 0:
            self._is_all_in[seat] = True
        return amount

    def _apply_bet(self, seat: int, atype: ActionType) -> int:
        if atype not in self._SUPPORTED_BET_TYPES:
            raise ValueError(
                f"GenericTreeEnv only supports {[a.name for a in self._SUPPORTED_BET_TYPES]} "
                f"as bet actions (fixed-tree games have no continuous "
                f"sizing); got {atype.name}"
            )
        if atype == ActionType.ALL_IN:
            amount = self._stack[seat]
        else:  # BET_100 — pot-sized bet, relative to the FULL pot
            pot = self._pot_carry + sum(self._committed)
            amount = min(pot, self._stack[seat])

        self._committed[seat] += amount
        self._stack[seat] -= amount
        if self._stack[seat] == 0:
            self._is_all_in[seat] = True
        return amount

    def _advance_through_deal_nodes(self, node_id: str) -> str:
        while not self.config.is_terminal(node_id):
            node = self.config.nodes[node_id]
            if not isinstance(node, DealNodeConfig):
                return node_id
            self._revealed_nodes.update(node.node_indices)
            node_id = node.next
        return node_id

    def _visible_node_cards(self) -> Tuple[Optional[int], ...]:
        return tuple(
            c if i in self._revealed_nodes else None
            for i, c in enumerate(self._node_cards_full)
        )

    def _record_history(
        self, seat: int, atype: ActionType, pot_before: int, amount: int, is_aggressor: bool
    ) -> None:
        self._history.append(
            HistoryEventObs(
                seat=seat,
                action_type=int(atype),
                street_index=self.config.deal.street_index,
                pot_fraction=(amount / pot_before if pot_before > 0 else 0.0),
                is_aggressor=is_aggressor,
            )
        )

    # ------------------------------------------------------------------
    # Terminal resolution
    # ------------------------------------------------------------------

    def _resolve_terminal(self, terminal_id: str) -> TreeStepResult:
        term = self.config.terminals[terminal_id]
        self._done = True

        if term.type == "fold":
            winner_seats = [
                self.config.seat_index(sid) for sid in self.config.fold_winners(terminal_id)
            ]
            rewards = self._split_flat(winner_seats)
            info = {"outcome": "fold", "terminal": terminal_id, "folders": term.folders}
        else:
            if self._use_real_rules_showdown:
                payouts = resolve_generic_tree_showdown(
                    game_name=self.config.game_name,
                    hole_masks=self._hole_masks,
                    committed=self._committed,
                    pot_carry=self._pot_carry,
                    node_cards_full=self._node_cards_full,
                    node_count=self.config.deal.node_count,
                    scoring_engine=self._scoring_engine,
                    num_showdown_samples=self._num_showdown_samples,
                    runout_rng=self._runout_rng,
                )
                n = len(self.config.seats)
                final_stack = [self._stack[i] + payouts.get(i, 0) for i in range(n)]
                rewards = {
                    seat: float(final_stack[seat] - self._starting_stack[seat])
                    for seat in range(n)
                }
            else:
                rewards = self._resolve_showdown()
            info = {"outcome": "showdown", "terminal": terminal_id}

        return TreeStepResult(obs=None, rewards=rewards, done=True, info=info)

    # def _resolve_terminal(self, terminal_id: str) -> TreeStepResult:
    #     term = self.config.terminals[terminal_id]
    #     self._done = True

    #     if term.type == "fold":
    #         winner_seats = [
    #             self.config.seat_index(sid) for sid in self.config.fold_winners(terminal_id)
    #         ]
    #         rewards = self._split_flat(winner_seats)
    #         info = {"outcome": "fold", "terminal": terminal_id, "folders": term.folders}
    #     else:
    #         rewards = self._resolve_showdown()
    #         info = {"outcome": "showdown", "terminal": terminal_id}

    #     return TreeStepResult(obs=None, rewards=rewards, done=True, info=info)

    def _split_flat(self, winner_seats: List[int]) -> Dict[int, float]:
        n = len(self.config.seats)
        final_stack = list(self._stack)
        pot = self._pot_carry + sum(self._committed)

        share = pot // len(winner_seats)
        remainder = pot % len(winner_seats)
        for i, seat in enumerate(winner_seats):
            final_stack[seat] += share + (1 if i < remainder else 0)

        return {
            seat: float(final_stack[seat] - self._starting_stack[seat])
            for seat in range(n)
        }

    # ------------------------------------------------------------------
    # Points-aware showdown (PATCHED — see module docstring)
    # ------------------------------------------------------------------

    def _node_set_mask(self, node_set) -> int:
        """Card mask for one point's node_set, from the hand's true
        dealt cards (self._node_cards_full — always fully populated by
        the scenario generator regardless of what's been *revealed* to
        the policy so far; showdown resolution needs the truth, not the
        observation)."""
        mask = 0
        for idx in node_set:
            if idx < len(self._node_cards_full) and self._node_cards_full[idx] is not None:
                mask |= 1 << self._node_cards_full[idx]
        return mask

    def _showdown_boards_incomplete(self) -> bool:
        """True if ANY point's ANY node_set has fewer than 5 known
        cards — drives whether Monte Carlo averaging (num_samples > 1)
        is needed at all. Mirrors the old single-board `needed > 0`
        check, generalized across every point/node_set."""
        for point_def in self._points:
            for node_set in point_def.node_sets:
                mask = self._node_set_mask(node_set)
                if 5 - bin(mask).count("1") > 0:
                    return True
        return False

    def _complete_point_board_for_showdown(
        self, board_mask: int, reserved_mask: int, player_masks: List[int]
    ) -> int:
        """
        Fill one point's board mask up to 5 cards with a random runout,
        excluding every card already known (hole cards, this board's
        own dealt cards) AND every card already reserved for a
        DIFFERENT point's board this same sample (`reserved_mask`) —
        two independent physical boards can never legitimately share a
        card, so runouts must not overlap across points within one
        Monte Carlo sample.
        """
        needed = 5 - bin(board_mask).count("1")
        if needed <= 0:
            return board_mask

        used_mask = board_mask | reserved_mask
        for m in player_masks:
            used_mask |= m
        available = [c for c in range(52) if not (used_mask >> c) & 1]

        if len(available) < needed:
            raise RuntimeError(
                f"Not enough cards left to complete a point's board for "
                f"showdown: need {needed}, have {len(available)}."
            )
        runout = self._runout_rng.sample(available, needed)
        mask = board_mask
        for c in runout:
            mask |= 1 << c
        return mask

    def _resolve_showdown(self) -> Dict[int, float]:
        """
        Side-pot-aware, POINTS-aware, Monte-Carlo-averaged showdown
        payout. See module docstring's "Showdown resolution" section
        for the full rationale — this replaces the old single-flat-
        board implementation.

        Supports both rules.payout_type values — "split_pot" and
        "points" — see module docstring's "Showdown resolution"
        section for which algorithm each uses. Any other payout_type
        raises NotImplementedError loudly rather than silently
        mis-paying.
        """
        if self._payout_type not in ("split_pot", "points"):
            raise NotImplementedError(
                f"GenericTreeEnv._resolve_showdown only supports "
                f"payout_type 'split_pot' or 'points' (game "
                f"{self.config.game_name!r} declares "
                f"{self._payout_type!r})."
            )

        n = len(self.config.seats)
        final_stack_start = list(self._stack)

        contributions = sorted(
            ((i, self._committed[i]) for i in range(n) if self._committed[i] > 0),
            key=lambda x: x[1],
        )
        side_pots: List[Tuple[int, List[int]]] = []
        prev = 0
        remaining = [i for i, _ in contributions]
        for i, amount in contributions:
            if amount > prev:
                layer = amount - prev
                side_pots.append((layer * len(remaining), list(remaining)))
                prev = amount
            remaining.remove(i)

        if self._pot_carry and side_pots:
            amount, eligible = side_pots[-1]
            side_pots[-1] = (amount + self._pot_carry, eligible)
        elif self._pot_carry:
            side_pots = [(self._pot_carry, list(range(n)))]

        player_masks = [self._hole_masks[i] for i in range(n)]
        num_points = len(self._points)

        num_samples = (
            self._num_showdown_samples if self._showdown_boards_incomplete() else 1
        )

        accumulated_payout = [0.0] * n

        for _ in range(num_samples):
            # -- score every point once per sample (best-of across each
            #    point's own node_sets), reserving each point's runout
            #    cards before completing the next point's board --------
            point_scores: List[List] = []
            reserved_mask = 0

            for point_def in self._points:
                showdown_type = (
                    point_def.showdown_type_override
                    if point_def.showdown_type_override is not None
                    else self._showdown_type
                )

                best_scores: List = [None] * n
                for node_set in point_def.node_sets:
                    raw_mask = self._node_set_mask(node_set)
                    board_mask = self._complete_point_board_for_showdown(
                        raw_mask, reserved_mask, player_masks
                    )
                    reserved_mask |= board_mask

                    scores = self._scoring_engine.evaluate(
                        player_masks, board_mask, point_def.score_type, showdown_type
                    )
                    for seat in range(n):
                        s = scores[seat]
                        if best_scores[seat] is None or s > best_scores[seat]:
                            best_scores[seat] = s

                point_scores.append(best_scores)

            # -- resolve every side pot, per the game's payout_type -----
            for pot_amount, eligible in side_pots:
                if self._payout_type == "split_pot":
                    # Each point independently wins an even share of
                    # this pot layer — matches
                    # ShowdownResolver._resolve_split_pot.
                    base_share = pot_amount // num_points
                    remainder = pot_amount % num_points

                    for point_idx in range(num_points):
                        component_pot = base_share + (1 if point_idx < remainder else 0)
                        scores = point_scores[point_idx]

                        best = max(scores[i] for i in eligible)
                        winners = [i for i in eligible if scores[i] == best]

                        share = component_pot // len(winners)
                        rem = component_pot % len(winners)
                        for j, seat in enumerate(winners):
                            accumulated_payout[seat] += share + (1 if j < rem else 0)

                else:  # "points"
                    # Tally 1/len(winners) per point a seat wins, then
                    # the ENTIRE pot layer goes to whoever has the most
                    # points (tie = chop) — matches
                    # ShowdownResolver._resolve_points_game.
                    point_tally: Dict[int, float] = {}
                    for point_idx in range(num_points):
                        scores = point_scores[point_idx]
                        best = max(scores[i] for i in eligible)
                        winners = [i for i in eligible if scores[i] == best]
                        share = 1.0 / len(winners)
                        for w in winners:
                            point_tally[w] = point_tally.get(w, 0.0) + share

                    if point_tally:
                        max_points = max(point_tally.values())
                        pot_winners = [
                            i for i in eligible
                            if point_tally.get(i, 0.0) >= max_points - 1e-9
                        ]
                    else:
                        pot_winners = list(eligible)

                    share = pot_amount // len(pot_winners)
                    rem = pot_amount % len(pot_winners)
                    for j, seat in enumerate(pot_winners):
                        accumulated_payout[seat] += share + (1 if j < rem else 0)

        final_stack = [
            final_stack_start[i] + accumulated_payout[i] / num_samples
            for i in range(n)
        ]

        return {
            seat: float(final_stack[seat] - self._starting_stack[seat])
            for seat in range(n)
        }

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @staticmethod
    def _legal_mask_for_node(node) -> Tuple[bool, ...]:
        mask = [False] * NUM_ACTIONS
        for atype in node.actions:
            mask[atype] = True
        return tuple(mask)

    def _build_obs(self) -> StructuredObservation:
        n = len(self.config.seats)

        players = tuple(
            PlayerObs(
                seat=i,
                stack=self._stack[i],
                current_bet=self._committed[i],
                has_folded=False,
                is_all_in=self._is_all_in[i],
                is_hero=(i == self.current_seat),
            )
            for i in range(n)
        )

        pot = self._pot_carry + sum(self._committed)
        to_match = max(
            (self._committed[i] for i in range(n) if i != self.current_seat),
            default=0,
        )
        bet_to_call = max(to_match - self._committed[self.current_seat], 0)

        node = self.config.nodes[self._current_node]
        dealer_seat_id = self.config.dealer_seat or self.config.first_seat_id
        dealer_position = self.config.seat_index(dealer_seat_id)

        return StructuredObservation(
            hero_seat=self.current_seat,
            hole_cards=HoleCardsObs(cards=_mask_to_cards(self._hole_masks[self.current_seat])),
            board=BoardObs(node_cards=self._visible_node_cards()),
            players=players,
            betting=BettingObs(
                pot=pot,
                bet_to_call=bet_to_call,
                min_raise=self.config.big_blind,
                street_index=self.config.deal.street_index,
                raises_this_street=0,
            ),
            legal_mask=self._legal_mask_for_node(node),
            game_name=self.config.game_name,
            history=HistoryObs(events=tuple(self._history)),
            dealer_position=dealer_position,
            is_terminal=False,
        )
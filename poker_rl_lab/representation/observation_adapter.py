"""
representation/observation_adapter.py — Adapter from StructuredObservation
(envs/observations.py) to RepresentationInput (representation_model.py).

Made-hand board extraction (added)
--------------------------------------
made_hand_boards_from_points() builds ONE board-card list PER POINT —
the union of dealt cards across that point's own node_sets — as input to
RepresentationModel's made_hand_encoder (hole ∪ board, decomposed into
legal 5-card candidate hands per showdown_type).

This is deliberately a SEPARATE function from boards_from_node_cards(),
not a reuse of its output, because the two serve different purposes:
boards_from_node_cards() DEDUPLICATES node_sets across all points into a
list of distinct BOARDS (for the texture role, where "board 1" / "board
2" are genuinely different physical boards, e.g. double_board_plo's two
independent 5-card boards). made_hand_boards_from_points() instead needs
one entry per POINT, in points order, even when two points share the
same underlying node_set (e.g. a hi/lo game's "high" and "low" points
over the same single board) — those are two DIFFERENT made-hand
evaluations (different showdown_type / qualifier) over the SAME cards,
and must remain two separate entries so points[i].showdown_type lines up
1:1 with made_hand_board_card_ids[i], which RepresentationModel.
forward_batch() relies on directly (see that module's per-point-index
pooling loop).

For a point spanning MULTIPLE node_sets (e.g. a "best-of-N-boards"
point), all of that point's node_sets are unioned together — every
board card that point could possibly use is made available to the
made-hand decomposer; showdown_constraints.py's CardUsageConstraint
still bounds how many of them can end up in any single candidate 5-card
hand.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from poker_rl_lab.envs.observations import HistoryEventObs, StructuredObservation
from poker_rl_lab.representation.representation_model import RepresentationInput
from poker_rl_lab.representation.point_encoder import PointFeatures, point_features_from_definition
from poker_rl_lab.representation.game_encoder import GameFeatures, game_features_from_definition
from poker_rl_lab.representation.position_encoder import PositionFeatures, position_features_from_observation
from poker_rl_lab.representation.history_encoder import HistoryEvent
from poker_rl_lab.representation.stack_encoder import StackFeatures, stack_features_from_observation


# ---------------------------------------------------------------------------
# Board extraction — texture role (unchanged)
# ---------------------------------------------------------------------------

def boards_from_node_cards(
    node_cards: Sequence[Optional[int]],
    points,  # Sequence[rules.point_definition.PointDefinition]
) -> List[List[int]]:
    """Unchanged — see original docstring. Deduplicated, one entry per
    distinct node_set, first-appearance order. Feeds board_encoder
    (texture role)."""
    seen: List[Tuple[int, ...]] = []

    for point_def in points:
        for node_set in point_def.node_sets:
            key = tuple(node_set)
            if key in seen:
                continue
            seen.append(key)

    boards = [
        [node_cards[n] for n in node_set if n < len(node_cards) and node_cards[n] is not None]
        for node_set in seen
    ]

    if not boards:
        boards = [[c for c in node_cards if c is not None]]

    return boards


# ---------------------------------------------------------------------------
# Board extraction — made-hand role (NEW)
# ---------------------------------------------------------------------------

def made_hand_boards_from_points(
    node_cards: Sequence[Optional[int]],
    points,  # Sequence[rules.point_definition.PointDefinition]
) -> List[List[int]]:
    """
    One board-card list PER POINT — union of dealt cards across that
    point's own node_sets, in points order. See module docstring for
    why this differs from boards_from_node_cards().

    Parameters
    ----------
    node_cards : Sequence[Optional[int]]
        StructuredObservation.board.node_cards. None = not yet dealt.
    points : Sequence[PointDefinition]
        rules.points — SAME sequence (same order) passed to
        point_features_from_definition() for each index, so
        made_hand_board_card_ids[i] and points[i] line up.

    Returns
    -------
    List[List[int]]
        Length len(points). Each entry may be empty (nothing dealt yet
        for that point) — made_hand_encoder's constraint.valid_splits()
        already handles n_board=0 (returns no valid splits for most
        showdown_types, falling back to the learned placeholder — see
        composite_board_encoder.py).
    """
    result: List[List[int]] = []
    for point_def in points:
        seen_indices: set = set()
        for node_set in point_def.node_sets:
            seen_indices.update(node_set)
        cards = [
            node_cards[n]
            for n in sorted(seen_indices)
            if n < len(node_cards) and node_cards[n] is not None
        ]
        result.append(cards)
    return result


# ---------------------------------------------------------------------------
# History extraction (unchanged)
# ---------------------------------------------------------------------------

def history_event_from_obs(
    event: HistoryEventObs,
    hero_seat: int,
    dealer_position: int,
    num_players: int,
) -> HistoryEvent:
    seat_fraction = ((event.seat - dealer_position) % num_players) / num_players
    return HistoryEvent(
        action_type=event.action_type,
        street_index=event.street_index,
        seat_fraction=seat_fraction,
        pot_fraction=event.pot_fraction,
        is_hero=(event.seat == hero_seat),
        is_aggressor=event.is_aggressor,
    )


def history_from_obs(obs: StructuredObservation) -> List[HistoryEvent]:
    n = len(obs.players)
    return [
        history_event_from_obs(e, obs.hero_seat, obs.dealer_position, n)
        for e in obs.history.events
    ]


# ---------------------------------------------------------------------------
# Top-level adapter
# ---------------------------------------------------------------------------

def observation_to_representation_input(
    obs: StructuredObservation,
    game_def,
    rules,
) -> RepresentationInput:
    """
    Convert one StructuredObservation into a RepresentationInput.

    Unchanged signature; now also populates made_hand_board_card_ids
    via made_hand_boards_from_points(), in points order.
    """
    hole_card_ids = list(obs.hole_cards.cards)

    board_card_ids = boards_from_node_cards(obs.board.node_cards, rules.points)
    made_hand_board_card_ids = made_hand_boards_from_points(obs.board.node_cards, rules.points)

    points = [
        point_features_from_definition(point_def, rules)
        for point_def in rules.points
    ]

    game = game_features_from_definition(game_def, rules)
    position = position_features_from_observation(obs)

    stack = stack_features_from_observation(
        obs,
        small_blind=getattr(game_def, "small_blind", 1) or 1,
        big_blind=getattr(game_def, "big_blind", 2) or 2,
    )

    history = history_from_obs(obs)

    return RepresentationInput(
        hole_card_ids=hole_card_ids,
        board_card_ids=board_card_ids,
        made_hand_board_card_ids=made_hand_board_card_ids,
        points=points,
        game=game,
        position=position,
        stack=stack,
        history=history,
    )


# ---------------------------------------------------------------------------
# Caching convenience wrapper (unchanged)
# ---------------------------------------------------------------------------

class ObservationAdapter:
    def __init__(self) -> None:
        self._cache: Dict[str, tuple] = {}

    def adapt(self, obs: StructuredObservation) -> RepresentationInput:
        game_def, rules = self._get_game(obs.game_name)
        return observation_to_representation_input(obs, game_def, rules)

    def _get_game(self, game_name: str) -> tuple:
        if game_name not in self._cache:
            from poker_engine.games.loader import load_game
            self._cache[game_name] = load_game(game_name)
        return self._cache[game_name]

    def preload(self, game_name: str) -> None:
        self._get_game(game_name)
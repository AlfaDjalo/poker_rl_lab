"""
representation/observation_adapter.py — Adapter from StructuredObservation
(envs/observations.py) to RepresentationInput (representation_model.py).

This is the missing link named in representation_model.py's own docstring:

    "That adapter is env-specific and lives outside this module, per the
    guide's 'Environment must remain independent of the neural network
    architecture'."

Both push_fold_env.py and river_env.py already produce StructuredObservation
at every hero decision point. This module converts one of those into a
RepresentationInput, ready to be passed to
RepresentationModel.forward()/PokerActorCritic.act().

Two engine-shaped inputs are needed that StructuredObservation does NOT
carry itself (deliberately — see observations.py's docstring: the
Environment must never duplicate poker rules): the game's GameDefinition
and GameRules (rules/game_definition.py, rules/game_rules.py), i.e. exactly
what loader.load_game(name) returns. StructuredObservation.game_name is
the string key to look them up.

Design notes
------------
- Board grouping: RepresentationInput.board_card_ids expects "one entry
  per board" (board_encoder.py encodes one board's texture at a time).
  Neither BaseScenario nor StructuredObservation expose an explicit
  board->node mapping — but rules.points[*].node_sets already ARE board
  definitions (see PointDefinition.generate_board_masks, and compare
  holdem.yaml's single node_set [[0,1,2,3,4]] against
  hopscotch_bomb_pot.yaml's two node_sets, one per board). So boards here
  are derived by collecting the unique node_sets referenced across all
  points, in first-appearance order, and mapping node indices to dealt
  card ids. This generalises correctly to today's single-board games
  (holdem, push-fold, river) and to already-defined multi-board YAMLs
  without needing a new "board" concept in the Engine.
- History: HistoryEventObs (env-facing, observations.py) is deliberately
  thinner than HistoryEvent (encoder-facing, history_encoder.py) — the
  latter additionally wants seat_fraction and is_hero, which depend on
  table geometry (dealer_position, hero_seat, player count) that the
  Environment doesn't need just to log an event. Computed here using the
  exact same seat_fraction formula as position_encoder.py, so the two
  encoders never disagree about what "seat_fraction" means.
- Stack/Pot: StructuredObservation.betting already carries pot and
  bet_to_call (accurate, per-decision), and obs.players carries every
  seat's live stack — but neither small_blind nor big_blind is
  StructuredObservation's to carry (see observations.py's docstring:
  the Environment stays independent of blind-size bookkeeping beyond
  what BettingObs already needs). Rather than widen
  StructuredObservation or push a new parameter onto every call site,
  stack_features_from_observation() sources small_blind/big_blind from
  `game_def` — exactly the same pattern game_encoder.py's own
  game_features_from_definition() already uses for `ante_bb` (ante /
  game_def.big_blind). This does mean the blinds used for BB-unit
  normalisation are the variant's YAML-declared blinds, not necessarily
  whatever non-standard blinds a specific scenario/generator used (see
  stack_encoder.py's own docstring) — an existing, pre-established
  limitation shared with ante_bb, not a new one introduced here.
- Loading (game_def, rules) from a game_name touches the real Engine
  (poker_eval C++ extension, via loader.load_game / rules.game_rules).
  ObservationAdapter caches the lookup per game_name so a training loop
  doesn't re-parse YAML on every single step.

Dependencies
------------
    envs.observations, representation.representation_model,
    representation.point_encoder, representation.game_encoder,
    representation.position_encoder, representation.history_encoder,
    representation.stack_encoder.
    loader.load_game is imported lazily (inside ObservationAdapter, not
    at module level) so this file can still be imported in a poker_eval-
    less test environment as long as callers build RepresentationInput
    via the plain function below with their own (game_def, rules).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from envs.observations import HistoryEventObs, StructuredObservation
from representation.representation_model import RepresentationInput
from representation.point_encoder import PointFeatures, point_features_from_definition
from representation.game_encoder import GameFeatures, game_features_from_definition
from representation.position_encoder import PositionFeatures, position_features_from_observation
from representation.history_encoder import HistoryEvent
from representation.stack_encoder import StackFeatures, stack_features_from_observation


# ---------------------------------------------------------------------------
# Board extraction
# ---------------------------------------------------------------------------

def boards_from_node_cards(
    node_cards: Sequence[Optional[int]],
    points,  # Sequence[rules.point_definition.PointDefinition]
) -> List[List[int]]:
    """
    Group dealt board cards into per-board lists using each point's
    node_sets as the board definition.

    Parameters
    ----------
    node_cards : Sequence[Optional[int]]
        StructuredObservation.board.node_cards. None = not yet dealt.
    points : Sequence[PointDefinition]
        rules.points for the current game.

    Returns
    -------
    List[List[int]]
        One list of dealt card ids per unique node_set (board), in
        first-appearance order across points. Never empty — falls back
        to a single board containing every dealt card if no points are
        given (defensive; every loaded game has >= 1 point in practice).
    """
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
# History extraction
# ---------------------------------------------------------------------------

def history_event_from_obs(
    event: HistoryEventObs,
    hero_seat: int,
    dealer_position: int,
    num_players: int,
) -> HistoryEvent:
    """
    Convert one env-facing HistoryEventObs into an encoder-facing
    HistoryEvent, filling in seat_fraction / is_hero from table geometry.

    seat_fraction uses the identical formula to
    representation.position_encoder's PositionFeatures.seat_fraction:
    ((seat - dealer) % n) / n — so "who acted" is directly comparable to
    "who is about to act" inside the same latent space.
    """
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
    """Convert every event in obs.history into encoder-facing HistoryEvents."""
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
    game_def,   # rules.game_definition.GameDefinition
    rules,      # rules.game_rules.GameRules
) -> RepresentationInput:
    """
    Convert one StructuredObservation into a RepresentationInput.

    Parameters
    ----------
    obs : StructuredObservation
        Produced by any BasePokerEnv at a hero decision point.
    game_def : rules.game_definition.GameDefinition
    rules : rules.game_rules.GameRules
        The (game_def, rules) pair for obs.game_name, e.g. from
        loader.load_game(obs.game_name). See ObservationAdapter below
        for a caching convenience wrapper.

    Returns
    -------
    RepresentationInput
        Ready to pass to RepresentationModel.forward() /
        PokerActorCritic.act().
    """
    hole_card_ids = list(obs.hole_cards.cards)

    board_card_ids = boards_from_node_cards(obs.board.node_cards, rules.points)

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
        points=points,
        game=game,
        position=position,
        stack=stack,
        history=history,
    )


# ---------------------------------------------------------------------------
# Caching convenience wrapper
# ---------------------------------------------------------------------------

class ObservationAdapter:
    """
    Caches (game_def, rules) per game_name so a training loop doesn't
    re-parse a YAML file (and re-touch the poker_eval C++ extension) on
    every single environment step.

    Usage
    -----
        adapter = ObservationAdapter()
        obs = env.reset(scenario)
        rep_input = adapter.adapt(obs)
    """

    def __init__(self) -> None:
        self._cache: Dict[str, tuple] = {}

    def adapt(self, obs: StructuredObservation) -> RepresentationInput:
        game_def, rules = self._get_game(obs.game_name)
        return observation_to_representation_input(obs, game_def, rules)

    def _get_game(self, game_name: str) -> tuple:
        if game_name not in self._cache:
            # Lazy import: touches the real Engine / poker_eval extension
            # only when a game actually needs loading, not at module
            # import time.
            from poker_engine.games.loader import load_game
            self._cache[game_name] = load_game(game_name)
        return self._cache[game_name]

    def preload(self, game_name: str) -> None:
        """Force-load (and cache) a game ahead of time, e.g. at startup."""
        self._get_game(game_name)
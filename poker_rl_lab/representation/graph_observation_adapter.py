"""
representation/graph_observation_adapter.py — Adapter from
envs.graph_scenario_env.GraphScenarioEnv's live GraphGameState +
GraphDecisionObs to representation.representation_model.RepresentationInput,
so the EXISTING RepresentationModel / PolicyValueNetwork stack can
consume graph-engine-driven decisions unchanged.

Supersedes representation/observation_adapter.py's role for the new
engine. Reuses several of that module's functions directly
(boards_from_node_cards, made_hand_boards_from_points,
point_features_from_definition) since those only ever needed
`node_cards` + `rules.points`, both of which GraphGameState/GameRules
still provide identically — no reason to duplicate them.

What's genuinely new here (vs. the old observation_adapter.py)
------------------------------------------------------------------
GameFeatures can no longer be built by game_encoder.py's own
game_features_from_definition(): the graph-native GameDefinition
(poker_engine.rules.graph_loader.load_game_graph()) does NOT expose
hole_cards / betting_type / street_nodes / ante / big_blind (see
CAP_Technical_Quick_Reference.md §12.4) — those facts now live in the
graph/flow instead. _graph_game_features() below rebuilds an equivalent
GameFeatures from: the live hand's own hole-card count (dynamic, read
off state at adapt-time rather than a fixed per-game constant — hold'em
vs. Omaha's 2-vs-4 already shows up correctly this way, without needing
a static field at all), GraphScenarioEnv.num_streets (precomputed once
per env from the compiled graph's node metadata — see that class), and
the CURRENT decision's own betting rules kind if this adapt() call is
for a BETTING decision (betting_type is no longer a single fixed
per-game constant now that different streets/nodes may use different
BettingRules — see flow_loader.py's `betting` step type; this adapter
approximates it as "whatever rules the node being decided right now
uses", which is exact for every variant shipped so far since none of
them actually mix betting-rule kinds across streets yet).

Known gaps (documented, not silently guessed)
--------------------------------------------------
    - has_ante / ante_bb: the graph-native GameDefinition has no ante
      amount to read (see above). has_ante is approximated as "does
      this variant's compiled graph contain a post_antes AUTO node at
      all" (cheap, exact for presence); ante_bb is left at 0.0 always
      — getting the real amount would mean inspecting that node's
      config, which isn't threaded through GraphScenarioEnv today.
      Flagged here rather than silently wrong; fix by extending
      GraphScenarioEnv to surface it if a variant actually needs
      ante-aware training.
    - History is BETTING-only (see graph_scenario_env.py's
      GraphHistoryEvent docstring) — CARD_SELECT/CARD_PASS/BOOLEAN/
      CHOICE actions never appear in the HistoryEncoder token sequence
      yet, same pre-existing scope limit the old pipeline had.

Dependencies
------------
    envs.graph_scenario_env (GraphScenarioEnv, GraphDecisionObs,
    GraphHistoryEvent), representation.representation_model
    (RepresentationInput), representation.point_encoder,
    representation.game_encoder, representation.position_encoder,
    representation.history_encoder, representation.stack_encoder,
    representation.card_utils.mask_to_card_ids,
    representation.observation_adapter (boards_from_node_cards,
    made_hand_boards_from_points — reused).
"""

from __future__ import annotations

from typing import List

from poker_rl_lab.envs.graph_scenario_env import GraphDecisionObs, GraphHistoryEvent, GraphScenarioEnv
from poker_rl_lab.representation.card_utils import mask_to_card_ids
from poker_rl_lab.representation.representation_model import RepresentationInput
from poker_rl_lab.representation.point_encoder import point_features_from_definition
from poker_rl_lab.representation.game_encoder import GameFeatures
from poker_rl_lab.representation.position_encoder import PositionFeatures, _compute_position_features
from poker_rl_lab.representation.history_encoder import HistoryEvent
from poker_rl_lab.representation.stack_encoder import StackFeatures, _compute_stack_features
from poker_rl_lab.representation.observation_adapter import (
    boards_from_node_cards,
    made_hand_boards_from_points,
)


# ---------------------------------------------------------------------------
# Game features (see module docstring — cannot reuse game_encoder.py's
# game_features_from_definition() against a graph-native GameDefinition)
# ---------------------------------------------------------------------------

_BETTING_RULES_CLASS_NAME_TO_ID = {
    "NoLimitRules": "no_limit",
    "PotLimitRules": "pot_limit",
    "FixedLimitRules": "fixed_limit",
    # AllInOrFoldRules (push/fold) and anything else fall through to
    # game_encoder.py's own "other" bucket via _betting_type_id()'s
    # unknown-string handling — no entry needed here for that.
}


def _current_betting_type_string(env: GraphScenarioEnv, obs: GraphDecisionObs) -> str:
    """
    Best-effort betting_type string for GameFeatures — see module
    docstring's "What's genuinely new here" section. Returns "other"
    for any non-BETTING decision or unrecognised BettingRules subclass.
    """
    if obs.domain != "BETTING":
        return "other"
    node = env.graph.node(obs.node_id)
    rules_obj = node.config.get("rules")
    class_name = type(rules_obj).__name__
    return _BETTING_RULES_CLASS_NAME_TO_ID.get(class_name, "other")


def _has_post_antes(env: GraphScenarioEnv) -> bool:
    return any(n.auto_type == "post_antes" for n in env.graph.nodes.values())


def _graph_game_features(env: GraphScenarioEnv, obs: GraphDecisionObs, hole_card_count: int) -> GameFeatures:
    from poker_rl_lab.representation.game_encoder import _betting_type_id, _payout_type_id

    return GameFeatures(
        betting_type_id=_betting_type_id(_current_betting_type_string(env, obs)),
        payout_type_id=_payout_type_id(getattr(env.rules, "payout_type", None)),
        showdown_type=int(env.rules.showdown_type),
        hole_cards=hole_card_count,
        node_count=int(env.game_def.node_count),
        num_streets=env.num_streets,
        num_points=len(env.rules.points),
        has_ante=_has_post_antes(env),
        ante_bb=0.0,   # see module docstring's "Known gaps" section
        has_low=any(
            int(st) in (1, 2) for st in getattr(env.rules, "score_types", [])
        ),  # 1/2 == LOW_A5/LOW_27, matching game_encoder.py's own _LOW_SCORE_TYPES
    )


# ---------------------------------------------------------------------------
# Position / Stack features (both reuse the generic, engine-agnostic
# compute functions already defined in position_encoder.py / stack_encoder.py
# — those only ever needed seat/dealer/current_player/has_folded/is_all_in
# and hero/opponent stack + pot + blind sizes respectively, all of which
# GraphGameState + GraphScenarioConfig.stack still provide identically)
# ---------------------------------------------------------------------------

def _graph_position_features(env: GraphScenarioEnv, seat: int) -> PositionFeatures:
    state = env.state
    n = len(state.players)
    has_folded = [p.has_folded for p in state.players]
    is_all_in = [p.is_all_in for p in state.players]
    return _compute_position_features(
        seat, state.dealer_position, n, state.current_player, has_folded, is_all_in
    )


def _graph_stack_features(env: GraphScenarioEnv, seat: int) -> StackFeatures:
    state = env.state
    hero = state.players[seat]
    opponents = [p.stack for i, p in enumerate(state.players) if i != seat]
    hero_bet = hero.current_bet
    # bet_to_call is only meaningful mid-BETTING-decision; state itself
    # doesn't carry a generic "bet_to_call" outside GameState's own
    # betting-round bookkeeping (bet_to_call attribute set by
    # apply_action/reset_betting_round — present on every GameState).
    bet_to_call = max(getattr(state, "bet_to_call", 0) - hero_bet, 0)
    return _compute_stack_features(
        hero_stack=hero.stack,
        opponent_stacks=opponents,
        pot=state.pot,
        bet_to_call=bet_to_call,
        small_blind=env.config.stack.small_blind_size,
        big_blind=env.config.stack.big_blind_size,
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _graph_history_events(env: GraphScenarioEnv, seat: int) -> List[HistoryEvent]:
    state = env.state
    n = len(state.players)
    dealer = state.dealer_position
    events: List[HistoryEvent] = []
    for e in env.history:
        seat_fraction = ((e.seat - dealer) % n) / n
        events.append(
            HistoryEvent(
                # engine_action_type is an int-valued Enum (FOLD/CHECK/
                # CALL/BET/RAISE) — HistoryEncoder's own action_type_dim
                # embedding table is sized NUM_ACTIONS (the OLD 9-value
                # abstract space) purely because that's what every other
                # caller in this codebase currently uses; the raw engine
                # int (0-4ish) fits well within that range so this is
                # safe today, but is NOT the same vocabulary — see
                # graph_scenario_env.GraphHistoryEvent's own docstring.
                # A HistoryEncoder trained on this adapter's output is
                # learning the ENGINE's action_type ids, not the abstract
                # bet-size buckets a BETTING-domain GraphActionOption
                # resolves to.
                action_type=e.engine_action_type.value,
                street_index=e.street_index,
                seat_fraction=seat_fraction,
                pot_fraction=e.pot_fraction,
                is_hero=(e.seat == seat),
                is_aggressor=e.is_aggressor,
            )
        )
    return events


# ---------------------------------------------------------------------------
# Top-level adapter
# ---------------------------------------------------------------------------

def graph_observation_to_representation_input(
    env: GraphScenarioEnv, obs: GraphDecisionObs
) -> RepresentationInput:
    """
    Convert one GraphDecisionObs (+ env's live state) into a
    RepresentationInput, for whichever hero seat obs.seat names.
    """
    seat = obs.seat
    state = env.state

    hole_card_ids = mask_to_card_ids(state.players[seat].hand_mask)
    board_card_ids = boards_from_node_cards(state.node_cards, env.rules.points)
    made_hand_board_card_ids = made_hand_boards_from_points(state.node_cards, env.rules.points)

    points = [point_features_from_definition(p, env.rules) for p in env.rules.points]
    game = _graph_game_features(env, obs, hole_card_count=len(hole_card_ids))
    position = _graph_position_features(env, seat)
    stack = _graph_stack_features(env, seat)
    history = _graph_history_events(env, seat)

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


class GraphObservationAdapter:
    """
    Thin stateful convenience wrapper mirroring
    representation.observation_adapter.ObservationAdapter's own
    call shape (`.adapt(obs) -> RepresentationInput`), for drop-in use
    in training loops expecting that interface. Unlike the old adapter,
    there's no per-game_name cache to maintain here — env already holds
    game_def/rules for its own lifetime — so this only exists for
    interface parity, not because it does real caching work.
    """

    def __init__(self, env: GraphScenarioEnv) -> None:
        self.env = env

    def adapt(self, obs: GraphDecisionObs) -> RepresentationInput:
        return graph_observation_to_representation_input(self.env, obs)
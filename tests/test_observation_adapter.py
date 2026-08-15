"""
tests/test_observation_adapter.py — Unit tests for
representation/observation_adapter.py.

Uses the mock_poker_eval shim (see that module's docstring) so these
tests run without the compiled poker_eval C++ extension — loading
holdem.yaml via loader.load_game("holdem") only needs ScoreType /
ShowdownType to exist as enums, not real hand evaluation.
"""

import sys

import pytest

# --- poker_eval shim, must happen before any rules/* or loader import ---
import poker_rl_lab.mock_poker_eval as mock_poker_eval
sys.modules.setdefault("poker_eval", mock_poker_eval)

from poker_engine.games import loader
from poker_engine.games.loader import load_game

from poker_rl_lab.scenarios.push_fold_generator import PushFoldScenarioGenerator
from poker_rl_lab.scenarios.river_generator import RiverScenarioGenerator
from poker_rl_lab.envs.push_fold_env import PushFoldEnv
from poker_rl_lab.envs.river_env import RiverEnv
from poker_rl_lab.envs.observations import HistoryEventObs
from poker_rl_lab.actions.abstract_action import ActionType

from poker_rl_lab.representation.observation_adapter import (
    ObservationAdapter,
    boards_from_node_cards,
    history_event_from_obs,
    history_from_obs,
    observation_to_representation_input,
)
from poker_rl_lab.representation.representation_model import RepresentationInput
from poker_rl_lab.representation.history_encoder import HistoryEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def holdem_game():
    return load_game("holdem")


@pytest.fixture
def push_fold_obs():
    gen = PushFoldScenarioGenerator(player_count=2)
    gen.seed(7)
    scenario = gen.generate()

    env = PushFoldEnv()
    return env.reset(scenario)


# ---------------------------------------------------------------------------
# boards_from_node_cards
# ---------------------------------------------------------------------------

class _FakePoint:
    """Minimal stand-in for rules.point_definition.PointDefinition."""
    def __init__(self, node_sets):
        self.node_sets = node_sets


def test_boards_single_board_holdem_layout():
    node_cards = [10, 20, 30, 40, 50]
    points = [_FakePoint([[0, 1, 2, 3, 4]])]

    boards = boards_from_node_cards(node_cards, points)

    assert boards == [[10, 20, 30, 40, 50]]


def test_boards_two_boards_dedup_across_points():
    node_cards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # Two points sharing the same two node_sets (e.g. high/low on same boards)
    points = [
        _FakePoint([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
        _FakePoint([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
    ]

    boards = boards_from_node_cards(node_cards, points)

    assert boards == [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]


def test_boards_skips_undealt_nodes():
    node_cards = [10, None, 30, None, None]
    points = [_FakePoint([[0, 1, 2, 3, 4]])]

    boards = boards_from_node_cards(node_cards, points)

    assert boards == [[10, 30]]


def test_boards_falls_back_to_single_board_with_no_points():
    node_cards = [1, 2, None, 4]

    boards = boards_from_node_cards(node_cards, [])

    assert boards == [[1, 2, 4]]


# ---------------------------------------------------------------------------
# history_event_from_obs / history_from_obs
# ---------------------------------------------------------------------------

def test_history_event_seat_fraction_and_is_hero():
    event = HistoryEventObs(
        seat=0, action_type=int(ActionType.ALL_IN), street_index=0,
        pot_fraction=1.5, is_aggressor=True,
    )

    result = history_event_from_obs(
        event, hero_seat=1, dealer_position=0, num_players=2,
    )

    assert isinstance(result, HistoryEvent)
    assert result.action_type == int(ActionType.ALL_IN)
    assert result.street_index == 0
    assert result.seat_fraction == pytest.approx(0.0)  # seat == dealer
    assert result.pot_fraction == pytest.approx(1.5)
    assert result.is_hero is False
    assert result.is_aggressor is True


def test_history_event_hero_seat_fraction():
    event = HistoryEventObs(seat=1, action_type=int(ActionType.CALL), street_index=0)

    result = history_event_from_obs(
        event, hero_seat=1, dealer_position=0, num_players=2,
    )

    assert result.seat_fraction == pytest.approx(0.5)  # (1 - 0) % 2 / 2
    assert result.is_hero is True


def test_history_from_obs_matches_event_count(push_fold_obs):
    events = history_from_obs(push_fold_obs)
    assert len(events) == len(push_fold_obs.history.events)
    assert all(isinstance(e, HistoryEvent) for e in events)


# ---------------------------------------------------------------------------
# observation_to_representation_input — end to end
# ---------------------------------------------------------------------------

def test_observation_to_representation_input_push_fold(push_fold_obs, holdem_game):
    game_def, rules = holdem_game

    rep_input = observation_to_representation_input(push_fold_obs, game_def, rules)

    assert isinstance(rep_input, RepresentationInput)
    assert rep_input.hole_card_ids == list(push_fold_obs.hole_cards.cards)
    assert len(rep_input.hole_card_ids) == 2

    # holdem.yaml has exactly one point ("main") -> one board
    assert len(rep_input.points) == 1
    assert len(rep_input.board_card_ids) == 1

    # push-fold scenarios have no board dealt yet
    assert rep_input.board_card_ids[0] == []

    assert rep_input.game.hole_cards == 2
    assert rep_input.position.is_next_to_act is True  # hero always acts in push-fold
    assert len(rep_input.history) == len(push_fold_obs.history.events)


def test_observation_to_representation_input_river(holdem_game):
    game_def, rules = holdem_game

    gen = RiverScenarioGenerator(player_count=2)
    gen.seed(3)
    scenario = gen.generate()

    env = RiverEnv()
    obs = env.reset(scenario)

    rep_input = observation_to_representation_input(obs, game_def, rules)

    # River scenarios have all 5 board cards dealt, one board.
    assert len(rep_input.board_card_ids) == 1
    assert len(rep_input.board_card_ids[0]) == 5
    assert all(isinstance(c, int) for c in rep_input.board_card_ids[0])


# ---------------------------------------------------------------------------
# ObservationAdapter caching
# ---------------------------------------------------------------------------

def test_observation_adapter_caches_game_lookup(push_fold_obs, monkeypatch):
    calls = []
    real_load_game = load_game

    def spy_load_game(name):
        calls.append(name)
        return real_load_game(name)

    monkeypatch.setattr(loader, "load_game", spy_load_game)

    adapter = ObservationAdapter()
    adapter.adapt(push_fold_obs)
    adapter.adapt(push_fold_obs)

    assert calls == ["holdem"]  # only loaded once despite two calls


def test_observation_adapter_preload(monkeypatch):
    calls = []
    real_load_game = load_game

    def spy_load_game(name):
        calls.append(name)
        return real_load_game(name)

    monkeypatch.setattr(loader, "load_game", spy_load_game)

    adapter = ObservationAdapter()
    adapter.preload("holdem")
    adapter.preload("holdem")

    assert calls == ["holdem"]


def test_observation_adapter_adapt_matches_plain_function(push_fold_obs, holdem_game):
    game_def, rules = holdem_game

    adapter = ObservationAdapter()
    via_adapter = adapter.adapt(push_fold_obs)
    via_function = observation_to_representation_input(push_fold_obs, game_def, rules)

    assert via_adapter.hole_card_ids == via_function.hole_card_ids
    assert via_adapter.board_card_ids == via_function.board_card_ids
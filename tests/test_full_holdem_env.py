"""
test_full_holdem_env.py — Tests for envs/full_holdem_env.py.

Runs against poker_engine/state/poker_state.py, which in this checkout
is a TEST DOUBLE (see that file's docstring) standing in for the real
Engine internals (GameState/PlayerState/Deck/ShowdownResolver aren't in
the project yet). Everything else touched here — actions.abstract_action,
scenarios.*, poker_engine.games.loader, poker_engine.rules.* — is the
real, unmodified project code.
"""

from __future__ import annotations

import sys

try:
    import poker_eval  # noqa: F401
except ImportError:
    import mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import pytest

from actions.abstract_action import AbstractAction, ActionType
from envs.full_holdem_env import FullHoldemEnv
from envs.river_env import (
    CheckFoldVillainPolicy,
    CallAnyVillainPolicy,
    FoldAnyVillainPolicy,
)
from scenarios.base_scenario import BaseScenario, PlayerScenario
from scenarios.full_holdem_generator import FullHoldemScenarioGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_heads_up_scenario(
    hero_stack=200, villain_stack=200, hero_mask=0b11, villain_mask=(0b11 << 2)
) -> BaseScenario:
    """
    2-player preflop scenario: dealer=seat0 (=villain, SB), seat1=hero (BB).
    First to act preflop (heads-up, per FullHoldemScenarioGenerator's own
    engine-matching formula) = (dealer + 3) % 2 = seat 1 = hero.
    """
    villain = PlayerScenario(
        stack=villain_stack - 1, hand_mask=villain_mask,
        current_bet=1, total_contribution=1, has_folded=False, is_all_in=False,
    )
    hero = PlayerScenario(
        stack=hero_stack - 2, hand_mask=hero_mask,
        current_bet=2, total_contribution=2, has_folded=False, is_all_in=False,
    )
    return BaseScenario(
        game_name="holdem",
        street_index=0,
        pot=0,
        dealer_position=0,
        current_player=1,
        bet_to_call=2,
        min_raise=2,
        players=(villain, hero),
        node_cards=tuple(None for _ in range(5)),
        small_blind=1,
        big_blind=2,
        raises_this_street=0,
        last_aggressor=None,
    )


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_reset_returns_nonterminal_hero_observation():
    env = FullHoldemEnv()
    scenario = make_heads_up_scenario()
    obs = env.reset(scenario)

    assert obs.is_terminal is False
    assert obs.hero_seat == 1
    assert any(obs.legal_mask)
    assert obs.legal_mask[ActionType.FOLD] is True


def test_reset_with_generator_produces_playable_scenario():
    gen = FullHoldemScenarioGenerator(player_count=2)
    gen.seed(0)
    env = FullHoldemEnv()
    obs = env.reset(gen.generate())
    assert obs.is_terminal is False
    assert obs.legal_mask[ActionType.CALL] is True


# ---------------------------------------------------------------------------
# step(): fold
# ---------------------------------------------------------------------------

def test_hero_fold_ends_hand_with_negative_reward():
    env = FullHoldemEnv()
    obs = env.reset(make_heads_up_scenario())

    obs, reward, done, info = env.step(AbstractAction(ActionType.FOLD))

    assert done is True
    assert obs.is_terminal is True
    assert not any(obs.legal_mask)
    # Hero posted BB=2 preflop; folding loses exactly that.
    assert reward == -2.0
    assert info["phase"] == "HAND_COMPLETE"


# ---------------------------------------------------------------------------
# step(): check through to showdown
# ---------------------------------------------------------------------------

def test_check_check_reaches_showdown_and_terminates():
    env = FullHoldemEnv(default_villain_policy=CheckFoldVillainPolicy())
    scenario = make_heads_up_scenario()
    # Make bet_to_call 0 so hero can check preflop (simplifies this scenario
    # into a pure check/showdown path across all streets).
    scenario = BaseScenario(
        game_name="holdem",
        street_index=0,
        pot=0,
        dealer_position=0,
        current_player=1,
        bet_to_call=0,
        min_raise=2,
        players=(
            PlayerScenario(stack=199, hand_mask=0b11, current_bet=0,
                            total_contribution=0, has_folded=False, is_all_in=False),
            PlayerScenario(stack=198, hand_mask=(0b1111 << 4), current_bet=0,
                            total_contribution=0, has_folded=False, is_all_in=False),
        ),
        node_cards=tuple(None for _ in range(5)),
        small_blind=1,
        big_blind=2,
    )
    obs = env.reset(scenario)
    assert obs.legal_mask[ActionType.CHECK] is True

    done = False
    steps = 0
    while not done and steps < 20:
        obs, reward, done, info = env.step(AbstractAction(ActionType.CHECK))
        steps += 1

    assert done is True
    assert info["phase"] in ("SHOWDOWN", "HAND_COMPLETE")
    # Villain (seat 1) has the higher hand_mask -> wins the (empty) pot.
    assert reward == 0.0


# ---------------------------------------------------------------------------
# step(): bet, villain folds
# ---------------------------------------------------------------------------

def test_hero_bet_villain_folds_awards_pot():
    env = FullHoldemEnv(default_villain_policy=FoldAnyVillainPolicy())
    scenario = BaseScenario(
        game_name="holdem",
        street_index=0,
        pot=0,
        dealer_position=0,
        current_player=1,
        bet_to_call=0,
        min_raise=2,
        players=(
            PlayerScenario(stack=199, hand_mask=0b11, current_bet=0,
                            total_contribution=0, has_folded=False, is_all_in=False),
            PlayerScenario(stack=198, hand_mask=(0b1111 << 4), current_bet=0,
                            total_contribution=0, has_folded=False, is_all_in=False),
        ),
        node_cards=tuple(None for _ in range(5)),
        small_blind=1,
        big_blind=2,
    )
    obs = env.reset(scenario)
    assert obs.legal_mask[ActionType.BET_100] is True

    obs, reward, done, info = env.step(AbstractAction(ActionType.BET_100))

    assert done is True
    # Pot was 0, hero bets pot-sized -> effectively min_raise (2) since
    # 100% of 0 pot = 0, clamped up to min_raise. Villain folds; hero wins
    # back exactly what they put in (no one else contributed).
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Illegal-turn / lifecycle guards
# ---------------------------------------------------------------------------

def test_step_before_reset_raises():
    env = FullHoldemEnv()
    with pytest.raises(RuntimeError):
        env.step(AbstractAction(ActionType.FOLD))


def test_step_after_done_raises():
    env = FullHoldemEnv()
    env.reset(make_heads_up_scenario())
    env.step(AbstractAction(ActionType.FOLD))
    with pytest.raises(RuntimeError):
        env.step(AbstractAction(ActionType.FOLD))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
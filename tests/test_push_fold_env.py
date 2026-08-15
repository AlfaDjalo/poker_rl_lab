"""
tests/test_push_fold_env_history.py — Tests for HistoryObs wiring in
envs/push_fold_env.py.
"""

from __future__ import annotations

import pytest

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.envs.push_fold_env import PushFoldEnv, HERO_SEAT
from poker_rl_lab.scenarios.base_scenario import BaseScenario, PlayerScenario


def make_scenario(
    shove_amount=100,
    small_blind=1,
    big_blind=2,
    starting_stack=200,
):
    shover = PlayerScenario(
        stack=0, hand_mask=0b11, current_bet=shove_amount,
        total_contribution=shove_amount, has_folded=False, is_all_in=True,
    )
    caller_bet = min(big_blind, starting_stack)
    caller = PlayerScenario(
        stack=starting_stack - caller_bet, hand_mask=0b1100,
        current_bet=caller_bet, total_contribution=caller_bet,
        has_folded=False, is_all_in=False,
    )
    return BaseScenario(
        game_name="holdem",
        street_index=0,
        pot=small_blind,
        dealer_position=0,
        current_player=1,
        bet_to_call=shove_amount,
        min_raise=shove_amount * 2,
        players=(shover, caller),
        node_cards=(),
        small_blind=small_blind,
        big_blind=big_blind,
        discard_pile=(),
        raises_this_street=1,
        last_aggressor=0,
    )


class TestPushFoldHistoryOnReset:
    def test_reset_seeds_villain_shove_event(self):
        sc = make_scenario(shove_amount=100, small_blind=1, big_blind=2)
        env = PushFoldEnv()
        obs = env.reset(sc)

        events = obs.history.events
        assert len(events) == 1
        shove = events[0]
        assert shove.seat == 0
        assert shove.action_type == int(ActionType.ALL_IN)
        assert shove.street_index == 0
        assert shove.is_aggressor is True
        assert shove.pot_fraction >= 0.0

    def test_shove_pot_fraction_scales_with_shove_size(self):
        small = make_scenario(shove_amount=10, small_blind=1, big_blind=2)
        big = make_scenario(shove_amount=1000, small_blind=1, big_blind=2)
        env = PushFoldEnv()

        small_frac = env.reset(small).history.events[0].pot_fraction
        env2 = PushFoldEnv()
        big_frac = env2.reset(big).history.events[0].pot_fraction
        assert big_frac > small_frac

    def test_history_event_validates_cleanly(self):
        """HistoryEventObs.__post_init__ must not raise for any generated scenario."""
        for shove in (2, 20, 200, 2000):
            sc = make_scenario(shove_amount=shove)
            env = PushFoldEnv()
            obs = env.reset(sc)
            assert obs.history.events[0].pot_fraction >= 0.0


class TestPushFoldHistoryOnStep:
    def test_fold_appends_hero_event_with_zero_pot_fraction(self):
        sc = make_scenario(shove_amount=100)
        env = PushFoldEnv()
        env.reset(sc)
        obs, reward, done, info = env.step(AbstractAction(ActionType.FOLD))

        events = obs.history.events
        assert len(events) == 2
        hero_event = events[-1]
        assert hero_event.seat == HERO_SEAT
        assert hero_event.action_type == int(ActionType.FOLD)
        assert hero_event.pot_fraction == 0.0
        assert hero_event.is_aggressor is False

    def test_call_appends_hero_event_with_positive_pot_fraction(self):
        sc = make_scenario(shove_amount=100)
        env = PushFoldEnv()
        env.reset(sc)
        obs, reward, done, info = env.step(AbstractAction(ActionType.CALL))

        events = obs.history.events
        assert len(events) == 2
        hero_event = events[-1]
        assert hero_event.action_type == int(ActionType.CALL)
        assert hero_event.pot_fraction > 0.0
        assert hero_event.is_aggressor is False

    def test_history_preserved_at_terminal_observation(self):
        sc = make_scenario(shove_amount=50)
        env = PushFoldEnv()
        env.reset(sc)
        obs, *_ = env.step(AbstractAction(ActionType.CALL))
        assert obs.is_terminal is True
        assert len(obs.history.events) == 2

    def test_history_events_length_never_exceeds_two(self):
        """Push-fold is exactly one hero decision after one villain shove."""
        sc = make_scenario(shove_amount=75)
        env = PushFoldEnv()
        env.reset(sc)
        obs, *_ = env.step(AbstractAction(ActionType.FOLD))
        assert len(obs.history.events) == 2

    def test_independent_episodes_do_not_share_history_state(self):
        """Regression guard: self._history must be reset, not appended across episodes."""
        env = PushFoldEnv()
        sc1 = make_scenario(shove_amount=50)
        env.reset(sc1)
        env.step(AbstractAction(ActionType.FOLD))

        sc2 = make_scenario(shove_amount=80)
        obs2 = env.reset(sc2)
        # Fresh reset should start with exactly the villain shove, not carry
        # over the previous episode's hero fold event.
        assert len(obs2.history.events) == 1
        assert obs2.history.events[0].seat == 0


if __name__ == "__main__":
    import sys
    import pytest as _pytest
    sys.exit(_pytest.main([__file__, "-v"]))
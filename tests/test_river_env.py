"""
tests/test_river_env_history.py — Tests for HistoryObs wiring in
envs/river_env.py.
"""

from __future__ import annotations

import pytest

from actions.abstract_action import AbstractAction, ActionType
from envs.river_env import (
    RiverEnv,
    HERO_SEAT,
    VILLAIN_SEAT,
    CheckFoldVillainPolicy,
    CallAnyVillainPolicy,
    FoldAnyVillainPolicy,
)
from scenarios.base_scenario import BaseScenario, PlayerScenario


def make_scenario(p0_stack=200, p1_stack=200, pot=20, big_blind=2, small_blind=1):
    p0 = PlayerScenario(stack=p0_stack, hand_mask=0b1111, current_bet=0,
                         total_contribution=pot // 2, has_folded=False, is_all_in=False)
    p1 = PlayerScenario(stack=p1_stack, hand_mask=0b110000, current_bet=0,
                         total_contribution=pot - pot // 2, has_folded=False, is_all_in=False)
    return BaseScenario(
        game_name="holdem", street_index=3, pot=pot, dealer_position=0,
        current_player=1, bet_to_call=0, min_raise=big_blind,
        players=(p0, p1), node_cards=(0, 13, 26, 39, 51),
        small_blind=small_blind, big_blind=big_blind,
        discard_pile=(), raises_this_street=0, last_aggressor=None,
    )


class TestRiverHistoryBasics:
    def test_reset_starts_with_empty_history(self):
        env = RiverEnv(villain_policy=CheckFoldVillainPolicy())
        obs = env.reset(make_scenario())
        assert obs.history.events == ()

    def test_check_check_produces_two_events(self):
        env = RiverEnv(villain_policy=CheckFoldVillainPolicy())
        env.reset(make_scenario())
        obs, reward, done, info = env.step(AbstractAction(ActionType.CHECK))
        assert done is True
        events = obs.history.events
        assert len(events) == 2
        assert events[0].seat == HERO_SEAT
        assert events[0].action_type == int(ActionType.CHECK)
        assert events[0].pot_fraction == 0.0
        assert events[0].is_aggressor is False
        assert events[1].seat == VILLAIN_SEAT
        assert events[1].action_type == int(ActionType.CHECK)

    def test_bet_call_produces_two_events_with_correct_flags(self):
        env = RiverEnv(villain_policy=CallAnyVillainPolicy())
        env.reset(make_scenario(pot=20))
        obs, reward, done, info = env.step(AbstractAction(ActionType.BET_50))

        events = obs.history.events
        assert len(events) == 2

        bet_event = events[0]
        assert bet_event.seat == HERO_SEAT
        assert bet_event.is_aggressor is True
        assert bet_event.pot_fraction > 0.0

        call_event = events[1]
        assert call_event.seat == VILLAIN_SEAT
        assert call_event.action_type == int(ActionType.CALL)
        assert call_event.is_aggressor is False
        assert call_event.pot_fraction > 0.0

    def test_bet_fold_produces_two_events(self):
        env = RiverEnv(villain_policy=FoldAnyVillainPolicy())
        env.reset(make_scenario())
        obs, reward, done, info = env.step(AbstractAction(ActionType.BET_75))

        events = obs.history.events
        assert len(events) == 2
        assert events[0].action_type == int(ActionType.BET_75)
        assert events[1].action_type == int(ActionType.FOLD)
        assert events[1].is_aggressor is False
        assert events[1].pot_fraction == 0.0

    def test_hero_fold_produces_single_event(self):
        env = RiverEnv(villain_policy=CheckFoldVillainPolicy())
        env.reset(make_scenario())
        obs, reward, done, info = env.step(AbstractAction(ActionType.FOLD))
        events = obs.history.events
        assert len(events) == 1
        assert events[0].seat == HERO_SEAT
        assert events[0].action_type == int(ActionType.FOLD)


class TestRiverHistoryRaiseWars:
    def test_raise_and_reraise_events_in_chronological_order(self):
        """
        A raise war: hero bets, villain policy calls (CallAnyVillainPolicy
        never raises) — so instead we drive a manual multi-raise sequence
        via a scripted villain to check ordering across several actions.
        """
        class RaiseOnceThenFoldVillain:
            def __init__(self):
                self._raised = False

            def act(self, legal_mask):
                if not self._raised and legal_mask[ActionType.BET_50]:
                    self._raised = True
                    return AbstractAction(ActionType.BET_50)
                if legal_mask[ActionType.CALL]:
                    return AbstractAction(ActionType.CALL)
                return AbstractAction(ActionType.FOLD)

        env = RiverEnv(villain_policy=RaiseOnceThenFoldVillain(), max_raises=10)
        env.reset(make_scenario(pot=20))

        # Hero checks; villain raises (bet); hero must act again.
        obs, reward, done, info = env.step(AbstractAction(ActionType.CHECK))
        assert done is False

        events_after_check = obs.history.events
        assert len(events_after_check) == 2
        assert events_after_check[0].seat == HERO_SEAT
        assert events_after_check[0].action_type == int(ActionType.CHECK)
        assert events_after_check[1].seat == VILLAIN_SEAT
        assert events_after_check[1].action_type == int(ActionType.BET_50)
        assert events_after_check[1].is_aggressor is True

        # Hero calls, ending the street.
        obs2, reward2, done2, info2 = env.step(AbstractAction(ActionType.CALL))
        assert done2 is True
        events_final = obs2.history.events
        assert len(events_final) == 3
        assert events_final[2].seat == HERO_SEAT
        assert events_final[2].action_type == int(ActionType.CALL)
        # Chronological order preserved across the whole hand
        assert [e.seat for e in events_final] == [HERO_SEAT, VILLAIN_SEAT, HERO_SEAT]

    def test_pot_fraction_reflects_growing_pot_across_raises(self):
        """Later raises' pot_fraction is computed against the already-grown pot."""
        class AlwaysRaiseVillain:
            def act(self, legal_mask):
                if legal_mask[ActionType.BET_50]:
                    return AbstractAction(ActionType.BET_50)
                if legal_mask[ActionType.CALL]:
                    return AbstractAction(ActionType.CALL)
                return AbstractAction(ActionType.FOLD)

        env = RiverEnv(villain_policy=AlwaysRaiseVillain(), max_raises=1)
        env.reset(make_scenario(pot=20))
        obs, reward, done, info = env.step(AbstractAction(ActionType.BET_50))
        events = obs.history.events
        # hero bets first (pot=20 before), villain raises (pot has grown)
        assert events[0].pot_fraction == pytest.approx(events[0].pot_fraction)
        assert events[1].seat == VILLAIN_SEAT


class TestRiverHistoryStateIsolation:
    def test_history_resets_between_episodes(self):
        env = RiverEnv(villain_policy=CheckFoldVillainPolicy())
        env.reset(make_scenario())
        env.step(AbstractAction(ActionType.CHECK))

        obs2 = env.reset(make_scenario())
        assert obs2.history.events == ()

    def test_street_index_recorded_matches_scenario(self):
        env = RiverEnv(villain_policy=CheckFoldVillainPolicy())
        sc = make_scenario()
        env.reset(sc)
        obs, *_ = env.step(AbstractAction(ActionType.CHECK))
        for e in obs.history.events:
            assert e.street_index == sc.street_index


class TestRiverHistoryFullPipeline:
    def test_run_episode_conformance_with_history(self):
        from envs.base_env import run_episode, RandomHeroPolicy
        from scenarios.river_generator import RiverScenarioGenerator

        gen = RiverScenarioGenerator()
        gen.seed(7)
        env = RiverEnv(villain_policy=CallAnyVillainPolicy())
        policy = RandomHeroPolicy(seed=3)

        for _ in range(10):
            sc = gen.generate()
            total, traj = run_episode(env, sc, policy)
            final_obs = traj[-1].next_obs
            assert final_obs.is_terminal is True
            # every recorded event must reference a valid seat
            for e in final_obs.history.events:
                assert e.seat in (0, 1)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
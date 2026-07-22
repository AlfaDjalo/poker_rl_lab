"""
tests/test_position_encoder.py — Unit tests for representation/position_encoder.py.

Covers:
  - PositionFeatures validation (rejects out-of-range fractions / table_size)
  - position_features_from_scenario() against hand-computed expected values,
    including the heads-up (n==2) small-blind special case and a 4-handed
    table with a folded seat
  - position_features_from_scenario() vs position_features_from_observation()
    agreement at reset() time (they must never silently drift apart)
  - PositionEncoder shape/determinism contracts
"""

import math

import pytest
import torch

from scenarios.base_scenario import BaseScenario, PlayerScenario
from scenarios.push_fold_generator import PushFoldScenarioGenerator
from scenarios.river_generator import RiverScenarioGenerator
from envs.push_fold_env import PushFoldEnv
from envs.river_env import RiverEnv
from representation.position_encoder import (
    PositionFeatures,
    PositionEncoder,
    position_features_from_scenario,
    position_features_from_observation,
)


# ---------------------------------------------------------------------------
# PositionFeatures validation
# ---------------------------------------------------------------------------

class TestPositionFeaturesValidation:

    def test_valid_construction_does_not_raise(self):
        PositionFeatures(
            seat_fraction=0.25,
            active_fraction=0.5,
            to_act_after_fraction=1.0,
            table_size=4,
        )

    @pytest.mark.parametrize("bad_seat_fraction", [-0.01, 1.0, 1.5])
    def test_rejects_seat_fraction_out_of_range(self, bad_seat_fraction):
        with pytest.raises(ValueError):
            PositionFeatures(
                seat_fraction=bad_seat_fraction,
                active_fraction=0.5,
                to_act_after_fraction=0.0,
                table_size=4,
            )

    @pytest.mark.parametrize("bad_active_fraction", [0.0, -0.1, 1.1])
    def test_rejects_active_fraction_out_of_range(self, bad_active_fraction):
        with pytest.raises(ValueError):
            PositionFeatures(
                seat_fraction=0.0,
                active_fraction=bad_active_fraction,
                to_act_after_fraction=0.0,
                table_size=4,
            )

    @pytest.mark.parametrize("bad_to_act_after", [-0.1, 1.1])
    def test_rejects_to_act_after_fraction_out_of_range(self, bad_to_act_after):
        with pytest.raises(ValueError):
            PositionFeatures(
                seat_fraction=0.0,
                active_fraction=1.0,
                to_act_after_fraction=bad_to_act_after,
                table_size=4,
            )

    def test_rejects_table_size_below_two(self):
        with pytest.raises(ValueError):
            PositionFeatures(
                seat_fraction=0.0,
                active_fraction=1.0,
                to_act_after_fraction=0.0,
                table_size=1,
            )


# ---------------------------------------------------------------------------
# Manual BaseScenario construction helpers
# ---------------------------------------------------------------------------

def _make_scenario(
    dealer_position: int,
    current_player: int,
    has_folded: list,
    is_all_in: list = None,
) -> BaseScenario:
    """Build a minimal, otherwise-arbitrary BaseScenario for position math only."""
    n = len(has_folded)
    is_all_in = is_all_in or [False] * n
    players = tuple(
        PlayerScenario(
            stack=100,
            hand_mask=0,
            current_bet=0,
            total_contribution=0,
            has_folded=has_folded[i],
            is_all_in=is_all_in[i],
        )
        for i in range(n)
    )
    return BaseScenario(
        game_name="holdem",
        street_index=1,
        pot=10,
        dealer_position=dealer_position,
        current_player=current_player,
        bet_to_call=0,
        min_raise=2,
        players=players,
        node_cards=tuple(),
    )


# ---------------------------------------------------------------------------
# position_features_from_scenario — hand-computed expected values
# ---------------------------------------------------------------------------

class TestPositionFeaturesFromScenarioHeadsUp:
    """n == 2: button is the small blind (see module docstring)."""

    def test_button_seat_is_small_blind(self):
        sc = _make_scenario(dealer_position=0, current_player=0, has_folded=[False, False])
        f = position_features_from_scenario(sc, seat=0)
        assert f.is_button is True
        assert f.is_small_blind is True
        assert f.is_big_blind is False
        assert f.seat_fraction == pytest.approx(0.0)
        assert f.table_size == 2

    def test_other_seat_is_big_blind(self):
        sc = _make_scenario(dealer_position=0, current_player=0, has_folded=[False, False])
        f = position_features_from_scenario(sc, seat=1)
        assert f.is_button is False
        assert f.is_small_blind is False
        assert f.is_big_blind is True
        assert f.seat_fraction == pytest.approx(0.5)


class TestPositionFeaturesFromScenarioFourHanded:
    """
    n == 4, dealer=1, current_player=3, seat 2 folded.
    Expected values hand-derived in the accompanying design notes:
      rotation (active, not all-in) = [0, 1, 3]
      ordered from current_player=3 -> [3, 0, 1]
    """

    def _scenario(self):
        return _make_scenario(
            dealer_position=1,
            current_player=3,
            has_folded=[False, False, True, False],
        )

    def test_dealer_seat_is_button(self):
        f = position_features_from_scenario(self._scenario(), seat=1)
        assert f.is_button is True
        assert f.seat_fraction == pytest.approx(0.0)

    def test_blind_seats(self):
        sc = self._scenario()
        assert position_features_from_scenario(sc, seat=2).is_small_blind is True
        assert position_features_from_scenario(sc, seat=3).is_big_blind is True

    def test_active_fraction_excludes_folded_seat(self):
        f = position_features_from_scenario(self._scenario(), seat=0)
        assert f.active_fraction == pytest.approx(3 / 4)

    def test_current_player_is_next_to_act_and_maximally_behind(self):
        f = position_features_from_scenario(self._scenario(), seat=3)
        assert f.is_next_to_act is True
        # everyone else still active (0, 1) acts after seat 3 this rotation
        assert f.to_act_after_fraction == pytest.approx(1.0)

    def test_seat_just_before_current_player_in_rotation_has_zero_to_act_after(self):
        # rotation order from current_player(3) is [3, 0, 1]; seat 1 is last
        f = position_features_from_scenario(self._scenario(), seat=1)
        assert f.is_next_to_act is False
        assert f.to_act_after_fraction == pytest.approx(0.0)

    def test_folded_seat_has_zero_to_act_after(self):
        f = position_features_from_scenario(self._scenario(), seat=2)
        assert f.to_act_after_fraction == pytest.approx(0.0)

    def test_seat_out_of_range_raises(self):
        with pytest.raises(ValueError):
            position_features_from_scenario(self._scenario(), seat=4)


# ---------------------------------------------------------------------------
# Scenario adapter vs Observation adapter agreement (through real envs)
# ---------------------------------------------------------------------------

class TestScenarioObservationAgreement:
    """
    At reset() time (the only point where both a BaseScenario and a
    StructuredObservation exist for the same state), the two adapters
    must produce identical features for the hero seat. This is the
    regression test for the "must never silently drift apart" concern
    raised in the module docstring.
    """

    def test_push_fold_env_agreement(self):
        gen = PushFoldScenarioGenerator(player_count=2)
        gen.seed(7)
        scenario = gen.generate()

        env = PushFoldEnv()
        obs = env.reset(scenario)

        from_scenario = position_features_from_scenario(scenario, seat=obs.hero_seat)
        from_obs = position_features_from_observation(obs)

        assert from_scenario == from_obs

    def test_river_env_agreement(self):
        gen = RiverScenarioGenerator(player_count=2)
        gen.seed(11)
        scenario = gen.generate()

        env = RiverEnv()
        obs = env.reset(scenario)

        from_scenario = position_features_from_scenario(scenario, seat=obs.hero_seat)
        from_obs = position_features_from_observation(obs)

        assert from_scenario == from_obs

    def test_dealer_position_is_threaded_through_observation(self):
        gen = RiverScenarioGenerator(player_count=2)
        gen.seed(3)
        scenario = gen.generate()

        env = RiverEnv()
        obs = env.reset(scenario)

        assert obs.dealer_position == scenario.dealer_position


# ---------------------------------------------------------------------------
# PositionEncoder
# ---------------------------------------------------------------------------

class TestPositionEncoder:

    def _sample_features(self) -> PositionFeatures:
        return PositionFeatures(
            seat_fraction=0.25,
            active_fraction=0.75,
            to_act_after_fraction=0.5,
            is_button=False,
            is_small_blind=True,
            is_big_blind=False,
            is_next_to_act=True,
            table_size=4,
        )

    def test_forward_single_output_shape(self):
        encoder = PositionEncoder(hidden_dim=16, output_dim=12)
        out = encoder(self._sample_features())
        assert out.shape == (12,)

    def test_forward_batch_output_shape(self):
        encoder = PositionEncoder(hidden_dim=16, output_dim=12)
        batch = [self._sample_features() for _ in range(5)]
        out = encoder.forward_batch(batch)
        assert out.shape == (5, 12)

    def test_forward_batch_rejects_empty(self):
        encoder = PositionEncoder()
        with pytest.raises(ValueError):
            encoder.forward_batch([])

    def test_identical_features_produce_identical_embeddings(self):
        encoder = PositionEncoder(hidden_dim=8, output_dim=8)
        encoder.eval()
        f = self._sample_features()
        out1 = encoder(f)
        out2 = encoder(f)
        assert torch.allclose(out1, out2)

    def test_output_is_finite(self):
        encoder = PositionEncoder()
        out = encoder(self._sample_features())
        assert torch.all(torch.isfinite(out))

    def test_rejects_nonpositive_dims(self):
        with pytest.raises(ValueError):
            PositionEncoder(hidden_dim=0)
        with pytest.raises(ValueError):
            PositionEncoder(output_dim=0)
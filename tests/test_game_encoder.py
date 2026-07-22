"""
tests/test_game_encoder.py — Unit tests for representation/game_encoder.py

Run with: pytest tests/test_game_encoder.py -v

These tests deliberately do NOT import the real poker_eval C++ extension.
game_features_from_definition() is documented to duck-type on whatever
GameDefinition/GameRules-shaped objects are passed in, so we exercise it
with plain SimpleNamespace stand-ins (mirroring mock_poker_eval.py's
ScoreType/ShowdownType values) rather than requiring the compiled engine
or the real rules/game_definition.py + rules/game_rules.py to be present.
"""

from __future__ import annotations

from types import SimpleNamespace

import math
import pytest
import torch

from representation.game_encoder import (
    GameEncoder,
    GameFeatures,
    NUM_BETTING_TYPES,
    NUM_PAYOUT_TYPES,
    NUM_SHOWDOWN_TYPES,
    game_features_from_definition,
)

# Mirrors mock_poker_eval.ScoreType / poker_eval.ScoreType by value.
HIGH, LOW_A5, LOW_27, BADUGI = 0, 1, 2, 3
# Mirrors mock_poker_eval.ShowdownType / poker_eval.ShowdownType by value.
HOLDEM, OMAHA, MAKE5, DRAW, BADUGI_SD = 0, 1, 2, 3, 4


def make_game_def(**overrides):
    defaults = dict(
        hole_cards=2,
        node_count=5,
        street_nodes=[[], [0, 1, 2], [3], [4]],
        betting_type="no_limit",
        small_blind=1,
        big_blind=2,
        ante=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_rules(**overrides):
    defaults = dict(
        score_types=[HIGH],
        showdown_type=HOLDEM,
        points=[SimpleNamespace(name="main")],
        payout_type="points",
        low_qualifier=8,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# GameFeatures validation
# ---------------------------------------------------------------------------

class TestGameFeaturesValidation:
    def test_valid_construction(self):
        gf = GameFeatures(
            betting_type_id=0,
            payout_type_id=0,
            showdown_type=HOLDEM,
            hole_cards=2,
            node_count=5,
            num_streets=4,
            num_points=1,
        )
        assert gf.hole_cards == 2
        assert gf.has_ante is False
        assert gf.ante_bb == 0.0

    @pytest.mark.parametrize("bad_id", [-1, NUM_BETTING_TYPES])
    def test_invalid_betting_type_id(self, bad_id):
        with pytest.raises(ValueError):
            GameFeatures(
                betting_type_id=bad_id, payout_type_id=0, showdown_type=HOLDEM,
                hole_cards=2, node_count=5, num_streets=4, num_points=1,
            )

    @pytest.mark.parametrize("bad_id", [-1, NUM_PAYOUT_TYPES])
    def test_invalid_payout_type_id(self, bad_id):
        with pytest.raises(ValueError):
            GameFeatures(
                betting_type_id=0, payout_type_id=bad_id, showdown_type=HOLDEM,
                hole_cards=2, node_count=5, num_streets=4, num_points=1,
            )

    @pytest.mark.parametrize("bad_id", [-1, NUM_SHOWDOWN_TYPES])
    def test_invalid_showdown_type(self, bad_id):
        with pytest.raises(ValueError):
            GameFeatures(
                betting_type_id=0, payout_type_id=0, showdown_type=bad_id,
                hole_cards=2, node_count=5, num_streets=4, num_points=1,
            )

    def test_negative_scalar_fields_rejected(self):
        for field in ("hole_cards", "node_count", "num_streets", "num_points"):
            kwargs = dict(
                betting_type_id=0, payout_type_id=0, showdown_type=HOLDEM,
                hole_cards=2, node_count=5, num_streets=4, num_points=1,
            )
            kwargs[field] = -1
            with pytest.raises(ValueError):
                GameFeatures(**kwargs)

    def test_negative_ante_bb_rejected(self):
        with pytest.raises(ValueError):
            GameFeatures(
                betting_type_id=0, payout_type_id=0, showdown_type=HOLDEM,
                hole_cards=2, node_count=5, num_streets=4, num_points=1,
                ante_bb=-0.5,
            )


# ---------------------------------------------------------------------------
# game_features_from_definition adapter
# ---------------------------------------------------------------------------

class TestAdapter:
    def test_holdem_like_game(self):
        gd = make_game_def()
        rules = make_rules()
        gf = game_features_from_definition(gd, rules)

        assert gf.betting_type_id == 0          # "no_limit" -> vocab index 0
        assert gf.payout_type_id == 0            # "points" -> vocab index 0
        assert gf.showdown_type == HOLDEM
        assert gf.hole_cards == 2
        assert gf.node_count == 5
        assert gf.num_streets == 4
        assert gf.num_points == 1
        assert gf.has_ante is False
        assert gf.ante_bb == 0.0
        assert gf.has_low is False

    def test_low_game_flagged(self):
        gd = make_game_def(hole_cards=4)
        rules = make_rules(score_types=[HIGH, LOW_A5])
        gf = game_features_from_definition(gd, rules)
        assert gf.has_low is True

    def test_ante_bb_computed(self):
        gd = make_game_def(ante=5, big_blind=2)
        rules = make_rules()
        gf = game_features_from_definition(gd, rules)
        assert gf.has_ante is True
        assert gf.ante_bb == pytest.approx(2.5)

    def test_ante_with_zero_big_blind_does_not_divide_by_zero(self):
        gd = make_game_def(ante=5, big_blind=0)
        rules = make_rules()
        gf = game_features_from_definition(gd, rules)
        assert gf.ante_bb == 0.0

    def test_unknown_betting_type_falls_back_to_other_bucket(self):
        gd = make_game_def(betting_type="short_deck_special")
        rules = make_rules()
        gf = game_features_from_definition(gd, rules)
        assert gf.betting_type_id == NUM_BETTING_TYPES - 1

    def test_unknown_payout_type_falls_back_to_other_bucket(self):
        gd = make_game_def()
        rules = make_rules(payout_type="weird_new_payout")
        gf = game_features_from_definition(gd, rules)
        assert gf.payout_type_id == NUM_PAYOUT_TYPES - 1

    def test_showdown_type_override_not_used_here(self):
        # game_encoder embeds the GAME's default showdown_type, not any
        # per-point override (that's point_encoder's job) — sanity check
        # that this adapter just reads rules.showdown_type directly.
        gd = make_game_def()
        rules = make_rules(showdown_type=OMAHA)
        gf = game_features_from_definition(gd, rules)
        assert gf.showdown_type == OMAHA

    def test_double_board_style_game(self):
        gd = make_game_def(
            hole_cards=4, node_count=10,
            street_nodes=[[], [0, 1, 2, 5, 6, 7], [3, 8], [4, 9]],
        )
        rules = make_rules(
            showdown_type=OMAHA,
            points=[SimpleNamespace(name="board1"), SimpleNamespace(name="board2"),
                    SimpleNamespace(name="hand")],
        )
        gf = game_features_from_definition(gd, rules)
        assert gf.node_count == 10
        assert gf.num_points == 3


# ---------------------------------------------------------------------------
# GameEncoder forward passes
# ---------------------------------------------------------------------------

class TestGameEncoderForward:
    def test_single_forward_shape(self):
        enc = GameEncoder(output_dim=16)
        gf = GameFeatures(
            betting_type_id=0, payout_type_id=0, showdown_type=HOLDEM,
            hole_cards=2, node_count=5, num_streets=4, num_points=1,
        )
        out = enc(gf)
        assert out.shape == (16,)

    def test_batch_forward_shape(self):
        enc = GameEncoder(output_dim=16)
        games = [
            GameFeatures(0, 0, HOLDEM, 2, 5, 4, 1),
            GameFeatures(1, 0, OMAHA, 4, 10, 4, 3, has_ante=True, ante_bb=1.5),
            GameFeatures(0, 1, MAKE5, 4, 10, 3, 2, has_low=True),
        ]
        out = enc.forward_batch(games)
        assert out.shape == (3, 16)

    def test_forward_matches_forward_batch_single(self):
        enc = GameEncoder(output_dim=16)
        gf = GameFeatures(0, 0, HOLDEM, 2, 5, 4, 1)
        single = enc(gf)
        batched = enc.forward_batch([gf])[0]
        assert torch.allclose(single, batched)

    def test_empty_batch_raises(self):
        enc = GameEncoder(output_dim=16)
        with pytest.raises(ValueError):
            enc.forward_batch([])

    def test_default_output_dim(self):
        enc = GameEncoder()
        assert enc.output_dim == 16

    def test_custom_dims_respected(self):
        enc = GameEncoder(
            betting_type_dim=4, payout_type_dim=4, showdown_type_dim=4,
            scalar_hidden_dim=8, output_dim=32,
        )
        gf = GameFeatures(0, 0, HOLDEM, 2, 5, 4, 1)
        out = enc(gf)
        assert out.shape == (32,)

    def test_gradients_flow_to_all_embedding_tables(self):
        enc = GameEncoder(output_dim=16)
        games = [
            GameFeatures(0, 0, HOLDEM, 2, 5, 4, 1),
            GameFeatures(2, 1, DRAW, 5, 8, 3, 2, has_ante=True, ante_bb=0.5),
        ]
        out = enc.forward_batch(games)
        out.sum().backward()

        assert enc.betting_type_embed.weight.grad is not None
        assert enc.payout_type_embed.weight.grad is not None
        assert enc.showdown_type_embed.weight.grad is not None
        assert not torch.all(enc.betting_type_embed.weight.grad == 0)

    def test_different_games_produce_different_embeddings(self):
        torch.manual_seed(0)
        enc = GameEncoder(output_dim=16)
        holdem = GameFeatures(0, 0, HOLDEM, 2, 5, 4, 1)
        double_board = GameFeatures(0, 0, OMAHA, 4, 10, 3, 3)
        out = enc.forward_batch([holdem, double_board])
        assert not torch.allclose(out[0], out[1])

    def test_end_to_end_from_definition_through_encoder(self):
        """Full pipeline: engine-shaped objects -> GameFeatures -> embedding."""
        torch.manual_seed(0)
        gd = make_game_def(hole_cards=4, node_count=10, ante=5, big_blind=2)
        rules = make_rules(showdown_type=OMAHA, score_types=[HIGH, LOW_A5])
        gf = game_features_from_definition(gd, rules)

        enc = GameEncoder(output_dim=16)
        out = enc(gf)
        assert out.shape == (16,)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
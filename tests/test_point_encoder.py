"""
tests/test_point_encoder.py — Unit tests for representation/point_encoder.py

Run with:
    python -m pytest rl_lab/tests/test_point_encoder.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from poker_rl_lab.representation.point_encoder import (
    PointEncoder,
    PointFeatures,
    point_features_from_definition,
    NUM_SCORE_TYPES,
    NUM_SHOWDOWN_TYPES,
)

OUTPUT_DIM = 16


@pytest.fixture
def encoder():
    return PointEncoder(output_dim=OUTPUT_DIM)


HIGH_POINT = PointFeatures(score_type=0, showdown_type=0)  # HIGH / HOLDEM
LOW_A5_POINT = PointFeatures(
    score_type=1, showdown_type=0, is_low_qualified=True, qualifier_strength=0.7
)


# ── PointFeatures validation ─────────────────────────────────────────────────

def test_point_features_rejects_bad_score_type():
    with pytest.raises(ValueError):
        PointFeatures(score_type=NUM_SCORE_TYPES, showdown_type=0)


def test_point_features_rejects_bad_showdown_type():
    with pytest.raises(ValueError):
        PointFeatures(score_type=0, showdown_type=NUM_SHOWDOWN_TYPES)


def test_point_features_rejects_bad_qualifier_strength():
    with pytest.raises(ValueError):
        PointFeatures(score_type=1, showdown_type=0, qualifier_strength=1.5)


def test_point_features_rejects_zero_node_sets():
    with pytest.raises(ValueError):
        PointFeatures(score_type=0, showdown_type=0, num_node_sets=0)


# ── Forward ──────────────────────────────────────────────────────────────────

def test_single_forward_shape(encoder):
    out = encoder(HIGH_POINT)
    assert out.shape == (OUTPUT_DIM,)


def test_batch_forward_shape(encoder):
    out = encoder.forward_batch([HIGH_POINT, LOW_A5_POINT])
    assert out.shape == (2, OUTPUT_DIM)


def test_batch_matches_single(encoder):
    single = encoder(HIGH_POINT)
    batch = encoder.forward_batch([HIGH_POINT])
    assert torch.allclose(batch[0], single)


def test_empty_batch_raises(encoder):
    with pytest.raises(ValueError):
        encoder.forward_batch([])


def test_high_and_low_produce_different_embeddings(encoder):
    high = encoder(HIGH_POINT)
    low = encoder(LOW_A5_POINT)
    assert not torch.allclose(high, low)


def test_qualifier_strength_affects_embedding(encoder):
    """Same score_type/showdown_type but different qualifier tightness
    must produce different embeddings — otherwise an 8-or-better low and
    a loose A5 low would be indistinguishable to the policy."""
    loose = PointFeatures(
        score_type=1, showdown_type=0, is_low_qualified=True, qualifier_strength=0.1
    )
    tight = PointFeatures(
        score_type=1, showdown_type=0, is_low_qualified=True, qualifier_strength=0.9
    )
    out_loose = encoder(loose)
    out_tight = encoder(tight)
    assert not torch.allclose(out_loose, out_tight)


def test_scoop_pairing_affects_embedding(encoder):
    no_scoop = PointFeatures(score_type=1, showdown_type=0, has_scoop_pairing=False)
    scoop = PointFeatures(score_type=1, showdown_type=0, has_scoop_pairing=True)
    assert not torch.allclose(encoder(no_scoop), encoder(scoop))


def test_gradients_flow(encoder):
    out = encoder(HIGH_POINT)
    out.sum().backward()
    assert encoder.score_type_embed.weight.grad is not None
    assert encoder.score_type_embed.weight.grad.abs().sum().item() > 0
    assert encoder.showdown_type_embed.weight.grad is not None


def test_invalid_output_dim_raises():
    with pytest.raises(ValueError):
        PointEncoder(output_dim=0)


# ── Engine adapter (using mock_poker_eval-style plain objects) ──────────────

class _FakePointDef:
    """Duck-types rules.point_definition.PointDefinition's public fields."""
    def __init__(self, score_type, node_sets, showdown_type_override=None, scoop_from=None):
        self.score_type = score_type
        self.node_sets = node_sets
        self.showdown_type_override = showdown_type_override
        self.scoop_from = scoop_from


class _FakeRules:
    """Duck-types rules.game_rules.GameRules's public fields/methods."""
    def __init__(self, showdown_type, low_qualifier=None):
        self.showdown_type = showdown_type
        self.low_qualifier = low_qualifier

    def is_low_type(self, score_type):
        return score_type in (1, 2)  # LOW_A5, LOW_27


def test_adapter_high_point_no_qualifier():
    point_def = _FakePointDef(score_type=0, node_sets=[(0, 1, 2, 3, 4)])
    rules = _FakeRules(showdown_type=0)
    feats = point_features_from_definition(point_def, rules)
    assert feats.score_type == 0
    assert feats.showdown_type == 0
    assert feats.is_low_qualified is False
    assert feats.qualifier_strength == 0.0
    assert feats.num_node_sets == 1


def test_adapter_low_point_with_qualifier():
    point_def = _FakePointDef(score_type=1, node_sets=[(0, 1, 2, 3, 4)], scoop_from="high")
    rules = _FakeRules(showdown_type=0, low_qualifier=8)
    feats = point_features_from_definition(point_def, rules)
    assert feats.is_low_qualified is True
    assert 0.0 < feats.qualifier_strength <= 1.0
    assert feats.has_scoop_pairing is True


def test_adapter_showdown_type_override():
    point_def = _FakePointDef(score_type=0, node_sets=[(0, 1, 2)], showdown_type_override=2)
    rules = _FakeRules(showdown_type=0)  # game default is 0, override should win
    feats = point_features_from_definition(point_def, rules)
    assert feats.showdown_type == 2


def test_adapter_multiple_node_sets():
    point_def = _FakePointDef(score_type=0, node_sets=[(0, 1, 2, 3, 4), (5, 6, 7, 3, 4)])
    rules = _FakeRules(showdown_type=1)
    feats = point_features_from_definition(point_def, rules)
    assert feats.num_node_sets == 2


def test_adapter_output_feeds_encoder(encoder):
    """End-to-end: adapter output must be directly usable by PointEncoder."""
    point_def = _FakePointDef(score_type=1, node_sets=[(0, 1, 2, 3, 4)])
    rules = _FakeRules(showdown_type=0, low_qualifier=8)
    feats = point_features_from_definition(point_def, rules)
    out = encoder(feats)
    assert out.shape == (OUTPUT_DIM,)
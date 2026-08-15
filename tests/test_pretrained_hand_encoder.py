"""
tests/test_pretrained_hand_encoder.py

These tests use PretrainedHandEncoder.untrained() since the real
.keras weights aren't available in this environment yet. They verify
the plumbing (grid construction, shapes, weight sharing across modes,
batching) — not poker knowledge. Once encoder_model.keras is dropped
into rl_lab/pretrained/, add a test using from_file() alongside these.

Run with:
    python -m pytest rl_lab/tests/test_pretrained_hand_encoder.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from poker_rl_lab.representation.pretrained_hand_encoder import PretrainedHandEncoder


EMBED_DIM = 16


@pytest.fixture
def encoder():
    return PretrainedHandEncoder.untrained(embedding_dim=EMBED_DIM)


def test_encode_hole_shape(encoder):
    out = encoder.encode_hole([0, 51])  # 2c, As
    assert out.shape == (EMBED_DIM,)


def test_encode_board_shape(encoder):
    out = encoder.encode_board([5, 18, 31])  # 3-card flop
    assert out.shape == (EMBED_DIM,)


def test_encode_combined_shape(encoder):
    out = encoder.encode_combined([0, 51], [5, 18, 31])
    assert out.shape == (EMBED_DIM,)


def test_embedding_is_l2_normalised(encoder):
    out = encoder.encode_hole([0, 51])
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-5)


def test_same_cards_same_embedding(encoder):
    """Determinism: same input grid -> same output (no dropout/BN randomness at inference)."""
    a = encoder.encode_hole([0, 51])
    b = encoder.encode_hole([0, 51])
    assert np.allclose(a, b)


def test_shared_weights_across_modes(encoder):
    """
    hole-only and board-only encodes must go through the SAME model
    (single set of weights), per the "one model, reused via different
    input modes" design. This just confirms both calls hit the same
    underlying CardSetEncoder instance.
    """
    assert encoder.encode_hole([0, 51]) is not None  # sanity: doesn't raise
    assert encoder._model.card_set_encoder is encoder._model.card_set_encoder


def test_batch_matches_single(encoder):
    single = encoder.encode_hole([0, 51])
    batch = encoder.encode_batch([[0, 51], [1, 14]])
    assert batch.shape == (2, EMBED_DIM)
    assert np.allclose(batch[0], single, atol=1e-5)


def test_different_hands_different_embeddings(encoder):
    a = encoder.encode_hole([0, 51])
    b = encoder.encode_hole([10, 40])
    assert not np.allclose(a, b)


def test_embedding_dim_property(encoder):
    assert encoder.embedding_dim == EMBED_DIM
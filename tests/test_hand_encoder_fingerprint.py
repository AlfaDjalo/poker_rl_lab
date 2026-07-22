"""
tests/test_hand_encoder_fingerprint.py — Tests for the from_file()
architecture fingerprint check in pretrained_hand_encoder.py.

Since the real encoder_model.keras isn't available in this environment,
these tests save/reload an untrained-but-architecturally-real
CardStateEncoder to/from an actual .keras file, so from_file() and its
validation run against a genuine Keras artifact rather than mocks.

Run with:
    python -m pytest rl_lab/tests/test_hand_encoder_fingerprint.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import tensorflow as tf

from representation.encoders import CardStateEncoder
from representation.pretrained_hand_encoder import (
    PretrainedHandEncoder,
    HandEncoderArchitectureMismatch,
)
from representation.card_utils import GRID_SHAPE

EMBED_DIM = 16


@pytest.fixture
def saved_model_path(tmp_path):
    """Build a real (untrained) CardStateEncoder and save it as .keras,
    so from_file() has a genuine file to load."""
    config = {"encoder_input_shape": GRID_SHAPE, "embedding_dim": EMBED_DIM}
    model = CardStateEncoder(config)
    # Build the model by running one dummy call before saving.
    dummy = tf.zeros((1,) + GRID_SHAPE)
    model(dummy, training=False)

    path = tmp_path / "test_encoder.keras"
    model.save(str(path))
    return str(path)


def test_from_file_loads_and_validates(saved_model_path):
    encoder = PretrainedHandEncoder.from_file(saved_model_path)
    assert encoder.embedding_dim == EMBED_DIM


def test_from_file_matches_expected_dim(saved_model_path):
    encoder = PretrainedHandEncoder.from_file(
        saved_model_path, expected_embedding_dim=EMBED_DIM
    )
    assert encoder.embedding_dim == EMBED_DIM


def test_from_file_rejects_wrong_expected_dim(saved_model_path):
    with pytest.raises(HandEncoderArchitectureMismatch, match="embedding_dim"):
        PretrainedHandEncoder.from_file(
            saved_model_path, expected_embedding_dim=EMBED_DIM + 1
        )


def test_from_file_matches_expected_input_shape(saved_model_path):
    encoder = PretrainedHandEncoder.from_file(
        saved_model_path, expected_input_shape=GRID_SHAPE
    )
    assert encoder.embedding_dim == EMBED_DIM


def test_from_file_rejects_wrong_expected_input_shape(saved_model_path):
    with pytest.raises(HandEncoderArchitectureMismatch, match="input shape"):
        PretrainedHandEncoder.from_file(
            saved_model_path, expected_input_shape=(13, 4, 2)
        )


def test_from_file_loaded_model_still_usable(saved_model_path):
    """The validated encoder must still work normally after passing
    the fingerprint check."""
    encoder = PretrainedHandEncoder.from_file(saved_model_path)
    out = encoder.encode_hole([0, 51])
    assert out.shape == (EMBED_DIM,)
    assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-3)


def test_skip_validation_bypasses_check(saved_model_path):
    """skip_validation=True must load without running the forward-pass
    fingerprint check at all (used only for debugging a mismatch)."""
    encoder = PretrainedHandEncoder.from_file(
        saved_model_path,
        expected_embedding_dim=EMBED_DIM + 1,  # would normally fail
        skip_validation=True,
    )
    assert encoder.embedding_dim == EMBED_DIM


def test_non_normalized_output_is_rejected(monkeypatch, saved_model_path):
    """Directly exercise the L2-norm check by monkeypatching the loaded
    model to return an un-normalized vector, simulating a checkpoint
    whose final l2_norm layer wasn't wired up as CardSetEncoder expects."""
    encoder = PretrainedHandEncoder.from_file(saved_model_path, skip_validation=True)

    class _FakeModel:
        card_set_encoder = encoder._model.card_set_encoder

        def __call__(self, x, training=False):
            batch = x.shape[0]
            # Deliberately NOT unit-norm.
            return tf.ones((batch, EMBED_DIM)) * 5.0

    encoder._model = _FakeModel()
    with pytest.raises(HandEncoderArchitectureMismatch, match="not L2-normalized"):
        encoder._validate_fingerprint(None, None)


def test_missing_input_shape_is_rejected(saved_model_path):
    encoder = PretrainedHandEncoder.from_file(saved_model_path, skip_validation=True)

    class _NoShapeCardSetEncoder:
        pass

    class _FakeModel:
        card_set_encoder = _NoShapeCardSetEncoder()

    encoder._model = _FakeModel()
    with pytest.raises(HandEncoderArchitectureMismatch, match="no encoder_input_shape"):
        encoder._validate_fingerprint(None, None)
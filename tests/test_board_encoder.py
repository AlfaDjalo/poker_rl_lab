"""
tests/test_board_encoder.py — Unit tests for representation/board_encoder.py

Run with:
    python -m pytest rl_lab/tests/test_board_encoder.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

from poker_rl_lab.representation.card_encoder import CardEncoder
from poker_rl_lab.representation.board_encoder import BoardEncoder


CARD_DIM = 24     # 16 rank + 8 suit, must be divisible by num_heads
OUTPUT_DIM = 32


@pytest.fixture
def shared_card_encoder():
    return CardEncoder(rank_dim=16, suit_dim=8)


@pytest.fixture
def board_encoder(shared_card_encoder):
    return BoardEncoder(
        card_encoder=shared_card_encoder, output_dim=OUTPUT_DIM, num_heads=4
    )


# ── Single-board forward ────────────────────────────────────────────────────

def test_flop_shape(board_encoder):
    out = board_encoder([0, 13, 26])  # 3-card flop
    assert out.shape == (OUTPUT_DIM,)


def test_turn_shape(board_encoder):
    out = board_encoder([0, 13, 26, 39])
    assert out.shape == (OUTPUT_DIM,)


def test_river_shape(board_encoder):
    out = board_encoder([0, 13, 26, 39, 51])
    assert out.shape == (OUTPUT_DIM,)


def test_double_board_river_shape(board_encoder):
    """Ten cards (two independent boards concatenated) still just embeds
    as one token sequence — callers pass one board's nodes at a time,
    but the module itself places no upper bound on card count."""
    out = board_encoder(list(range(10)))
    assert out.shape == (OUTPUT_DIM,)


def test_empty_board_uses_placeholder(board_encoder):
    out_none = board_encoder(None)
    out_empty = board_encoder([])
    assert torch.equal(out_none, out_empty)
    assert torch.equal(out_none, board_encoder.empty_board_embed)


def test_permutation_invariance(board_encoder):
    """Board texture must not depend on card dealing order."""
    a = board_encoder([0, 13, 26])
    b = board_encoder([26, 0, 13])
    assert torch.allclose(a, b, atol=1e-5)


def test_different_boards_different_embeddings(board_encoder):
    a = board_encoder([0, 13, 26])       # unpaired rainbow-ish
    b = board_encoder([0, 1, 2])          # very different texture
    assert not torch.allclose(a, b, atol=1e-4)


def test_gradients_flow_into_shared_card_encoder(shared_card_encoder, board_encoder):
    """Confirm CardEncoder weights (shared elsewhere) get gradient signal
    when trained through the board texture path."""
    out = board_encoder([0, 13, 26])
    out.sum().backward()
    assert shared_card_encoder.rank_embed.weight.grad is not None
    assert shared_card_encoder.rank_embed.weight.grad.abs().sum().item() > 0


def test_invalid_head_count_raises(shared_card_encoder):
    with pytest.raises(ValueError):
        BoardEncoder(card_encoder=shared_card_encoder, output_dim=32, num_heads=5)


# ── Batched forward ──────────────────────────────────────────────────────────

def test_batch_shape_mixed_lengths(board_encoder):
    batch = [
        None,                    # preflop
        [0, 13, 26],              # flop
        [0, 13, 26, 39, 51],       # river
    ]
    out = board_encoder.forward_batch(batch)
    assert out.shape == (3, OUTPUT_DIM)


def test_batch_matches_single(board_encoder):
    single = board_encoder([0, 13, 26, 39])
    batch = board_encoder.forward_batch([[0, 13, 26, 39]])
    assert torch.allclose(batch[0], single, atol=1e-5)


def test_batch_empty_rows_use_placeholder(board_encoder):
    out = board_encoder.forward_batch([None, [0, 13, 26]])
    assert torch.allclose(out[0], board_encoder.empty_board_embed, atol=1e-5)


def test_batch_all_empty(board_encoder):
    out = board_encoder.forward_batch([None, None, []])
    assert out.shape == (3, OUTPUT_DIM)
    for row in out:
        assert torch.allclose(row, board_encoder.empty_board_embed)


def test_batch_padding_does_not_affect_shorter_board(board_encoder):
    """A 3-card board's embedding must be identical whether it's batched
    alongside a longer board (and thus padded) or run alone."""
    solo = board_encoder([0, 13, 26])
    batched = board_encoder.forward_batch([[0, 13, 26], [0, 13, 26, 39, 51]])
    assert torch.allclose(batched[0], solo, atol=1e-4)
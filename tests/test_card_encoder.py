"""
tests/test_card_encoder.py  —  Unit tests for CardEncoder.

Run directly:
    python tests/test_card_encoder.py

Or with pytest:
    pytest tests/test_card_encoder.py -v

No CAP Engine, no C++ extension required.  PyTorch only.
All tests are deterministic (no random initialisation matters for shape /
structure checks; gradient / update tests use a fixed seed).

Card convention (engine standard)
----------------------------------
    rank = card_id % 13   (0=2 ... 12=A)
    suit = card_id // 13  (0=clubs, 1=diamonds, 2=hearts, 3=spades)

Notable cards used in tests
---------------------------
    0  = 2 of clubs
    1  = 3 of clubs
    12 = Ace of clubs
    13 = 2 of diamonds
    26 = 2 of hearts
    39 = 2 of spades
    51 = Ace of spades
"""

from __future__ import annotations

import sys
import os

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from poker_rl_lab.representation.card_encoder import CardEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encoder(rank_dim: int = 16, suit_dim: int = 8) -> CardEncoder:
    torch.manual_seed(0)
    return CardEncoder(rank_dim=rank_dim, suit_dim=suit_dim)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------

def test_output_shape_1d():
    """1-D batch of cards → (N, output_dim)."""
    enc = _make_encoder(rank_dim=16, suit_dim=8)
    cards = torch.tensor([0, 13, 26, 39, 51], dtype=torch.long)
    emb = enc(cards)
    assert emb.shape == (5, 24), f"Expected (5, 24), got {emb.shape}"
    print("PASS test_output_shape_1d")


def test_output_shape_scalar():
    """Scalar tensor → (output_dim,)."""
    enc = _make_encoder()
    emb = enc(torch.tensor(0, dtype=torch.long))
    assert emb.shape == (24,), f"Expected (24,), got {emb.shape}"
    print("PASS test_output_shape_scalar")


def test_output_shape_2d():
    """2-D batch (e.g. players × hole cards) → (B, N, output_dim)."""
    enc = _make_encoder()
    cards = torch.tensor([[0, 1], [13, 14], [26, 27]], dtype=torch.long)
    emb = enc(cards)
    assert emb.shape == (3, 2, 24), f"Expected (3, 2, 24), got {emb.shape}"
    print("PASS test_output_shape_2d")


def test_output_dim_attribute():
    enc = CardEncoder(rank_dim=32, suit_dim=12)
    assert enc.output_dim == 44
    print("PASS test_output_dim_attribute")


# ---------------------------------------------------------------------------
# Inductive-bias tests (the key correctness checks)
# ---------------------------------------------------------------------------

def test_same_rank_shares_rank_component():
    """
    All four 2s (clubs/diamonds/hearts/spades) must have identical rank
    embedding components because they share the rank=0 entry.
    """
    enc = _make_encoder()
    # 2c=0, 2d=13, 2h=26, 2s=39 — all rank 0
    cards = torch.tensor([0, 13, 26, 39], dtype=torch.long)
    emb = enc(cards)

    rank_parts = emb[:, :enc.rank_dim]   # first rank_dim dims
    for i in range(1, 4):
        assert torch.allclose(rank_parts[0], rank_parts[i]), \
            f"Rank components differ between card 0 and card {i*13}"
    print("PASS test_same_rank_shares_rank_component")


def test_different_ranks_different_rank_component():
    """Cards with different ranks must have different rank embeddings."""
    enc = _make_encoder()
    # 2c=0 (rank 0), Ac=12 (rank 12) — same suit, different rank
    cards = torch.tensor([0, 12], dtype=torch.long)
    emb = enc(cards)
    rank_parts = emb[:, :enc.rank_dim]
    assert not torch.allclose(rank_parts[0], rank_parts[1]), \
        "Expected different rank components for different ranks"
    print("PASS test_different_ranks_different_rank_component")


def test_same_suit_shares_suit_component():
    """
    All clubs (suit 0) must have identical suit embedding components
    regardless of rank.
    """
    enc = _make_encoder()
    # 2c=0, 3c=1, 4c=2, Ac=12 — all clubs
    cards = torch.tensor([0, 1, 2, 12], dtype=torch.long)
    emb = enc(cards)

    suit_parts = emb[:, enc.rank_dim:]   # last suit_dim dims
    for i in range(1, 4):
        assert torch.allclose(suit_parts[0], suit_parts[i]), \
            f"Suit components differ for clubs card index {i}"
    print("PASS test_same_suit_shares_suit_component")


def test_different_suits_different_suit_component():
    """Cards with different suits must have different suit embeddings."""
    enc = _make_encoder()
    # Ac=12 (suit 0), As=51 (suit 3) — same rank, different suit
    cards = torch.tensor([12, 51], dtype=torch.long)
    emb = enc(cards)
    suit_parts = emb[:, enc.rank_dim:]
    assert not torch.allclose(suit_parts[0], suit_parts[1]), \
        "Expected different suit components for different suits"
    print("PASS test_different_suits_different_suit_component")


# ---------------------------------------------------------------------------
# embed_mask helper
# ---------------------------------------------------------------------------

def test_embed_mask_correct_count():
    """embed_mask returns one row per set bit."""
    enc = _make_encoder()
    # Cards 0, 1, 2 → mask = 0b111 = 7
    mask = (1 << 0) | (1 << 1) | (1 << 2)
    emb = enc.embed_mask(mask)
    assert emb.shape == (3, 24), f"Expected (3, 24), got {emb.shape}"
    print("PASS test_embed_mask_correct_count")


def test_embed_mask_empty():
    """embed_mask with mask=0 returns shape (0, output_dim)."""
    enc = _make_encoder()
    emb = enc.embed_mask(0)
    assert emb.shape == (0, 24), f"Expected (0, 24), got {emb.shape}"
    print("PASS test_embed_mask_empty")


def test_embed_mask_matches_forward():
    """embed_mask and forward produce the same values for the same cards."""
    enc = _make_encoder()
    card_ids = [0, 14, 51]
    mask = sum(1 << c for c in card_ids)

    via_mask    = enc.embed_mask(mask)
    via_forward = enc(torch.tensor(card_ids, dtype=torch.long))

    assert torch.allclose(via_mask, via_forward), \
        "embed_mask and forward disagree"
    print("PASS test_embed_mask_matches_forward")


# ---------------------------------------------------------------------------
# embed_cards helper
# ---------------------------------------------------------------------------

def test_embed_cards_empty():
    enc = _make_encoder()
    emb = enc.embed_cards([])
    assert emb.shape == (0, 24)
    print("PASS test_embed_cards_empty")


def test_embed_cards_single():
    enc = _make_encoder()
    emb = enc.embed_cards([0])
    assert emb.shape == (1, 24)
    print("PASS test_embed_cards_single")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_invalid_rank_dim():
    try:
        CardEncoder(rank_dim=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS test_invalid_rank_dim")


def test_invalid_suit_dim():
    try:
        CardEncoder(suit_dim=-1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS test_invalid_suit_dim")


# ---------------------------------------------------------------------------
# Gradient / trainability test
# ---------------------------------------------------------------------------

def test_gradients_flow():
    """
    A trivial training step should update both embedding tables.
    Ensures the module is properly registered as a leaf with grad.
    """
    torch.manual_seed(42)
    enc = CardEncoder(rank_dim=8, suit_dim=4)
    opt = torch.optim.SGD(enc.parameters(), lr=0.01)

    cards = torch.tensor([0, 13, 26, 39], dtype=torch.long)

    rank_before = enc.rank_embed.weight.data.clone()
    suit_before = enc.suit_embed.weight.data.clone()

    emb  = enc(cards)
    loss = emb.sum()
    loss.backward()
    opt.step()

    assert not torch.allclose(enc.rank_embed.weight.data, rank_before), \
        "Rank embedding did not update"
    assert not torch.allclose(enc.suit_embed.weight.data, suit_before), \
        "Suit embedding did not update"
    print("PASS test_gradients_flow")


# ---------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------

def test_parameter_count():
    """Verify total parameter count matches expected tables."""
    enc = CardEncoder(rank_dim=16, suit_dim=8)
    total = sum(p.numel() for p in enc.parameters())
    expected = 13 * 16 + 4 * 8   # rank table + suit table
    assert total == expected, f"Expected {expected} params, got {total}"
    print("PASS test_parameter_count")


def test_repr_contains_dims():
    enc = CardEncoder(rank_dim=16, suit_dim=8)
    r = repr(enc)
    assert "rank_dim=16" in r
    assert "suit_dim=8" in r
    print("PASS test_repr_contains_dims")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_output_shape_1d,
        test_output_shape_scalar,
        test_output_shape_2d,
        test_output_dim_attribute,
        test_same_rank_shares_rank_component,
        test_different_ranks_different_rank_component,
        test_same_suit_shares_suit_component,
        test_different_suits_different_suit_component,
        test_embed_mask_correct_count,
        test_embed_mask_empty,
        test_embed_mask_matches_forward,
        test_embed_cards_empty,
        test_embed_cards_single,
        test_invalid_rank_dim,
        test_invalid_suit_dim,
        test_gradients_flow,
        test_parameter_count,
        test_repr_contains_dims,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed.")
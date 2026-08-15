"""
tests/test_card_utils.py  —  Unit tests for representation.card_utils.

Run directly:
    python tests/test_card_utils.py

Or with pytest:
    pytest tests/test_card_utils.py -v

No PyTorch, no CAP Engine, no C++ extension required.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from poker_rl_lab.representation.card_utils import (
    card_id_to_str,
    str_to_card_id,
    card_ids_to_strs,
    strs_to_card_ids,
    mask_to_card_ids,
    card_ids_to_mask,
    card_ids_to_grid,
    mask_to_grid,
    GRID_SHAPE,
)


# ---------------------------------------------------------------------------
# card_id_to_str
# ---------------------------------------------------------------------------

def test_card_id_to_str_corners():
    """0 is 2c (lowest), 51 is As (highest)."""
    assert card_id_to_str(0)  == "2c"
    assert card_id_to_str(51) == "As"
    print("PASS test_card_id_to_str_corners")


def test_card_id_to_str_suit_stride():
    """Same rank across all four suits changes every 13 ids."""
    # 2c=0, 2d=13, 2h=26, 2s=39
    assert card_id_to_str(0)  == "2c"
    assert card_id_to_str(13) == "2d"
    assert card_id_to_str(26) == "2h"
    assert card_id_to_str(39) == "2s"
    print("PASS test_card_id_to_str_suit_stride")


def test_card_id_to_str_rank_run():
    """Rank characters cycle through 2..A within each suit."""
    expected = ["2c","3c","4c","5c","6c","7c","8c","9c","Tc","Jc","Qc","Kc","Ac"]
    for i, s in enumerate(expected):
        assert card_id_to_str(i) == s, f"card_id {i}: expected {s}, got {card_id_to_str(i)}"
    print("PASS test_card_id_to_str_rank_run")


def test_card_id_to_str_invalid():
    """Out-of-range ids raise ValueError."""
    for bad in (-1, 52, 100):
        try:
            card_id_to_str(bad)
            assert False, f"Should have raised ValueError for {bad}"
        except ValueError:
            pass
    print("PASS test_card_id_to_str_invalid")


# ---------------------------------------------------------------------------
# str_to_card_id
# ---------------------------------------------------------------------------

def test_str_to_card_id_corners():
    assert str_to_card_id("2c") == 0
    assert str_to_card_id("As") == 51
    print("PASS test_str_to_card_id_corners")


def test_str_to_card_id_case_insensitive_rank():
    """Rank character is case-insensitive."""
    assert str_to_card_id("as") == str_to_card_id("As")
    assert str_to_card_id("kd") == str_to_card_id("Kd")
    print("PASS test_str_to_card_id_case_insensitive_rank")


def test_str_to_card_id_invalid_length():
    try:
        str_to_card_id("Ace")
        assert False
    except ValueError:
        pass
    print("PASS test_str_to_card_id_invalid_length")


def test_str_to_card_id_invalid_rank():
    try:
        str_to_card_id("Xc")
        assert False
    except ValueError:
        pass
    print("PASS test_str_to_card_id_invalid_rank")


# ---------------------------------------------------------------------------
# Round-trip: card_id_to_str ↔ str_to_card_id
# ---------------------------------------------------------------------------

def test_round_trip_all_52():
    """All 52 card ids survive a round-trip through string and back."""
    for cid in range(52):
        s   = card_id_to_str(cid)
        cid2 = str_to_card_id(s)
        assert cid2 == cid, f"Round-trip failed at {cid}: {s} -> {cid2}"
    print("PASS test_round_trip_all_52")


def test_round_trip_known_cards():
    for s in ("As", "Kd", "2c", "Th", "9s", "Jc"):
        assert card_id_to_str(str_to_card_id(s)) == s
    print("PASS test_round_trip_known_cards")


# ---------------------------------------------------------------------------
# Batch conversions
# ---------------------------------------------------------------------------

def test_card_ids_to_strs_and_back():
    ids  = [0, 13, 26, 39, 51]
    strs = card_ids_to_strs(ids)
    assert strs == ["2c", "2d", "2h", "2s", "As"]
    assert strs_to_card_ids(strs) == ids
    print("PASS test_card_ids_to_strs_and_back")


# ---------------------------------------------------------------------------
# mask_to_card_ids / card_ids_to_mask
# ---------------------------------------------------------------------------

def test_mask_empty():
    assert mask_to_card_ids(0) == []
    print("PASS test_mask_empty")


def test_mask_single():
    for cid in [0, 25, 51]:
        mask = 1 << cid
        assert mask_to_card_ids(mask) == [cid]
    print("PASS test_mask_single")


def test_mask_round_trip():
    ids  = [0, 7, 14, 51]
    mask = card_ids_to_mask(ids)
    assert mask_to_card_ids(mask) == sorted(ids)
    print("PASS test_mask_round_trip")


def test_mask_all_52():
    full_mask = (1 << 52) - 1
    ids = mask_to_card_ids(full_mask)
    assert ids == list(range(52))
    print("PASS test_mask_all_52")


# ---------------------------------------------------------------------------
# card_ids_to_grid — shape and values
# ---------------------------------------------------------------------------

def test_grid_shape():
    g = card_ids_to_grid([])
    assert g.shape == GRID_SHAPE
    assert g.dtype == np.float32
    print("PASS test_grid_shape")


def test_grid_empty():
    g = card_ids_to_grid([])
    assert g.sum() == 0.0
    print("PASS test_grid_empty")


def test_grid_one_card_set():
    """Each card sets exactly one cell to 1.0."""
    for cid in range(52):
        g = card_ids_to_grid([cid])
        assert g.sum() == 1.0, f"Expected sum 1.0 for card_id={cid}, got {g.sum()}"
    print("PASS test_grid_one_card_set")


def test_grid_n_cards_set():
    """n distinct cards set exactly n cells."""
    ids = [0, 13, 26, 39]   # four 2s
    g = card_ids_to_grid(ids)
    assert g.sum() == 4.0
    print("PASS test_grid_n_cards_set")


def test_grid_row_13_always_zero():
    """Row 13 is the historical unused row; it should always be 0."""
    for cid in range(52):
        g = card_ids_to_grid([cid])
        assert g[13, :, :].sum() == 0.0, f"Row 13 was set for card_id={cid}"
    print("PASS test_grid_row_13_always_zero")


def test_grid_coordinates_as():
    """As (card_id=51) should land at row=0, col=0."""
    g = card_ids_to_grid([51])
    assert g[0, 0, 0] == 1.0,  f"As should be at (0,0,0), got {g[0,0,0]}"
    assert g.sum()    == 1.0
    print("PASS test_grid_coordinates_as")


def test_grid_coordinates_2c():
    """2c (card_id=0) should land at row=12, col=3."""
    g = card_ids_to_grid([0])
    assert g[12, 3, 0] == 1.0, f"2c should be at (12,3,0), got {g[12,3,0]}"
    assert g.sum()     == 1.0
    print("PASS test_grid_coordinates_2c")


def test_grid_coordinates_2s():
    """2s (card_id=39) should land at row=12, col=0."""
    g = card_ids_to_grid([39])
    assert g[12, 0, 0] == 1.0, f"2s should be at (12,0,0), got {g[12,0,0]}"
    print("PASS test_grid_coordinates_2s")


def test_grid_coordinates_ac():
    """Ac (card_id=12) should land at row=0, col=3."""
    g = card_ids_to_grid([12])
    assert g[0, 3, 0] == 1.0,  f"Ac should be at (0,3,0), got {g[0,3,0]}"
    print("PASS test_grid_coordinates_ac")


def test_grid_matches_old_cards_to_tensor():
    """
    Validate against the original cards_to_tensor logic from push_fold_env.py.

    We reimplement the original directly here (without importing TF) and
    compare both outputs on a random hand.
    """
    # --- original logic (pure numpy, no TF) ---
    rank_to_idx = {'A':0,'K':1,'Q':2,'J':3,'T':4,'9':5,'8':6,'7':7,
                   '6':8,'5':9,'4':10,'3':11,'2':12}
    suit_to_idx = {'s':0,'h':1,'d':2,'c':3}

    def original_grid(card_strs):
        g = np.zeros((14, 4, 1), dtype=np.float32)
        for card in card_strs:
            r, s = card[0].upper(), card[1].lower()
            g[rank_to_idx[r], suit_to_idx[s], 0] = 1.0
        return g

    # Test hands
    test_hands = [
        [51, 50],          # As, Ks
        [0, 1, 2, 3],      # 2c, 3c, 4c, 5c
        [12, 25, 38, 51],  # Ac, Jh, Qs, As — but 51=As already, let's keep it
    ]
    # Safer hands with no duplicates:
    test_hands = [
        [51, 50],     # As Ks
        [0, 13],      # 2c 2d
        [21, 33],     # Td 9h
        [12, 26, 39, 25],  # Ac 2h 2s Jh — Omaha-style 4 cards
    ]

    for ids in test_hands:
        strs    = [card_id_to_str(c) for c in ids]
        ref     = original_grid(strs)
        ours    = card_ids_to_grid(ids)
        assert np.array_equal(ref, ours), (
            f"Grid mismatch for {strs}: ref sum={ref.sum()}, ours sum={ours.sum()}"
        )
    print("PASS test_grid_matches_old_cards_to_tensor")


# ---------------------------------------------------------------------------
# mask_to_grid
# ---------------------------------------------------------------------------

def test_mask_to_grid_matches_card_ids_to_grid():
    """mask_to_grid and card_ids_to_grid agree for the same cards."""
    ids  = [0, 14, 51]
    mask = card_ids_to_mask(ids)
    g1   = card_ids_to_grid(ids)
    g2   = mask_to_grid(mask)
    assert np.array_equal(g1, g2)
    print("PASS test_mask_to_grid_matches_card_ids_to_grid")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_card_id_to_str_corners,
        test_card_id_to_str_suit_stride,
        test_card_id_to_str_rank_run,
        test_card_id_to_str_invalid,
        test_str_to_card_id_corners,
        test_str_to_card_id_case_insensitive_rank,
        test_str_to_card_id_invalid_length,
        test_str_to_card_id_invalid_rank,
        test_round_trip_all_52,
        test_round_trip_known_cards,
        test_card_ids_to_strs_and_back,
        test_mask_empty,
        test_mask_single,
        test_mask_round_trip,
        test_mask_all_52,
        test_grid_shape,
        test_grid_empty,
        test_grid_one_card_set,
        test_grid_n_cards_set,
        test_grid_row_13_always_zero,
        test_grid_coordinates_as,
        test_grid_coordinates_2c,
        test_grid_coordinates_2s,
        test_grid_coordinates_ac,
        test_grid_matches_old_cards_to_tensor,
        test_mask_to_grid_matches_card_ids_to_grid,
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
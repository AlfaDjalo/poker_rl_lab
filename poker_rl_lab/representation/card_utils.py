"""
representation/card_utils.py  —  Card representation conversion utilities.

Bridges two card naming systems used in this project:

  Engine convention (used everywhere in the new RL Lab)
  -------------------------------------------------------
      card_id in [0, 51]
      rank = card_id % 13   (0=2, 1=3, ..., 12=A)
      suit = card_id // 13  (0=clubs, 1=diamonds, 2=hearts, 3=spades)

  String convention (used in the old push-fold project and the Keras encoder)
  ---------------------------------------------------------------------------
      "<rank><suit>" e.g. "As", "Kd", "2c"
      rank chars: 2 3 4 5 6 7 8 9 T J Q K A  (low to high)
      suit chars: c d h s  (clubs diamonds hearts spades)

  Grid convention (input format for the pre-trained Keras hand encoder)
  ---------------------------------------------------------------------
      numpy array of shape (14, 4, 1)
      row = 12 - new_rank   (A=row 0, K=row 1, ..., 2=row 12; row 13 unused)
      col = 3  - new_suit   (spades=col 0, hearts=col 1, diamonds=col 2, clubs=col 3)
      channel 0: 1.0 if that card is present, else 0.0

  The grid layout matches the Keras encoder's expected input exactly.
  Only rows 0-12 are ever populated; row 13 is always zero (historical artifact).

Notes
-----
  No neural-network or PyTorch dependency in this file.
  All functions are pure Python / numpy so they can be used in both the
  environment and the representation model.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Lookup tables (constants, never mutated)
# ---------------------------------------------------------------------------

# New-system rank index → rank character (2 … A)
_RANK_CHARS: tuple[str, ...] = (
    '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'
)

# New-system suit index → suit character (clubs … spades)
_SUIT_CHARS: tuple[str, ...] = ('c', 'd', 'h', 's')

# Reverse lookups
_CHAR_TO_RANK: dict[str, int] = {c: i for i, c in enumerate(_RANK_CHARS)}
_CHAR_TO_SUIT: dict[str, int] = {c: i for i, c in enumerate(_SUIT_CHARS)}

# Shape of the grid expected by the pre-trained Keras encoder
GRID_SHAPE: tuple[int, int, int] = (14, 4, 1)

# Number of used rank rows in the grid (rows 0-12; row 13 is always zero)
_GRID_RANK_ROWS: int = 13


# ---------------------------------------------------------------------------
# Single-card conversions
# ---------------------------------------------------------------------------

def card_id_to_str(card_id: int) -> str:
    """
    Convert an engine card id to a card string.

    Parameters
    ----------
    card_id : int
        Value in [0, 51].

    Returns
    -------
    str
        Two-character string e.g. "As", "Kd", "2c".

    Examples
    --------
    >>> card_id_to_str(0)
    '2c'
    >>> card_id_to_str(51)
    'As'
    """
    if not 0 <= card_id <= 51:
        raise ValueError(f"card_id must be in [0, 51], got {card_id}")
    rank = card_id % 13
    suit = card_id // 13
    return _RANK_CHARS[rank] + _SUIT_CHARS[suit]


def str_to_card_id(card_str: str) -> int:
    """
    Convert a card string to an engine card id.

    Parameters
    ----------
    card_str : str
        Two-character string e.g. "As", "Kd", "2c".
        Rank is case-insensitive; suit is lower-case.

    Returns
    -------
    int
        Value in [0, 51].

    Examples
    --------
    >>> str_to_card_id('2c')
    0
    >>> str_to_card_id('As')
    51
    """
    if len(card_str) != 2:
        raise ValueError(f"card_str must be 2 characters, got {card_str!r}")
    rank_char = card_str[0].upper()
    suit_char = card_str[1].lower()
    if rank_char not in _CHAR_TO_RANK:
        raise ValueError(f"Unknown rank character {rank_char!r} in {card_str!r}")
    if suit_char not in _CHAR_TO_SUIT:
        raise ValueError(f"Unknown suit character {suit_char!r} in {card_str!r}")
    rank = _CHAR_TO_RANK[rank_char]
    suit = _CHAR_TO_SUIT[suit_char]
    return suit * 13 + rank


# ---------------------------------------------------------------------------
# Batch conversions
# ---------------------------------------------------------------------------

def card_ids_to_strs(card_ids: Iterable[int]) -> list[str]:
    """Convert an iterable of engine card ids to a list of card strings."""
    return [card_id_to_str(c) for c in card_ids]


def strs_to_card_ids(card_strs: Iterable[str]) -> list[int]:
    """Convert an iterable of card strings to a list of engine card ids."""
    return [str_to_card_id(s) for s in card_strs]


def mask_to_card_ids(mask: int) -> list[int]:
    """
    Expand a 52-bit bitmask into a sorted list of card ids.

    Bit N set means card N is present.  Returns cards in ascending id order.

    Parameters
    ----------
    mask : int
        52-bit hand bitmask (engine convention).

    Returns
    -------
    list[int]
        Card ids in [0, 51], ascending.
    """
    return [i for i in range(52) if (mask >> i) & 1]


def card_ids_to_mask(card_ids: Iterable[int]) -> int:
    """Convert an iterable of card ids to a 52-bit bitmask."""
    mask = 0
    for c in card_ids:
        mask |= 1 << c
    return mask


# ---------------------------------------------------------------------------
# Grid builder  (for pre-trained Keras encoder)
# ---------------------------------------------------------------------------

def card_ids_to_grid(card_ids: Iterable[int]) -> np.ndarray:
    """
    Build the (14, 4, 1) numpy grid used by the pre-trained Keras hand encoder.

    Grid layout
    -----------
      row = 12 - (card_id % 13)   →  A=0, K=1, …, 2=12  (row 13 always zero)
      col = 3  - (card_id // 13)  →  spades=0, hearts=1, diamonds=2, clubs=3
      channel 0 is set to 1.0 for each present card.

    This matches the ``cards_to_tensor(mode="hand")`` function from the
    original push-fold project exactly, using engine card ids as input
    instead of card strings.

    Parameters
    ----------
    card_ids : Iterable[int]
        Engine card ids in [0, 51].  Duplicates are silently ignored.

    Returns
    -------
    np.ndarray  float32
        Shape (14, 4, 1).

    Examples
    --------
    >>> g = card_ids_to_grid([51, 50])   # As, Ks
    >>> g[0, 0, 0]                        # As at row 0, col 0
    1.0
    >>> g[1, 0, 0]                        # Ks at row 1, col 0
    1.0
    >>> g.sum()                           # exactly 2 cards set
    2.0
    """
    grid = np.zeros(GRID_SHAPE, dtype=np.float32)
    for card_id in card_ids:
        if not 0 <= card_id <= 51:
            raise ValueError(f"card_id must be in [0, 51], got {card_id}")
        row = 12 - (card_id % 13)    # A → row 0, 2 → row 12
        col = 3  - (card_id // 13)   # spades → col 0, clubs → col 3
        grid[row, col, 0] = 1.0
    return grid


def mask_to_grid(hand_mask: int) -> np.ndarray:
    """
    Build the (14, 4, 1) grid from a 52-bit bitmask.

    Convenience wrapper around card_ids_to_grid for code that stores
    hands as bitmasks (the engine's native format).

    Parameters
    ----------
    hand_mask : int
        52-bit bitmask.

    Returns
    -------
    np.ndarray  float32
        Shape (14, 4, 1).
    """
    return card_ids_to_grid(mask_to_card_ids(hand_mask))
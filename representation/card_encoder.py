"""
representation/card_encoder.py  —  Learnable card embedding.

Each card in [0, 51] is decomposed into:
    rank = card_id % 13   (0=2, 1=3, ..., 12=A)
    suit = card_id // 13  (0=clubs, 1=diamonds, 2=hearts, 3=spades)

The card embedding is the concatenation of a learned rank embedding
and a learned suit embedding:

    card_embed = [rank_embed ‖ suit_embed]
    output_dim = rank_dim + suit_dim

Why decompose rather than embed all 52 cards independently?
-----------------------------------------------------------
  - All four aces share one rank embedding.  All clubs share one suit
    embedding.  The model cannot memorise 52 independent cards; it must
    learn what rank and suit *mean*.
  - Poker strategy is strongly driven by rank (hand strength) and weakly
    by suit (flush potential).  Separate embedding tables let the model
    allocate capacity accordingly.
  - Decomposition is the standard approach in poker AI literature and
    matches the pretrained embeddings described in the RL Lab guide.
  - The pretrained 2-card, 3-card and 5-card hand embeddings (Phase 5
    guide, §Existing Embeddings) will be initialised using weights
    derived from these rank and suit components.

Dependencies
------------
  PyTorch only.  No CAP Engine, no C++ extension.

Usage
-----
    import torch
    from representation.card_encoder import CardEncoder

    encoder = CardEncoder(rank_dim=16, suit_dim=8)

    # Single card
    emb = encoder(torch.tensor(0))          # shape (24,)

    # Batch of cards
    emb = encoder(torch.tensor([0, 14, 51]))  # shape (3, 24)

    # From a hand bitmask (engine convention)
    emb = encoder.embed_mask(0b11)          # first two cards dealt
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CardEncoder(nn.Module):
    """
    Learnable embedding for individual playing cards.

    Decomposes each card id into rank and suit indices, looks up
    independent learned embeddings for each, and concatenates them.

    Parameters
    ----------
    rank_dim : int
        Dimensionality of the rank embedding.  Default 16.
    suit_dim : int
        Dimensionality of the suit embedding.  Default 8.

    Attributes
    ----------
    output_dim : int
        Total embedding size: rank_dim + suit_dim.
    rank_embed : nn.Embedding
        Learnable table of shape (13, rank_dim).
    suit_embed : nn.Embedding
        Learnable table of shape (4, suit_dim).
    """

    def __init__(self, rank_dim: int = 16, suit_dim: int = 8) -> None:
        super().__init__()

        if rank_dim <= 0:
            raise ValueError(f"rank_dim must be > 0, got {rank_dim}")
        if suit_dim <= 0:
            raise ValueError(f"suit_dim must be > 0, got {suit_dim}")

        self.rank_dim   = rank_dim
        self.suit_dim   = suit_dim
        self.output_dim = rank_dim + suit_dim

        # 13 ranks (2 … A), 4 suits (clubs, diamonds, hearts, spades)
        self.rank_embed = nn.Embedding(13, rank_dim)
        self.suit_embed = nn.Embedding(4,  suit_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed one or more card ids.

        Parameters
        ----------
        card_ids : torch.Tensor  (dtype=torch.long)
            Card ids in [0, 51].  Any shape is accepted.

        Returns
        -------
        torch.Tensor  float32
            Shape ``(*card_ids.shape, output_dim)``.

        Examples
        --------
        Single card (scalar tensor)::

            encoder(torch.tensor(0))                 # (24,)

        1-D batch::

            encoder(torch.tensor([0, 13, 26, 39]))   # (4, 24)

        2-D batch (e.g. 3 players × 2 hole cards)::

            encoder(torch.zeros(3, 2, dtype=torch.long))  # (3, 2, 24)
        """
        ranks = card_ids % 13      # values in [0, 12]
        suits = card_ids // 13     # values in [0,  3]

        rank_emb = self.rank_embed(ranks)   # (..., rank_dim)
        suit_emb = self.suit_embed(suits)   # (..., suit_dim)

        return torch.cat([rank_emb, suit_emb], dim=-1)  # (..., output_dim)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Device the embedding tables currently live on."""
        return self.rank_embed.weight.device

    def embed_mask(self, hand_mask: int) -> torch.Tensor:
        """
        Embed all cards present in *hand_mask*.

        Useful when a hand is stored as a bitmask (CAP Engine convention)
        rather than a tuple of card ids.

        Parameters
        ----------
        hand_mask : int
            52-bit bitmask.  Bit N set means card N is in the hand.

        Returns
        -------
        torch.Tensor  float32
            Shape (n_cards, output_dim) where n_cards = popcount(hand_mask).
            Returns an empty tensor of shape (0, output_dim) when mask is 0.
        """
        card_ids = [i for i in range(52) if (hand_mask >> i) & 1]
        if not card_ids:
            return torch.zeros(0, self.output_dim, device=self.device)
        ids = torch.tensor(card_ids, dtype=torch.long, device=self.device)
        return self.forward(ids)

    def embed_cards(self, card_ids: list[int]) -> torch.Tensor:
        """
        Embed a list of card ids.

        Thin wrapper around forward() for callers that have a list rather
        than a tensor.

        Parameters
        ----------
        card_ids : list[int]
            Card ids in [0, 51].

        Returns
        -------
        torch.Tensor  float32
            Shape (len(card_ids), output_dim).
        """
        if not card_ids:
            return torch.zeros(0, self.output_dim, device=self.device)
        ids = torch.tensor(card_ids, dtype=torch.long, device=self.device)
        return self.forward(ids)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"rank_dim={self.rank_dim}, suit_dim={self.suit_dim}, "
            f"output_dim={self.output_dim}"
        )
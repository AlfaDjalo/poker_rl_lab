"""
representation/board_encoder.py — Learned board-texture embedding.

Per the RL Lab implementation guide (§5, "Additional Learned Embeddings"):

    Board: Represents the strategic properties of an individual board.

This is deliberately separate from PretrainedHandEncoder.encode_board():
that method (via the pretrained Keras CardSetEncoder) captures *hand
strength given a board*, using an existing model with poker knowledge
baked in from training on card sets in general. BoardEncoder instead
learns board *texture* from scratch — features like how connected,
paired, or flush-heavy a board is — independent of any particular
hand. It is a from-scratch PyTorch module trained jointly with the
rest of the Representation Model via PPO.

Design
------
A board is a variable-length, unordered set of 0-5 cards (0 preflop,
3 on the flop, 4 on the turn, 5 on the river; more nodes for
multi-board CAP variants, but this module encodes ONE board at a time
— callers with multiple boards run it once per board).

  1. Each dealt card is embedded via the shared CardEncoder (the same
     per-card rank/suit embedding used elsewhere in the Representation
     Model — NOT a separate embedding table, so rank/suit knowledge is
     shared across every place cards appear).
  2. A small Transformer encoder layer lets cards attend to each other
     (this is how "texture" — pairs, straights, flush draws — is
     detected; a card's contribution to board texture depends on the
     *other* cards present, which a permutation-invariant sum alone
     cannot capture).
  3. Mean-pooling over the (attended) card tokens gives a fixed-size,
     permutation-invariant board summary regardless of card count.
  4. An empty board (preflop, street_index == 0) has no cards to embed;
     a dedicated learned "empty board" parameter is used instead so the
     model can still distinguish "no board yet" from any real board.

Dependencies
------------
  PyTorch only. Depends on representation.card_encoder.CardEncoder,
  which the caller must construct and pass in (shared, not owned) so
  the same rank/suit embedding table is reused everywhere cards are
  encoded — see the `card_encoder` parameter below.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from representation.card_encoder import CardEncoder


class BoardEncoder(nn.Module):
    """
    Encodes one board's cards into a fixed-size texture embedding.

    Parameters
    ----------
    card_encoder : CardEncoder
        Shared per-card embedding module. Pass the SAME instance used
        elsewhere in the Representation Model (e.g. for hole cards),
        so rank/suit weights are shared, not duplicated.
    output_dim : int
        Dimensionality of the board embedding. Default 32.
    num_heads : int
        Attention heads in the internal Transformer layer. Must divide
        card_encoder.output_dim. Default 4.
    num_layers : int
        Number of stacked Transformer encoder layers. Default 1
        (boards are small — at most 5-10 cards — so one layer of
        full attention is normally enough).
    dropout : float
        Dropout inside the Transformer layer. Default 0.0 (RL training
        is typically low-data-per-update; keep it off unless overfitting
        is observed).
    """

    def __init__(
        self,
        card_encoder: CardEncoder,
        output_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        card_dim = card_encoder.output_dim
        if card_dim % num_heads != 0:
            raise ValueError(
                f"card_encoder.output_dim ({card_dim}) must be divisible "
                f"by num_heads ({num_heads})"
            )

        self.card_encoder = card_encoder
        self.output_dim = output_dim
        self.card_dim = card_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=card_dim,
            nhead=num_heads,
            dim_feedforward=card_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.project = nn.Linear(card_dim, output_dim)

        # Learned placeholder for a board with zero cards (preflop).
        self.empty_board_embed = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.empty_board_embed, std=0.02)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, card_ids: Optional[Sequence[int]]) -> torch.Tensor:
        """
        Embed a single board.

        Parameters
        ----------
        card_ids : Sequence[int] or None
            Card ids in [0, 51] currently dealt on this board. Empty
            sequence or None means no cards dealt yet (preflop).

        Returns
        -------
        torch.Tensor
            Shape (output_dim,).
        """
        if not card_ids:
            return self.empty_board_embed

        device = self.empty_board_embed.device
        ids = torch.tensor(list(card_ids), dtype=torch.long, device=device)
        card_embeds = self.card_encoder(ids)              # (n_cards, card_dim)
        card_embeds = card_embeds.unsqueeze(0)             # (1, n_cards, card_dim)
        attended = self.attention(card_embeds)              # (1, n_cards, card_dim)
        pooled = attended.mean(dim=1).squeeze(0)            # (card_dim,)
        return self.project(pooled)                          # (output_dim,)

    def forward_batch(
        self, batch_card_ids: Sequence[Optional[Sequence[int]]]
    ) -> torch.Tensor:
        """
        Embed a batch of boards, each with a possibly different card count.

        Parameters
        ----------
        batch_card_ids : Sequence[Optional[Sequence[int]]]
            One board per example. Empty/None entries are preflop boards.

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).

        Notes
        -----
        Boards are padded to the max length in the batch and masked so
        padding never contributes to attention or pooling. Empty boards
        in the batch fall back to empty_board_embed directly (skipping
        the Transformer, since there's nothing to attend over).
        """
        non_empty = [list(c) for c in batch_card_ids if c]
        if not non_empty:
            return self.empty_board_embed.unsqueeze(0).expand(
                len(batch_card_ids), -1
            )

        max_len = max(len(c) for c in non_empty)
        batch_size = len(batch_card_ids)
        device = self.empty_board_embed.device

        ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        pad_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)  # True = ignore

        for i, cards in enumerate(batch_card_ids):
            if not cards:
                continue
            n = len(cards)
            ids[i, :n] = torch.tensor(cards, dtype=torch.long, device=device)
            pad_mask[i, :n] = False

        card_embeds = self.card_encoder(ids)  # (batch, max_len, card_dim)
        attended = self.attention(card_embeds, src_key_padding_mask=pad_mask)

        # Mean-pool only over real (non-padded) positions.
        valid = (~pad_mask).unsqueeze(-1).float()            # (batch, max_len, 1)
        summed = (attended * valid).sum(dim=1)                # (batch, card_dim)
        counts = valid.sum(dim=1).clamp(min=1.0)               # (batch, 1)
        pooled = summed / counts

        out = self.project(pooled)  # (batch, output_dim)

        # Overwrite rows that were empty boards with the learned placeholder.
        empty_rows = torch.tensor(
            [not bool(c) for c in batch_card_ids], dtype=torch.bool, device=device
        )
        if empty_rows.any():
            out = out.clone()
            out[empty_rows] = self.empty_board_embed

        return out
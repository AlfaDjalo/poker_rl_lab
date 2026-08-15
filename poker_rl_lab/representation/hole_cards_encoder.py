"""
representation/hole_cards_encoder.py  —  Hole-cards → hand embedding.

Converts a player's hole cards (2 for hold'em, 4 for Omaha) into a
single fixed-size hand embedding that the downstream policy network can
use as a token.

Architecture
------------
    card_ids  →  CardEncoder  →  (n_cards, card_dim)
                              →  pool  →  (card_dim,)
                              →  [optional Linear projection]  →  (output_dim,)

Pooling over the card dimension is permutation-invariant, which is the
correct inductive bias: {A♠, K♦} and {K♦, A♠} are the same hand.

Pre-trained weights
-------------------
The pre-trained Keras hand encoder uses a different input format (a
(14, 4, 1) binary grid — see card_utils.card_ids_to_grid) and an
unknown CNN/MLP architecture.  Integrating those weights requires:

  1. The architecture definition (ML/training/trainer.py or equivalent).
  2. The saved Keras model (encoder_model.keras).

Once those are available, the recommended path is:

  a) Mirror the Keras architecture here as a PyTorch nn.Module.
  b) Extract weights with keras_model.get_weights() and load them into
     the PyTorch layer-by-layer via load_pretrained_weights().
  c) Keep the weights trainable so RL continues to refine them.

This class currently provides the clean interface that the rest of the
system uses.  The internal encoder can be swapped for a grid-based
encoder without changing any other code.

Usage
-----
    from poker_rl_lab.representation.card_encoder    import CardEncoder
    from poker_rl_lab.representation.hole_cards_encoder import HoleCardsEncoder
    from poker_rl_lab.envs.observations              import HoleCardsObs
    import torch

    card_enc = CardEncoder(rank_dim=16, suit_dim=8)       # 24-dim per card
    hand_enc = HoleCardsEncoder(card_enc, output_dim=64)  # 64-dim hand

    # From a batch of hands  (batch_size=32, 2 hole cards each)
    card_ids = torch.randint(0, 52, (32, 2))
    emb = hand_enc(card_ids)   # (32, 64)

    # From a single HoleCardsObs (at inference time)
    obs  = HoleCardsObs(cards=(0, 51))    # 2c, As
    emb  = hand_enc.encode_obs(obs)       # (64,)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from poker_rl_lab.representation.card_encoder import CardEncoder


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

_POOL_STRATEGIES = ("mean", "sum", "max")


class HoleCardsEncoder(nn.Module):
    """
    Encodes hole cards into a fixed-size hand embedding.

    Parameters
    ----------
    card_encoder : CardEncoder
        Shared card embedding module.  Weights are shared with any other
        encoder (board, discard) that uses the same CardEncoder instance.
    output_dim : int, optional
        Desired embedding size.  If None or equal to card_encoder.output_dim,
        no projection layer is added and output_dim = card_encoder.output_dim.
        If different, a trainable Linear projection is appended.
    pool : str
        How to combine per-card embeddings into one hand embedding.
        One of "mean" (default), "sum", or "max".

    Attributes
    ----------
    output_dim : int
        Size of the embedding produced by forward().

    Notes
    -----
    Permutation-invariant: forward({A♠, K♦}) == forward({K♦, A♠}).
    Variable-length: works for any number of cards (2 for hold'em, 4 for Omaha).
    """

    def __init__(
        self,
        card_encoder: CardEncoder,
        output_dim: Optional[int] = None,
        pool: str = "mean",
    ) -> None:
        super().__init__()

        if pool not in _POOL_STRATEGIES:
            raise ValueError(
                f"pool must be one of {_POOL_STRATEGIES}, got {pool!r}"
            )

        self.card_encoder = card_encoder
        self.pool         = pool
        card_dim          = card_encoder.output_dim

        if output_dim is not None and output_dim != card_dim:
            self.proj      = nn.Linear(card_dim, output_dim)
            self.output_dim = output_dim
        else:
            self.proj       = None
            self.output_dim = card_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode one or more hands.

        Parameters
        ----------
        card_ids : torch.Tensor  (dtype=torch.long)
            Shape (..., n_cards).  The last dimension is the cards in
            each hand.  Leading dimensions are batched freely.

        Returns
        -------
        torch.Tensor  float32
            Shape (..., output_dim).  One embedding per hand.

        Examples
        --------
        Single hand (2 hole cards)::

            ids = torch.tensor([0, 51], dtype=torch.long)   # 2c, As
            emb = encoder(ids)                               # (output_dim,)

        Batch of 32 hands (4 Omaha hole cards each)::

            ids = torch.randint(0, 52, (32, 4))
            emb = encoder(ids)                               # (32, output_dim)
        """
        # (..., n_cards, card_dim)
        card_embs = self.card_encoder(card_ids)

        # Pool across the cards dimension (axis -2)
        if self.pool == "mean":
            hand_emb = card_embs.mean(dim=-2)
        elif self.pool == "sum":
            hand_emb = card_embs.sum(dim=-2)
        elif self.pool == "max":
            hand_emb = card_embs.max(dim=-2).values

        # Optional projection
        if self.proj is not None:
            hand_emb = self.proj(hand_emb)

        return hand_emb

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def encode_obs(self, obs) -> torch.Tensor:
        """
        Encode a HoleCardsObs directly.

        Parameters
        ----------
        obs : HoleCardsObs
            From StructuredObservation.hole_cards.  obs.cards is a tuple
            of card ids.

        Returns
        -------
        torch.Tensor  float32
            Shape (output_dim,).
        """
        ids = torch.tensor(obs.cards, dtype=torch.long, device=self.card_encoder.device)
        return self.forward(ids)

    # ------------------------------------------------------------------
    # Pre-trained weight loading (stub — architecture TBD)
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        keras_weights: list,
        card_encoder: CardEncoder,
        output_dim: Optional[int] = None,
        pool: str = "mean",
    ) -> "HoleCardsEncoder":
        """
        [STUB]  Construct a HoleCardsEncoder initialised with weights
        extracted from the pre-trained Keras hand encoder.

        This method is a placeholder.  Implementing it requires:
          1. The Keras encoder architecture from ML/training/trainer.py.
          2. A confirmed mapping between Keras layer names and the
             PyTorch parameters of this module (or a companion
             GridHandEncoder that mirrors the Keras architecture).

        Parameters
        ----------
        keras_weights : list
            Output of keras_model.get_weights() — a list of numpy arrays.
        card_encoder : CardEncoder
            Shared card embedding module.
        output_dim : int, optional
            Desired output size.
        pool : str
            Pooling strategy.

        Returns
        -------
        HoleCardsEncoder
            Instance with pre-trained weights loaded.

        Raises
        ------
        NotImplementedError
            Until the Keras architecture is known and matched here.
        """
        raise NotImplementedError(
            "from_pretrained() requires the Keras encoder architecture.\n"
            "Please share ML/training/trainer.py (or the file that defines\n"
            "get_custom_objects and the encoder model) so the architecture\n"
            "can be mirrored in PyTorch and the weights transferred."
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"pool={self.pool!r}, "
            f"card_dim={self.card_encoder.output_dim}, "
            f"output_dim={self.output_dim}"
        )
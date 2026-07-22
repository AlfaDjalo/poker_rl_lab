"""
representation/representation_model.py — Combines all component encoders
into one latent representation, per the RL Lab implementation guide (§5,
"Hierarchical Representation"):

    Hole Cards -> Hole Embedding
    Board      -> Board Embedding
    Point      -> Point Embedding
    Game       -> Game Embedding
    History    -> History Embedding
                        v
                Representation Model
                        v
                  Latent State

Position is not in the guide's hierarchical diagram but IS listed in
§5's "Additional Learned Embeddings" and in §6's transformer token list,
so it is included here as a sixth input, pooled in the same way as the
others.

Architecture: MLP-concatenation now, transformer-of-tokens later
------------------------------------------------------------------
§6 ("Neural Architecture") is explicit that the RL Lab should support
multiple architectures, starting with MLP, with the "preferred long-term
architecture" being a transformer where each observation component is
its own token (Hole, Board 1, Board 2, Point 1, Point 2, Position,
Stack, Pot, Betting History, Game).

This module implements the MLP stage: each component encoder still
produces its own embedding (unchanged, and reusable verbatim once a
transformer variant is built), but because a plain MLP needs a
fixed-size input, variable-count components (boards, points) are
mean-pooled down to one vector each before concatenation. This is a
real architectural simplification relative to the guide's preferred
design (mean-pooling discards "which board" / "which point" identity
that per-token attention would preserve) — acceptable for the initial
curriculum stages (push-fold, single-board river) where there is
exactly one board and one point, so pooling is a no-op. It will matter
starting at "Double Board" in the curriculum (§8); a TokenRepresentationModel
variant belongs in a follow-up file, reusing every sub-encoder here
unchanged and replacing only the final combination step.

TensorFlow / PyTorch boundary
------------------------------
BaseHandEncoder (hole-card embedding, base_hand_encoder.py) is
implemented today by PretrainedHandEncoder, which is TF/Keras — see
that file's closing NOTE for the unresolved Option A/B/C decision. This
module takes **Option B** for now: the hand encoder is treated as a
frozen, non-PyTorch feature source. Its numpy output is converted to a
torch tensor with requires_grad=False every forward pass, so the rest
of the Representation Model (which IS PyTorch, per card_encoder.py /
board_encoder.py / point_encoder.py / game_encoder.py /
position_encoder.py / history_encoder.py) can still be trained with
PPO; only the hole-card embedding itself is frozen until Option A
(porting the Keras weights into a torch.nn.Module) is done. That is a
one-line swap here (replace self.hand_encoder's type) once available.

Dependencies
------------
    PyTorch, numpy. Depends on every encoder in representation/:
    card_encoder (indirectly, via board_encoder), board_encoder,
    point_encoder, game_encoder, position_encoder, history_encoder,
    base_hand_encoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn

from representation.base_hand_encoder import BaseHandEncoder
from representation.board_encoder import BoardEncoder
from representation.point_encoder import PointEncoder, PointFeatures
from representation.game_encoder import GameEncoder, GameFeatures
from representation.position_encoder import PositionEncoder, PositionFeatures
from representation.history_encoder import HistoryEncoder, HistoryEvent


# ---------------------------------------------------------------------------
# Plain-data input bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepresentationInput:
    """
    Everything the Representation Model needs for one decision point.

    Framework-independent (no torch types) so callers building this from
    a StructuredObservation don't need to import torch just to construct
    one. All fields are plain Python / the encoder-facing dataclasses
    already defined in the sibling encoder modules.

    Attributes
    ----------
    hole_card_ids : Sequence[int]
        Hero's hole card ids, engine convention (card_utils.py).
    board_card_ids : Sequence[Sequence[int]]
        One entry per board. A single-board game (e.g. holdem) passes a
        length-1 sequence; double_board passes length-2. Each inner
        sequence may be empty (preflop, no cards dealt on that board yet).
    points : Sequence[PointFeatures]
        One entry per scoring point (rules.points). Length 1 for holdem's
        "main" point; length 2+ for multi-point games (see game_encoder.py
        docstring).
    game : GameFeatures
        This variant's betting/board/payout/showdown characteristics.
    position : PositionFeatures
        Hero's seat position at this decision point.
    history : Sequence[HistoryEvent]
        Actions taken so far this hand, chronological. Empty at the
        first decision point.
    """
    hole_card_ids: Sequence[int]
    board_card_ids: Sequence[Sequence[int]]
    points: Sequence[PointFeatures]
    game: GameFeatures
    position: PositionFeatures
    history: Sequence[HistoryEvent] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.board_card_ids) < 1:
            raise ValueError("board_card_ids must have at least one board")
        if len(self.points) < 1:
            raise ValueError("points must have at least one point")


# ---------------------------------------------------------------------------
# Representation Model
# ---------------------------------------------------------------------------

class RepresentationModel(nn.Module):
    """
    Combines hole/board/point/game/position/history embeddings into one
    fixed-size latent vector.

    Parameters
    ----------
    hand_encoder : BaseHandEncoder
        Hole-card embedding source. Currently TF/Keras-backed
        (PretrainedHandEncoder) — see module docstring's "TensorFlow /
        PyTorch boundary" section. Treated as frozen: its output is
        detached from any graph before concatenation.
    board_encoder : BoardEncoder
    point_encoder : PointEncoder
    game_encoder : GameEncoder
    position_encoder : PositionEncoder
    history_encoder : HistoryEncoder
        The six component encoders. Pass already-constructed instances
        so callers control hyperparameters and, critically, so the SAME
        card_encoder.CardEncoder instance can be shared between
        board_encoder and anywhere else per-card embeddings are needed
        (see board_encoder.py's docstring on why sharing matters).
    hidden_dim : int
        Hidden width of the final combination MLP. Default 256.
    output_dim : int
        Final latent state dimensionality — what the Policy Network
        consumes. Default 128.
    dropout : float
        Dropout in the combination MLP. Default 0.0.
    """

    def __init__(
        self,
        hand_encoder: BaseHandEncoder,
        board_encoder: BoardEncoder,
        point_encoder: PointEncoder,
        game_encoder: GameEncoder,
        position_encoder: PositionEncoder,
        history_encoder: HistoryEncoder,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")

        self.hand_encoder = hand_encoder  # NOT an nn.Module (TF-backed); stored, not registered
        self.board_encoder = board_encoder
        self.point_encoder = point_encoder
        self.game_encoder = game_encoder
        self.position_encoder = position_encoder
        self.history_encoder = history_encoder

        self.output_dim = output_dim

        combined_dim = (
            hand_encoder.embedding_dim
            + board_encoder.output_dim
            + point_encoder.output_dim
            + game_encoder.output_dim
            + position_encoder.output_dim
            + history_encoder.output_dim
        )

        self.combine = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: RepresentationInput) -> torch.Tensor:
        """
        Embed one decision point into a latent vector.

        Parameters
        ----------
        x : RepresentationInput

        Returns
        -------
        torch.Tensor
            Shape (output_dim,).
        """
        device = self.combine[0].weight.device

        # -- Hole cards (frozen TF source; see module docstring) --------
        hole_np = self.hand_encoder.encode_hole(x.hole_card_ids)
        hole_embed = torch.from_numpy(hole_np).float().to(device)
        hole_embed = hole_embed.detach()

        # -- Boards: encode each, mean-pool across boards ----------------
        board_embeds = torch.stack(
            [self.board_encoder(cards) for cards in x.board_card_ids],
            dim=0,
        )  # (num_boards, board_dim)
        board_pooled = board_embeds.mean(dim=0)  # (board_dim,)

        # -- Points: batch-encode, mean-pool across points ---------------
        point_embeds = self.point_encoder.forward_batch(x.points)  # (num_points, point_dim)
        point_pooled = point_embeds.mean(dim=0)  # (point_dim,)

        # -- Game (single token) ------------------------------------------
        game_embed = self.game_encoder(x.game)  # (game_dim,)

        # -- Position (single token) ---------------------------------------
        position_embed = self.position_encoder(x.position)  # (position_dim,)

        # -- History (single token, GRU-pooled internally) ------------------
        history_embed = self.history_encoder(x.history)  # (history_dim,)

        combined = torch.cat(
            [
                hole_embed,
                board_pooled,
                point_pooled,
                game_embed,
                position_embed,
                history_embed,
            ],
            dim=-1,
        )

        return self.combine(combined)

    # ------------------------------------------------------------------
    # Batch forward
    # ------------------------------------------------------------------

    def forward_batch(self, batch: Sequence[RepresentationInput]) -> torch.Tensor:
        """
        Embed a batch of decision points.

        Parameters
        ----------
        batch : Sequence[RepresentationInput]

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).

        Notes
        -----
        This is a straightforward per-example loop, NOT a vectorised
        batch implementation — board_encoder/point_encoder/etc. all
        expose forward_batch() variants of their own, but wiring those
        through here (padding boards/points per example, building a
        batched hole-card grid for the TF hand encoder, ...) is real
        additional work deferred to a follow-up once this un-batched
        version is confirmed correct end-to-end. Fine for the curriculum's
        early stages (push-fold, single-board river) where episodes are
        short and PPO rollout batch sizes are modest; revisit if this
        becomes a training-speed bottleneck.
        """
        if not batch:
            raise ValueError("batch must be non-empty")
        return torch.stack([self.forward(x) for x in batch], dim=0)
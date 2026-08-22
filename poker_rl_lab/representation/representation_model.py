"""
representation/representation_model.py — Combines all component encoders
into one latent representation.

Made-hand token (added)
--------------------------
A third pretrained-backed component, `made_hand_encoder`
(representation.composite_board_encoder.CompositeBoardEncoder in its
"made_hand" role — see that module's docstring), is pooled alongside
Board/Point: for each point index, hole_card_ids (shared across points)
plus that point's own made_hand_board_card_ids are decomposed into legal
5-card candidate hands (per showdown_type — Hold'em/Omaha/Make-5, see
representation/showdown_constraints.py) and pooled through the SAME
frozen pretrained base as hole cards and board texture, via its own
separately-trained adapter. This closes the gap where the architecture
previously had no representation of actual made-hand strength (hole and
board were encoded in total isolation from each other).

`board_encoder` here is still the "board texture" role instance of
CompositeBoardEncoder (window=3) — NOT the old from-scratch BoardEncoder
class, which has been removed from the wiring (see
models/actor_critic_factory.py). Both board_encoder and made_hand_encoder
are duck-typed on their respective forward_batch()/
forward_batch_made_hand() surfaces, not on a specific class, so this
module has no import-time dependency on CompositeBoardEncoder itself.

[... everything else from the original module docstring unchanged ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn

from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.point_encoder import PointEncoder, PointFeatures
from poker_rl_lab.representation.game_encoder import GameEncoder, GameFeatures
from poker_rl_lab.representation.position_encoder import PositionEncoder, PositionFeatures
from poker_rl_lab.representation.history_encoder import HistoryEncoder, HistoryEvent
from poker_rl_lab.representation.stack_encoder import StackEncoder, StackFeatures


# ---------------------------------------------------------------------------
# Plain-data input bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepresentationInput:
    """
    Everything the Representation Model needs for one decision point.

    Attributes
    ----------
    hole_card_ids : Sequence[int]
        Hero's hole card ids. Shared across every point/board — CAP
        games in this codebase deal one hole-card set per hero, used
        identically against every point.
    board_card_ids : Sequence[Sequence[int]]
        One entry per BOARD (see boards_from_node_cards in
        observation_adapter.py — deduplicated node_sets, first-
        appearance order). Feeds board_encoder (texture role).
    made_hand_board_card_ids : Sequence[Sequence[int]]
        One entry per POINT (rules.points order — same length and
        order as `points` below), each the board cards belonging to
        THAT point specifically (union of that point's own
        node_sets — built directly from point_def.node_sets, not from
        the deduplicated `board_card_ids` list, since a point spanning
        multiple node_sets, e.g. a "best-of" point, needs its own union
        regardless of how those node_sets were bucketed into
        `board_card_ids`). Combined with hole_card_ids and that same
        index's points[i].showdown_type, feeds made_hand_encoder.
    points : Sequence[PointFeatures]
        One entry per scoring point.
    game : GameFeatures
    position : PositionFeatures
    stack : StackFeatures
    history : Sequence[HistoryEvent]
    """
    hole_card_ids: Sequence[int]
    board_card_ids: Sequence[Sequence[int]]
    made_hand_board_card_ids: Sequence[Sequence[int]]
    points: Sequence[PointFeatures]
    game: GameFeatures
    position: PositionFeatures
    stack: StackFeatures
    history: Sequence[HistoryEvent] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(self.board_card_ids) < 1:
            raise ValueError("board_card_ids must have at least one board")
        if len(self.points) < 1:
            raise ValueError("points must have at least one point")
        if len(self.made_hand_board_card_ids) != len(self.points):
            raise ValueError(
                f"made_hand_board_card_ids must have one entry per point "
                f"({len(self.points)}), got {len(self.made_hand_board_card_ids)}"
            )


# ---------------------------------------------------------------------------
# Representation Model
# ---------------------------------------------------------------------------

class RepresentationModel(nn.Module):
    """
    Combines hole/board/made-hand/point/game/position/stack/history
    embeddings into one fixed-size latent vector.

    Parameters
    ----------
    hand_encoder : BaseHandEncoder
        Hole-card embedding (frozen base + adapter, or TF-native).
    board_encoder
        Board-TEXTURE role. Duck-typed: forward_batch(batch_card_ids)
        -> Tensor(batch, output_dim), .output_dim.
    made_hand_encoder
        Made-HAND role (NEW). Duck-typed:
        forward_batch_made_hand(batch_hole_ids, batch_board_ids,
        batch_showdown_types) -> Tensor(batch, output_dim), .output_dim.
    point_encoder, game_encoder, position_encoder, history_encoder,
    stack_encoder
        Unchanged.
    hidden_dim, output_dim, dropout
        Unchanged.
    """

    def __init__(
        self,
        hand_encoder: BaseHandEncoder,
        board_encoder,
        made_hand_encoder,
        point_encoder: PointEncoder,
        game_encoder: GameEncoder,
        position_encoder: PositionEncoder,
        history_encoder: HistoryEncoder,
        stack_encoder: StackEncoder,
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
        self.made_hand_encoder = made_hand_encoder
        self.point_encoder = point_encoder
        self.game_encoder = game_encoder
        self.position_encoder = position_encoder
        self.history_encoder = history_encoder
        self.stack_encoder = stack_encoder

        self.output_dim = output_dim

        combined_dim = (
            hand_encoder.embedding_dim
            + board_encoder.output_dim
            + made_hand_encoder.output_dim
            + point_encoder.output_dim
            + game_encoder.output_dim
            + position_encoder.output_dim
            + history_encoder.output_dim
            + stack_encoder.output_dim
        )

        self.combine = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    # ------------------------------------------------------------------
    # Forward (single example)
    # ------------------------------------------------------------------

    def forward(self, x: RepresentationInput) -> torch.Tensor:
        return self.forward_batch([x])[0]

    # ------------------------------------------------------------------
    # Batch forward
    # ------------------------------------------------------------------

    def forward_batch(self, batch: Sequence[RepresentationInput]) -> torch.Tensor:
        """
        Embed a batch of decision points. See class/module docstrings
        for the made-hand addition; everything else matches the
        original batching contract (one sub-encoder call per component
        per batch, boards/points/made-hands pooled per-index across the
        batch, not once per example).
        """
        if not batch:
            raise ValueError("batch must be non-empty")

        device = self.combine[0].weight.device

        # -- Hole cards ------------------------------------------------
        if hasattr(self.hand_encoder, "forward_batch_torch"):
            hole_embed = self.hand_encoder.forward_batch_torch(
                [x.hole_card_ids for x in batch]
            ).to(device)
        else:
            hole_np = self.hand_encoder.encode_batch([x.hole_card_ids for x in batch])
            hole_embed = torch.from_numpy(hole_np).float().to(device).detach()

        # -- Boards (texture) -------------------------------------------
        num_boards = len(batch[0].board_card_ids)
        if any(len(x.board_card_ids) != num_boards for x in batch):
            raise ValueError(
                "RepresentationModel.forward_batch() requires every example "
                "in the batch to have the same number of boards."
            )
        if num_boards == 1:
            board_pooled = self.board_encoder.forward_batch(
                [x.board_card_ids[0] for x in batch]
            )
        else:
            board_per_index = torch.stack(
                [
                    self.board_encoder.forward_batch([x.board_card_ids[b] for x in batch])
                    for b in range(num_boards)
                ],
                dim=0,
            )
            board_pooled = board_per_index.mean(dim=0)

        # -- Points (+ made hands, pooled per point index) --------------
        num_points = len(batch[0].points)
        if any(len(x.points) != num_points for x in batch):
            raise ValueError(
                "RepresentationModel.forward_batch() requires every example "
                "in the batch to have the same number of points."
            )

        if num_points == 1:
            point_pooled = self.point_encoder.forward_batch([x.points[0] for x in batch])
            made_hand_pooled = self.made_hand_encoder.forward_batch_made_hand(
                [x.hole_card_ids for x in batch],
                [x.made_hand_board_card_ids[0] for x in batch],
                [x.points[0].showdown_type for x in batch],
            )
        else:
            point_per_index = torch.stack(
                [
                    self.point_encoder.forward_batch([x.points[p] for x in batch])
                    for p in range(num_points)
                ],
                dim=0,
            )
            point_pooled = point_per_index.mean(dim=0)

            made_hand_per_index = torch.stack(
                [
                    self.made_hand_encoder.forward_batch_made_hand(
                        [x.hole_card_ids for x in batch],
                        [x.made_hand_board_card_ids[p] for x in batch],
                        [x.points[p].showdown_type for x in batch],
                    )
                    for p in range(num_points)
                ],
                dim=0,
            )
            made_hand_pooled = made_hand_per_index.mean(dim=0)

        # -- Game / Position / History / Stack ---------------------------
        game_embed = self.game_encoder.forward_batch([x.game for x in batch])
        position_embed = self.position_encoder.forward_batch([x.position for x in batch])
        history_embed = self.history_encoder.forward_batch([x.history for x in batch])
        stack_embed = self.stack_encoder.forward_batch([x.stack for x in batch])

        combined = torch.cat(
            [
                hole_embed,
                board_pooled,
                made_hand_pooled,
                point_pooled,
                game_embed,
                position_embed,
                history_embed,
                stack_embed,
            ],
            dim=-1,
        )

        return self.combine(combined)
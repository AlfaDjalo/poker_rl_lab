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

Stack/Pot (added)
------------------
§6's transformer token list is ("Hole", "Board 1", "Board 2", "Point 1",
"Point 2", "Position", "Stack", "Pot", "Betting History", "Game") — this
module previously implemented every token EXCEPT "Stack"/"Pot": neither
RepresentationInput nor this class's forward() carried any signal for
the hero's remaining stack, the pot size, or bet-to-call.
StackEncoder (representation/stack_encoder.py) closes that gap as a
seventh input, pooled into the combine MLP exactly like Position — a
single per-decision token, not a variable-count one like Board/Point.
(NOTE: this wiring was previously missing from this file entirely even
though actor_critic_factory.py and observation_adapter.py already
assumed it existed — build_actor_critic() would have raised
"unexpected keyword argument 'stack_encoder'" the moment it tried to
construct a RepresentationModel. Restored here.)

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
position_encoder.py / history_encoder.py / stack_encoder.py) can still
be trained with PPO; only the hole-card embedding itself is frozen
until Option A (porting the Keras weights into a torch.nn.Module) is
done. That is a one-line swap here (replace self.hand_encoder's type)
once available.

Batched forward_batch() (previously a per-example Python loop)
------------------------------------------------------------------
forward_batch() used to call self.forward(x) once per example in a
plain Python for-loop — including, per example, one TF/Keras call to
encode a single (14,4,1) hole-card grid. TF's per-call dispatch
overhead dominates that cost (the actual convolution is tiny), so a
batch of N examples paid N separate TF invocations instead of one. This
is the dominant cost of PPOTrainer.update()'s minibatch passes (each
one calls forward_batch()), so it is fixed here: every sub-encoder
already exposes its own forward_batch()/encode_batch() (board_encoder,
point_encoder, game_encoder, position_encoder, history_encoder,
stack_encoder, and BaseHandEncoder.encode_batch()) — this method now
calls each of those exactly ONCE per forward_batch() call, instead of
once per example. Boards/points are handled per-board-index /
per-point-index across the whole batch (see forward_batch()'s docstring
for the uniform-count assumption this requires).

Dependencies
------------
    PyTorch, numpy. Depends on every encoder in representation/:
    card_encoder (indirectly, via board_encoder), board_encoder,
    point_encoder, game_encoder, position_encoder, history_encoder,
    stack_encoder, base_hand_encoder.
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
from representation.stack_encoder import StackEncoder, StackFeatures


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
    stack : StackFeatures
        Hero's stack/pot/bet-to-call state at this decision point (see
        stack_encoder.py). Unlike `game` (one constant value per hand),
        this changes at every decision point within a hand, same as
        `position` and `history`.
    history : Sequence[HistoryEvent]
        Actions taken so far this hand, chronological. Empty at the
        first decision point.
    """
    hole_card_ids: Sequence[int]
    board_card_ids: Sequence[Sequence[int]]
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


# ---------------------------------------------------------------------------
# Representation Model
# ---------------------------------------------------------------------------

class RepresentationModel(nn.Module):
    """
    Combines hole/board/point/game/position/stack/history embeddings
    into one fixed-size latent vector.

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
    stack_encoder : StackEncoder
        The seven component encoders. Pass already-constructed instances
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
        self.point_encoder = point_encoder
        self.game_encoder = game_encoder
        self.position_encoder = position_encoder
        self.history_encoder = history_encoder
        self.stack_encoder = stack_encoder

        self.output_dim = output_dim

        combined_dim = (
            hand_encoder.embedding_dim
            + board_encoder.output_dim
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
        """
        Embed one decision point into a latent vector.

        Parameters
        ----------
        x : RepresentationInput

        Returns
        -------
        torch.Tensor
            Shape (output_dim,).

        Notes
        -----
        Used at rollout-collection time (one decision at a time — see
        models.policy_value_network.PokerActorCritic.act()), where
        batching isn't available because actions are chosen
        sequentially as the episode unfolds. For anything batched
        (PPO's evaluate_actions() / update()), forward_batch() below is
        used instead and is dramatically cheaper per-example — prefer
        it whenever a batch of RepresentationInputs is available.
        """
        return self.forward_batch([x])[0]

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
        Calls each sub-encoder's own forward_batch()/encode_batch()
        exactly ONCE for the whole batch (board_encoder, point_encoder,
        game_encoder, position_encoder, history_encoder, stack_encoder,
        and hand_encoder.encode_batch()) — not once per example. This
        matters most for hand_encoder: it's TF/Keras-backed, and TF's
        per-call dispatch overhead (tens of ms) dwarfs the actual
        computation for a single (14,4,1) grid, so collapsing N calls
        into 1 is the single biggest win available in this class.

        Boards and points are pooled per-example (mean across that
        example's boards / points, unchanged from forward()'s own
        semantics), but the encoder itself is invoked once per board
        INDEX / point INDEX across the whole batch — e.g. for a
        2-board game, board_encoder.forward_batch() is called twice
        total (once for every example's board 0, once for every
        example's board 1), not 2*batch_size times. This requires
        every example in the batch to have the SAME number of boards
        and the SAME number of points — true for any batch drawn from
        one game/curriculum-stage's rollouts (the normal case: one
        PPOTrainer trains against one env/game at a time), and checked
        explicitly below so a genuinely mixed batch fails loudly rather
        than silently mis-pooling. If you need to mix variants with
        different board/point counts in one batch, call forward() per
        example instead (or partition the batch by game_name first).
        """
        if not batch:
            raise ValueError("batch must be non-empty")

        device = self.combine[0].weight.device
        batch_size = len(batch)

        # -- Hole cards: ONE encode_batch call for the whole batch ---------
        hole_np = self.hand_encoder.encode_batch([x.hole_card_ids for x in batch])
        hole_embed = torch.from_numpy(hole_np).float().to(device).detach()  # (batch, hand_dim)

        # -- Boards: one forward_batch() call per board index --------------
        num_boards = len(batch[0].board_card_ids)
        if any(len(x.board_card_ids) != num_boards for x in batch):
            raise ValueError(
                "RepresentationModel.forward_batch() requires every example "
                "in the batch to have the same number of boards (mixing "
                "e.g. holdem and double_board examples in one batch isn't "
                "supported by the batched path). Call forward() per example "
                "instead, or partition the batch by game_name first."
            )
        if num_boards == 1:
            board_pooled = self.board_encoder.forward_batch(
                [x.board_card_ids[0] for x in batch]
            )  # (batch, board_dim) — mean over 1 board is a no-op
        else:
            board_per_index = torch.stack(
                [
                    self.board_encoder.forward_batch([x.board_card_ids[b] for x in batch])
                    for b in range(num_boards)
                ],
                dim=0,
            )  # (num_boards, batch, board_dim)
            board_pooled = board_per_index.mean(dim=0)  # (batch, board_dim)

        # -- Points: one forward_batch() call per point index ----------------
        num_points = len(batch[0].points)
        if any(len(x.points) != num_points for x in batch):
            raise ValueError(
                "RepresentationModel.forward_batch() requires every example "
                "in the batch to have the same number of points — same "
                "rationale as the board-count check above."
            )
        if num_points == 1:
            point_pooled = self.point_encoder.forward_batch(
                [x.points[0] for x in batch]
            )  # (batch, point_dim)
        else:
            point_per_index = torch.stack(
                [
                    self.point_encoder.forward_batch([x.points[p] for x in batch])
                    for p in range(num_points)
                ],
                dim=0,
            )  # (num_points, batch, point_dim)
            point_pooled = point_per_index.mean(dim=0)  # (batch, point_dim)

        # -- Game (single token per example) -----------------------------------
        game_embed = self.game_encoder.forward_batch([x.game for x in batch])  # (batch, game_dim)

        # -- Position (single token per example) ---------------------------------
        position_embed = self.position_encoder.forward_batch(
            [x.position for x in batch]
        )  # (batch, position_dim)

        # -- History (single token per example, GRU-pooled internally) -------------
        history_embed = self.history_encoder.forward_batch(
            [x.history for x in batch]
        )  # (batch, history_dim)

        # -- Stack/Pot (single token per example) ----------------------------------
        stack_embed = self.stack_encoder.forward_batch(
            [x.stack for x in batch]
        )  # (batch, stack_dim)

        combined = torch.cat(
            [
                hole_embed,
                board_pooled,
                point_pooled,
                game_embed,
                position_embed,
                history_embed,
                stack_embed,
            ],
            dim=-1,
        )

        return self.combine(combined)
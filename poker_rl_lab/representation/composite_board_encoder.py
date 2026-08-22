"""
representation/composite_board_encoder.py — Pretrained-encoder-backed
card-set embedding, serving TWO roles depending on construction:

  1. Board texture (mode="board"): decomposes a single N-card board into
     native-size windows (default 3, since a 3-card set is the closest
     match to what the pretrained encoder was actually trained to
     evaluate standalone — see module history / actor_critic_factory.py
     for why 5-card windows were WRONG for a bare board).
  2. Made hand (mode="made_hand"): decomposes hole ∪ board into legal
     5-card candidate hands, respecting a showdown_type-specific
     CardUsageConstraint (representation/showdown_constraints.py) —
     Hold'em (any split), Omaha (exactly 2 hole + 3 board), Make-5
     (exactly all board cards + however many hole cards fill out 5).

Both roles share the same decompose -> batch-encode-via-frozen-base ->
learned-attention-pool -> project pipeline; only how the index subsets
are built differs. A single instance is constructed for ONE role only
(via the `window`/`mode` constructor args) — board texture and made
hand should be two SEPARATE instances (own attn_scorer/project), even
though they may share the same frozen `hand_encoder` base. See
models/actor_critic_factory.py for how the three roles (hole-only,
board-texture, made-hand) are wired as three adapters over one base.

Frozen base + trainable adapter
----------------------------------
Unchanged from the original design: hand_encoder's output is ALWAYS
detached before pooling; only attn_scorer and project are trained.

Dependencies
------------
    PyTorch, numpy. representation.base_hand_encoder.BaseHandEncoder
    (duck-typed). representation.showdown_constraints (made_hand mode
    only — not imported at all for mode="board" usage beyond the
    top-level import, which is cheap/pure-Python).
"""

from __future__ import annotations

import itertools
import logging
import sys
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.showdown_constraints import (
    CardUsageConstraint,
    get_constraint,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-list decomposition (board-texture role, and anything else that
# wants "decompose one N-card set into native-size windows")
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _decompose_indices(
    n: int, max_subsets: int, window: int = 5
) -> Tuple[Tuple[int, ...], ...]:
    """
    Index-tuples (positions into an N-card list) covering the N-card set
    with subsets sized to the pretrained encoder's native training sizes.

    Parameters
    ----------
    n : int
        Card-set size. Must be >= 2 (0 and 1 are handled by dedicated
        placeholder paths in the calling encoder, not here).
    max_subsets : int
        Hard cap on returned subsets for large N — see class docstring.
    window : int
        Preferred subset size for n >= 5 (the only case with real
        choice — n in {2,3,4} always has a fixed, unambiguous native
        decomposition regardless of `window`). Callers pass window=3
        for board-texture (closest to the encoder's flop-sized training
        distribution) or window=5 for made-hand (a real candidate final
        hand). Clamped to n if window > n.
    """
    if n == 2:
        return ((0, 1),)
    if n == 3:
        return ((0, 1, 2),)
    if n == 4:
        # Four 3-card "drop-one" subsets — reasonable coverage
        # regardless of role at this size.
        return tuple(itertools.combinations(range(4), 3))

    w = min(window, n)
    combos = list(itertools.combinations(range(n), w))
    if len(combos) > max_subsets:
        stride = len(combos) / max_subsets
        combos = [combos[int(i * stride)] for i in range(max_subsets)]
    return tuple(combos)


# ---------------------------------------------------------------------------
# Two-list, constraint-aware decomposition (made-hand role)
# ---------------------------------------------------------------------------

def _decompose_made_hand_indices(
    n_hole: int,
    n_board: int,
    constraint: CardUsageConstraint,
    max_subsets: int,
) -> Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]:
    """
    Every legal (hole_index_tuple, board_index_tuple) pair — each
    totaling 5 cards — satisfying `constraint`, given the actual number
    of hole/board cards currently available.

    Returns
    -------
    Tuple[Tuple[Tuple[int, ...], Tuple[int, ...]], ...]
        Possibly EMPTY — e.g. Omaha before the flop (n_board < 3), or
        any street where n_hole + n_board < 5 (true of every street
        before the river in Hold'em). Callers must handle the empty
        case via a learned placeholder (see CompositeBoardEncoder.
        no_valid_hand_embed), not assume >= 1 split always exists.

    Notes
    -----
    Capped/strided at max_subsets across the FULL combined list of
    (hole_combo, board_combo) pairs — not per-split — so the same
    determinism guarantee _decompose_indices already documents (fixed
    itertools order, evenly strided sample) applies here too.
    """
    splits = constraint.valid_splits(n_hole, n_board)
    if not splits:
        return ()

    all_combos: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
    for hole_count, board_count in splits:
        hole_combos = (
            list(itertools.combinations(range(n_hole), hole_count))
            if hole_count > 0 else [()]
        )
        board_combos = (
            list(itertools.combinations(range(n_board), board_count))
            if board_count > 0 else [()]
        )
        for hc in hole_combos:
            for bc in board_combos:
                all_combos.append((hc, bc))

    if len(all_combos) > max_subsets:
        stride = len(all_combos) / max_subsets
        all_combos = [all_combos[int(i * stride)] for i in range(max_subsets)]
    return tuple(all_combos)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CompositeBoardEncoder(nn.Module):
    """
    Card-set embedding built from a pretrained BaseHandEncoder, serving
    either the board-texture role or the made-hand role — see module
    docstring. One instance = one role; construct TWO instances (with
    separate trainable adapters) to use both.

    Parameters
    ----------
    hand_encoder : BaseHandEncoder
        Frozen. NOT registered as an nn.Module.
    output_dim : int
        Default 32.
    attn_hidden_dim : int
        Default 32.
    project_hidden_dim : int, optional
        2-layer MLP adapter width if given, else a single Linear.
    max_subsets : int
        Decomposition cap. Default 12.
    window : int
        Preferred native window size for the board-texture (single-
        list) path — see _decompose_indices. Default 5 (unchanged
        behaviour for any existing caller not yet updated); pass 3
        explicitly for board texture, per module docstring.
    debug : bool
        If True, prints pre/post-project variance for the first few
        forward_batch()/forward_batch_made_hand() calls. Default False.
    """

    def __init__(
        self,
        hand_encoder: BaseHandEncoder,
        output_dim: int = 32,
        attn_hidden_dim: int = 32,
        project_hidden_dim: Optional[int] = None,
        max_subsets: int = 12,
        window: int = 5,
        debug: bool = False,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        if max_subsets < 1:
            raise ValueError(f"max_subsets must be >= 1, got {max_subsets}")
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")

        self.hand_encoder = hand_encoder  # NOT an nn.Module (frozen, TF-backed)
        self.output_dim = output_dim
        self.max_subsets = max_subsets
        self.window = window
        self.debug = debug
        self._debug_calls_board = 0
        self._debug_calls_made_hand = 0

        embed_dim = hand_encoder.embedding_dim

        self.attn_scorer = nn.Sequential(
            nn.Linear(embed_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )

        if project_hidden_dim is not None:
            self.project = nn.Sequential(
                nn.Linear(embed_dim, project_hidden_dim),
                nn.ReLU(),
                nn.Linear(project_hidden_dim, output_dim),
            )
        else:
            self.project = nn.Linear(embed_dim, output_dim)

        # N == 0 (board-texture path): preflop / no board dealt.
        self.empty_board_embed = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.empty_board_embed, std=0.02)

        # N == 1 (board-texture path): lone dealt node.
        self.single_card_embed = nn.Embedding(52, output_dim)

        # No legal 5-card split yet (made-hand path) — e.g. any street
        # before enough hole+board cards exist, or Omaha before the
        # flop. Distinct from empty_board_embed: this fires even when
        # hole cards ARE known, just not enough board (or vice versa).
        self.no_valid_hand_embed = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.no_valid_hand_embed, std=0.02)

    # ------------------------------------------------------------------
    # Board-texture role (single list) — forward / forward_batch
    # ------------------------------------------------------------------

    def forward(self, card_ids: Optional[Sequence[int]]) -> torch.Tensor:
        """Embed a single board. Returns shape (output_dim,)."""
        return self.forward_batch([card_ids])[0]

    def forward_batch(
        self, batch_card_ids: Sequence[Optional[Sequence[int]]]
    ) -> torch.Tensor:
        """
        Embed a batch of boards (or any single card-set list), each of
        potentially different size, decomposed via self.window.
        """
        batch_size = len(batch_card_ids)
        if batch_size == 0:
            raise ValueError("batch_card_ids must be non-empty")

        device = self.empty_board_embed.device

        empty_idx: List[int] = []
        single_idx: List[int] = []
        decomposed_idx: List[int] = []
        subsets_per_example: List[List[Tuple[int, ...]]] = [None] * batch_size  # type: ignore[list-item]

        for i, cards in enumerate(batch_card_ids):
            n = len(cards) if cards else 0
            if n == 0:
                empty_idx.append(i)
            elif n == 1:
                single_idx.append(i)
            else:
                idx_tuples = _decompose_indices(n, self.max_subsets, self.window)
                subsets_per_example[i] = [
                    tuple(cards[p] for p in idxs) for idxs in idx_tuples
                ]
                decomposed_idx.append(i)

        out = torch.zeros(batch_size, self.output_dim, device=device)

        if empty_idx:
            out[empty_idx] = self.empty_board_embed

        if single_idx:
            card_ids_t = torch.tensor(
                [batch_card_ids[i][0] for i in single_idx],
                dtype=torch.long, device=device,
            )
            out[single_idx] = self.single_card_embed(card_ids_t)

        if decomposed_idx:
            pooled = self._encode_and_pool(
                [subsets_per_example[i] for i in decomposed_idx], device
            )
            projected = self.project(pooled)
            out[decomposed_idx] = projected
            self._maybe_debug_log("board", pooled, projected)

        return out

    # ------------------------------------------------------------------
    # Made-hand role (hole + board, constraint-aware)
    # ------------------------------------------------------------------

    def forward_made_hand(
        self,
        hole_card_ids: Sequence[int],
        board_card_ids: Sequence[int],
        showdown_type: int,
    ) -> torch.Tensor:
        """Embed one made hand. Returns shape (output_dim,)."""
        return self.forward_batch_made_hand(
            [hole_card_ids], [board_card_ids], [showdown_type]
        )[0]

    def forward_batch_made_hand(
        self,
        batch_hole_card_ids: Sequence[Sequence[int]],
        batch_board_card_ids: Sequence[Sequence[int]],
        batch_showdown_types: Sequence[int],
    ) -> torch.Tensor:
        """
        Embed a batch of (hole, board) pairs as pooled 5-card
        candidate-made-hand embeddings, per each example's showdown_type
        constraint (representation/showdown_constraints.py).

        Parameters
        ----------
        batch_hole_card_ids, batch_board_card_ids : Sequence[Sequence[int]]
            Same length, one entry per example.
        batch_showdown_types : Sequence[int]
            Same length — showdown_type int per example (matches
            poker_eval.ShowdownType / point_encoder.py's vocabulary).

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim). Examples with no legal 5-card
            split yet (see _decompose_made_hand_indices) get
            self.no_valid_hand_embed.
        """
        batch_size = len(batch_hole_card_ids)
        if batch_size == 0:
            raise ValueError("batch_hole_card_ids must be non-empty")
        if not (len(batch_board_card_ids) == len(batch_showdown_types) == batch_size):
            raise ValueError(
                f"batch_hole_card_ids ({batch_size}), batch_board_card_ids "
                f"({len(batch_board_card_ids)}), and batch_showdown_types "
                f"({len(batch_showdown_types)}) must all have the same length"
            )

        device = self.no_valid_hand_embed.device

        none_idx: List[int] = []
        decomposed_idx: List[int] = []
        subsets_per_example: List[List[Tuple[int, ...]]] = [None] * batch_size  # type: ignore[list-item]

        for i in range(batch_size):
            hole = batch_hole_card_ids[i] or []
            board = batch_board_card_ids[i] or []
            constraint = get_constraint(batch_showdown_types[i])
            idx_pairs = _decompose_made_hand_indices(
                len(hole), len(board), constraint, self.max_subsets
            )
            if not idx_pairs:
                none_idx.append(i)
                continue
            subsets_per_example[i] = [
                tuple(hole[p] for p in hc) + tuple(board[p] for p in bc)
                for hc, bc in idx_pairs
            ]
            decomposed_idx.append(i)

        out = torch.zeros(batch_size, self.output_dim, device=device)

        if none_idx:
            out[none_idx] = self.no_valid_hand_embed

        if decomposed_idx:
            pooled = self._encode_and_pool(
                [subsets_per_example[i] for i in decomposed_idx], device
            )
            projected = self.project(pooled)
            out[decomposed_idx] = projected
            self._maybe_debug_log("made_hand", pooled, projected)

        return out

    # ------------------------------------------------------------------
    # Shared: batched frozen-encoder call + learned attention pool
    # ------------------------------------------------------------------

    def _encode_and_pool(
        self, subsets_per_example: List[List[Tuple[int, ...]]], device
    ) -> torch.Tensor:
        """
        One batched hand_encoder.encode_batch() call across every
        subset of every example in subsets_per_example, then per-example
        learned-attention pooling. Shared by both forward_batch() and
        forward_batch_made_hand() — identical mechanics, only how the
        subsets were built differs upstream.
        """
        flat_subsets: List[Tuple[int, ...]] = []
        counts: List[int] = []
        for subs in subsets_per_example:
            counts.append(len(subs))
            flat_subsets.extend(subs)

        flat_np = self.hand_encoder.encode_batch([list(s) for s in flat_subsets])
        flat_embed = torch.from_numpy(flat_np).float().to(device).detach()

        max_subsets_in_batch = max(counts)
        embed_dim = flat_embed.shape[-1]
        rows = len(subsets_per_example)
        padded = torch.zeros(rows, max_subsets_in_batch, embed_dim, device=device)
        pad_mask = torch.ones(rows, max_subsets_in_batch, dtype=torch.bool, device=device)

        cursor = 0
        for row, count in enumerate(counts):
            padded[row, :count] = flat_embed[cursor: cursor + count]
            pad_mask[row, :count] = False
            cursor += count

        scores = self.attn_scorer(padded).squeeze(-1)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = (weights.unsqueeze(-1) * padded).sum(dim=1)  # (rows, embed_dim)
        return pooled

    def _maybe_debug_log(self, role: str, pooled: torch.Tensor, projected: torch.Tensor) -> None:
        counter_attr = f"_debug_calls_{role}"
        calls = getattr(self, counter_attr)
        if not self.debug or calls >= 3:
            return
        setattr(self, counter_attr, calls + 1)
        with torch.no_grad():
            pre_std = pooled.std(dim=0).mean().item()
            post_std = projected.std(dim=0).mean().item()
        msg = (
            f"[CompositeBoardEncoder:{role}] debug ({calls + 1}/3): "
            f"n_examples={pooled.shape[0]}, window={self.window}, "
            f"pre_project_std={pre_std:.6f}, post_project_std={post_std:.6f}"
        )
        logger.info(msg)
        print(msg, file=sys.stderr)
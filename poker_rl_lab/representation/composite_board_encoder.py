"""
representation/composite_board_encoder.py — Pretrained-encoder-backed
board embedding, decomposing any N-card board into subsets the
pretrained CardSetEncoder was actually trained on (2, 3, or 5 cards),
batch-encoding those subsets, and learned-pooling the results.

Why this exists
-----------------
representation.pretrained_hand_encoder.PretrainedHandEncoder wraps a
Keras CardSetEncoder trained on 2-card, 3-card, and 5-card grids (see
that module's own docstring — "existing pretrained embeddings for
two-card, three-card, and five-card hands"), with the additional
property that it was trained to FACTORIZE: 2- and 3-card subsets of a
larger set carry meaningful information on their own, not just the full
5-card composite. Feeding it a board of any other cardinality (0, 1, 4,
6+) — which CAP's board_layout.streets configs genuinely produce, from
hopscotch_bomb_pot's 6-node overlapping boards to a bare single-card
node — is out-of-distribution: the grid shape never changes
((14,4,1) regardless of occupancy — see card_utils.card_ids_to_grid),
so it will always RUN, but nothing in training ever taught the
conv/dense stack what a 1-, 4-, or 6-card occupancy pattern means.

Design: decompose-and-pool
----------------------------
Rather than ever calling the pretrained encoder outside {2, 3, 5}, this
module:
  1. Splits an N-card board into a covering set of native-size subsets
     (see _decompose_indices below for the exact rule per N).
  2. Batch-encodes EVERY subset, across the WHOLE forward_batch() call,
     in a single hand_encoder.encode_batch() dispatch — this is the
     one-TF-call-per-forward_batch() discipline
     representation_model.py's own docstring already establishes as the
     dominant cost lever (TF per-call dispatch overhead dwarfs the
     actual conv for a single grid).
  3. Learned-pools (small attention scorer, trainable) the resulting
     per-subset embeddings into one fixed-size board embedding per
     example.

N=0 (preflop) and N=1 (a lone dealt node, e.g. an odd bomb-pot layout)
have no native-size decomposition — see EMPTY/SINGLE handling below,
each with its own small dedicated learned parameter/table rather than
ever calling the pretrained encoder on an out-of-distribution input.

Frozen base + trainable adapter
----------------------------------
The pretrained hand_encoder's output is ALWAYS detached before pooling
— no gradient ever flows into the frozen Keras weights (matching the
documented TF/PyTorch boundary in representation_model.py /
pretrained_hand_encoder.py). Only attn_scorer and project are
nn.Parameters PPO actually updates. `project` is a 2-layer MLP
(Linear -> ReLU -> Linear) rather than a single Linear whenever
project_hidden_dim is given, so the trainable adapter has real capacity
to reshape the frozen representation for the policy's actual decision
task, not just re-scale it — see the conversation this module was
built in for the fuller "why freeze the base, not just conveniently
but correctly" argument.

Contract (matches representation.board_encoder.BoardEncoder exactly)
----------------------------------------------------------------------
    forward(card_ids: Optional[Sequence[int]]) -> Tensor(output_dim,)
    forward_batch(batch_card_ids: Sequence[Optional[Sequence[int]]])
        -> Tensor(batch, output_dim)

This is the ENTIRE surface representation_model.py calls on
board_encoder — swapping BoardEncoder for CompositeBoardEncoder in
models/actor_critic_factory.py's construction of RepresentationModel is
therefore a one-line change with no other file touched.

Dependencies
------------
    PyTorch, numpy. representation.base_hand_encoder.BaseHandEncoder
    (duck-typed — .encode_batch(), .embedding_dim). itertools,
    functools (subset-index caching; pure Python, no per-call cost for
    repeated board sizes).
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder

# Board sizes the pretrained CardSetEncoder was actually trained on —
# see module docstring. Preference order when decomposing: try 5-card
# windows first (most information per subset), then fall back to
# 3-card for boards too small for a 5-window.
_NATIVE_SIZES = (5, 3, 2)


# ---------------------------------------------------------------------------
# Subset decomposition (indices only — cached per board size, not per
# actual hand, since which POSITIONS form a subset never depends on
# which cards occupy them). Shared with composite_hand_encoder.py.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _decompose_indices(n: int, max_subsets: int) -> Tuple[Tuple[int, ...], ...]:
    """
    Return a tuple of index-tuples (positions into an N-card list)
    covering the N-card set with native-size ({2,3,5}) subsets.

    Parameters
    ----------
    n : int
        Card-set size. Must be >= 2 (0 and 1 are handled by dedicated
        paths in the calling encoder classes, not by decomposition).
    max_subsets : int
        Hard cap on how many subsets are returned, for large N (CAP
        boards up to and beyond 6 cards, Omaha hole+board composites up
        to 9 cards) — bounds both the hand_encoder.encode_batch() cost
        and the pooling attention's sequence length. When C(n, k)
        exceeds this cap, a deterministic evenly-strided sample of the
        full combination list is taken (not random — see determinism
        note below) so the same size always decomposes identically
        run-to-run.

    Returns
    -------
    Tuple[Tuple[int, ...], ...]
        Each inner tuple is a set of 0-based positions into the N-card
        list, length 2, 3, or 5.

    Notes
    -----
    Determinism matters here for the same reason it matters everywhere
    else in this codebase's scenario generators (scenarios/
    scenario_generator.py's seed() contract): two runs dealing the same
    board/hand must decompose it identically, or "same seed -> same
    training trajectory" silently breaks. itertools.combinations()
    already produces a fixed, deterministic order for a given n/k, so
    a strided sample of it is reproducible without needing its own RNG.
    """
    if n == 2:
        return ((0, 1),)
    if n == 3:
        return ((0, 1, 2),)
    if n == 4:
        # Four 3-card "drop-one" subsets — every card appears in 3 of
        # the 4 subsets, giving reasonably even coverage of pairwise
        # interactions without an explosion in subset count. Also
        # exactly the set of legal Omaha-relevant triples when n=4 is a
        # board (not a hole) — see composite_hand_encoder.py for the
        # distinct, rules-motivated n=4 treatment used for Omaha holes.
        return tuple(itertools.combinations(range(4), 3))

    # n >= 5: prefer 5-card windows (richest native size).
    combos = list(itertools.combinations(range(n), 5))
    if len(combos) > max_subsets:
        stride = len(combos) / max_subsets
        combos = [combos[int(i * stride)] for i in range(max_subsets)]
    return tuple(combos)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class CompositeBoardEncoder(nn.Module):
    """
    Board embedding built from a pretrained BaseHandEncoder, valid for
    ANY board size (0 through arbitrarily many CAP board nodes), by
    decomposing into native-size ({2,3,5}) subsets — see module
    docstring for the full rationale.

    Parameters
    ----------
    hand_encoder : BaseHandEncoder
        The pretrained (or untrained, for pipeline testing) card-set
        encoder — same object type used for hole cards
        (representation/pretrained_hand_encoder.py). NOT registered as
        an nn.Module (it's TF/Keras-backed, frozen) — same treatment
        RepresentationModel gives its own hand_encoder.
    output_dim : int
        Final board embedding dimensionality. Default 32, matching
        BoardEncoder's own default so this is a drop-in swap with no
        downstream dimension changes required.
    attn_hidden_dim : int
        Hidden width of the small learned attention-scorer MLP used to
        pool subset embeddings. Default 32.
    project_hidden_dim : int, optional
        If given, the final projection from the frozen embedding_dim to
        output_dim is a 2-layer MLP (Linear -> ReLU -> Linear) with
        this hidden width, instead of a single Linear — gives the
        trainable adapter real capacity to reshape the frozen
        representation. Default None (single Linear).
    max_subsets : int
        Forwarded to _decompose_indices — caps decomposition cost for
        large boards. Default 12 (covers every board size in CAP's
        shipped YAMLs — up to hopscotch_bomb_pot's 6-card boards, which
        decompose to C(6,5)=6 subsets, well under the cap — with
        headroom for larger future variants).
    """

    def __init__(
        self,
        hand_encoder: BaseHandEncoder,
        output_dim: int = 32,
        attn_hidden_dim: int = 32,
        project_hidden_dim: Optional[int] = None,
        max_subsets: int = 12,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        if max_subsets < 1:
            raise ValueError(f"max_subsets must be >= 1, got {max_subsets}")

        self.hand_encoder = hand_encoder  # NOT an nn.Module (frozen, TF-backed)
        self.output_dim = output_dim
        self.max_subsets = max_subsets

        embed_dim = hand_encoder.embedding_dim

        # Learned attention scorer over subset embeddings: one scalar
        # score per subset, softmax-normalised, weighted sum — the
        # trainable "which decomposition best represents this board"
        # step. Mirrors BoardEncoder's own use of a small learned
        # module (there, a Transformer layer) on top of a frozen/shared
        # per-card representation.
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

        # N == 0 (preflop / no board dealt yet): dedicated learned
        # placeholder, same pattern as BoardEncoder.empty_board_embed.
        self.empty_board_embed = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.empty_board_embed, std=0.02)

        # N == 1 (a lone dealt node — no native subset size fits one
        # card): dedicated small embedding table over the 52 possible
        # cards, NOT the pretrained encoder (which has never seen a
        # single-card grid and has no factorization guarantee at that
        # size). Rare case in practice; doesn't need to be
        # sophisticated, just needs to exist and be trainable.
        self.single_card_embed = nn.Embedding(52, output_dim)

    # ------------------------------------------------------------------
    # Forward (single example)
    # ------------------------------------------------------------------

    def forward(self, card_ids: Optional[Sequence[int]]) -> torch.Tensor:
        """Embed a single board. Returns shape (output_dim,)."""
        return self.forward_batch([card_ids])[0]

    # ------------------------------------------------------------------
    # Forward (batch)
    # ------------------------------------------------------------------

    def forward_batch(
        self, batch_card_ids: Sequence[Optional[Sequence[int]]]
    ) -> torch.Tensor:
        """
        Embed a batch of boards, each of potentially different size —
        exactly the RepresentationModel.forward_batch() calling
        convention (see representation_model.py's own board-count
        handling for how per-example board LISTS are already grouped
        before reaching here; this method still handles per-example
        SIZE variation within one board index, e.g. mixed preflop/flop/
        turn/river examples in one training batch).

        Parameters
        ----------
        batch_card_ids : Sequence[Optional[Sequence[int]]]
            One board's dealt card ids per example. Empty/None = no
            board dealt yet for that example.

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).
        """
        batch_size = len(batch_card_ids)
        if batch_size == 0:
            raise ValueError("batch_card_ids must be non-empty")

        device = self.empty_board_embed.device

        # --- classify each example -----------------------------------
        empty_idx: List[int] = []
        single_idx: List[int] = []
        decomposed_idx: List[int] = []
        # Per decomposed example: list of (subset card-id tuples)
        subsets_per_example: List[List[Tuple[int, ...]]] = [None] * batch_size  # type: ignore[list-item]

        for i, cards in enumerate(batch_card_ids):
            n = len(cards) if cards else 0
            if n == 0:
                empty_idx.append(i)
            elif n == 1:
                single_idx.append(i)
            else:
                idx_tuples = _decompose_indices(n, self.max_subsets)
                subsets_per_example[i] = [
                    tuple(cards[p] for p in idxs) for idxs in idx_tuples
                ]
                decomposed_idx.append(i)

        out = torch.zeros(batch_size, self.output_dim, device=device)

        # --- N == 0: learned placeholder -------------------------------
        if empty_idx:
            out[empty_idx] = self.empty_board_embed

        # --- N == 1: small embedding table -----------------------------
        if single_idx:
            card_ids_t = torch.tensor(
                [batch_card_ids[i][0] for i in single_idx],
                dtype=torch.long, device=device,
            )
            out[single_idx] = self.single_card_embed(card_ids_t)

        # --- N >= 2: decompose, ONE batched pretrained-encoder call,
        #     then learned attention-pool per example -------------------
        if decomposed_idx:
            flat_subsets: List[Tuple[int, ...]] = []
            counts: List[int] = []
            for i in decomposed_idx:
                subs = subsets_per_example[i]
                counts.append(len(subs))
                flat_subsets.extend(subs)

            # Single batched TF dispatch for every subset of every
            # decomposed example in this call — the throughput property
            # this whole module exists to preserve (see module
            # docstring's step 2).
            flat_np = self.hand_encoder.encode_batch(
                [list(s) for s in flat_subsets]
            )  # (total_subsets, embed_dim)
            flat_embed = torch.from_numpy(flat_np).float().to(device).detach()

            max_subsets_in_batch = max(counts)
            embed_dim = flat_embed.shape[-1]
            padded = torch.zeros(
                len(decomposed_idx), max_subsets_in_batch, embed_dim, device=device
            )
            pad_mask = torch.ones(
                len(decomposed_idx), max_subsets_in_batch, dtype=torch.bool, device=device
            )  # True = padding (ignore)

            cursor = 0
            for row, count in enumerate(counts):
                padded[row, :count] = flat_embed[cursor: cursor + count]
                pad_mask[row, :count] = False
                cursor += count

            # Learned attention pooling over the (padded) subset axis.
            scores = self.attn_scorer(padded).squeeze(-1)          # (rows, max_subsets)
            scores = scores.masked_fill(pad_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)                 # (rows, max_subsets)
            pooled = (weights.unsqueeze(-1) * padded).sum(dim=1)    # (rows, embed_dim)

            projected = self.project(pooled)                        # (rows, output_dim)
            out[decomposed_idx] = projected

        return out 
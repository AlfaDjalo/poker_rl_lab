"""
representation/composite_hand_encoder.py — Pretrained-encoder-backed
hand embedding for ANY hole-card count, via the same decompose-and-pool
strategy as representation.composite_board_encoder.CompositeBoardEncoder.

Why this exists
-----------------
representation.hole_cards_encoder.HoleCardsEncoder and
RepresentationModel's own direct use of hand_encoder.encode_batch()
both assume the raw hole_card_ids list is fed straight to the
pretrained CardSetEncoder. That's correct for hold'em (2 cards — a
native trained size — see pretrained_hand_encoder.py's own docstring)
but wrong for Omaha's 4-card hole: feeding a 4-card grid directly is
out-of-distribution for an encoder trained on {2, 3, 5}-card sets (see
composite_board_encoder.py's module docstring for the same argument
applied to boards).

Omaha specifically: C(4,2) is not an arbitrary choice
------------------------------------------------------------
Unlike a board (where "which subsets to sample" is purely an
engineering decomposition choice), Omaha's showdown rule REQUIRES
exactly 2 of the 4 hole cards be used in the final 5-card hand (see
poker_engine's OMAHA ShowdownType). So decomposing the 4-card hole into
its six 2-card pairs isn't just a convenient native-size split — it
mirrors the actual legal-hand structure the policy needs to reason
about ("which two of my four cards are the ones that matter here").
This is a stronger case for decompose-and-pool than the board case,
not just a reuse of the same trick.

Two call surfaces — numpy (BaseHandEncoder contract) vs. torch (gradient-preserving)
------------------------------------------------------------------------------------
BaseHandEncoder.encode_batch() is a plain-numpy-in/plain-numpy-out
contract (see that module's own docstring: "All methods return plain
numpy arrays regardless of the underlying framework... so callers never
need to know or care which backend produced them"). That's the right
contract for a frozen TF encoder, but this class ALSO owns real
trainable PyTorch parameters (attn_scorer, project) — round-tripping
through numpy on every call would silently sever autograd and leave
those parameters untrained forever, since numpy arrays carry no
gradient graph.

So this class exposes BOTH:
    forward_batch_torch(card_id_lists) -> torch.Tensor
        Gradient-preserving. RepresentationModel.forward_batch() calls
        THIS directly (see models/actor_critic_factory.py wiring +
        the corresponding representation_model.py patch) whenever the
        configured hand_encoder exposes it, precisely so attn_scorer/
        project actually receive PPO's gradients.
    encode_batch(card_id_lists) -> np.ndarray
        The BaseHandEncoder-contract numpy wrapper around the above,
        for any OTHER caller that still expects the plain-numpy
        interface (e.g. a standalone script, a notebook, a future
        caller not yet updated to the torch-native path). Always runs
        under torch.no_grad() — no caller reaching this method can
        possibly need gradients, since the return type has none.

Contract (matches representation.base_hand_encoder.BaseHandEncoder)
----------------------------------------------------------------------
    encode_hole(card_ids)                    -> np.ndarray(embedding_dim,)
    encode_board(card_ids)                   -> np.ndarray(embedding_dim,)
    encode_combined(hole_ids, board_ids)      -> np.ndarray(embedding_dim,)
    encode_batch(card_id_lists)               -> np.ndarray(batch, embedding_dim)
    embedding_dim                              -> int
    forward_batch_torch(card_id_lists)         -> torch.Tensor(batch, embedding_dim)  [NEW, non-BaseHandEncoder]

Dependencies
------------
    PyTorch, numpy. representation.base_hand_encoder.BaseHandEncoder,
    representation.composite_board_encoder._decompose_indices (shared
    decomposition logic — see that module's docstring for the subset
    scheme per board/hand size).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.composite_board_encoder import _decompose_indices


class CompositeHandEncoder(BaseHandEncoder, nn.Module):
    """
    Hand/composite-card-set embedding valid for ANY card count (hold'em
    2-card hole, Omaha 4-card hole, hole+board composites of any size),
    built from a frozen pretrained BaseHandEncoder plus a small
    trainable pooling adapter — see module docstring.

    Parameters
    ----------
    base_encoder : BaseHandEncoder
        The frozen pretrained card-set encoder (e.g.
        PretrainedHandEncoder.from_file(...)). Never has gradients
        flow into it from this class.
    output_dim : int, optional
        Final embedding dimensionality. Defaults to
        base_encoder.embedding_dim (no projection) if omitted, matching
        HoleCardsEncoder's own "no-op if already the right size"
        convention. Pass an explicit value to project to a different
        size, e.g. to match RepresentationModel's expected hand_encoder
        .embedding_dim.
    attn_hidden_dim : int
        Hidden width of the learned attention-scorer MLP. Default 32.
    project_hidden_dim : int, optional
        If given, the final projection is a 2-layer MLP
        (Linear -> ReLU -> Linear) with this hidden width instead of a
        single Linear — more adapter capacity to reshape the frozen
        embedding (see module docstring). Default None (single Linear,
        matching the original minimal adapter).
    max_subsets : int
        Forwarded to _decompose_indices. Default 12 — see
        composite_board_encoder.py's identical parameter for the
        rationale (bounds cost for large composite sets, e.g.
        Omaha hole + full board = 9 cards).
    """

    def __init__(
        self,
        base_encoder: BaseHandEncoder,
        output_dim: int = None,
        attn_hidden_dim: int = 32,
        project_hidden_dim: int = None,
        max_subsets: int = 12,
    ) -> None:
        nn.Module.__init__(self)

        base_dim = base_encoder.embedding_dim
        self._output_dim = output_dim if output_dim is not None else base_dim

        self.base_encoder = base_encoder  # NOT registered as nn.Module (TF-backed, frozen)
        self.max_subsets = max_subsets

        self.attn_scorer = nn.Sequential(
            nn.Linear(base_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1),
        )

        if project_hidden_dim is not None:
            self.project = nn.Sequential(
                nn.Linear(base_dim, project_hidden_dim),
                nn.ReLU(),
                nn.Linear(project_hidden_dim, self._output_dim),
            )
        else:
            self.project = nn.Linear(base_dim, self._output_dim)

        # N == 0 (e.g. a fully-masked/unknown hand — defensive; every
        # real caller has >= 2 hole cards, but keeps this class total
        # over its declared interface rather than raising on an edge
        # case a future caller might legitimately hit).
        self.empty_embed = nn.Parameter(torch.zeros(self._output_dim))
        nn.init.normal_(self.empty_embed, std=0.02)

    # ------------------------------------------------------------------
    # BaseHandEncoder interface (numpy contract)
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._output_dim

    def encode_hole(self, card_ids: Sequence[int]) -> np.ndarray:
        return self.encode_batch([card_ids])[0]

    def encode_board(self, card_ids: Sequence[int]) -> np.ndarray:
        return self.encode_batch([card_ids])[0]

    def encode_combined(self, hole_ids: Sequence[int], board_ids: Sequence[int]) -> np.ndarray:
        return self.encode_batch([list(hole_ids) + list(board_ids)])[0]

    def encode_batch(self, card_id_lists: Sequence[Sequence[int]]) -> np.ndarray:
        """
        Embed a batch of card sets, each of potentially different size.

        Plain-numpy BaseHandEncoder-contract wrapper — see module
        docstring's "Two call surfaces" section. Always runs under
        torch.no_grad(): a numpy return value can never carry a
        gradient, so tracking one here would be pure waste. Callers
        that need gradients to flow (i.e. RepresentationModel, when
        this class is the configured hand_encoder) MUST use
        forward_batch_torch() instead — see that method and the
        corresponding patch in representation_model.py.

        Returns
        -------
        np.ndarray
            Shape (batch, embedding_dim).
        """
        with torch.no_grad():
            out = self.forward_batch_torch(card_id_lists)
        return out.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # Torch-native contract (gradient-preserving — NOT part of
    # BaseHandEncoder; RepresentationModel duck-types for this)
    # ------------------------------------------------------------------

    def forward_batch_torch(
        self, card_id_lists: Sequence[Sequence[int]]
    ) -> torch.Tensor:
        """
        Embed a batch of card sets, each of potentially different size,
        returning a torch.Tensor with gradients intact through
        attn_scorer/project (the base_encoder call itself is always
        detached — see module docstring's "Frozen base + trainable
        adapter" discussion carried over from composite_board_encoder.py).

        This is what RepresentationModel.forward_batch() calls directly
        when self.hand_encoder is a CompositeHandEncoder (or anything
        else exposing this method) — see representation_model.py's
        patched hole-embedding section.

        Parameters
        ----------
        card_id_lists : Sequence[Sequence[int]]
            One card-id list per example — hold'em 2-card holes, Omaha
            4-card holes, hole+board composites, all freely mixed in
            one batch.

        Returns
        -------
        torch.Tensor
            Shape (batch, embedding_dim).
        """
        batch_size = len(card_id_lists)
        if batch_size == 0:
            raise ValueError("card_id_lists must be non-empty")

        device = self.empty_embed.device

        empty_idx: List[int] = []
        native_idx: List[int] = []      # N in {2, 3, 5} — pass straight through
        decomposed_idx: List[int] = []  # everything else — decompose + pool
        subsets_per_example: List[List[Tuple[int, ...]]] = [None] * batch_size  # type: ignore[list-item]

        for i, cards in enumerate(card_id_lists):
            n = len(cards) if cards else 0
            if n == 0:
                empty_idx.append(i)
            elif n in (2, 3, 5):
                native_idx.append(i)
            else:
                idx_tuples = _decompose_indices(n, self.max_subsets)
                subsets_per_example[i] = [
                    tuple(cards[p] for p in idxs) for idxs in idx_tuples
                ]
                decomposed_idx.append(i)

        out = torch.zeros(batch_size, self._output_dim, device=device)

        if empty_idx:
            out[empty_idx] = self.empty_embed

        # --- native sizes: one direct base_encoder call, one project() ---
        if native_idx:
            native_np = self.base_encoder.encode_batch(
                [list(card_id_lists[i]) for i in native_idx]
            )
            native_embed = torch.from_numpy(native_np).float().to(device).detach()
            out[native_idx] = self.project(native_embed)

        # --- decomposed sizes: batched subset encode + attention pool ---
        if decomposed_idx:
            flat_subsets: List[Tuple[int, ...]] = []
            counts: List[int] = []
            for i in decomposed_idx:
                subs = subsets_per_example[i]
                counts.append(len(subs))
                flat_subsets.extend(subs)

            flat_np = self.base_encoder.encode_batch([list(s) for s in flat_subsets])
            flat_embed = torch.from_numpy(flat_np).float().to(device).detach()

            max_subsets_in_batch = max(counts)
            base_dim = flat_embed.shape[-1]
            padded = torch.zeros(
                len(decomposed_idx), max_subsets_in_batch, base_dim, device=device
            )
            pad_mask = torch.ones(
                len(decomposed_idx), max_subsets_in_batch, dtype=torch.bool, device=device
            )

            cursor = 0
            for row, count in enumerate(counts):
                padded[row, :count] = flat_embed[cursor: cursor + count]
                pad_mask[row, :count] = False
                cursor += count

            scores = self.attn_scorer(padded).squeeze(-1)
            scores = scores.masked_fill(pad_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            pooled = (weights.unsqueeze(-1) * padded).sum(dim=1)  # (rows, base_dim)

            out[decomposed_idx] = self.project(pooled)

        return out
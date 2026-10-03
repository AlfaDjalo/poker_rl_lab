"""
representation/history_encoder.py — Learned embedding for betting history.

Per the RL Lab implementation guide (§5, "Additional Learned Embeddings"):

    Betting History: Represents previous actions taken during the current
    hand.

...and (§6, "Preferred Architecture") betting history is one token among
the eventual transformer's inputs ("Hole", "Board 1", "Board 2", "Point 1",
"Point 2", "Position", "Stack", "Pot", "Betting History", "Game").

Why this file does NOT read from StructuredObservation directly
-------------------------------------------------------------------
This module defines the plain-data event type (`HistoryEvent`) and the
encoder (`HistoryEncoder`) against that type only. Adapters that build
HistoryEvents live elsewhere (representation/observation_adapter.py for
the legacy envs, representation/graph_observation_adapter.py for the
graph-native env), keeping this module engine-independent.

Representation
--------------
A hand's history is a variable-length, ORDER-DEPENDENT sequence of
actions (unlike a board, which is an unordered card set — see
board_encoder.py's mean-pooling over an unordered set). Order matters
here: "bet, raise, call" is a different story than "call, raise, bet"
even though as an unordered multiset they're identical. So this encoder
uses a GRU rather than pooled/permutation-invariant attention, and reads
out the final hidden state as the sequence embedding.

Each `HistoryEvent` is intentionally minimal and engine-independent:

    action_type   : index into actions.abstract_action.ActionType (or,
                    for the graph-native adapter, the engine's own
                    ActionType int) — what was actually done.
    street_index  : which street the action happened on (0=preflop, ...).
                    Not embedded categorically (games/CAP variants vary
                    in street count — see game_encoder.py's num_streets
                    treatment) — log1p-scaled instead.
    seat_fraction : ((seat - dealer) % n) / n at the time of the action —
                    same definition as position_encoder.py's
                    PositionFeatures.seat_fraction, so "who acted" is
                    comparable across variable table sizes without a
                    fixed per-seat embedding table.
    pot_fraction  : chips this action put into the pot, divided by the
                    pot size immediately before the action. 0.0 for
                    FOLD/CHECK (no chips move).
    is_hero       : whether the hero (not an opponent) took this action.
    is_aggressor  : whether this action set a new bet_to_call (bet/raise),
                    as opposed to a call that merely matched a bet.

PERFORMANCE CHANGES IN THIS VERSION (from profiling)
------------------------------------------------------
Profiling showed HistoryEncoder.forward_batch at ~25% of full hold'em
wall time (~11 ms/call): it built token tensors one example at a time
(several torch.tensor() calls + an embedding lookup + an MLP call PER
example, in a Python loop) and ran the GRU over zero-padded rows for
examples with no history at all.

forward_batch() is now fully vectorised:
  - ONE torch.tensor() build for every event across the whole batch,
    ONE embedding lookup and ONE scalar-MLP call over all events.
  - Tokens scattered into the padded (rows, max_len, token_dim) tensor
    with a single index_put (differentiable).
  - The GRU runs ONLY on rows with >= 1 event; empty rows take the
    learned placeholder directly.
Output is numerically identical to the previous per-example
implementation (same parameters, same math). _events_to_tokens() is
kept for any external caller / test that still uses it.

Dependencies
------------
    PyTorch, plus actions.abstract_action for NUM_ACTIONS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import torch
import torch.nn as nn

from poker_rl_lab.actions.abstract_action import NUM_ACTIONS


# ---------------------------------------------------------------------------
# Plain-data event type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryEvent:
    """
    One action taken by one seat during the current hand. Framework- and
    engine-independent — HistoryEncoder only ever sees a sequence of
    these, never a real Action/GameState object.
    """
    action_type: int
    street_index: int
    seat_fraction: float
    pot_fraction: float = 0.0
    is_hero: bool = False
    is_aggressor: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.action_type < NUM_ACTIONS):
            raise ValueError(
                f"action_type must be in [0, {NUM_ACTIONS}), got {self.action_type}"
            )
        if self.street_index < 0:
            raise ValueError(f"street_index must be >= 0, got {self.street_index}")
        if not (0.0 <= self.seat_fraction < 1.0):
            raise ValueError(
                f"seat_fraction must be in [0, 1), got {self.seat_fraction}"
            )
        if self.pot_fraction < 0:
            raise ValueError(f"pot_fraction must be >= 0, got {self.pot_fraction}")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class HistoryEncoder(nn.Module):
    """
    Encodes one hand's action sequence into a fixed-size, order-aware
    embedding.

    Parameters
    ----------
    action_type_dim : int
        Embedding size for the action_type lookup table. Default 8.
    scalar_hidden_dim : int
        Hidden width of the small MLP over the per-event scalar/boolean
        features (log1p(street_index), seat_fraction, log1p(pot_fraction),
        is_hero, is_aggressor). Default 8.
    gru_hidden_dim : int
        Hidden size of the GRU that reads the token sequence. Default 32.
    output_dim : int
        Final embedding dimensionality. Default 16.
    """

    _NUM_SCALAR_FEATURES = 5

    def __init__(
        self,
        action_type_dim: int = 8,
        scalar_hidden_dim: int = 8,
        gru_hidden_dim: int = 32,
        output_dim: int = 16,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")
        if gru_hidden_dim <= 0:
            raise ValueError(f"gru_hidden_dim must be > 0, got {gru_hidden_dim}")

        self.output_dim = output_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.action_type_embed = nn.Embedding(NUM_ACTIONS, action_type_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(self._NUM_SCALAR_FEATURES, scalar_hidden_dim),
            nn.ReLU(),
        )

        token_dim = action_type_dim + scalar_hidden_dim
        self._token_dim = token_dim
        self.gru = nn.GRU(input_size=token_dim, hidden_size=gru_hidden_dim, batch_first=True)

        self.project = nn.Linear(gru_hidden_dim, output_dim)

        # Learned placeholder for a hand with zero actions so far
        # (the very first decision point in a hand).
        self.empty_history_embed = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.empty_history_embed, std=0.02)

    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------

    def _tokens_for_events(self, events: Sequence[HistoryEvent]) -> torch.Tensor:
        """
        Vectorised token build for a FLAT list of events (from any
        number of sequences). Returns (len(events), token_dim).
        """
        device = self.action_type_embed.weight.device
        action_ids = torch.tensor(
            [e.action_type for e in events], dtype=torch.long, device=device
        )
        raw = torch.tensor(
            [
                [
                    e.street_index,
                    e.seat_fraction,
                    e.pot_fraction,
                    float(e.is_hero),
                    float(e.is_aggressor),
                ]
                for e in events
            ],
            dtype=torch.float32,
            device=device,
        )
        scalars = torch.stack(
            [
                torch.log1p(raw[:, 0]),   # log1p(street_index)
                raw[:, 1],                # seat_fraction
                torch.log1p(raw[:, 2]),   # log1p(pot_fraction)
                raw[:, 3],                # is_hero
                raw[:, 4],                # is_aggressor
            ],
            dim=1,
        )
        action_embed = self.action_type_embed(action_ids)   # (E, action_type_dim)
        scalar_embed = self.scalar_mlp(scalars)              # (E, scalar_hidden_dim)
        return torch.cat([action_embed, scalar_embed], dim=-1)

    def _events_to_tokens(self, events: Sequence[HistoryEvent]) -> torch.Tensor:
        """Build the (len(events), token_dim) token tensor for one sequence.
        Kept for backward compatibility; same result as _tokens_for_events."""
        return self._tokens_for_events(events)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, events: Sequence[HistoryEvent]) -> torch.Tensor:
        """
        Embed one hand's action sequence, in order.

        Parameters
        ----------
        events : Sequence[HistoryEvent]
            Actions taken so far, in chronological order. Empty sequence
            means no actions have been taken yet (start of the hand).

        Returns
        -------
        torch.Tensor
            Shape (output_dim,).
        """
        if not events:
            return self.empty_history_embed

        tokens = self._tokens_for_events(events).unsqueeze(0)  # (1, len, token_dim)
        output, _ = self.gru(tokens)                            # (1, len, gru_hidden_dim)
        last = output[0, -1]                                     # (gru_hidden_dim,)
        return self.project(last)                                 # (output_dim,)

    def forward_batch(
        self, batch_events: Sequence[Sequence[HistoryEvent]]
    ) -> torch.Tensor:
        """
        Embed a batch of action sequences, each with a possibly different
        length.

        Parameters
        ----------
        batch_events : Sequence[Sequence[HistoryEvent]]
            One action sequence per example. Empty sub-sequences are
            hands with no actions taken yet.

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).

        Notes
        -----
        Sequences are right-padded to the max length among NON-EMPTY
        sequences and run through the GRU together; because a GRU is
        strictly causal, padding appended after a sequence's true length
        cannot affect the hidden state at that true length, so gathering
        each sequence's output at its own (length - 1) index is exact.
        Wholly empty sequences skip the GRU and use the learned
        placeholder directly.
        """
        batch_size = len(batch_events)
        if batch_size == 0:
            raise ValueError("batch_events must be non-empty")

        lengths = [len(evs) for evs in batch_events]
        non_empty = [i for i, n in enumerate(lengths) if n > 0]

        if not non_empty:
            return self.empty_history_embed.unsqueeze(0).expand(batch_size, -1)

        device = self.action_type_embed.weight.device
        ne_lengths = [lengths[i] for i in non_empty]
        num_rows = len(non_empty)
        max_len = max(ne_lengths)

        # -- one flat token build across every event in the batch --------
        flat_events = [e for i in non_empty for e in batch_events[i]]
        tokens = self._tokens_for_events(flat_events)           # (E, token_dim)

        # -- scatter into the padded (rows, max_len, token_dim) tensor ---
        len_t = torch.tensor(ne_lengths, dtype=torch.long, device=device)
        rows = torch.repeat_interleave(torch.arange(num_rows, device=device), len_t)
        offsets = torch.cumsum(len_t, dim=0) - len_t
        cols = torch.arange(tokens.shape[0], device=device) - torch.repeat_interleave(
            offsets, len_t
        )
        padded = tokens.new_zeros(num_rows, max_len, self._token_dim)
        padded = padded.index_put((rows, cols), tokens)

        output, _ = self.gru(padded)                             # (rows, max_len, H)
        gathered = output[torch.arange(num_rows, device=device), len_t - 1]
        projected = self.project(gathered)                       # (rows, output_dim)

        if num_rows == batch_size:
            return projected

        out = self.empty_history_embed.unsqueeze(0).expand(batch_size, -1).clone()
        idx = torch.tensor(non_empty, dtype=torch.long, device=device)
        return out.index_copy(0, idx, projected)
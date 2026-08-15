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
Unlike point_encoder.py/game_encoder.py/position_encoder.py, there is
currently no history-of-actions field anywhere in the codebase:
envs/observations.py's StructuredObservation exposes hole cards, board,
players, and current betting state (pot / bet_to_call / min_raise /
raises_this_street) — but not a log of individual past actions — and
neither push_fold_env.py nor river_env.py records one internally as the
episode progresses. So, deliberately mirroring the pattern already used
for engine-shaped inputs elsewhere in this package (see point_encoder.py
and game_encoder.py's adapter-function docstrings): this module defines
the plain-data event type (`HistoryEvent`) and the encoder
(`HistoryEncoder`) against that type only, with NO adapter function,
because there is nothing yet to adapt *from*. A follow-up change to
envs/observations.py (adding a `HistoryObs` field) and to the two
environments (appending a `HistoryEvent` on every `_apply_action` call)
is required before this encoder can be wired into the live pipeline —
see the module-level TODO at the bottom of this file.

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

    action_type   : index into actions.abstract_action.ActionType — the
                    dominant signal; what was actually done.
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
                    FOLD/CHECK (no chips move). Raw bet-sizing signal —
                    "bet 25% of pot" vs "shoved 3x the pot" are very
                    different actions even though both are ActionType
                    BET_*/ALL_IN.
    is_hero       : whether the hero (not an opponent) took this action.
    is_aggressor  : whether this action set a new bet_to_call (i.e. was
                    a bet/raise, matching ActionType.is_bet), as opposed
                    to a call that merely matched an existing bet. Kept
                    as an explicit flag rather than derived solely from
                    action_type inside the encoder, since it is cheap for
                    the caller building each HistoryEvent to know for
                    certain (it already knows whether bet_to_call changed)
                    while re-deriving "is this a raise vs a bet" from
                    action_type alone would require also knowing whether
                    someone had already bet this street.

Dependencies
------------
    PyTorch, plus actions.abstract_action for NUM_ACTIONS/ActionType
    (pure Python — no C++ extension, no poker_eval — see that module's
    own docstring). No CAP Engine dependency.
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
        self.gru = nn.GRU(input_size=token_dim, hidden_size=gru_hidden_dim, batch_first=True)

        self.project = nn.Linear(gru_hidden_dim, output_dim)

        # Learned placeholder for a hand with zero actions so far
        # (the very first decision point in a hand).
        self.empty_history_embed = nn.Parameter(torch.zeros(output_dim))
        nn.init.normal_(self.empty_history_embed, std=0.02)

    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------

    def _events_to_tokens(self, events: Sequence[HistoryEvent]) -> torch.Tensor:
        """Build the (len(events), token_dim) token tensor for one sequence."""
        device = self.action_type_embed.weight.device
        action_ids = torch.tensor(
            [e.action_type for e in events], dtype=torch.long, device=device
        )
        scalars = torch.tensor(
            [
                [
                    math.log1p(e.street_index),
                    e.seat_fraction,
                    math.log1p(e.pot_fraction),
                    float(e.is_hero),
                    float(e.is_aggressor),
                ]
                for e in events
            ],
            dtype=torch.float32,
            device=device,
        )
        action_embed = self.action_type_embed(action_ids)   # (len, action_type_dim)
        scalar_embed = self.scalar_mlp(scalars)              # (len, scalar_hidden_dim)
        return torch.cat([action_embed, scalar_embed], dim=-1)  # (len, token_dim)

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

        tokens = self._events_to_tokens(events).unsqueeze(0)  # (1, len, token_dim)
        output, _ = self.gru(tokens)                           # (1, len, gru_hidden_dim)
        last = output[0, -1]                                    # (gru_hidden_dim,)
        return self.project(last)                                # (output_dim,)

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
        Sequences are right-padded to the batch's max length and run
        through the GRU together; because a GRU is strictly causal (each
        step only depends on earlier steps, never later ones), padding
        appended *after* a sequence's true length cannot affect the
        hidden state already computed at that true length — so gathering
        each sequence's output at its own (length - 1) index is exact,
        not an approximation. This avoids needing PackedSequence. Wholly
        empty sequences skip the GRU and use the learned placeholder
        directly (mirroring board_encoder.forward_batch's handling of
        empty boards).
        """
        batch_size = len(batch_events)
        if batch_size == 0:
            raise ValueError("batch_events must be non-empty")

        lengths = [len(evs) for evs in batch_events]
        non_empty = [i for i, n in enumerate(lengths) if n > 0]

        if not non_empty:
            return self.empty_history_embed.unsqueeze(0).expand(batch_size, -1)

        max_len = max(lengths)
        token_dim = self.action_type_embed.embedding_dim + self.scalar_mlp[0].out_features
        device = self.action_type_embed.weight.device

        padded = torch.zeros(batch_size, max_len, token_dim, device=device)
        for i, evs in enumerate(batch_events):
            if not evs:
                continue
            padded[i, : len(evs)] = self._events_to_tokens(evs)

        output, _ = self.gru(padded)  # (batch, max_len, gru_hidden_dim)

        idx = torch.tensor([max(n - 1, 0) for n in lengths], dtype=torch.long, device=device)
        gathered = output[torch.arange(batch_size, device=device), idx]  # (batch, gru_hidden_dim)
        projected = self.project(gathered)                  # (batch, output_dim)

        empty_rows = torch.tensor(
            [n == 0 for n in lengths], dtype=torch.bool, device=device
        )
        if empty_rows.any():
            projected = projected.clone()
            projected[empty_rows] = self.empty_history_embed

        return projected


# ---------------------------------------------------------------------------
# TODO — wiring into the live pipeline (not done in this file):
#
#   1. Add a `HistoryObs` dataclass to envs/observations.py, e.g.:
#          events: Tuple[HistoryEvent, ...]
#      and a corresponding field on StructuredObservation.
#   2. In push_fold_env.py / river_env.py, append one HistoryEvent per
#      call to _apply_action() (river_env.py) / the single step()
#      decision (push_fold_env.py), using pot-before/pot-after to derive
#      pot_fraction, and ActionType.is_bet (abstract_action.py) to derive
#      is_aggressor.
#   3. Thread the accumulated event list through _build_obs() into the
#      new HistoryObs field.
#
# Left for a follow-up change since it touches two existing, tested env
# files rather than adding a new one.
# ---------------------------------------------------------------------------
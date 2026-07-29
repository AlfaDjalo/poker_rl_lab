"""
representation/position_encoder.py — Learned embedding for a player's
seat position at the table.

Per the RL Lab implementation guide (§5, "Additional Learned Embeddings"):

    Position: Represents the acting player's position.

Why relative, table-normalised features rather than a seat lookup table
-------------------------------------------------------------------------
The guide's design goals (§1) require the architecture to support
"variable player counts" without redesign. A raw seat index has no
consistent strategic meaning across table sizes: seat 4 is one off the
button at a 6-max table and doesn't exist heads-up. An nn.Embedding
keyed on literal seat number would need re-keying per table size and
would have to relearn "distance from the button" independently at each
size instead of sharing it.

So PositionEncoder consumes only:
  - continuous, table-size-normalised features that carry the same
    meaning at any player count (fraction of the way around the table
    from the button, fraction of seats still active, ...)
  - a few poker-meaningful booleans (button / small blind / big blind /
    "it's currently my turn")
  - table size itself as a continuous, log-scaled scalar (matching how
    point_encoder.py treats num_node_sets), not a categorical embedding,
    since a fixed-size embedding table would silently cap max_players.

Why this file does NOT reconstruct betting turn order
--------------------------------------------------------
The guide is explicit: the RL Lab must never duplicate poker/betting
rules — that belongs entirely to the Engine (rules/game_rules.py,
poker_engine/state/game_state.py). Real turn-order state (e.g.
GameState.last_to_act, whose computation depends on reset_betting_round)
is Engine state that neither BaseScenario nor StructuredObservation
carries, and this module does not attempt to re-derive it.

`to_act_after_fraction` below answers a purely topological question —
"going around the table in seat order starting from whoever acts now,
how many of the currently-active seats come after me before it wraps
back around" — using only fields already exposed (dealer_position /
current_player-equivalent, has_folded, is_all_in). It is NOT a claim
about betting-round closure (who has matched the current bet, etc.);
that distinction matters because seats can still act again later in the
same round after a raise, which this feature does not and should not
try to model.

Two adapters are provided:
  position_features_from_scenario()     — from a BaseScenario (reset()
                                            time; used by scenario tests
                                            and by any code that only has
                                            the raw scenario).
  position_features_from_observation()  — from a StructuredObservation
                                            (any hero decision point at
                                            runtime, via the Environment).
Both funnel into the same internal computation so the two can never
silently drift apart.

Current scope note
--------------------
For today's two-seat environments (PushFoldEnv, RiverEnv) hero is
always seat 1, villain always seat 0, dealer always seat 0, for the
whole episode — so features computed via either adapter are identical
throughout an episode. This starts to matter (values changing street to
street as players fold) only once a 3+ seat environment exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import torch
import torch.nn as nn

from scenarios.base_scenario import BaseScenario
from envs.observations import StructuredObservation


# ---------------------------------------------------------------------------
# Plain-data feature bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionFeatures:
    """
    Framework- and engine-independent feature bundle for one seat's
    position at one decision point. Downstream (PositionEncoder) only
    ever sees this, never a BaseScenario/StructuredObservation object.

    Attributes
    ----------
    seat_fraction : float
        Distance from the button, normalised to [0, 1) by table size.
        0.0 = on the button. Increases moving away from the button in
        seat order: (seat - dealer) % n / n.
    active_fraction : float
        Fraction of seats still in the hand (not folded), in (0, 1].
    to_act_after_fraction : float
        Fraction of the *other* currently-active seats that come after
        this seat, going around the table from whoever acts now, before
        the rotation would wrap back to the start. In [0, 1]. See the
        module docstring — this is seat topology, not betting closure.
    is_button : bool
    is_small_blind : bool
    is_big_blind : bool
    is_next_to_act : bool
        True iff this seat is the current actor.
    table_size : int
        Raw seat count. Log-scaled inside the encoder, not looked up.
    """
    seat_fraction: float
    active_fraction: float
    to_act_after_fraction: float
    is_button: bool = False
    is_small_blind: bool = False
    is_big_blind: bool = False
    is_next_to_act: bool = False
    table_size: int = 2

    def __post_init__(self) -> None:
        if not (0.0 <= self.seat_fraction < 1.0):
            raise ValueError(f"seat_fraction must be in [0, 1), got {self.seat_fraction}")
        if not (0.0 < self.active_fraction <= 1.0):
            raise ValueError(f"active_fraction must be in (0, 1], got {self.active_fraction}")
        if not (0.0 <= self.to_act_after_fraction <= 1.0):
            raise ValueError(
                f"to_act_after_fraction must be in [0, 1], got {self.to_act_after_fraction}"
            )
        if self.table_size < 2:
            raise ValueError(f"table_size must be >= 2, got {self.table_size}")


# ---------------------------------------------------------------------------
# Shared computation
# ---------------------------------------------------------------------------

def _compute_position_features(
    seat: int,
    dealer: int,
    n: int,
    current_player: int,
    has_folded: Sequence[bool],
    is_all_in: Sequence[bool],
) -> PositionFeatures:
    if not (0 <= seat < n):
        raise ValueError(f"seat {seat} out of range for {n} players")

    seat_fraction = ((seat - dealer) % n) / n

    # Heads-up special case: the button posts the small blind. This
    # matches scenarios/push_fold_generator.py's own convention (seat 0
    # is documented as "dealer / small blind / shover" for n==2) — see
    # the module docstring for a discrepancy this surfaced against
    # GameState._post_blinds, which does not special-case n == 2.
    if n == 2:
        sb_seat = dealer
        bb_seat = (dealer + 1) % n
    else:
        sb_seat = (dealer + 1) % n
        bb_seat = (dealer + 2) % n

    is_button = (seat == dealer)
    is_small_blind = (seat == sb_seat)
    is_big_blind = (seat == bb_seat)
    is_next_to_act = (seat == current_player)

    active_seats = [i for i in range(n) if not has_folded[i]]
    active_fraction = len(active_seats) / n

    rotation = [i for i in range(n) if not has_folded[i] and not is_all_in[i]]
    if seat in rotation and len(rotation) > 1:
        start_idx = rotation.index(current_player) if current_player in rotation else 0
        ordered = rotation[start_idx:] + rotation[:start_idx]
        seat_pos = ordered.index(seat)
        to_act_after = len(ordered) - seat_pos - 1
        to_act_after_fraction = to_act_after / (len(ordered) - 1)
    else:
        to_act_after_fraction = 0.0

    return PositionFeatures(
        seat_fraction=seat_fraction,
        active_fraction=active_fraction,
        to_act_after_fraction=to_act_after_fraction,
        is_button=is_button,
        is_small_blind=is_small_blind,
        is_big_blind=is_big_blind,
        is_next_to_act=is_next_to_act,
        table_size=n,
    )


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def position_features_from_scenario(scenario: BaseScenario, seat: int) -> PositionFeatures:
    """Build PositionFeatures for one seat from a BaseScenario (reset() time)."""
    n = scenario.player_count
    has_folded = [p.has_folded for p in scenario.players]
    is_all_in = [p.is_all_in for p in scenario.players]
    return _compute_position_features(
        seat, scenario.dealer_position, n, scenario.current_player, has_folded, is_all_in
    )


def position_features_from_observation(obs: StructuredObservation) -> PositionFeatures:
    """
    Build PositionFeatures for the hero from a StructuredObservation.

    A StructuredObservation is only ever handed back at the hero's own
    decision point (see envs/base_env.py's contract), so the hero is
    necessarily the current actor here.
    """
    n = len(obs.players)
    has_folded = [p.has_folded for p in obs.players]
    is_all_in = [p.is_all_in for p in obs.players]
    return _compute_position_features(
        obs.hero_seat, obs.dealer_position, n, obs.hero_seat, has_folded, is_all_in
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class PositionEncoder(nn.Module):
    """
    Encodes one seat's table position into a fixed-size embedding.

    Unlike PointEncoder there are no categorical fields here (no fixed
    enum of "position types"), so this is a single small MLP over
    scalar/boolean features rather than embedding-table lookups.

    Parameters
    ----------
    hidden_dim : int
        Hidden width of the MLP. Default 16.
    output_dim : int
        Final embedding dimensionality. Default 16.
    """

    _NUM_SCALAR_FEATURES = 8  # seat_fraction, active_fraction,
                               # to_act_after_fraction, is_button,
                               # is_small_blind, is_big_blind,
                               # is_next_to_act, log1p(table_size)

    def __init__(self, hidden_dim: int = 16, output_dim: int = 16) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")

        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(self._NUM_SCALAR_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, position: PositionFeatures) -> torch.Tensor:
        """Embed a single seat's position. Returns shape (output_dim,)."""
        return self.forward_batch([position])[0]

    def forward_batch(self, positions: Sequence[PositionFeatures]) -> torch.Tensor:
        """
        Embed a batch of positions.

        Parameters
        ----------
        positions : Sequence[PositionFeatures]

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).
        """
        if not positions:
            raise ValueError("positions must be non-empty")

        device = next(self.mlp.parameters()).device

        scalars = torch.tensor(
            [
                [
                    p.seat_fraction,
                    p.active_fraction,
                    p.to_act_after_fraction,
                    float(p.is_button),
                    float(p.is_small_blind),
                    float(p.is_big_blind),
                    float(p.is_next_to_act),
                    math.log1p(p.table_size),
                ]
                for p in positions
            ],
            dtype=torch.float32,
            device=device,
        )

        return self.mlp(scalars)
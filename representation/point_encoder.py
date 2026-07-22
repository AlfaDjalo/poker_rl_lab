"""
representation/point_encoder.py — Learned embedding for a scoring point.

Per the RL Lab implementation guide (§5, "Additional Learned Embeddings"):

    Point: Represents a scoring point.
    Examples include: High, Low, Badugi, Make-5.

A "point" here matches the engine's PointDefinition (rules/point_definition.py):
one component of a game's payout structure, e.g. holdem's single "main"
HIGH point, or a hi/lo split game's "high" and "low" points, or hopscotch's
"board" point with two node_sets. Games with multiple points feed one
PointEncoder call per point — per the guide's hierarchical diagram these
become separate tokens ("Point 1", "Point 2", ...) for the eventual
transformer, not a single pooled point vector.

Why this takes plain primitives, not engine objects
------------------------------------------------------
rules/point_definition.py and rules/game_rules.py both import ScoreType
and ShowdownType directly from `poker_eval` — the compiled C++ engine
extension. That's correct for the engine (it's the single source of
truth for poker logic, per the guide's core philosophy), but it means
importing those modules requires the compiled extension to be present.
The RL Lab must not inherit that hard dependency purely to embed a
point's *category* — PointEncoder only needs a few small integers/flags,
which can be pulled out of a PointDefinition/GameRules pair once and
handed in as plain Python values. See `PointFeatures` and
`point_features_from_definition()` below: the latter is the one place
that would import the real engine types, and it degrades gracefully
(mock_poker_eval.py's ScoreType/ShowdownType — see that file's own
docstring for the sys.modules shim — work here too, since only the
enums' integer .value is ever used).

What features actually go into the embedding
-----------------------------------------------
  score_type       : HIGH / LOW_A5 / LOW_27 / BADUGI (categorical) —
                      the dominant strategic signal; a "low" point is
                      played completely differently from a "high" one.
  showdown_type     : HOLDEM / OMAHA / MAKE5 / DRAW / BADUGI (categorical)
                      — how the best 5-card hand is assembled from hole +
                      board cards for THIS point (a point's
                      showdown_type_override can differ from the game's
                      default).
  is_low_qualified  : whether a qualifier applies (bool) — a low point
                      that never qualifies plays very differently near
                      the river than one that reliably does.
  qualifier_strength: low_qualifier normalised to [0, 1] (0 = no
                      qualifier / not a low point). Lower low_qualifier
                      (e.g. 8-or-better) means a tighter, more
                      predictable range qualifies.
  has_scoop_pairing : whether this point's no-qualify chips scoop to a
                      paired point (point_def.scoop_from is not None) —
                      changes the effective payout structure.
  num_node_sets     : how many independent board configurations this
                      point draws its best-of from (len(node_sets)),
                      log-scaled — more ways to make the point raises
                      its effective value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import math
import torch
import torch.nn as nn

# Matches mock_poker_eval.ScoreType / the real poker_eval.ScoreType by value.
NUM_SCORE_TYPES = 4       # HIGH, LOW_A5, LOW_27, BADUGI
# Matches mock_poker_eval.ShowdownType / the real poker_eval.ShowdownType by value.
NUM_SHOWDOWN_TYPES = 5    # HOLDEM, OMAHA, MAKE5, DRAW, BADUGI

_LOW_SCORE_TYPES = frozenset({1, 2})  # LOW_A5.value, LOW_27.value


@dataclass(frozen=True)
class PointFeatures:
    """
    Plain-data feature bundle for one scoring point. Framework- and
    engine-independent — everything downstream (PointEncoder) only ever
    sees this, never a real PointDefinition/GameRules pair.
    """
    score_type: int
    showdown_type: int
    is_low_qualified: bool = False
    qualifier_strength: float = 0.0   # 0.0 if not applicable
    has_scoop_pairing: bool = False
    num_node_sets: int = 1

    def __post_init__(self) -> None:
        if not (0 <= self.score_type < NUM_SCORE_TYPES):
            raise ValueError(f"score_type must be in [0, {NUM_SCORE_TYPES}), got {self.score_type}")
        if not (0 <= self.showdown_type < NUM_SHOWDOWN_TYPES):
            raise ValueError(f"showdown_type must be in [0, {NUM_SHOWDOWN_TYPES}), got {self.showdown_type}")
        if not (0.0 <= self.qualifier_strength <= 1.0):
            raise ValueError(f"qualifier_strength must be in [0, 1], got {self.qualifier_strength}")
        if self.num_node_sets < 1:
            raise ValueError(f"num_node_sets must be >= 1, got {self.num_node_sets}")


def point_features_from_definition(point_def, rules) -> PointFeatures:
    """
    Adapter: build a PointFeatures from the engine's real PointDefinition
    + GameRules objects.

    This is the one function in this module that touches engine types.
    It does NOT import rules.point_definition / rules.game_rules itself
    (avoiding a module-level hard dependency on the poker_eval C++
    extension) — it just duck-types on whatever PointDefinition/GameRules
    instances are passed in, reading the same fields those classes
    already expose (see rules/point_definition.py, rules/game_rules.py).

    Parameters
    ----------
    point_def : rules.point_definition.PointDefinition
    rules : rules.game_rules.GameRules
        The GameRules this point_def belongs to (for is_low_type() and
        low_qualifier).

    Returns
    -------
    PointFeatures
    """
    score_type = int(point_def.score_type)
    showdown_type = int(
        point_def.showdown_type_override
        if point_def.showdown_type_override is not None
        else rules.showdown_type
    )
    is_low = rules.is_low_type(point_def.score_type)
    qualifier_strength = 0.0
    if is_low and rules.low_qualifier:
        # Smaller low_qualifier ("8-or-better") = stricter = higher
        # qualifier_strength. 13 is the loosest possible A5 low qualifier.
        qualifier_strength = max(0.0, min(1.0, 1.0 - (rules.low_qualifier - 1) / 12.0))

    return PointFeatures(
        score_type=score_type,
        showdown_type=showdown_type,
        is_low_qualified=bool(is_low and rules.low_qualifier),
        qualifier_strength=qualifier_strength,
        has_scoop_pairing=point_def.scoop_from is not None,
        num_node_sets=len(point_def.node_sets),
    )


class PointEncoder(nn.Module):
    """
    Encodes one scoring point's category + qualifier structure into a
    fixed-size embedding.

    Parameters
    ----------
    score_type_dim : int
        Embedding size for the score_type lookup table. Default 8.
    showdown_type_dim : int
        Embedding size for the showdown_type lookup table. Default 8.
    scalar_hidden_dim : int
        Hidden width of the small MLP over the scalar/boolean features
        (is_low_qualified, qualifier_strength, has_scoop_pairing,
        log(num_node_sets)). Default 8.
    output_dim : int
        Final embedding dimensionality. Default 16.
    """

    _NUM_SCALAR_FEATURES = 4

    def __init__(
        self,
        score_type_dim: int = 8,
        showdown_type_dim: int = 8,
        scalar_hidden_dim: int = 8,
        output_dim: int = 16,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")

        self.output_dim = output_dim

        self.score_type_embed = nn.Embedding(NUM_SCORE_TYPES, score_type_dim)
        self.showdown_type_embed = nn.Embedding(NUM_SHOWDOWN_TYPES, showdown_type_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(self._NUM_SCALAR_FEATURES, scalar_hidden_dim),
            nn.ReLU(),
        )

        combined_dim = score_type_dim + showdown_type_dim + scalar_hidden_dim
        self.project = nn.Linear(combined_dim, output_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, point: PointFeatures) -> torch.Tensor:
        """Embed a single point. Returns shape (output_dim,)."""
        return self.forward_batch([point])[0]

    def forward_batch(self, points: Sequence[PointFeatures]) -> torch.Tensor:
        """
        Embed a batch of points.

        Parameters
        ----------
        points : Sequence[PointFeatures]
            One entry per point. Points from different games/hands can
            be mixed freely in one batch.

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).
        """
        if not points:
            raise ValueError("points must be non-empty")

        score_type_ids = torch.tensor([p.score_type for p in points], dtype=torch.long)
        showdown_type_ids = torch.tensor([p.showdown_type for p in points], dtype=torch.long)

        scalars = torch.tensor(
            [
                [
                    float(p.is_low_qualified),
                    p.qualifier_strength,
                    float(p.has_scoop_pairing),
                    math.log1p(p.num_node_sets),
                ]
                for p in points
            ],
            dtype=torch.float32,
        )

        score_embed = self.score_type_embed(score_type_ids)         # (batch, score_type_dim)
        showdown_embed = self.showdown_type_embed(showdown_type_ids)  # (batch, showdown_type_dim)
        scalar_embed = self.scalar_mlp(scalars)                       # (batch, scalar_hidden_dim)

        combined = torch.cat([score_embed, showdown_embed, scalar_embed], dim=-1)
        return self.project(combined)  # (batch, output_dim)
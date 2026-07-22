"""
representation/game_encoder.py — Learned embedding for game-specific
information.

Per the RL Lab implementation guide (§5, "Additional Learned Embeddings"):

    Game: Represents game-specific information including:
    - betting structure
    - board layout
    - payout model
    - showdown rules

A "game" here matches the engine's (GameDefinition, GameRules) pair
(rules/game_definition.py, rules/game_rules.py) produced by loader.load_game()
from a variant's YAML file (e.g. holdem.yaml, hopscotch_bomb_pot.yaml).
Unlike Point or Position, there is exactly one Game token per hand — see
the guide's transformer token list ("Hole", "Board 1", "Board 2", "Point
1", "Point 2", "Position", "Stack", "Pot", "Betting History", "Game").

Why this takes plain primitives, not engine objects
------------------------------------------------------
Same rationale as representation/point_encoder.py: rules/game_rules.py
and rules/point_definition.py both import ScoreType/ShowdownType from the
compiled `poker_eval` C++ extension. GameEncoder must not inherit that
hard dependency just to embed a game's *category*. `GameFeatures` is the
plain-data bundle; `game_features_from_definition()` is the one adapter
that would import the real engine types, and it degrades gracefully with
mock_poker_eval.py's shim (only integer `.value`s are ever read).

Two structural fields deliberately are NOT read directly off
GameDefinition/GameRules by strict type — `game_features_from_definition`
duck-types on whatever is passed in (reading `.betting_type`,
`.hole_cards`, `.node_count`, `.street_nodes`, `.ante`, `.big_blind`,
`.score_types`, `.points`, `.payout_type`), matching the duck-typing
pattern already established in point_encoder.py and position_encoder.py.

What features actually go into the embedding
-----------------------------------------------
Categorical:
  betting_type   : no_limit / pot_limit / fixed_limit / other — per the
                    CAP Game Guide, betting structure is one of the
                    biggest strategic levers ("Hold'em-style games are
                    usually played no limit... Fixed limit games are
                    strongly discouraged"). An open vocabulary (with an
                    "other" bucket) is used rather than a hand-picked
                    closed enum, since variant YAMLs are free text here
                    and new betting types may be added without touching
                    this file.
  payout_type    : points / split_pot / other — changes how components
                    combine into a single pot outcome; see
                    showdown_resolver.py's two separate resolution paths.
  showdown_type  : the game's default showdown type (HOLDEM / OMAHA /
                    MAKE5 / DRAW / BADUGI), reusing the same
                    NUM_SHOWDOWN_TYPES vocabulary as point_encoder.py so
                    the two encoders share the same categorical meaning
                    for this field (a point's showdown_type_override can
                    still differ — that's captured per-point, not here).

Scalar / boolean (mirrors the log1p-scalar treatment used for
num_node_sets in PointEncoder and table_size in PositionEncoder):
  hole_cards      : log1p(hole_cards) — 2 (hold'em) vs 4 (Omaha-style CAP
                    games) materially changes range construction.
  node_count      : log1p(node_count) — total board complexity, e.g. 5
                    (single board) vs 10 (double board) vs more for
                    exotic CAP layouts.
  num_streets     : log1p(len(street_nodes)) — how many betting/dealing
                    rounds the hand has.
  num_points      : log1p(len(rules.points)) — payout structure
                    complexity (e.g. holdem's 1 point vs a 6-point
                    double-board game).
  has_ante        : bool — antes are uncommon in CAP (per the Game
                    Guide) and change preflop incentives when present.
  ante_bb         : ante / big_blind, 0.0 if inapplicable — magnitude of
                    the ante relative to the blind structure, not just
                    its presence.
  has_low         : bool — whether any of the game's score_types is a
                    low type. A game with a low component plays very
                    differently near showdown than an all-high game.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import math
import torch
import torch.nn as nn

# Matches mock_poker_eval.ShowdownType / the real poker_eval.ShowdownType by
# value. Kept in sync with point_encoder.NUM_SHOWDOWN_TYPES deliberately —
# both encoders embed the same categorical concept.
NUM_SHOWDOWN_TYPES = 5    # HOLDEM, OMAHA, MAKE5, DRAW, BADUGI

# Low-type score_type values, matching poker_eval.ScoreType /
# mock_poker_eval.ScoreType by value (LOW_A5=1, LOW_27=2).
_LOW_SCORE_TYPES = frozenset({1, 2})

# Open vocabularies for the two free-text categorical fields. Unknown
# strings fall back to the last "other" bucket rather than raising, since
# variant YAMLs are free text and new values may appear without this file
# being updated first.
_BETTING_TYPE_VOCAB = {"no_limit": 0, "pot_limit": 1, "fixed_limit": 2}
NUM_BETTING_TYPES = len(_BETTING_TYPE_VOCAB) + 1   # + 1 "other" bucket
_BETTING_TYPE_OTHER = NUM_BETTING_TYPES - 1

_PAYOUT_TYPE_VOCAB = {"points": 0, "split_pot": 1}
NUM_PAYOUT_TYPES = len(_PAYOUT_TYPE_VOCAB) + 1     # + 1 "other" bucket
_PAYOUT_TYPE_OTHER = NUM_PAYOUT_TYPES - 1


def _betting_type_id(betting_type: Optional[str]) -> int:
    """Look up a betting_type string, defaulting to the 'other' bucket."""
    if betting_type is None:
        return _BETTING_TYPE_OTHER
    return _BETTING_TYPE_VOCAB.get(betting_type, _BETTING_TYPE_OTHER)


def _payout_type_id(payout_type: Optional[str]) -> int:
    """Look up a payout_type string, defaulting to the 'other' bucket."""
    if payout_type is None:
        return _PAYOUT_TYPE_OTHER
    return _PAYOUT_TYPE_VOCAB.get(payout_type, _PAYOUT_TYPE_OTHER)


@dataclass(frozen=True)
class GameFeatures:
    """
    Plain-data feature bundle for one game/variant. Framework- and
    engine-independent — everything downstream (GameEncoder) only ever
    sees this, never a real GameDefinition/GameRules pair.
    """
    betting_type_id: int
    payout_type_id: int
    showdown_type: int
    hole_cards: int
    node_count: int
    num_streets: int
    num_points: int
    has_ante: bool = False
    ante_bb: float = 0.0
    has_low: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.betting_type_id < NUM_BETTING_TYPES):
            raise ValueError(
                f"betting_type_id must be in [0, {NUM_BETTING_TYPES}), "
                f"got {self.betting_type_id}"
            )
        if not (0 <= self.payout_type_id < NUM_PAYOUT_TYPES):
            raise ValueError(
                f"payout_type_id must be in [0, {NUM_PAYOUT_TYPES}), "
                f"got {self.payout_type_id}"
            )
        if not (0 <= self.showdown_type < NUM_SHOWDOWN_TYPES):
            raise ValueError(
                f"showdown_type must be in [0, {NUM_SHOWDOWN_TYPES}), "
                f"got {self.showdown_type}"
            )
        if self.hole_cards < 0:
            raise ValueError(f"hole_cards must be >= 0, got {self.hole_cards}")
        if self.node_count < 0:
            raise ValueError(f"node_count must be >= 0, got {self.node_count}")
        if self.num_streets < 0:
            raise ValueError(f"num_streets must be >= 0, got {self.num_streets}")
        if self.num_points < 0:
            raise ValueError(f"num_points must be >= 0, got {self.num_points}")
        if self.ante_bb < 0:
            raise ValueError(f"ante_bb must be >= 0, got {self.ante_bb}")


def game_features_from_definition(game_def, rules) -> GameFeatures:
    """
    Adapter: build a GameFeatures from the engine's real GameDefinition +
    GameRules objects (see loader.load_game()).

    This is the one function in this module that touches engine-shaped
    types. It does not import rules.game_definition / rules.game_rules
    itself (avoiding a module-level hard dependency on the poker_eval C++
    extension) — it just duck-types on whatever objects are passed in,
    reading the same fields those classes already expose.

    Parameters
    ----------
    game_def : rules.game_definition.GameDefinition
    rules : rules.game_rules.GameRules
        The GameRules produced alongside game_def by loader.load_game().

    Returns
    -------
    GameFeatures
    """
    big_blind = getattr(game_def, "big_blind", 0) or 0
    ante = getattr(game_def, "ante", 0) or 0

    has_low = any(
        int(st) in _LOW_SCORE_TYPES for st in getattr(rules, "score_types", [])
    )

    return GameFeatures(
        betting_type_id=_betting_type_id(getattr(game_def, "betting_type", None)),
        payout_type_id=_payout_type_id(getattr(rules, "payout_type", None)),
        showdown_type=int(rules.showdown_type),
        hole_cards=int(game_def.hole_cards),
        node_count=int(game_def.node_count),
        num_streets=len(game_def.street_nodes),
        num_points=len(rules.points),
        has_ante=bool(ante > 0),
        ante_bb=(ante / big_blind) if big_blind > 0 else 0.0,
        has_low=has_low,
    )


class GameEncoder(nn.Module):
    """
    Encodes one game/variant's betting structure, board layout, payout
    model, and showdown rules into a fixed-size embedding.

    Architecturally mirrors PointEncoder (representation/point_encoder.py):
    small embedding tables for categorical fields, a small MLP over
    scalar/boolean features, concatenated and projected to output_dim.

    Parameters
    ----------
    betting_type_dim : int
        Embedding size for the betting_type lookup table. Default 8.
    payout_type_dim : int
        Embedding size for the payout_type lookup table. Default 8.
    showdown_type_dim : int
        Embedding size for the showdown_type lookup table. Default 8.
    scalar_hidden_dim : int
        Hidden width of the small MLP over the scalar/boolean features
        (log1p(hole_cards), log1p(node_count), log1p(num_streets),
        log1p(num_points), has_ante, ante_bb, has_low). Default 16.
    output_dim : int
        Final embedding dimensionality. Default 16.
    """

    _NUM_SCALAR_FEATURES = 7

    def __init__(
        self,
        betting_type_dim: int = 8,
        payout_type_dim: int = 8,
        showdown_type_dim: int = 8,
        scalar_hidden_dim: int = 16,
        output_dim: int = 16,
    ) -> None:
        super().__init__()

        if output_dim <= 0:
            raise ValueError(f"output_dim must be > 0, got {output_dim}")

        self.output_dim = output_dim

        self.betting_type_embed = nn.Embedding(NUM_BETTING_TYPES, betting_type_dim)
        self.payout_type_embed = nn.Embedding(NUM_PAYOUT_TYPES, payout_type_dim)
        self.showdown_type_embed = nn.Embedding(NUM_SHOWDOWN_TYPES, showdown_type_dim)

        self.scalar_mlp = nn.Sequential(
            nn.Linear(self._NUM_SCALAR_FEATURES, scalar_hidden_dim),
            nn.ReLU(),
        )

        combined_dim = (
            betting_type_dim + payout_type_dim + showdown_type_dim + scalar_hidden_dim
        )
        self.project = nn.Linear(combined_dim, output_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, game: GameFeatures) -> torch.Tensor:
        """Embed a single game. Returns shape (output_dim,)."""
        return self.forward_batch([game])[0]

    def forward_batch(self, games: Sequence[GameFeatures]) -> torch.Tensor:
        """
        Embed a batch of games.

        Parameters
        ----------
        games : Sequence[GameFeatures]
            One entry per game/variant. Different variants can be mixed
            freely in one batch (e.g. self-play across a curriculum).

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).
        """
        if not games:
            raise ValueError("games must be non-empty")

        betting_type_ids = torch.tensor(
            [g.betting_type_id for g in games], dtype=torch.long
        )
        payout_type_ids = torch.tensor(
            [g.payout_type_id for g in games], dtype=torch.long
        )
        showdown_type_ids = torch.tensor(
            [g.showdown_type for g in games], dtype=torch.long
        )

        scalars = torch.tensor(
            [
                [
                    math.log1p(g.hole_cards),
                    math.log1p(g.node_count),
                    math.log1p(g.num_streets),
                    math.log1p(g.num_points),
                    float(g.has_ante),
                    g.ante_bb,
                    float(g.has_low),
                ]
                for g in games
            ],
            dtype=torch.float32,
        )

        betting_embed = self.betting_type_embed(betting_type_ids)     # (batch, betting_type_dim)
        payout_embed = self.payout_type_embed(payout_type_ids)        # (batch, payout_type_dim)
        showdown_embed = self.showdown_type_embed(showdown_type_ids)  # (batch, showdown_type_dim)
        scalar_embed = self.scalar_mlp(scalars)                       # (batch, scalar_hidden_dim)

        combined = torch.cat(
            [betting_embed, payout_embed, showdown_embed, scalar_embed], dim=-1
        )
        return self.project(combined)  # (batch, output_dim)
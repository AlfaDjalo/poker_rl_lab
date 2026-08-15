"""
representation/stack_encoder.py — Learned embedding for stack/pot state
at the current decision point.

Per the RL Lab implementation guide (§6, "Preferred Architecture")'s
transformer token list:

    Hole, Board 1, Board 2, Point 1, Point 2, Position, Stack, Pot,
    Betting History, Game

"Stack" and "Pot" are named as their own tokens, distinct from "Game"
(betting *structure*: no-limit/pot-limit, ante presence, showdown
rules — see game_encoder.py) and from "Position" (seat topology — see
position_encoder.py). Until this module, neither RepresentationInput
nor RepresentationModel carried ANY stack/pot signal at all: the hero's
remaining stack, the pot size, and bet-to-call never appeared anywhere
in the trunk's input. This was silently starving every downstream
consumer of stack-depth-dependent behaviour — see
push_fold_duo_hand_grids.py's SB push grid, which came out identical
across every stack_bb value because nothing in RepresentationInput
varied with it (BB's grid only happened to vary because
HistoryEncoder incidentally saw SB's push amount as one event's
pot_fraction — a side channel, not real stack awareness).

Why NOT fold this into GameEncoder
--------------------------------------
GameEncoder already owns ante_bb (ante relative to big_blind) — but that
is a STATIC property of the variant (fixed for the whole hand/game,
read once from GameDefinition/GameRules). Stack, pot, and bet-to-call
are DYNAMIC — they change every decision point within a single hand.
Mixing a per-decision quantity into the one-token-per-hand Game
embedding would mean GameEncoder's output is no longer constant across
a hand's decision points, breaking the "exactly one Game token per
hand" invariant the guide's diagram and game_encoder.py's own docstring
both rely on. Blinds themselves (small_blind/big_blind) stay implicit
here too: every chip quantity below is normalised to big-blind units
(matching the BB/100 convention used throughout evaluation/), so the
blind SIZE is the unit of measurement, not a separate feature — a
`sb_bb` ratio (small_blind/big_blind) is included below purely as a
robustness feature for non-standard blind structures (normally always
0.5), which is a per-hand constant like ante_bb but travels with this
encoder since it's cheap and keeps every blind-derived ratio in one
place.

Why plain primitives, not engine objects
--------------------------------------------
Same rationale as every other encoder here: no PointDefinition/GameRules
poker_eval dependency. `StackFeatures` is pure data; the two adapter
functions below (`stack_features_from_scenario`,
`stack_features_from_observation`) only ever read fields already
exposed by BaseScenario / StructuredObservation.

What features actually go into the embedding
-----------------------------------------------
  stack_bb            : hero's remaining stack, in big blinds, log1p-scaled.
  effective_stack_bb   : min(hero stack, largest still-live opponent
                          stack), in big blinds, log1p-scaled — the
                          actual number of chips that can move this hand,
                          which is what SPR is conventionally computed
                          against, not hero's raw stack alone.
  pot_bb               : current pot, in big blinds, log1p-scaled.
  bet_to_call_bb       : additional chips hero must add to call, in big
                          blinds, log1p-scaled. 0 when check is available.
  spr                  : effective_stack_bb (unscaled bb) / pot_bb
                          (unscaled bb) — stack-to-pot ratio, the
                          single most common shorthand poker players use
                          for "how committed are we already / how much
                          room is left to play"; log1p-scaled since SPR
                          is unbounded above and heavily right-skewed.
  pot_odds             : bet_to_call / (pot + bet_to_call), in [0, 1),
                          0 when bet_to_call == 0 — the price hero is
                          being offered to continue, already normalised
                          so it needs no further scaling.
  sb_bb                : small_blind / big_blind. Normally a constant
                          0.5; kept as a feature (not assumed) so a
                          future non-standard blind structure is still
                          representable.

Dependencies
------------
    PyTorch. scenarios.base_scenario.BaseScenario,
    envs.observations.StructuredObservation for the two adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import math
import torch
import torch.nn as nn

from poker_rl_lab.scenarios.base_scenario import BaseScenario
from poker_rl_lab.envs.observations import StructuredObservation

# Floor to avoid a divide-by-zero when pot_bb is 0 (e.g. a fresh preflop
# push-fold-duo scenario before SB has acted — see
# push_fold_duo_generator.py: pot=0, both blinds live in current_bet).
_MIN_POT_BB = 1e-3


# ---------------------------------------------------------------------------
# Plain-data feature bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackFeatures:
    """
    Framework- and engine-independent feature bundle for the stack/pot
    state at one decision point. Downstream (StackEncoder) only ever
    sees this, never a BaseScenario/StructuredObservation object.
    """
    stack_bb: float
    effective_stack_bb: float
    pot_bb: float
    bet_to_call_bb: float
    spr: float
    pot_odds: float
    sb_bb: float = 0.5

    def __post_init__(self) -> None:
        if self.stack_bb < 0:
            raise ValueError(f"stack_bb must be >= 0, got {self.stack_bb}")
        if self.effective_stack_bb < 0:
            raise ValueError(
                f"effective_stack_bb must be >= 0, got {self.effective_stack_bb}"
            )
        if self.pot_bb < 0:
            raise ValueError(f"pot_bb must be >= 0, got {self.pot_bb}")
        if self.bet_to_call_bb < 0:
            raise ValueError(
                f"bet_to_call_bb must be >= 0, got {self.bet_to_call_bb}"
            )
        if self.spr < 0:
            raise ValueError(f"spr must be >= 0, got {self.spr}")
        if not (0.0 <= self.pot_odds <= 1.0):
            raise ValueError(f"pot_odds must be in [0, 1], got {self.pot_odds}")
        if self.sb_bb <= 0:
            raise ValueError(f"sb_bb must be > 0, got {self.sb_bb}")


# ---------------------------------------------------------------------------
# Shared computation
# ---------------------------------------------------------------------------

def _compute_stack_features(
    hero_stack: int,
    opponent_stacks: Sequence[int],
    pot: int,
    bet_to_call: int,
    small_blind: int,
    big_blind: int,
) -> StackFeatures:
    if big_blind <= 0:
        raise ValueError(f"big_blind must be > 0, got {big_blind}")

    live_opponents = [s for s in opponent_stacks if s > 0]
    # If every opponent is already covered/all-in for 0, effective stack
    # collapses to hero's own — nothing more can move this hand from the
    # opponent side.
    max_opponent_stack = max(live_opponents) if live_opponents else 0
    effective_stack = min(hero_stack, max_opponent_stack) if live_opponents else hero_stack

    stack_bb = hero_stack / big_blind
    effective_stack_bb = effective_stack / big_blind
    pot_bb = pot / big_blind
    bet_to_call_bb = bet_to_call / big_blind

    spr = effective_stack_bb / max(pot_bb, _MIN_POT_BB)
    pot_odds = (
        bet_to_call / (pot + bet_to_call) if (pot + bet_to_call) > 0 else 0.0
    )

    return StackFeatures(
        stack_bb=stack_bb,
        effective_stack_bb=effective_stack_bb,
        pot_bb=pot_bb,
        bet_to_call_bb=bet_to_call_bb,
        spr=spr,
        pot_odds=pot_odds,
        sb_bb=small_blind / big_blind,
    )


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def stack_features_from_scenario(scenario: BaseScenario, seat: int) -> StackFeatures:
    """Build StackFeatures for one seat from a BaseScenario (reset() time)."""
    hero = scenario.players[seat]
    opponents = [p.stack for i, p in enumerate(scenario.players) if i != seat]

    return _compute_stack_features(
        hero_stack=hero.stack,
        opponent_stacks=opponents,
        pot=scenario.pot,
        bet_to_call=max(scenario.bet_to_call - hero.current_bet, 0),
        small_blind=scenario.small_blind,
        big_blind=scenario.big_blind,
    )


def stack_features_from_observation(
    obs: StructuredObservation, small_blind: int, big_blind: int
) -> StackFeatures:
    """
    Build StackFeatures for the hero from a StructuredObservation.

    Parameters
    ----------
    obs : StructuredObservation
        Only ever handed back at the hero's own decision point (see
        envs/base_env.py's contract), so obs.players' `is_hero` flag
        unambiguously identifies the hero's own entry.
    small_blind, big_blind : int
        StructuredObservation carries neither (see observations.py's
        docstring — the Environment stays independent of blind-size
        bookkeeping beyond what BettingObs already needs); callers
        already have these on hand from whatever generated the episode
        (ObservationAdapter's callers, or the scenario/config that
        produced the hand) — same pattern game_encoder.py's adapter
        takes a `rules` argument for game-level values not on the obs.
    """
    hero = next(p for p in obs.players if p.is_hero)
    opponents = [p.stack for p in obs.players if not p.is_hero and not p.has_folded]

    return _compute_stack_features(
        hero_stack=hero.stack,
        opponent_stacks=opponents,
        pot=obs.betting.pot,
        bet_to_call=obs.betting.bet_to_call,
        small_blind=small_blind,
        big_blind=big_blind,
    )


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class StackEncoder(nn.Module):
    """
    Encodes the current stack/pot state into a fixed-size embedding.

    Like PositionEncoder, every feature here is continuous — no
    categorical lookup tables — so this is a single small MLP.

    Parameters
    ----------
    hidden_dim : int
        Hidden width of the MLP. Default 16.
    output_dim : int
        Final embedding dimensionality. Default 16.
    """

    _NUM_SCALAR_FEATURES = 7  # log1p(stack_bb), log1p(effective_stack_bb),
                               # log1p(pot_bb), log1p(bet_to_call_bb),
                               # log1p(spr), pot_odds, sb_bb

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

    def forward(self, stack: StackFeatures) -> torch.Tensor:
        """Embed a single decision point's stack state. Returns shape (output_dim,)."""
        return self.forward_batch([stack])[0]

    def forward_batch(self, stacks: Sequence[StackFeatures]) -> torch.Tensor:
        """
        Embed a batch of stack states.

        Parameters
        ----------
        stacks : Sequence[StackFeatures]

        Returns
        -------
        torch.Tensor
            Shape (batch, output_dim).
        """
        if not stacks:
            raise ValueError("stacks must be non-empty")

        device = next(self.mlp.parameters()).device

        scalars = torch.tensor(
            [
                [
                    math.log1p(s.stack_bb),
                    math.log1p(s.effective_stack_bb),
                    math.log1p(s.pot_bb),
                    math.log1p(s.bet_to_call_bb),
                    math.log1p(s.spr),
                    s.pot_odds,
                    s.sb_bb,
                ]
                for s in stacks
            ],
            dtype=torch.float32,
            device=device,
        )

        return self.mlp(scalars)
"""
envs/observations.py — Structured observation types for the RL agent.

The Environment produces a StructuredObservation at every decision point.
This preserves the logical structure of the poker state without committing
to any particular neural network representation.

The Representation Model (Phase 5) is responsible for converting these
objects into embeddings. The Environment must remain independent of any
neural architecture decision.

Card ids follow the engine convention:
    rank = card_id % 13   (0=2, 1=3, ..., 12=A)
    suit = card_id // 13  (0=clubs, 1=diamonds, 2=hearts, 3=spades)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

from poker_rl_lab.actions.abstract_action import NUM_ACTIONS


# ---------------------------------------------------------------------------
# Component observations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HoleCardsObs:
    """The hero's private hole cards."""
    cards: Tuple[int, ...]   # card ids; length = game_def.hole_cards


@dataclass(frozen=True)
class BoardObs:
    """
    Current board state.

    Each element is a card id or None (node not yet dealt).
    Indexed by node position as defined in the game's board layout.
    """
    node_cards: Tuple[Optional[int], ...]


@dataclass(frozen=True)
class PlayerObs:
    """
    Public information about one player seat.

    Hole cards are NOT included — they live in HoleCardsObs for the hero
    only. Opponent hole cards are hidden.
    """
    seat: int
    stack: int
    current_bet: int    # chips committed this street
    has_folded: bool
    is_all_in: bool
    is_hero: bool


@dataclass(frozen=True)
class BettingObs:
    """Current betting state at the decision point."""
    pot: int
    bet_to_call: int      # additional chips the hero must add to call
    min_raise: int        # minimum additional raise amount
    street_index: int     # engine street index (0=pre-flop board deal, etc.)
    raises_this_street: int


@dataclass(frozen=True)
class HistoryEventObs:
    """
    One action taken by one seat during the current hand, as recorded
    live by an Environment.

    This is deliberately a thin, env-facing record — it captures just
    enough to reconstruct "who did what, when, with how much" from
    inside push_fold_env.py / river_env.py as the episode progresses.
    It is NOT the same type as representation.history_encoder.HistoryEvent
    (the encoder-facing plain-data type used once this is fed into the
    Representation Model) — that type additionally wants seat_fraction
    and is_hero, which depend on table geometry the Environment doesn't
    need to compute just to log an event. An adapter from
    HistoryEventObs (+ hero_seat, dealer_position) to
    representation.history_encoder.HistoryEvent belongs in the
    Representation Model layer, not here.

    Attributes
    ----------
    seat : int
        0-based seat index of the acting player.
    action_type : int
        Value of actions.abstract_action.ActionType for this action.
    street_index : int
        Engine street index the action occurred on.
    pot_fraction : float
        Chips this action added to the pot, divided by the pot size
        immediately before the action. 0.0 for FOLD/CHECK.
    is_aggressor : bool
        True if this action set a new bet_to_call (bet/raise/all-in
        that increases the bet), as opposed to a call/check.
    """
    seat: int
    action_type: int
    street_index: int
    pot_fraction: float = 0.0
    is_aggressor: bool = False

    def __post_init__(self) -> None:
        if self.seat < 0:
            raise ValueError(f"seat must be >= 0, got {self.seat}")
        if not (0 <= self.action_type < NUM_ACTIONS):
            raise ValueError(
                f"action_type must be in [0, {NUM_ACTIONS}), got {self.action_type}"
            )
        if self.street_index < 0:
            raise ValueError(f"street_index must be >= 0, got {self.street_index}")
        if self.pot_fraction < 0:
            raise ValueError(f"pot_fraction must be >= 0, got {self.pot_fraction}")


@dataclass(frozen=True)
class HistoryObs:
    """
    Ordered sequence of actions taken so far during the current hand.

    Empty tuple means no actions have been recorded yet (e.g. the very
    first decision point). Order matters and must be chronological.
    """
    events: Tuple[HistoryEventObs, ...] = ()


# ---------------------------------------------------------------------------
# Top-level observation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StructuredObservation:
    """
    ...
    dealer_position : int
        0-based seat index of the button. Added so Position-embedding
        features (representation/position_encoder.py) can be recomputed
        at every decision point, not just at reset() from the raw
        BaseScenario. Static for the lifetime of a hand.
    history : HistoryObs
        Actions taken so far this hand, in chronological order. Feeds
        representation/history_encoder.py (via an adapter — see
        HistoryEventObs's docstring). Defaults to empty for callers/tests
        that don't populate it.
    ...
    """
    hero_seat: int
    hole_cards: HoleCardsObs
    board: BoardObs
    players: Tuple[PlayerObs, ...]
    betting: BettingObs
    legal_mask: Tuple[bool, ...]
    game_name: str
    history: HistoryObs = field(default_factory=HistoryObs)
    dealer_position: int = 0
    is_terminal: bool = False

    def __post_init__(self) -> None:
        if len(self.legal_mask) != NUM_ACTIONS:
            raise ValueError(
                f"legal_mask must have length {NUM_ACTIONS}, "
                f"got {len(self.legal_mask)}"
            )
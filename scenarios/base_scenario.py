"""
base_scenario.py — Core scenario data structure.

A Scenario is an immutable snapshot of the initial conditions for one RL
episode. It contains exactly what is needed to construct a PokerState and
nothing else — no rewards, no policies, no neural network concepts.

Cards are represented as integers in [0, 51] matching the Engine convention:
    rank = card_id % 13   (0=2, 1=3, ..., 12=A)
    suit = card_id // 13  (0=clubs, 1=diamonds, 2=hearts, 3=spades)

Hand masks are integer bitmasks over the 52-card space. A card with id N
sets bit N:  mask = 1 << card_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class PlayerScenario:
    """
    Initial state for one player seat.

    Attributes
    ----------
    stack : int
        Chips available to bet, not including committed chips.
    hand_mask : int
        Bitmask of hole cards already dealt. 0 if not yet dealt.
    current_bet : int
        Chips committed to the pot this street (e.g. blind already posted).
    total_contribution : int
        Total chips moved from stack to pot across all streets so far.
    has_folded : bool
        True if this player has already folded.
    is_all_in : bool
        True if this player has no chips remaining (stack == 0, active).
    """

    stack: int
    hand_mask: int = 0
    current_bet: int = 0
    total_contribution: int = 0
    has_folded: bool = False
    is_all_in: bool = False

    def __post_init__(self) -> None:
        if self.stack < 0:
            raise ValueError(f"stack must be >= 0, got {self.stack}")
        if self.hand_mask < 0:
            raise ValueError(f"hand_mask must be >= 0, got {self.hand_mask}")
        if self.current_bet < 0:
            raise ValueError(f"current_bet must be >= 0, got {self.current_bet}")
        if self.total_contribution < 0:
            raise ValueError(
                f"total_contribution must be >= 0, got {self.total_contribution}"
            )


@dataclass(frozen=True)
class BaseScenario:
    """
    Initial conditions for one RL episode.

    A BaseScenario is the single object passed to Environment.reset() to
    start a new episode. It defines the complete poker position from which
    play continues, without any learning-related information.

    Attributes
    ----------
    game_name : str
        Variant identifier (e.g. "holdem", "omaha"). Corresponds to a
        YAML game definition in the Engine.
    street_index : int
        Current street: 0=preflop, 1=flop, 2=turn, 3=river.
    pot : int
        Chips already in the pot (excluding current_bet of players).
    dealer_position : int
        0-based seat index of the dealer button.
    current_player : int
        0-based seat index of the player who must act next.
    bet_to_call : int
        The total bet amount a player must match to call. 0 means check
        is available.
    min_raise : int
        Minimum legal raise amount (total, not delta).
    players : tuple[PlayerScenario, ...]
        One entry per seat, in seat order.
    node_cards : tuple[Optional[int], ...]
        Board cards by node index. None = not yet dealt. Card ids in [0,51].
    small_blind : int
        Small blind size for this game.
    big_blind : int
        Big blind size for this game.
    discard_pile : tuple[int, ...]
        Card ids that have been seen and removed from the live deck.
    raises_this_street : int
        Number of raises on the current street (used for fixed-limit cap).
    last_aggressor : Optional[int]
        0-based seat index of the last player to bet or raise, or None.
    """

    game_name: str
    street_index: int
    pot: int
    dealer_position: int
    current_player: int
    bet_to_call: int
    min_raise: int
    players: tuple
    node_cards: tuple
    small_blind: int = 1
    big_blind: int = 2
    discard_pile: tuple = field(default_factory=tuple)
    raises_this_street: int = 0
    last_aggressor: Optional[int] = None

    def __post_init__(self) -> None:
        if self.street_index < 0:
            raise ValueError(f"street_index must be >= 0, got {self.street_index}")
        if self.pot < 0:
            raise ValueError(f"pot must be >= 0, got {self.pot}")
        if len(self.players) < 2:
            raise ValueError(
                f"At least 2 players required, got {len(self.players)}"
            )
        if not (0 <= self.dealer_position < len(self.players)):
            raise ValueError(
                f"dealer_position {self.dealer_position} out of range "
                f"for {len(self.players)} players"
            )
        if not (0 <= self.current_player < len(self.players)):
            raise ValueError(
                f"current_player {self.current_player} out of range "
                f"for {len(self.players)} players"
            )
        if self.bet_to_call < 0:
            raise ValueError(f"bet_to_call must be >= 0, got {self.bet_to_call}")
        if self.min_raise < 0:
            raise ValueError(f"min_raise must be >= 0, got {self.min_raise}")
        if self.small_blind <= 0:
            raise ValueError(f"small_blind must be > 0, got {self.small_blind}")
        if self.big_blind <= 0:
            raise ValueError(f"big_blind must be > 0, got {self.big_blind}")

    @property
    def player_count(self) -> int:
        """Number of seats in this scenario."""
        return len(self.players)

    @property
    def active_players(self) -> List[int]:
        """Seat indices of players who have not folded."""
        return [i for i, p in enumerate(self.players) if not p.has_folded]

    @property
    def all_cards_mask(self) -> int:
        """Bitmask of every card currently in use (hole + board)."""
        mask = 0
        for p in self.players:
            mask |= p.hand_mask
        for card_id in self.node_cards:
            if card_id is not None:
                mask |= 1 << card_id
        for card_id in self.discard_pile:
            mask |= 1 << card_id
        return mask
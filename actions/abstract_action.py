"""
abstract_action.py — Abstract action space for RL agents.

Agents choose from a fixed discrete set of abstract action types. The
Environment is responsible for translating each abstract action into a
concrete Engine Action with a legal bet amount.

Why abstract actions?
---------------------
Bet sizing is continuous, but neural networks produce fixed-size outputs.
We discretise sizing into a small set of fractions of the pot, which
covers the strategically important bet sizes across all poker variants.
The Policy selects an action type index; the Environment maps it to chips.

Action types
------------
    FOLD    : Discard hand; forfeit any future claim on the pot.
    CHECK   : Pass action without betting (only legal when bet_to_call == 0).
    CALL    : Match the current bet_to_call.
    BET_25  : Bet 25% of the pot.
    BET_50  : Bet 50% of the pot.
    BET_75  : Bet 75% of the pot.
    BET_100 : Bet 100% of the pot (pot-sized bet).
    BET_150 : Bet 150% of the pot.
    ALL_IN  : Shove entire remaining stack.

The same set covers both bet (no prior bet this street) and raise
(someone has already bet). The Environment decides whether to call the
resulting Engine action a "bet" or a "raise".

Legal action masking
--------------------
Not all actions are legal in every state. The Environment provides a
boolean mask over the action space. The Policy multiplies its logits by
this mask before softmax so illegal actions have zero probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional


class ActionType(IntEnum):
    """
    Discrete action types available to an RL agent.

    IntEnum so that action indices can be used directly as neural network
    output indices without a separate mapping step.
    """
    FOLD    = 0
    CHECK   = 1
    CALL    = 2
    BET_25  = 3
    BET_50  = 4
    BET_75  = 5
    BET_100 = 6
    BET_150 = 7
    ALL_IN  = 8

    def __str__(self) -> str:
        return self.name

    @property
    def is_bet(self) -> bool:
        """True for any action that places chips into the pot aggressively."""
        return self in (
            ActionType.BET_25,
            ActionType.BET_50,
            ActionType.BET_75,
            ActionType.BET_100,
            ActionType.BET_150,
            ActionType.ALL_IN,
        )

    @property
    def pot_fraction(self) -> Optional[float]:
        """
        Fraction of the pot this action bets, or None for non-bet actions.

        ALL_IN is also None because its size is determined by the stack,
        not a pot fraction.
        """
        _fractions = {
            ActionType.BET_25:  0.25,
            ActionType.BET_50:  0.50,
            ActionType.BET_75:  0.75,
            ActionType.BET_100: 1.00,
            ActionType.BET_150: 1.50,
        }
        return _fractions.get(self)


# ---------------------------------------------------------------------------
# Full action space
# ---------------------------------------------------------------------------

NUM_ACTIONS: int = len(ActionType)
ALL_ACTION_TYPES: List[ActionType] = list(ActionType)


# ---------------------------------------------------------------------------
# Concrete abstract action
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AbstractAction:
    """
    A single agent action.

    Attributes
    ----------
    action_type : ActionType
        Which abstract action was chosen.

    Notes
    -----
    No amount field — sizing is resolved by the Environment. The agent
    only chooses the type; the Environment handles chips.
    """
    action_type: ActionType

    def __str__(self) -> str:
        return f"AbstractAction({self.action_type})"

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# Sizing helper (used by the Environment, not the Policy)
# ---------------------------------------------------------------------------

def compute_bet_amount(
    action_type: ActionType,
    pot: int,
    stack: int,
    to_call: int,
    min_raise: int,
    current_bet: int,
) -> int:
    """
    Translate an abstract bet/raise action into a chip amount.

    The amount returned is the **total** bet for this player on the
    current street (matching the Engine's BET/RAISE semantics).

    Parameters
    ----------
    action_type : ActionType
        Must be a bet-type action (action_type.is_bet == True).
    pot : int
        Current pot size in chips (before this action).
    stack : int
        Player's remaining stack (not including current_bet).
    to_call : int
        Additional chips needed to call (i.e. bet_to_call - current_bet).
    min_raise : int
        Minimum legal total bet (from GameState.min_raise).
    current_bet : int
        Chips this player has already committed on the CURRENT street
        (e.g. GameState.players[seat].current_bet / PlayerObs.current_bet).
        Required to compute the true all-in ceiling — see Notes.

    Returns
    -------
    int
        Total amount for the BET/RAISE Engine action, clamped to
        [min_raise, current_bet + stack].

    Raises
    ------
    ValueError
        If action_type is not a bet-type action.

    Notes
    -----
    The all-in ceiling is ``current_bet + stack`` — everything the player
    has, both already committed this street and still in the stack — NOT
    ``stack + to_call``. Those coincide only when ``to_call == current_bet``
    (a coincidence, not a general identity): whenever a player already
    faces a live bet with nothing yet committed this street (current_bet
    == 0, to_call == bet_to_call > 0), ``stack + to_call`` double-counts
    the bet being faced and overstates the true max by exactly
    ``bet_to_call - 2 * current_bet``. An earlier version of this
    function used ``stack + to_call`` and could compute an ALL_IN/BET_*
    amount exceeding the player's actual chips, driving the Engine's
    tracked stack negative once applied (surfaced by
    envs/full_holdem_env.py under multi-street, multi-raise play — see
    that module's test suite).
    """
    if not action_type.is_bet:
        raise ValueError(
            f"compute_bet_amount called with non-bet action: {action_type}"
        )

    max_amount = current_bet + stack

    if action_type == ActionType.ALL_IN:
        return max_amount

    fraction = action_type.pot_fraction
    assert fraction is not None

    effective_pot = pot + to_call
    raw_amount = int(fraction * effective_pot) + to_call

    return max(min_raise, min(raw_amount, max_amount))


# ---------------------------------------------------------------------------
# Legal action mask builder (used by the Environment)
# ---------------------------------------------------------------------------

def legal_action_mask(
    available: List[str],
    pot: int,
    stack: int,
    to_call: int,
    min_raise: int,
    max_raise: int,
) -> List[bool]:
    """
    Build a boolean mask over all action types for the current game state.

    Parameters
    ----------
    available : List[str]
        Engine legal action names in lowercase
        (e.g. ``["fold", "call", "bet"]`` or ``["check", "bet"]``).
    pot : int
        Current pot.
    stack : int
        Acting player's remaining stack.
    to_call : int
        Additional chips needed to call.
    min_raise : int
        Minimum legal total bet.
    max_raise : int
        Maximum legal total bet (stack ceiling).

    Returns
    -------
    List[bool]
        Length ``NUM_ACTIONS``. ``True`` at index ``i`` means
        ``ActionType(i)`` is legal in this state.
    """
    avail_set = set(available)
    mask = [False] * NUM_ACTIONS

    mask[ActionType.FOLD]  = "fold"  in avail_set
    mask[ActionType.CHECK] = "check" in avail_set
    mask[ActionType.CALL]  = "call"  in avail_set

    can_bet = "bet" in avail_set or "raise" in avail_set
    if can_bet:
        for action_type in (
            ActionType.BET_25,
            ActionType.BET_50,
            ActionType.BET_75,
            ActionType.BET_100,
            ActionType.BET_150,
        ):
            amount = compute_bet_amount(
                action_type, pot, stack, to_call, min_raise, 0
            )
            mask[action_type] = min_raise <= amount <= max_raise

        mask[ActionType.ALL_IN] = True

    return mask
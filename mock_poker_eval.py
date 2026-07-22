"""
mock_poker_eval.py — Pure-Python stand-in for the poker_eval C++ extension.

The real poker_eval module is a compiled C++ extension that is not
available in environments such as QuantConnect free tier. This module
provides the same public surface (ScoreType, ShowdownType enums and a
stub evaluate_hands function) so that all engine modules that import
from poker_eval can be loaded without the binary.

Usage (at the very top of any test or script, before engine imports):

    import sys
    try:
        import poker_eval          # use real extension if available
    except ImportError:
        import mock_poker_eval
        sys.modules["poker_eval"] = mock_poker_eval

The MockScoringEngine already bypasses evaluate_hands, so pairing this
shim with MockScoringEngine is sufficient for full unit-test coverage of
all engine logic that does not depend on actual hand evaluation quality.
"""

from enum import IntEnum
from typing import List


class ScoreType(IntEnum):
    HIGH   = 0
    LOW_A5 = 1
    LOW_27 = 2
    BADUGI = 3


class ShowdownType(IntEnum):
    HOLDEM = 0
    OMAHA  = 1
    MAKE5  = 2
    DRAW   = 3
    BADUGI = 4


def evaluate_hands(
    hole_masks: List[int],
    board_mask: int,
    score_type: ScoreType,
    showdown_type: ShowdownType,
) -> List[List[int]]:
    """
    Stub evaluator.  Returns [score, best_hand_mask] per player.
    The score is a monotonically increasing index so that the last player
    in the list always "wins" — identical to MockScoringEngine's contract.
    """
    return [[1_500_000 + i, mask] for i, mask in enumerate(hole_masks)]
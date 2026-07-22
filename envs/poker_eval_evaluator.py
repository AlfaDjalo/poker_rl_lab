"""
envs/poker_eval_evaluator.py — Real showdown evaluator for RL environments.

Drop-in replacement for envs.push_fold_env.MockEvaluator (and any other
env's board-comparison stub) that resolves showdowns using the actual
poker_eval C++ extension instead of a raw mask-integer placeholder.

Why a runout is needed
-----------------------
poker_eval's HOLDEM evaluator (evaluate_holdem_high) needs >= 5 cards
total across hole + board to produce a score (see evaluate_holdem.cpp:
"if (hole.size() < 2) return {0, 0}", and best-5-of-N combinations
requires N >= 5). Push-fold scenarios are dealt with an EMPTY board
(no node_cards) — see push_fold_generator.py — so at showdown there are
only 2+2 = 4 known cards. This evaluator completes the missing board
cards with a random runout (sampled from an internal random.Random,
excluding all known hero/villain/board cards) before scoring, matching
standard Monte-Carlo poker RL training (reward reflects one sampled
hand's outcome, not analytic equity).

No dependency on poker_engine.cards.mask by design — RL Lab keeps its
own minimal card-mask helpers (see push_fold_env.py's module docstring
"Card helpers (no dependency on cards.mask)"), so this module does the
same rather than pulling in the full poker_engine package.

poker_eval shim
----------------
Same pattern as push_fold_experiment.py / mock_poker_eval.py: prefer
the real compiled extension, fall back to the mock (which returns
"last player wins" and ignores the board entirely — fine for pipeline
smoke tests, but it carries no real poker knowledge; see
mock_poker_eval.py's own docstring).
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

try:
    import poker_engine.poker_eval as poker_eval # noqa: F401  (prefer the real extension if present)
except ImportError:
    import mock_poker_eval as poker_eval  # noqa: F401


# ── Local card-mask helpers (mirrors push_fold_env.py's pattern) ────────────

def _count_bits(mask: int) -> int:
    return bin(mask).count("1")


def _cards_to_mask(card_ids: Sequence[int]) -> int:
    mask = 0
    for cid in card_ids:
        mask |= 1 << cid
    return mask


# ── Evaluator ─────────────────────────────────────────────────────────────

class PokerEvalEvaluator:
    """
    Real showdown evaluator for HOLDEM-showdown environments (push-fold,
    river, full hold'em).

    Same public contract as envs.push_fold_env.MockEvaluator::

        hero_wins(hero_mask, villain_mask, board_mask) -> Optional[bool]

    ``True`` = hero wins, ``False`` = villain wins, ``None`` = chop.

    Parameters
    ----------
    seed : int, optional
        Seeds the internal random.Random used for board runouts. Pass a
        seed for reproducible evaluation runs (mirrors
        envs.base_env.RandomHeroPolicy's seeding convention); omit for
        an unseeded (fresh-RNG-state) runout each call.

    Notes
    -----
    Only HIGH / HOLDEM is implemented — sufficient for push-fold and
    standard river/full hold'em curriculum stages. A hi/lo or
    Omaha-showdown env would need its own evaluator (score_type /
    showdown_type are not env-configurable here on purpose, to keep
    this class a minimal drop-in for the existing MockEvaluator
    call sites).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def hero_wins(
        self,
        hero_mask: int,
        villain_mask: int,
        board_mask: int,
    ) -> Optional[bool]:
        full_board = self._complete_board(hero_mask, villain_mask, board_mask)

        raw: List[List[int]] = poker_eval.evaluate_hands(
            [hero_mask, villain_mask],
            full_board,
            poker_eval.ScoreType.HIGH,
            poker_eval.ShowdownType.HOLDEM,
        )
        hero_score = raw[0][0]
        villain_score = raw[1][0]

        if hero_score > villain_score:
            return True
        if hero_score < villain_score:
            return False
        return None

    # ── Runout ────────────────────────────────────────────────────────────

    def _complete_board(self, hero_mask: int, villain_mask: int, board_mask: int) -> int:
        """
        Fill board_mask up to 5 cards with a random runout, excluding
        every card already known (hero, villain, existing board).
        """
        needed = 5 - _count_bits(board_mask)
        if needed <= 0:
            return board_mask

        used_mask = hero_mask | villain_mask | board_mask
        available = [c for c in range(52) if not (used_mask >> c) & 1]

        if len(available) < needed:
            raise RuntimeError(
                f"Not enough cards left to complete the board: "
                f"need {needed}, have {len(available)}."
            )

        runout = self._rng.sample(available, needed)
        return board_mask | _cards_to_mask(runout)
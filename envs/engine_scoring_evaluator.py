"""
envs/engine_scoring_evaluator.py — Showdown evaluator for RL environments
that resolves hands through the real poker_engine scoring path
(poker_engine.scoring.scoring_engine.CppScoringEngine), instead of
calling the poker_eval extension directly (contrast with
envs.poker_eval_evaluator.PokerEvalEvaluator, which does exactly that
and is kept around for environments that don't want the poker_engine
dependency).

Why this exists
----------------
PushFoldEnv / RiverEnv are deliberately self-contained (see their own
docstrings) and were originally wired to PokerEvalEvaluator, which
calls `poker_eval.evaluate_hands(...)` directly — bypassing
poker_engine.scoring entirely. FullHoldemEnv, by contrast, is
Engine-backed and takes a `scoring_engine` (CppScoringEngine or
MockScoringEngine) that is the SAME object ShowdownResolver uses for
real games. This module lets PushFoldEnv / RiverEnv resolve showdowns
through that identical scoring_engine.evaluate(...) surface, so a hand
ruled a win/loss/chop during RL training is guaranteed consistent with
what the real Engine's ShowdownResolver would rule for the same cards.

Runout
------
Same rationale as PokerEvalEvaluator: poker_engine's HOLDEM evaluator
needs >= 5 known cards to produce a real score. Push-fold scenarios are
dealt with an EMPTY board (see push_fold_generator.py), so this
evaluator completes the missing board cards with a random runout
(sampled from an internal random.Random, excluding all known
hero/villain/board cards) before scoring.

Scoring engine selection
--------------------------
Defaults to poker_engine.scoring.scoring_engine.CppScoringEngine (the
real evaluator used by production/self-play games). If the poker_eval
C++ extension isn't importable, falls back to
poker_engine.scoring.mock_scoring_engine.MockScoringEngine — a
documented, deterministic stub within poker_engine itself (see that
module's own docstring) — so this evaluator stays usable without the
compiled extension, exactly like every other poker_eval shim in this
codebase. A caller can also pass an explicit `scoring_engine=` to
override either default.

Dependencies
------------
    poker_engine.scoring.scoring_engine.CppScoringEngine,
    poker_engine.scoring.mock_scoring_engine.MockScoringEngine.
    poker_eval / mock_poker_eval for ScoreType.HIGH / ShowdownType.HOLDEM
    — the shim below installs mock_poker_eval into sys.modules under
    the name "poker_eval" if the real extension is missing, matching
    every other entry point in this codebase (e.g.
    experiments/push_fold_experiment.py).
"""

from __future__ import annotations

import random
import sys
from typing import List, Optional, Sequence

try:
    import poker_eval  # noqa: F401  (prefer the real extension if present)
except ImportError:
    import mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

from poker_eval import ScoreType, ShowdownType


# ── Local card-mask helpers (mirrors poker_eval_evaluator.py's pattern) ─────

def _count_bits(mask: int) -> int:
    return bin(mask).count("1")


def _cards_to_mask(card_ids: Sequence[int]) -> int:
    mask = 0
    for cid in card_ids:
        mask |= 1 << cid
    return mask


def _default_scoring_engine():
    """
    Prefer the real Engine scoring path (CppScoringEngine, backed by
    poker_eval). Fall back to MockScoringEngine if the C++ extension
    isn't importable — poker_engine.scoring.scoring_engine imports
    poker_engine.poker_eval at MODULE level with no shim of its own, so
    that import is attempted lazily here and guarded.
    """
    try:
        from poker_engine.scoring.scoring_engine import CppScoringEngine

        return CppScoringEngine()
    except ImportError:
        from poker_engine.scoring.mock_scoring_engine import MockScoringEngine

        return MockScoringEngine()


# ── Evaluator ─────────────────────────────────────────────────────────────

class EngineScoringEvaluator:
    """
    Showdown evaluator backed by poker_engine's own scoring_engine
    surface (CppScoringEngine / MockScoringEngine), rather than calling
    poker_eval directly.

    Same public contract as envs.push_fold_env.MockEvaluator /
    envs.poker_eval_evaluator.PokerEvalEvaluator::

        hero_wins(hero_mask, villain_mask, board_mask) -> Optional[bool]

    ``True`` = hero wins, ``False`` = villain wins, ``None`` = chop.

    Parameters
    ----------
    scoring_engine : optional
        Any object duck-typing
        ``evaluate(player_masks, board_mask, score_type, showdown_type)
        -> List[ScoreResult]`` — i.e. CppScoringEngine or
        MockScoringEngine's own shared surface (see
        poker_engine/scoring/scoring_engine.py,
        poker_engine/scoring/mock_scoring_engine.py). Defaults to
        CppScoringEngine, falling back to MockScoringEngine if the
        poker_eval extension isn't available — see
        _default_scoring_engine() above.
    seed : int, optional
        Seeds the internal random.Random used for board runouts.
        Mirrors PokerEvalEvaluator's own seeding convention; omit for
        an unseeded (fresh-RNG-state) runout each call.

    Notes
    -----
    Only HIGH / HOLDEM is used — sufficient for push-fold and standard
    river/full hold'em curriculum stages, matching
    envs.poker_eval_evaluator.PokerEvalEvaluator's own scope note.
    """

    def __init__(self, scoring_engine=None, seed: Optional[int] = None) -> None:
        self._scoring_engine = scoring_engine or _default_scoring_engine()
        self._rng = random.Random(seed)

    def hero_wins(
        self,
        hero_mask: int,
        villain_mask: int,
        board_mask: int,
    ) -> Optional[bool]:
        full_board = self._complete_board(hero_mask, villain_mask, board_mask)

        results = self._scoring_engine.evaluate(
            [hero_mask, villain_mask],
            full_board,
            ScoreType.HIGH,
            ShowdownType.HOLDEM,
        )
        hero_score = results[0]
        villain_score = results[1]

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
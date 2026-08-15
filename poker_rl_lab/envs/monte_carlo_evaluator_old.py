"""
envs/monte_carlo_evaluator.py — Reusable batched Monte-Carlo outcome
sampler for stochastic showdown resolution.

Problem
--------
Existing evaluators (envs.poker_eval_evaluator.PokerEvalEvaluator,
envs.engine_scoring_evaluator.EngineScoringEvaluator, and
envs.generic_tree_env.GenericTreeEnv's own inline
_complete_board_for_showdown) resolve an incomplete board (e.g. the
empty preflop board in push-fold) by drawing exactly ONE random runout
and scoring it. That is an unbiased but high-variance single-sample
Monte Carlo estimate of the hand's true equity. The hero's action is
already fixed and known by the time this runout happens, so nothing
about the DECISION changes if more samples are used — only the
resulting reward's variance shrinks. Averaging K independent, disjoint
runouts is a strictly lower-variance estimate of the identical expected
reward (standard Monte-Carlo variance reduction — more samples, same
expectation), which reduces the noise PPO's advantage estimates have to
average out over training, at the cost of K (cheap, C-extension-backed)
extra evaluations per terminal step instead of one.

No information leakage, no change to game semantics
--------------------------------------------------------
All K runouts are drawn, evaluated, and averaged entirely INSIDE the
environment's reward computation, strictly after the hero's action has
already been chosen and applied — this module never sees or influences
an observation. StructuredObservation carries no board/runout
information beyond what was genuinely dealt; only the resulting scalar
reward changes from a 1-sample estimate to a K-sample average of the
SAME underlying random variable. The action space, legal-action
masking, and the game's true expected value under any given policy are
completely unchanged. This preserves every invariant PPO/self-play rely
on: reward is still an unbiased sample of a fixed (state, action)'s
outcome distribution, just averaged over more draws per call.

Reusability — this is NOT push-fold-specific
--------------------------------------------------
`equity()` draws `num_samples` independent random completions of a
board (as many cards as needed to reach 5), scores each with the SAME
scoring engine already used elsewhere (poker_engine.scoring's
CppScoringEngine / MockScoringEngine — the duck-typed
`evaluate(player_masks, board_mask, score_type, showdown_type) ->
List[ScoreResult]` surface), and returns win/chop/lose fractions.
`equity_over_villain_range()` reuses the identical K-sample-and-average
machinery but additionally samples the OPPONENT's hidden hand from a
supplied range each draw — the extension point named in the project
brief for "sampling opponent holdings on the river where appropriate".
Neither is wired into a live environment's decision loop; both are
strictly reward-computation-time utilities called after an action has
already been applied.

Dependencies
------------
    Standard library + whichever scoring_engine is passed in. poker_eval
    / mock_poker_eval only for ScoreType.HIGH / ShowdownType.HOLDEM
    constants, imported lazily with the same shim pattern used
    throughout this codebase (e.g. engine_scoring_evaluator.py).
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence


def _count_bits(mask: int) -> int:
    return bin(mask).count("1")


def _cards_to_mask(card_ids: Sequence[int]) -> int:
    mask = 0
    for c in card_ids:
        mask |= 1 << c
    return mask


def _default_scoring_engine():
    """Prefer the real Engine scoring path; fall back to the documented
    mock — identical pattern to engine_scoring_evaluator.py's own
    _default_scoring_engine()."""
    try:
        from poker_engine.scoring.scoring_engine import CppScoringEngine

        return CppScoringEngine()
    except ImportError:
        from poker_engine.scoring.mock_scoring_engine import MockScoringEngine

        return MockScoringEngine()


def _load_poker_eval_constants():
    try:
        import poker_eval as _pe
        print("Loading true poker eval")
    except ImportError:
        import mock_poker_eval as _pe
        print("Loading mock poker eval")

        sys.modules.setdefault("poker_eval", _pe)
    return _pe.ScoreType, _pe.ShowdownType


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EquityResult:
    """Aggregate outcome fractions across num_samples independent draws."""
    p_win: float
    p_chop: float
    p_lose: float
    num_samples: int

    def __post_init__(self) -> None:
        total = self.p_win + self.p_chop + self.p_lose
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"p_win+p_chop+p_lose must sum to ~1.0, got {total}")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MonteCarloEquityEvaluator:
    """
    Batched-Monte-Carlo showdown evaluator: averages hero equity over
    `num_samples` independent random board completions instead of one.

    Drop-in compatible with the existing single-sample evaluators via
    hero_wins() (envs.poker_eval_evaluator.PokerEvalEvaluator /
    envs.engine_scoring_evaluator.EngineScoringEvaluator's own
    contract), but callers wanting the actual variance-reduction
    benefit should call equity() directly and use the continuous
    win/chop/lose fractions to compute an EXPECTED reward, rather than
    collapsing back to one categorical outcome via hero_wins() (see
    that method's own docstring below).

    Parameters
    ----------
    scoring_engine : optional
        Duck-typed `evaluate(player_masks, board_mask, score_type,
        showdown_type) -> List[ScoreResult]`. Defaults to
        CppScoringEngine, falling back to MockScoringEngine.
    num_samples : int
        Default number of independent runouts averaged per equity()/
        hero_wins() call when the caller doesn't override it. Default
        30 — each runout is a cheap scoring-engine call (not a neural
        network forward pass), so this is a small, fixed per-terminal-
        step cost, negligible next to one PPO rollout tick's network
        call.
    seed : int, optional
        Seeds the internal random.Random for reproducible runouts.
    """

    def __init__(
        self,
        scoring_engine=None,
        num_samples: int = 30,
        seed: Optional[int] = None,
    ) -> None:
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}")
        self._scoring_engine = scoring_engine or _default_scoring_engine()
        self._default_num_samples = num_samples
        self._rng = random.Random(seed)
        self._ScoreType, self._ShowdownType = _load_poker_eval_constants()

    # ------------------------------------------------------------------
    # Core: equity via averaged board runouts
    # ------------------------------------------------------------------

    def equity(
        self,
        hero_mask: int,
        villain_mask: int,
        board_mask: int,
        num_samples: Optional[int] = None,
    ) -> EquityResult:
        """
        Average hero equity over `num_samples` independent random
        completions of `board_mask` up to 5 cards.

        Parameters
        ----------
        hero_mask, villain_mask : int
            Fully known hole-card bitmasks — both hands are known to
            the evaluator (mirrors every existing evaluator in this
            codebase). It is the BOARD that is stochastic for
            push-fold, not the opponent's revealed-to-the-evaluator
            hand.
        board_mask : int
            Already-dealt board cards, 0-5 cards. Completed to 5 cards
            independently for each sample.
        num_samples : int, optional
            Overrides the instance default for this call.

        Returns
        -------
        EquityResult
        """
        n = num_samples if num_samples is not None else self._default_num_samples
        needed = 5 - _count_bits(board_mask)

        if needed <= 0:
            # Board already complete: every sample is identical, one
            # evaluation suffices. Still report n so callers can treat
            # every equity() result uniformly (e.g. for logging).
            outcome = self._score_once(hero_mask, villain_mask, board_mask)
            wins = n if outcome > 0 else 0
            losses = n if outcome < 0 else 0
            chops = n if outcome == 0 else 0
            return EquityResult(wins / n, chops / n, losses / n, n)

        used_base = hero_mask | villain_mask | board_mask
        available = [c for c in range(52) if not (used_base >> c) & 1]
        if len(available) < needed:
            raise RuntimeError(
                f"Not enough cards left to complete the board: "
                f"need {needed}, have {len(available)}."
            )

        wins = chops = losses = 0
        for _ in range(n):
            runout = self._rng.sample(available, needed)
            full_board = board_mask | _cards_to_mask(runout)
            outcome = self._score_once(hero_mask, villain_mask, full_board)
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                chops += 1

        return EquityResult(wins / n, chops / n, losses / n, n)

    def _score_once(self, hero_mask: int, villain_mask: int, full_board: int) -> int:
        """+1 hero wins, -1 villain wins, 0 chop, for one fully-dealt board."""
        results = self._scoring_engine.evaluate(
            [hero_mask, villain_mask],
            full_board,
            self._ScoreType.HIGH,
            self._ShowdownType.HOLDEM,
        )
        hero_score, villain_score = results[0], results[1]
        if hero_score > villain_score:
            return 1
        if hero_score < villain_score:
            return -1
        return 0

    # ------------------------------------------------------------------
    # hero_wins() — single Optional[bool], for drop-in use wherever a
    # caller hasn't been updated to consume EquityResult directly.
    # ------------------------------------------------------------------

    def hero_wins(self, hero_mask: int, villain_mask: int, board_mask: int) -> Optional[bool]:
        """
        Collapses equity() back to a single categorical outcome (the
        majority result across num_samples draws), for compatibility
        with the existing hero_wins()-only call sites
        (envs.push_fold_env.PushFoldEnv, if constructed without the
        equity-aware reward path — see that module's updated CALL
        branch). Prefer equity() directly wherever the caller can use
        the continuous fractions: hero_wins() throws away the
        variance-reduction benefit by collapsing K samples back into
        one boolean/None.
        """
        result = self.equity(hero_mask, villain_mask, board_mask)
        if result.p_win > result.p_lose and result.p_win >= result.p_chop:
            return True
        if result.p_lose > result.p_win and result.p_lose >= result.p_chop:
            return False
        return None

    # ------------------------------------------------------------------
    # Extension point: sampling over an opponent HAND RANGE instead of
    # (or in addition to) board runouts. Not wired into any live
    # environment yet — provided so a future river/turn scenario with a
    # genuinely unknown opponent holding can reuse this exact
    # K-sample-and-average machinery rather than reinventing it.
    # ------------------------------------------------------------------

    def equity_over_villain_range(
        self,
        hero_mask: int,
        villain_range: Sequence[int],
        board_mask: int,
        num_samples: Optional[int] = None,
    ) -> EquityResult:
        """
        Average hero equity over `num_samples` independent draws of
        BOTH a villain hand (uniformly from `villain_range`) AND a
        board runout — for situations where the opponent's exact
        holding is not fixed (e.g. evaluating against a hand-strength-
        weighted range instead of one known hand).

        Parameters
        ----------
        villain_range : Sequence[int]
            Candidate villain hole-card bitmasks, sampled uniformly.
            Pass a pre-filtered/pre-weighted list (repeat entries to
            weight a hand more heavily) — this function has no opinion
            on range construction or card-removal weighting beyond
            excluding cards already used by hero/board on each draw.
        Other parameters as equity().

        Notes
        -----
        Strictly a reward-computation-time operation, exactly like
        equity() — the sampled villain hand is used only to average an
        already-fixed hero action's outcome and is never exposed to any
        observation the policy sees. Draws that can't be completed
        (not enough cards left after excluding hero/board/that
        particular sampled villain hand) are skipped rather than
        raising, so one unlucky villain-hand draw doesn't abort the
        whole average.
        """
        n = num_samples if num_samples is not None else self._default_num_samples
        if not villain_range:
            raise ValueError("villain_range must be non-empty")

        wins = chops = losses = 0
        for _ in range(n):
            villain_mask = self._rng.choice(villain_range)
            used = hero_mask | villain_mask | board_mask
            needed = 5 - _count_bits(board_mask)
            available = [c for c in range(52) if not (used >> c) & 1]
            if len(available) < needed:
                continue
            runout = self._rng.sample(available, needed) if needed > 0 else []
            full_board = board_mask | _cards_to_mask(runout)
            outcome = self._score_once(hero_mask, villain_mask, full_board)
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                chops += 1

        total = wins + chops + losses
        if total == 0:
            raise RuntimeError(
                "No realisable (villain, runout) draws in villain_range sampling."
            )
        return EquityResult(wins / total, chops / total, losses / total, total)
"""
trainers/showdown_equity.py — Expected showdown rewards from hand equity,
replacing Monte Carlo "replay the whole engine N times" resampling.

Problem this solves
-------------------
collect_graph_rollouts() used to reduce showdown variance by snapshotting
state+deck before the final decision and replaying board dealing +
showdown (through GraphEngine, deepcopy, ShowdownResolver) N-1 more
times per hand. Profiling showed ~65-75% of push_fold_duo wall time in
that replay, of which the C++ evaluator was only ~5%.

Approach
--------
For a HEADS-UP showdown there are only THREE possible outcome classes:
A wins, tie, B wins. So:

  1. holdem_matchup_counts() (C++, poker_eval) gives P(A wins), P(tie),
     P(B wins) for the two hands given the board known at decision time
     (exact when <= 2 board cards are missing, Monte Carlo otherwise).
  2. The payout for each outcome class is computed by the REAL
     ShowdownResolver (so side pots, odd-chip rules, payout_type etc.
     are exactly the engine's), fed a ForcedScoringEngine that returns
     fixed scores for the desired outcome instead of evaluating cards.
  3. Expected reward = sum over classes of P(class) * (payout - contribution).

Safety net
----------
Payouts are recomputed from the end-of-hand state (stacks minus the real
payouts, pot restored). If none of the three reconstructed payout vectors
equals the payout the REAL hand produced, the reconstruction doesn't
match the engine's behaviour for this game/state and rewards() returns
None, so the caller falls back to the slower replay. Likewise for
anything that isn't a 2-player HOLDEM / HIGH / single-point showdown.

Dependencies
------------
    poker_engine.poker_eval (needs holdem_matchup_counts — added in
    equity_mc.cpp), poker_engine.scoring.score_result.ScoreResult,
    poker_engine.showdown.showdown_resolver.ShowdownResolver.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:
    import poker_engine.poker_eval as _pe
except ImportError:  # pragma: no cover
    _pe = None

from poker_engine.scoring.score_result import ScoreResult
from poker_engine.showdown.showdown_resolver import ShowdownResolver

_WIN, _LOSE = 2, 1  # any distinct non-zero scores; resolver ignores score[0] == 0


def _as_int(x) -> int:
    return int(getattr(x, "value", x))


class _ForcedScoringEngine:
    """Scoring-engine stand-in returning pre-set scores (one per active
    player, in the order ShowdownResolver passes player_masks)."""

    __slots__ = ("values",)

    def __init__(self) -> None:
        self.values: Tuple[int, ...] = ()

    def evaluate(self, player_masks, board_mask, score_type, showdown_type):
        return [ScoreResult((v,), best_hand_mask=0) for v in self.values]


class HoldemEquityRewards:
    """
    Computes low-variance expected per-seat rewards for a just-finished
    heads-up hold'em showdown.

    Parameters
    ----------
    rules : GameRules
        env.rules of the GraphScenarioEnv (needs .points, .showdown_type).
    n_samples : int
        Monte Carlo runouts when >= 3 board cards are still missing at
        decision time (exact enumeration is used otherwise). ~18 us each.
    seed : int
        Base seed for the C++ sampler (varied per call).

    Attributes
    ----------
    supported : bool
        False if this rules/engine combination can't use the equity path;
        `reason` then says why. rewards() always returns None in that case.
    used, fallbacks : int
        Diagnostics: how many hands were handled here vs. returned None.
    """

    def __init__(self, rules, n_samples: int = 100, seed: int = 0) -> None:
        self.n_samples = n_samples
        self._base_seed = seed
        self._calls = 0
        self.used = 0
        self.fallbacks = 0

        self.supported, self.reason = self._check(rules)
        self._forced = _ForcedScoringEngine()
        self._resolver = ShowdownResolver(self._forced, rules) if self.supported else None
        self._board_nodes = (
            sorted({n for ns in rules.points[0].node_sets for n in ns}) if self.supported else []
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _check(rules) -> Tuple[bool, str]:
        if _pe is None or not hasattr(_pe, "holdem_matchup_counts"):
            return False, "poker_eval.holdem_matchup_counts not available (rebuild the extension)"
        if len(rules.points) != 1:
            return False, f"{len(rules.points)} scoring points (equity path supports exactly 1)"
        point = rules.points[0]
        if _as_int(point.score_type) != _as_int(_pe.ScoreType.HIGH):
            return False, "point is not a HIGH point"
        st = point.showdown_type_override if point.showdown_type_override is not None else rules.showdown_type
        if _as_int(st) != _as_int(_pe.ShowdownType.HOLDEM):
            return False, "showdown type is not HOLDEM"
        return True, ""

    def board_mask(self, state) -> int:
        """Bitmask of this point's board cards dealt so far. Call this
        BEFORE the final decision is submitted (cards dealt after it are
        what the equity calculation re-samples)."""
        mask = 0
        node_cards = state.node_cards
        for n in self._board_nodes:
            if n < len(node_cards) and node_cards[n] is not None:
                mask |= 1 << node_cards[n]
        return mask

    # ------------------------------------------------------------------

    def rewards(self, env, board_mask_pre: int) -> Optional[Dict[int, float]]:
        """
        Expected {seat: chip change} for a hand that just ended in a
        showdown, or None if the equity path doesn't apply (caller should
        fall back to replay-based resampling).

        `env` is the finished GraphScenarioEnv; its state/engine are
        restored exactly as found before returning.
        """
        if not self.supported:
            return None
        out = self._rewards(env, board_mask_pre)
        if out is None:
            self.fallbacks += 1
        else:
            self.used += 1
        return out

    def _rewards(self, env, board_mask_pre: int) -> Optional[Dict[int, float]]:
        engine = env.engine
        showdown = engine.last_showdown
        if showdown is None:
            return None

        state = engine.state
        players = state.players
        n = len(players)

        active = [i for i, p in enumerate(players) if not p.has_folded]
        if len(active) != 2:
            return None
        a, b = active

        real_payouts = tuple(int(showdown.payouts.get(i, 0)) for i in range(n))
        final_stacks = [p.stack for p in players]
        final_pot = state.pot
        baseline = [final_stacks[i] - real_payouts[i] for i in range(n)]
        pot_before = sum(real_payouts)

        self._calls += 1
        try:
            counts = _pe.holdem_matchup_counts(
                players[a].hand_mask,
                players[b].hand_mask,
                board_mask_pre,
                self.n_samples,
                self._base_seed + self._calls * 0x9E3779B1,
            )
        except (ValueError, RuntimeError):
            return None
        total = float(sum(counts))
        if total <= 0:
            return None
        probs = [c / total for c in counts]  # [A wins, tie, B wins]

        class_payouts = []
        try:
            state.pot = pot_before
            for values in ((_WIN, _LOSE), (_LOSE, _LOSE), (_LOSE, _WIN)):
                for i, p in enumerate(players):
                    p.stack = baseline[i]
                self._forced.values = values
                result = self._resolver.resolve(state)
                class_payouts.append(tuple(int(result.payouts.get(i, 0)) for i in range(n)))
        finally:
            for i, p in enumerate(players):
                p.stack = final_stacks[i]
            state.pot = final_pot

        # Safety net: the real hand's payout must be one of the three
        # reconstructed outcomes, or our reconstruction is wrong here.
        if real_payouts not in class_payouts:
            return None

        start = env._starting_stacks
        rewards: Dict[int, float] = {}
        for i in range(n):
            rewards[i] = sum(
                probs[k] * (baseline[i] + class_payouts[k][i] - start[i]) for k in range(3)
            )
        return rewards
"""
test_equity_mc.py — Validate poker_eval.holdem_matchup_counts (fast bitmask
evaluator) against an INDEPENDENT pure-Python brute-force reference
(best of all 21 five-card combinations), and separately report where the
engine's own poker_eval.evaluate_hands disagrees with that reference.

    python test_equity_mc.py            # 30k hands per mode
    python test_equity_mc.py 100000

PASS/FAIL is decided ONLY by fast-evaluator vs brute force.
Engine disagreements are reported as information. The current
evaluate_hands implementation ranks a wheel (A-2-3-4-5) below a
6-high straight/straight-flush, matching brute force and the fast
evaluator.

Deal modes (for coverage of rare hand classes):
  uniform : any 9 distinct cards.
  ranks7  : cards from 7 random ranks -> trips/quads/full houses/straights.
  suits2  : cards from 2 random suits -> flushes and straight flushes.
"""

import itertools
import random
import sys
import time
from collections import Counter

import poker_engine.poker_eval as pe


# ---------------------------------------------------------------- reference
def _rank(c): return c % 13
def _suit(c): return c // 13


def score5(cards):
    rs = sorted((_rank(c) for c in cards), reverse=True)
    flush = len({_suit(c) for c in cards}) == 1
    uniq = sorted(set(rs), reverse=True)
    straight_top = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_top = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_top = 3                      # wheel: five-high
    cnt = Counter(rs)
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = tuple(g[1] for g in groups)
    ordered = tuple(g[0] for g in groups)
    if straight_top is not None and flush: return (8, straight_top)
    if shape == (4, 1): return (7,) + ordered
    if shape == (3, 2): return (6,) + ordered
    if flush: return (5,) + tuple(rs)
    if straight_top is not None: return (4, straight_top)
    if shape == (3, 1, 1): return (3,) + ordered
    if shape == (2, 2, 1): return (2,) + ordered
    if shape == (2, 1, 1, 1): return (1,) + ordered
    return (0,) + tuple(rs)


def best7(cards):
    return max(score5(c) for c in itertools.combinations(cards, 5))


# ------------------------------------------------------------------ helpers
def mask(cards):
    m = 0
    for c in cards:
        m |= 1 << c
    return m


def verdict(sa, sb):
    return 0 if sa > sb else (2 if sa < sb else 1)    # index into [a_wins, ties, b_wins]


def engine_verdict(a, b, board):
    r = pe.evaluate_hands([a, b], board, pe.ScoreType.HIGH, pe.ShowdownType.HOLDEM)
    return verdict(r[0][0], r[1][0])


def fast_verdict(a, b, board):
    return pe.holdem_matchup_counts(a, b, board, 1, 1).index(1)


def deck_for(mode, rng):
    if mode == "uniform":
        return list(range(52))
    if mode == "ranks7":
        ranks = set(rng.sample(range(13), 7))
        return [c for c in range(52) if c % 13 in ranks]
    suits = set(rng.sample(range(4), 2))
    return [c for c in range(52) if c // 13 in suits]


# --------------------------------------------------------------------- main
def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30_000
    rng = random.Random(2024)
    fast_bad_total = 0
    engine_wheel_mismatch_total = 0
    engine_other_total = 0
    engine_other_examples = []

    for mode in ("uniform", "ranks7", "suits2"):
        fast_bad = engine_wheel_mismatch = engine_other = 0
        for _ in range(n):
            cards = rng.sample(deck_for(mode, rng), 9)
            hole_a, hole_b, board_c = cards[0:2], cards[2:4], cards[4:9]
            a, b, bd = mask(hole_a), mask(hole_b), mask(board_c)

            sa, sb = best7(hole_a + board_c), best7(hole_b + board_c)
            ref = verdict(sa, sb)

            if fast_verdict(a, b, bd) != ref:
                fast_bad += 1
                if fast_bad <= 5:
                    print(f"   FAST MISMATCH {hole_a} {hole_b} {board_c} ref={ref}")

            if engine_verdict(a, b, bd) != ref:
                wheel_mismatch = (
                    sa[0] == sb[0] and sa[0] in (4, 8)
                    and sorted((sa[1], sb[1])) == [3, 4]
                )
                if wheel_mismatch:
                    engine_wheel_mismatch += 1
                else:
                    engine_other += 1
                    if len(engine_other_examples) < 5:
                        engine_other_examples.append((hole_a, hole_b, board_c, ref))

        print(f"{mode:8s}: {n} hands | fast vs brute-force mismatches = {fast_bad} | "
              f"engine vs brute-force: wheel-vs-6high mismatches = {engine_wheel_mismatch}, "
              f"other = {engine_other}")
        fast_bad_total += fast_bad
        engine_wheel_mismatch_total += engine_wheel_mismatch
        engine_other_total += engine_other

    print("\nFAST EVALUATOR:", "PASS" if fast_bad_total == 0 else f"FAIL ({fast_bad_total} mismatches)")
    print(f"ENGINE evaluate_hands: {engine_wheel_mismatch_total} wheel-vs-6-high-straight mismatches, "
          f"{engine_other_total} other disagreements")
    for ex in engine_other_examples:
        print("   engine-other example:", ex)

    # Equity sanity: AcAd vs KcKd preflop
    r = pe.holdem_matchup_counts(mask([12, 25]), mask([11, 24]), 0, 200000, 1)
    t = float(sum(r))
    print(f"\nAA vs KK preflop (200k samples): A {r[0]/t:.3f}  tie {r[1]/t:.3f}  B {r[2]/t:.3f} "
          f"(expect ~0.82 / 0.005 / 0.17)")

    # Speed
    a, b = mask([12, 25]), mask([11, 24])
    for samples in (100, 1000):
        reps = max(20, 20000 // samples)
        t0 = time.perf_counter()
        for i in range(reps):
            pe.holdem_matchup_counts(a, b, 0, samples, i)
        dt = (time.perf_counter() - t0) / reps
        print(f"preflop, {samples:5d} samples: {1000 * dt:8.3f} ms/call  ({1e9 * dt / samples:7.1f} ns/sample)")
    flop = mask([5, 18, 31])
    t0 = time.perf_counter()
    for i in range(200):
        pe.holdem_matchup_counts(a, b, flop, 1, i)
    print(f"flop exact (990 runouts): {1000 * (time.perf_counter() - t0) / 200:8.3f} ms/call")


if __name__ == "__main__":
    main()
"""
test_evaluators.py — Differential test of poker_eval.evaluate_hands against
INDEPENDENT pure-Python brute-force references, for every score/showdown
combination the engine implements:

    HIGH    x  HOLDEM, OMAHA, MAKE5, DRAW
    LOW_A5  x  HOLDEM, OMAHA, DRAW        (ace-low, pairs disqualify)
    LOW_27  x  HOLDEM, OMAHA, DRAW        (ace-high, straights/flushes count)

Why: RL agents find and exploit evaluator bugs; nothing in training flags
them. This compares only WIN / TIE / LOSE verdicts between two hands, which
is what actually determines payouts.

    python test_evaluators.py             # 3000 hands per mode per variant
    python test_evaluators.py 10000
    python test_evaluators.py 3000 HIGH   # only variants whose name contains HIGH
    python test_evaluators.py 3000 OMAHA LOW_27

Deal modes (to force rare hand classes):
  uniform  : any distinct cards
  ranks7   : cards from 7 random ranks   (trips/quads/full houses/straights)
  suits2   : cards from 2 random suits   (flushes / straight flushes)
  lowcards : ranks A..8 only             (many valid A-5 lows, wheels)

Reference semantics
-------------------
  HIGH   : best 5-card high hand (wheel A-2-3-4-5 is the LOWEST straight).
  LOW_A5 : ace=1, no straights/flushes, any pair disqualifies a 5-card
           candidate; lower (sorted-descending) rank tuple is better; no
           qualifying candidate = no low (worst).
  LOW_27 : ace always HIGH, A-2-3-4-5 is NOT a straight, straights and
           flushes count against you; the LOWEST high-hand ranking wins.
Engine conventions: HIGH higher score wins; LOW_*: lower non-zero score
wins and 0 means "no low".
"""

import itertools
import random
import sys
from collections import Counter

import poker_engine.poker_eval as pe

RANK_CH = "23456789TJQKA"
SUIT_CH = "cdhs"


def cstr(c):
    return RANK_CH[c % 13] + SUIT_CH[c // 13]


def hstr(cs):
    return " ".join(cstr(c) for c in cs)


def mask(cards):
    m = 0
    for c in cards:
        m |= 1 << c
    return m


# ---------------------------------------------------------------------------
# Reference 5-card scorers
# ---------------------------------------------------------------------------

def high5(cards, wheel_is_straight=True):
    """Comparable tuple; larger = better high hand."""
    rs = sorted((c % 13 for c in cards), reverse=True)
    flush = len({c // 13 for c in cards}) == 1
    uniq = sorted(set(rs), reverse=True)
    top = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            top = uniq[0]
        elif wheel_is_straight and uniq == [12, 3, 2, 1, 0]:
            top = 3
    cnt = Counter(rs)
    groups = sorted(cnt.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = tuple(g[1] for g in groups)
    ordered = tuple(g[0] for g in groups)
    if top is not None and flush: return (8, top)
    if shape == (4, 1): return (7,) + ordered
    if shape == (3, 2): return (6,) + ordered
    if flush: return (5,) + tuple(rs)
    if top is not None: return (4, top)
    if shape == (3, 1, 1): return (3,) + ordered
    if shape == (2, 2, 1): return (2,) + ordered
    if shape == (2, 1, 1, 1): return (1,) + ordered
    return (0,) + tuple(rs)


def low_a5_key(cards):
    """Sorted-descending rank tuple (ace=1), or None if paired. Smaller = better."""
    rs = sorted(((1 if c % 13 == 12 else c % 13 + 2) for c in cards), reverse=True)
    if len(set(rs)) < 5:
        return None
    return tuple(rs)


# ---------------------------------------------------------------------------
# Candidate 5-card hands per showdown type
# ---------------------------------------------------------------------------

def candidates(sd, hole, board):
    if sd == "HOLDEM":
        allc = hole + board
        return list(itertools.combinations(allc, 5)) if len(allc) >= 5 else []
    if sd == "OMAHA":
        if len(hole) < 2 or len(board) < 3:
            return []
        return [h + b for h in itertools.combinations(hole, 2)
                for b in itertools.combinations(board, 3)]
    if sd == "MAKE5":
        need = 5 - len(board)
        if need < 0 or need > len(hole):
            return []
        return [h + tuple(board) for h in itertools.combinations(hole, need)]
    if sd == "DRAW":
        return list(itertools.combinations(hole, 5)) if len(hole) >= 5 else []
    raise ValueError(sd)


def ref_score(st, sd, hole, board):
    """Reference best score for one player, or None if no valid hand."""
    cands = candidates(sd, hole, board)
    if not cands:
        return None
    if st == "HIGH":
        return max(high5(c) for c in cands)
    if st == "LOW_A5":
        keys = [k for k in (low_a5_key(c) for c in cands) if k is not None]
        return min(keys) if keys else None
    if st == "LOW_27":
        return min(high5(c, wheel_is_straight=False) for c in cands)
    raise ValueError(st)


def ref_verdict(st, ra, rb):
    """0 = A wins, 1 = tie, 2 = B wins."""
    if st == "HIGH":
        return 0 if ra > rb else (2 if ra < rb else 1)
    # lower is better; None = no low = worst
    if ra is None and rb is None:
        return 1
    if ra is None:
        return 2
    if rb is None:
        return 0
    return 0 if ra < rb else (2 if ra > rb else 1)


def engine_verdict(st, sa, sb):
    if st == "HIGH":
        return 0 if sa > sb else (2 if sa < sb else 1)
    # LOW_*: 0 = no low (worst); lower non-zero is better
    ka = float("inf") if sa <= 0 else sa
    kb = float("inf") if sb <= 0 else sb
    return 0 if ka < kb else (2 if ka > kb else 1)


# ---------------------------------------------------------------------------
# Deal generation
# ---------------------------------------------------------------------------

def deck_for(mode, rng):
    if mode == "uniform":
        return list(range(52))
    if mode == "ranks7":
        ranks = set(rng.sample(range(13), 7))
        return [c for c in range(52) if c % 13 in ranks]
    if mode == "suits2":
        suits = set(rng.sample(range(4), 2))
        return [c for c in range(52) if c // 13 in suits]
    if mode == "lowcards":
        ranks = {12, 0, 1, 2, 3, 4, 5, 6}   # A,2,3,4,5,6,7,8
        return [c for c in range(52) if c % 13 in ranks]
    raise ValueError(mode)


MODES = ("uniform", "ranks7", "suits2", "lowcards")

# (score_type, showdown_type, hole_cards, board_card_choices)
VARIANTS = [
    ("HIGH",   "HOLDEM", 2, (5,)),
    ("HIGH",   "OMAHA",  4, (5,)),
    ("HIGH",   "MAKE5",  4, (1, 2, 3, 4)),
    ("HIGH",   "DRAW",   5, (0,)),
    ("LOW_A5", "HOLDEM", 2, (5,)),
    ("LOW_A5", "OMAHA",  4, (5,)),
    ("LOW_A5", "DRAW",   5, (0,)),
    ("LOW_27", "HOLDEM", 2, (5,)),
    ("LOW_27", "OMAHA",  4, (5,)),
    ("LOW_27", "DRAW",   5, (0,)),
]


def run_variant(st, sd, hole_n, board_choices, n, rng):
    st_enum = getattr(pe.ScoreType, st)
    sd_enum = getattr(pe.ShowdownType, sd)
    per_mode = {}
    examples = []

    for mode in MODES:
        bad = 0
        for _ in range(n):
            deck = deck_for(mode, rng)
            board_n = rng.choice(board_choices)
            cards = rng.sample(deck, 2 * hole_n + board_n)
            hole_a = cards[:hole_n]
            hole_b = cards[hole_n:2 * hole_n]
            board = cards[2 * hole_n:]

            ra = ref_score(st, sd, hole_a, board)
            rb = ref_score(st, sd, hole_b, board)
            want = ref_verdict(st, ra, rb)

            raw = pe.evaluate_hands([mask(hole_a), mask(hole_b)], mask(board), st_enum, sd_enum)
            got = engine_verdict(st, raw[0][0], raw[1][0])

            if want != got:
                bad += 1
                if len(examples) < 4:
                    names = ("A wins", "tie", "B wins")
                    examples.append(
                        f"[{mode}] A={hstr(hole_a)} B={hstr(hole_b)} board={hstr(board) or '-'} "
                        f"-> reference: {names[want]}, engine: {names[got]}  (ref scores {ra} vs {rb})"
                    )
        per_mode[mode] = bad
    return per_mode, examples


def main():
    args = sys.argv[1:]
    n = int(args[0]) if args and args[0].isdigit() else 3000
    filters = [a for a in args if not a.isdigit()]
    rng = random.Random(7)

    print(f"{n} hands per mode x {len(MODES)} modes per variant\n")
    overall_bad = 0
    summary = []
    for st, sd, hole_n, board_choices in VARIANTS:
        name = f"{st}/{sd}"
        if filters and not any(f.upper() in name for f in filters):
            continue
        per_mode, examples = run_variant(st, sd, hole_n, board_choices, n, rng)
        bad = sum(per_mode.values())
        overall_bad += bad
        summary.append((name, bad))
        modes = "  ".join(f"{m}={per_mode[m]}" for m in MODES)
        print(f"{name:16s} {'PASS' if bad == 0 else 'FAIL':4s}  mismatches: {bad:5d}   ({modes})")
        for ex in examples:
            print("     ", ex)

    print("\n" + "=" * 60)
    print("ALL EVALUATORS PASS" if overall_bad == 0 else f"FAILURES: {overall_bad} mismatching hands")
    print("=" * 60)
    return 0 if overall_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
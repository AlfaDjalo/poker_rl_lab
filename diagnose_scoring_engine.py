"""
diagnose_scoring_engine.py — Run this BEFORE re-training. Confirms
whether showdowns are being resolved by the real poker_eval-backed
CppScoringEngine or have silently fallen back to MockScoringEngine
(which ignores hole cards entirely — see GenericTreeEnv's module
docstring / run_generic_tree_smoke_test.py's own comments on this).

Usage
-----
    PYTHONPATH=. python3 diagnose_scoring_engine.py
"""
import sys

# 1) Is the real extension even importable?
try:
    import poker_engine.poker_eval as poker_eval # noqa: F401
    print("poker_eval C++ extension: IMPORTABLE (good)")
except ImportError as e:
    print("poker_eval C++ extension: NOT IMPORTABLE")
    print(f"  ImportError: {e}")
    print("  -> Every _default_scoring_engine() call in this codebase will")
    print("     silently fall back to MockScoringEngine. This is the bug.")

# 2) Directly check what GenericTreeEnv's own default resolves to.
from poker_rl_lab.envs.generic_tree_env import _default_scoring_engine
engine = _default_scoring_engine()
print(f"\nGenericTreeEnv._default_scoring_engine() resolved to: {type(engine).__name__}")
if type(engine).__name__ == "MockScoringEngine":
    print("  *** THIS IS MOCKSCORINGENGINE. Showdowns ignore hole cards. ***")

# 3) Concrete check: does AA beat 72o more often than not, over many
#    independent random boards, using whatever engine ended up wired in?
import random
from poker_rl_lab.representation.card_utils import str_to_card_id

def mask(cards):
    m = 0
    for c in cards:
        m |= 1 << str_to_card_id(c)
    return m

aa = mask(["As", "Ah"])
weak = mask(["7c", "2d"])

wins_for_aa = 0
trials = 500
rng = random.Random(0)
deck = list(range(52))

for _ in range(trials):
    used = aa | weak
    available = [c for c in deck if not (used >> c) & 1]
    board = rng.sample(available, 5)
    board_mask = 0
    for c in board:
        board_mask |= 1 << c

    from poker_engine.poker_eval import ScoreType, ShowdownType
    scores = engine.evaluate([aa, weak], board_mask, ScoreType.HIGH, ShowdownType.HOLDEM)
    if scores[0] > scores[1]:
        wins_for_aa += 1

win_rate = wins_for_aa / trials
print(f"\nAA vs 72o over {trials} random boards: AA won {win_rate:.1%} of showdowns")
print("Expected with a REAL evaluator: ~88% (AA is a big favourite preflop).")
print("If this comes back ~50% (or exactly determined by seat order, not")
print("cards), the scoring engine is not discriminating hand strength and")
print("training will not converge to a sensible equilibrium no matter how")
print("long you run it.")
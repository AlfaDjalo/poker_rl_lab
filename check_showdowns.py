import sys
try:
    import poker_engine.poker_eval
except ImportError:
    import poker_rl_lab.mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

from poker_rl_lab.envs.engine_scoring_evaluator import EngineScoringEvaluator
ev = EngineScoringEvaluator(seed=0)
print(type(ev._scoring_engine))   # CppScoringEngine or MockScoringEngine?

# AA vs 72o, run a few times — should win ~85% for AA if real evaluator
from poker_rl_lab.representation.card_utils import str_to_card_id
aa = 1 << str_to_card_id("Ah") | 1 << str_to_card_id("As")
trash = 1 << str_to_card_id("7c") | 1 << str_to_card_id("2d")
wins = sum(ev.hero_wins(aa, trash, 0) is True for _ in range(200))
print(wins, "/200")
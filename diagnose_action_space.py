"""
diagnose_action_space.py — confirms whether SB's push-fold decision
really offers FOLD/CALL/RAISE (3 options) instead of the intended
FOLD/RAISE (2). No training needed.
"""

from poker_engine.cards.deck import Deck
from poker_engine.graph.graph_engine import GraphEngine
from poker_engine.graph.graph_state import GraphGameState
from poker_engine.state.player_state import PlayerState

from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario
from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv

config = load_graph_scenario("push_fold_duo")
env = GraphScenarioEnv(config)

stack = 20  # 10bb at big_blind_size=2
players = [
    PlayerState(stack=stack, hand_mask=0, current_bet=0, total_contribution=0,
                has_folded=False, is_all_in=False, sitting_out=False)
    for _ in range(2)
]
state = GraphGameState(players, betting_type="no_limit")
state.node_cards = [None] * env.game_def.node_count
state.dealer_position = 1  # seat 0 = SB, first to act

deck = Deck()
deck.shuffle()

engine = GraphEngine(env.graph, state, deck, showdown_resolver=env._showdown_resolver)
env._engine = engine
env._history = []
engine.start()

req = engine.pending_request
print(f"acting seat: {req.player_index}")
print(f"raw engine options (min_amount, max_amount, action_type):")
for opt in req.options:
    print(f"  {opt.action_type}: min={opt.min_amount} max={opt.max_amount}")

obs = env._build_decision_obs(req)
print(f"\nresolved GraphActionOptions the POLICY actually sees:")
for o in obs.options:
    print(f"  label={o.label!r} engine_action_type={o.engine_action_type} amount={o.amount}")
print(f"\ntotal options exposed to the policy: {len(obs.options)}")
print("Expected for a true push-fold format: 2 (FOLD, RAISE/ALL_IN).")
print("If this prints 3 (FOLD, CALL, RAISE), SB is being trained on a "
      "fold/limp/shove decision instead of fold/shove.")
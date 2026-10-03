"""
diagnose_shove_call_variance.py — no neural network, no training.

Forces SB to ALWAYS shove and BB to ALWAYS call, for many hands (random
hole cards each time, hand strength ignored by both scripted
"policies"), and reports the reward distribution TWICE per hand:

  (a) SINGLE-SAMPLE — one real board runout, exactly what you'd get
      with num_showdown_samples=1.
  (b) AVERAGED — using the SAME _resample_showdown_rewards() function
      graph_training_loop.py's collect_graph_rollouts() actually calls
      during training (num_showdown_samples=30 by default), applied to
      the identical hand (same hole cards, same committed actions).

What this checks
------------------
1. EV sanity: mean SB reward (either version) should be roughly
   break-even, not wildly/consistently negative — SB is shoving 100%
   of hands including terrible ones, so BB profits a little on average
   by always calling back, but it should not look like shoving is
   structurally doomed regardless of hand strength.
2. Variance reduction: (b)'s std should be MEANINGFULLY smaller than
   (a)'s — that's the whole point of averaging 30 independent runouts.
   If they're statistically indistinguishable, the averaging path
   graph_training_loop.py relies on isn't doing its job for some
   reason (even though this script calls the identical function),
   which would point at a bug in _resample_showdown_rewards() itself
   or in how/when collect_graph_rollouts() decides to invoke it.

Usage
-----
    python diagnose_shove_call_variance.py --hands 300 --stack-bb 10
"""

from __future__ import annotations

import argparse
import copy
import random
import statistics

from poker_engine.actions.action import Action
from poker_engine.actions.action_type import ActionType as EngineActionType
from poker_engine.cards.deck import Deck
from poker_engine.graph.graph_engine import GraphEngine
from poker_engine.graph.graph_state import GraphGameState
from poker_engine.graph.core_types import DecisionDomain, DecisionResponse
from poker_engine.state.player_state import PlayerState

from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv
from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario
from poker_rl_lab.trainers.graph_training_loop import _resample_showdown_rewards

_SB_SEAT = 0
_BB_SEAT = 1
_DEALER_POSITION = 1


def play_one_hand(env: GraphScenarioEnv, stack: int, num_showdown_samples: int, rng: random.Random):
    players = [
        PlayerState(stack=stack, hand_mask=0, current_bet=0, total_contribution=0,
                    has_folded=False, is_all_in=False, sitting_out=False)
        for _ in range(2)
    ]
    state = GraphGameState(players, betting_type="no_limit")
    state.node_cards = [None] * env.game_def.node_count
    state.dealer_position = _DEALER_POSITION

    deck = Deck()
    deck.shuffle()

    engine = GraphEngine(env.graph, state, deck, showdown_resolver=env._showdown_resolver)
    env._engine = engine
    env._history = []
    env._starting_stacks = (stack, stack)
    env._rng = rng
    engine.start()

    req = engine.pending_request
    if req.player_index != _SB_SEAT:
        raise RuntimeError(f"expected SB to act first, got seat {req.player_index}")

    obs = env._build_decision_obs(req)
    shove = next(o for o in obs.options if o.engine_action_type == EngineActionType.RAISE)
    env._submit(DecisionResponse(
        node_id=req.node_id, domain=DecisionDomain.BETTING, player_index=_SB_SEAT,
        value=Action(type=EngineActionType.RAISE, amount=shove.amount),
    ))

    req2 = engine.pending_request
    if req2 is None or req2.player_index != _BB_SEAT:
        raise RuntimeError(f"expected BB to face the shove, got {req2!r}")
    obs2 = env._build_decision_obs(req2)
    call = next(o for o in obs2.options if o.engine_action_type == EngineActionType.CALL)

    # Snapshot immediately before BB's call is applied -- same point
    # collect_graph_rollouts() snapshots at.
    pre_request = req2
    pre_response = DecisionResponse(
        node_id=req2.node_id, domain=DecisionDomain.BETTING, player_index=_BB_SEAT,
        value=Action(type=EngineActionType.CALL, amount=call.amount),
    )
    snap_state = copy.deepcopy(engine.state)
    snap_deck = copy.deepcopy(engine.deck)

    env._submit(pre_response)
    if not engine.is_complete:
        raise RuntimeError("hand did not complete after SB shove + BB call")

    final = [p.stack for p in engine.state.players]
    single_sample = {i: float(final[i] - stack) for i in range(2)}

    averaged = _resample_showdown_rewards(
        env, snap_state, snap_deck, pre_request, pre_response,
        first_sample_rewards=single_sample, num_samples=num_showdown_samples,
    )
    return single_sample, averaged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-name", type=str, default="push_fold_duo")
    parser.add_argument("--scenario-path", type=str, default=None)
    parser.add_argument("--hands", type=int, default=300)
    parser.add_argument("--stack-bb", type=float, default=10.0)
    parser.add_argument("--num-showdown-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_graph_scenario(args.scenario_name, args.scenario_path)
    env = GraphScenarioEnv(config)
    big_blind = env.config.stack.big_blind_size
    stack = max(big_blind, int(round(args.stack_bb * big_blind)))
    rng = random.Random(args.seed)

    single_rewards, avg_rewards = [], []
    for _ in range(args.hands):
        single, averaged = play_one_hand(env, stack, args.num_showdown_samples, rng)
        single_rewards.append(single[_SB_SEAT])
        avg_rewards.append(averaged[_SB_SEAT])

    def report(name: str, values: list) -> None:
        mean = statistics.mean(values)
        std = statistics.pstdev(values)
        distinct = sorted(set(round(v, 4) for v in values))
        print(f"--- {name} ---")
        print(f"mean: {mean:+.4f}   std: {std:.4f}   min/max: {min(values):+.2f}/{max(values):+.2f}")
        print(f"distinct values: {len(distinct)} of {len(values)} hands  (first 10: {distinct[:10]})")
        print()

    print(f"hands: {args.hands}  stack_bb: {args.stack_bb}  stack_chips: {stack}  "
          f"num_showdown_samples: {args.num_showdown_samples}\n")
    report("SINGLE-SAMPLE (num_showdown_samples=1 equivalent)", single_rewards)
    report(f"AVERAGED (num_showdown_samples={args.num_showdown_samples}, via the real training code path)", avg_rewards)

    single_std = statistics.pstdev(single_rewards)
    avg_std = statistics.pstdev(avg_rewards)
    if single_std > 0 and avg_std / single_std > 0.7:
        print(
            f"WARNING: averaged std ({avg_std:.4f}) is not much smaller than "
            f"single-sample std ({single_std:.4f}) -- variance reduction does "
            f"NOT look like it's meaningfully active. Expected the averaged "
            f"std to shrink roughly by sqrt({args.num_showdown_samples}) "
            f"(~{args.num_showdown_samples ** 0.5:.1f}x) relative to single-sample."
        )
    else:
        print(
            f"Averaged std IS meaningfully smaller than single-sample std "
            f"({avg_std:.4f} vs {single_std:.4f}) -- variance reduction is "
            f"working as intended. EV sanity: check the 'mean' lines above "
            f"aren't wildly/consistently negative for SB."
        )


if __name__ == "__main__":
    main()
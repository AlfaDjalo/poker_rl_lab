"""
diagnose_sb_policy_drift.py — runs a handful of REAL training iterations
(same collect_graph_rollouts / _ppo_update_seat calls
GraphTrainingLoop.train() uses) and prints, before and after each one:

  - SB's P(shove) for a few FIXED, known hands (AA, 72o) -- lets you
    watch whether/how fast probability mass drains from RAISE.
  - Raw stats on SB's rollout buffer that iteration: mean/std of
    reward, value estimate, and (post-normalization) advantage --
    if advantages are behaving pathologically (e.g. wrong sign on
    average for RAISE transitions specifically) this is where it'll
    show up directly, not just as its downstream effect on P(shove).

Usage
-----
    python diagnose_sb_policy_drift.py --iterations 5 --episodes-per-iteration 512
"""

from __future__ import annotations

import argparse
import statistics

import torch

from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv
from poker_rl_lab.representation.card_utils import str_to_card_id, card_ids_to_mask
from poker_rl_lab.representation.graph_observation_adapter import GraphObservationAdapter
from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario
from poker_rl_lab.trainers.graph_training_loop import (
    PolicyVariant,
    build_dual_seat_actor_critic_for_graph,
    collect_graph_rollouts,
    _ppo_update_seat,
)

from poker_engine.actions.action_type import ActionType as EngineActionType
from poker_engine.cards.deck import Deck
from poker_engine.graph.graph_engine import GraphEngine
from poker_engine.graph.graph_state import GraphGameState
from poker_engine.state.player_state import PlayerState

_SB_SEAT = 0
_DEALER_POSITION = 1

_FIXED_HANDS = {
    "AA": [str_to_card_id("As"), str_to_card_id("Ad")],
    "72o": [str_to_card_id("7c"), str_to_card_id("2d")],
}


def sb_shove_prob(env, actor_critic, adapter, stack: int, hole_cards) -> float:
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
    engine.start()
    engine.state.players[_SB_SEAT].hand_mask = card_ids_to_mask(hole_cards)

    req = engine.pending_request
    obs = env._build_decision_obs(req)
    rep_input = adapter.adapt(obs)

    with torch.no_grad():
        result = actor_critic.act(_SB_SEAT, rep_input, obs.legal_mask, deterministic=False)
        # act() samples once -- for a real probability, average many samples,
        # or read the distribution directly via act_batch's internals. Cheap
        # approach: sample N times.
        n = 200
        raise_count = 0
        shove_idx = next(
            i for i, o in enumerate(obs.options) if o.engine_action_type == EngineActionType.RAISE
        )
        for _ in range(n):
            r = actor_critic.act(_SB_SEAT, rep_input, obs.legal_mask, deterministic=False)
            if r.action_type == shove_idx:
                raise_count += 1
    return raise_count / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-name", type=str, default="push_fold_duo")
    parser.add_argument("--scenario-path", type=str, default=None)
    parser.add_argument("--variant", type=str, default="separate_networks",
                         choices=[v.value for v in PolicyVariant])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes-per-iteration", type=int, default=512)
    parser.add_argument("--entropy-coef", type=float, default=0.03)
    parser.add_argument("--stack-bb", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = load_graph_scenario(args.scenario_name, args.scenario_path)
    env = GraphScenarioEnv(config)
    adapter = GraphObservationAdapter(env)

    actor_critic = build_dual_seat_actor_critic_for_graph(
        PolicyVariant(args.variant), action_dim=config.max_action_options,
    ).to(args.device)

    optimizer_sb = torch.optim.Adam(actor_critic.seat_parameters(0), lr=3e-4)
    optimizer_bb = torch.optim.Adam(actor_critic.seat_parameters(1), lr=3e-4)
    optimizers = {0: optimizer_sb, 1: optimizer_bb}

    big_blind = env.config.stack.big_blind_size
    stack = max(big_blind, int(round(args.stack_bb * big_blind)))

    def print_probs(label: str) -> None:
        for name, cards in _FIXED_HANDS.items():
            p = sb_shove_prob(env, actor_critic, adapter, stack, cards)
            print(f"  [{label}] P(SB shove | {name}) = {p:.3f}")

    print("=== BEFORE any training ===")
    print_probs("init")

    for it in range(args.iterations):
        buffers, reward_log = collect_graph_rollouts(
            env, actor_critic, adapter, args.episodes_per_iteration,
            num_showdown_samples=30,
        )
        buf0 = buffers[0]
        if len(buf0) > 0:
            rewards = [t.reward for t in buf0._transitions if t.done]
            values = [t.value for t in buf0._transitions]
            print(f"\n=== iteration {it}: SB buffer size={len(buf0)} ===")
            print(f"  reward: mean={statistics.mean(rewards):+.3f} std={statistics.pstdev(rewards) if len(rewards)>1 else 0:.3f}")
            print(f"  value : mean={statistics.mean(values):+.3f} std={statistics.pstdev(values) if len(values)>1 else 0:.3f}")
            if buf0.advantages:
                print(f"  raw advantages (pre-normalize) mean={statistics.mean(buf0.advantages):+.3f} "
                      f"std={statistics.pstdev(buf0.advantages) if len(buf0.advantages)>1 else 0:.3f}")
        else:
            print(f"\n=== iteration {it}: SB buffer EMPTY (should never happen -- SB acts every hand) ===")

        loss0 = _ppo_update_seat(
            actor_critic, 0, buf0, optimizers[0],
            epochs=4, minibatch_size=64, clip_epsilon=0.2, value_coef=0.5,
            entropy_coef=args.entropy_coef, max_grad_norm=0.5,
        )
        _ppo_update_seat(
            actor_critic, 1, buffers[1], optimizers[1],
            epochs=4, minibatch_size=64, clip_epsilon=0.2, value_coef=0.5,
            entropy_coef=args.entropy_coef, max_grad_norm=0.5,
        )
        print(f"  SB update: policy_loss={loss0['policy_loss']:+.4f} "
              f"value_loss={loss0['value_loss']:.4f} entropy={loss0['entropy']:.4f}")
        print_probs(f"after iter {it}")


if __name__ == "__main__":
    main()
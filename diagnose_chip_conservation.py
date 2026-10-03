"""
diagnose_chip_conservation.py — Run this directly, no training required.

Checks the ONE invariant that must hold for every hand regardless of
architecture, PPO hyperparameters, or entropy: chips are conserved.
sum(rewards.values()) must be EXACTLY 0.0 for every hand (money only
moves between seats). If this fails specifically on hands where SB
shoves (RAISE), that points straight at a bug in how the shove amount
is applied to the stack (likely a total-vs-delta mixup in whatever
resolver.apply() does for AllInOrFoldRules's BET/RAISE, amount=shove_total
— see betting_rules.py's AllInOrFoldRules docstring: shove_total is a
TOTAL target bet, current_bet + stack, not a delta to add).

If this script reports every hand conserved, the bug is NOT here and
we should look at PPO/architecture dynamics instead (or something more
subtle in reward attribution, not raw chip movement).

Usage
-----
    python diagnose_chip_conservation.py --scenario-name push_fold_duo --hands 2000
"""

from __future__ import annotations

import argparse
import random
import sys

from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv
from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario


def run(scenario_name: str, scenario_path: str | None, hands: int, seed: int) -> None:
    config = load_graph_scenario(scenario_name, path=scenario_path)
    env = GraphScenarioEnv(config, seed=seed)
    rng = random.Random(seed)

    n_bad = 0
    n_shove_hands = 0
    n_bad_shove_hands = 0
    max_abs_imbalance = 0.0

    for hand_idx in range(hands):
        obs = env.reset()
        starting_stacks = list(env._starting_stacks)

        shoved_this_hand = False
        action_log = []

        while True:
            legal = [i for i, ok in enumerate(obs.legal_mask) if ok]
            action_index = rng.choice(legal)
            chosen_opt = obs.options[action_index]
            action_log.append(
                f"seat={obs.seat} label={chosen_opt.label} "
                f"engine_type={chosen_opt.engine_action_type} amount={chosen_opt.amount}"
            )
            if chosen_opt.engine_action_type is not None and chosen_opt.engine_action_type.name in (
                "RAISE", "BET"
            ):
                shoved_this_hand = True

            result = env.step(action_index)
            if result.done:
                rewards = result.rewards
                break
            obs = result.obs

        total_reward = sum(rewards.values())
        imbalance = abs(total_reward)
        max_abs_imbalance = max(max_abs_imbalance, imbalance)

        if shoved_this_hand:
            n_shove_hands += 1

        if imbalance > 1e-6:
            n_bad += 1
            if shoved_this_hand:
                n_bad_shove_hands += 1
            if n_bad <= 10:
                print(f"\n--- IMBALANCED HAND #{hand_idx} ---")
                print(f"starting_stacks={starting_stacks}")
                print(f"rewards={rewards}  sum={total_reward}")
                print("actions:")
                for line in action_log:
                    print(f"  {line}")

    print(f"\n{'='*60}")
    print(f"hands played:            {hands}")
    print(f"hands with a shove:      {n_shove_hands}")
    print(f"imbalanced hands:        {n_bad}")
    print(f"imbalanced shove hands:  {n_bad_shove_hands}")
    print(f"max |sum(rewards)|:      {max_abs_imbalance}")
    if n_bad == 0:
        print("OK — chips conserved on every hand. The bug is NOT here.")
    else:
        print("BUG CONFIRMED — chips are not conserved. See imbalanced hands above.")
        if n_bad_shove_hands == n_bad:
            print(
                "Every imbalance occurred on a hand with a shove (RAISE) — "
                "points at the BETTING resolver's amount-application logic "
                "for AllInOrFoldRules specifically."
            )
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-name", type=str, default="push_fold_duo")
    parser.add_argument("--scenario-path", type=str, default=None)
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(args.scenario_name, args.scenario_path, args.hands, args.seed)
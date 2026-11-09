"""
profiling/analyze_scenario_strategy.py — Action frequencies by hand strength,
per decision node, for a trained heads-up HOLD'EM graph scenario.

    python -m poker_rl_lab.profiling.analyze_scenario_strategy \
        --scenario-name holdem_turn_river_034_third_shove \
        --checkpoint poker_rl_lab/checkpoints/river_duo/ckpt_generic.pt \
        --hands 3000

What it does
------------
Plays `--hands` self-play hands (the trained policy on every seat, sampling
from its distribution like training does; add --deterministic for argmax).
At EVERY decision it records:

    (node_id, seat, situation, chosen option label, hand equity)

  situation = "facing_bet" if FOLD is among the options, else "first_to_act"
              (or "after_check" if a check already happened on this street's
              node — i.e. the second actor with no bet to face).
  equity    = win% (+ half tie%) of the acting hand vs a random opponent
              hand, with any missing board cards run out randomly, scored by
              the ENGINE evaluator (poker_eval). Sampled `--strength-samples`
              times per decision.

Then, per (node, seat, situation), prints the overall action mix and the
mix inside equity quantile bins (weakest -> strongest), so you can see e.g.
"bets 3/4 pot with top 20% and bottom 10%, checks the middle".

Holdem-only (2 hole cards, HIGH, one 5-card runout). Not for Omaha.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict


def mask(cards):
    m = 0
    for c in cards:
        m |= 1 << c
    return m


def equity(pe, hole, board, rng, k):
    """Win + tie/2 of `hole` vs k random opponent hands, board run out to 5."""
    used = set(hole) | set(board)
    rest = [c for c in range(52) if c not in used]
    need_board = 5 - len(board)
    a = mask(hole)
    score = 0.0
    for _ in range(k):
        draw = rng.sample(rest, 2 + need_board)
        opp, runout = draw[:2], draw[2:]
        full = mask(list(board) + runout)
        r = pe.evaluate_hands(
            [a, mask(opp)], full, pe.ScoreType.HIGH, pe.ShowdownType.HOLDEM
        )
        if r[0][0] > r[1][0]:
            score += 1.0
        elif r[0][0] == r[1][0]:
            score += 0.5
    return score / k


def label_mix(labels):
    c = Counter(labels)
    n = len(labels)
    return "  ".join(f"{k}={v / n:.2f}" for k, v in sorted(c.items()))


def main():
    p = argparse.ArgumentParser(description="Strategy by hand strength for a graph scenario.")
    p.add_argument("--scenario-name", required=True)
    p.add_argument("--scenario-path", default=None)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hands", type=int, default=3000)
    p.add_argument("--strength-samples", type=int, default=100)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    import torch
    import poker_engine.poker_eval as pe

    from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario
    from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv
    from poker_rl_lab.representation.card_utils import mask_to_card_ids
    from poker_rl_lab.representation.graph_observation_adapter import GraphObservationAdapter
    from poker_rl_lab.trainers.graph_training_loop import load_dual_seat_actor_critic_for_graph

    config = load_graph_scenario(args.scenario_name, path=args.scenario_path)
    env = GraphScenarioEnv(config, seed=args.seed)
    adapter = GraphObservationAdapter(env)
    actor_critic = load_dual_seat_actor_critic_for_graph(args.checkpoint)
    rng = random.Random(args.seed)

    print(f"scenario={config.name}  checkpoint={args.checkpoint}  hands={args.hands}\n")

    rows = defaultdict(list)   # (node, seat, situation) -> [(equity, label)]
    outcomes = Counter()

    for _ in range(args.hands):
        obs = env.reset()
        checked_at_node = set()
        while True:
            labels = [o.label for o in obs.options]
            facing = any(l.startswith("FOLD") for l in labels)
            if facing:
                situation = "facing_bet"
            elif obs.node_id in checked_at_node:
                situation = "after_check"
            else:
                situation = "first_to_act"

            st = env.state
            hole = mask_to_card_ids(st.players[obs.seat].hand_mask)
            board = [c for c in st.node_cards if c is not None]
            eq = equity(pe, hole, board, rng, args.strength_samples)

            rep = adapter.adapt(obs)
            with torch.no_grad():
                r = actor_critic.act(
                    obs.seat, rep, obs.legal_mask, deterministic=args.deterministic
                )
            idx = r.action_type
            label = labels[idx]
            rows[(obs.node_id, obs.seat, situation)].append((eq, label))
            if label.startswith("CHECK"):
                checked_at_node.add(obs.node_id)

            res = env.step(idx)
            if res.done:
                outcomes[res.info.get("outcome")] += 1
                break
            obs = res.obs

    print(f"hand outcomes: {dict(outcomes)}\n")

    for key in sorted(rows):
        node, seat, situation = key
        data = sorted(rows[key])
        n = len(data)
        print(f"{node} | seat {seat} | {situation} | n={n}")
        print(f"  overall: {label_mix([l for _, l in data])}")
        bins = 5 if n >= 100 else (3 if n >= 30 else 0)
        if bins:
            for b in range(bins):
                chunk = data[b * n // bins:(b + 1) * n // bins]
                if not chunk:
                    continue
                mean_eq = sum(e for e, _ in chunk) / len(chunk)
                print(f"  bin {b + 1}/{bins}  eq={mean_eq:.2f}  n={len(chunk):4d}  "
                      f"{label_mix([l for _, l in chunk])}")
        print()

    print("Reading guide: a sensible policy bets/calls MORE as equity rises (later bins),")
    print("mixes in some bluffs at the bottom, and checks/folds the middle. A flat mix in")
    print("every bin means the policy is ignoring hand strength (undertrained or collapsed).")


if __name__ == "__main__":
    main()
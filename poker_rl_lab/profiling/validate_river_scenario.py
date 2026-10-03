"""
validate_river_scenario.py — Validate a river-only bet-or-check scenario
(omaha_river_bet_or_check, holdem_river_bet_or_check, ...) end to end.

    # Part A only (no checkpoint needed): engine wiring on random play
    python -m poker_rl_lab.profiling.validate_river_scenario \
        --scenario-name omaha_river_bet_or_check --rule omaha

    # + trained-policy checks (B: baselines, C: strategy by hand strength)
    python -m poker_rl_lab.profiling.validate_river_scenario \
        --scenario-name omaha_river_bet_or_check --rule omaha \
        --checkpoint ckpt_omaha_river.pt --made-hand-max-subsets 60

    # Hold'em: --rule holdem
    python -m poker_rl_lab.profiling.validate_river_scenario \
        --scenario-name holdem_river_bet_or_check --rule holdem --checkpoint ckpt_holdem_river.pt

PART A — ENGINE WIRING (random play, no policy involved)
  * Start-of-river state: pot, stacks behind, per-seat contribution, hole
    card count, board size, empty history, starting-chips baseline.
  * Rewards are zero-sum on every hand.
  * EVERY showdown's winner is checked against an independent brute-force
    reference using the scenario's showdown rule (Omaha: exactly 2 hole +
    3 board; Hold'em: best 5 of 7). Also counts the showdowns where the
    OTHER rule would give a different verdict, proving the engine
    followed the right one.
PART B — BASELINES (needs --checkpoint): trained policy (both seats) vs
  always-fold, calling-station, random, always-bet.
PART C — STRATEGY BY HAND STRENGTH (needs --checkpoint): self-play, each
  decision bucketed by the acting hand's equity vs random opponent hands:
  call/bet frequency per strength decile, compared with indicative theory
  for a pot-sized bet at SPR 1 (defender calls ~50% = minimum defence
  frequency; a balanced betting range is ~1/3 bluffs).
"""

from __future__ import annotations

import argparse
import itertools
import math
import random
import sys
import time
from collections import Counter, defaultdict

import poker_engine.poker_eval as pe

RANK_CH = "23456789TJQKA"
SUIT_CH = "cdhs"


def cstr(c): return RANK_CH[c % 13] + SUIT_CH[c // 13]
def hstr(cs): return " ".join(cstr(c) for c in cs)


def mask(cards):
    m = 0
    for c in cards:
        m |= 1 << c
    return m


# ---------------------------------------------------------------------------
# Independent brute-force reference (HIGH only)
# ---------------------------------------------------------------------------

def high5(cards):
    rs = sorted((c % 13 for c in cards), reverse=True)
    flush = len({c // 13 for c in cards}) == 1
    uniq = sorted(set(rs), reverse=True)
    top = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            top = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            top = 3                                  # wheel = lowest straight
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


def best_hand(rule, hole, board):
    if rule == "omaha":
        cands = [h + b for h in itertools.combinations(hole, 2)
                 for b in itertools.combinations(board, 3)]
    else:  # holdem
        cands = itertools.combinations(list(hole) + list(board), 5)
    return max(high5(c) for c in cands)


def verdict(rule, hole_a, hole_b, board):
    """0 = seat 0 wins, 1 = tie, 2 = seat 1 wins."""
    ra, rb = best_hand(rule, hole_a, board), best_hand(rule, hole_b, board)
    return 0 if ra > rb else (2 if ra < rb else 1)


def hand_strength(rule, hole, board, rng, k=100):
    """Equity (win + tie/2) of `hole` vs k random opponent hands on `board`,
    scored by the ENGINE evaluator (pe.evaluate_hands)."""
    used = set(hole) | set(board)
    rest = [c for c in range(52) if c not in used]
    sd = pe.ShowdownType.OMAHA if rule == "omaha" else pe.ShowdownType.HOLDEM
    a, b0 = mask(hole), mask(board)
    w = t = 0
    for _ in range(k):
        opp = rng.sample(rest, len(hole))
        r = pe.evaluate_hands([a, mask(opp)], b0, pe.ScoreType.HIGH, sd)
        if r[0][0] > r[1][0]: w += 1
        elif r[0][0] == r[1][0]: t += 1
    return (w + 0.5 * t) / k


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def label_kind(label):
    if label.startswith("FOLD"): return "FOLD"
    if label.startswith("CHECK"): return "CHECK"
    if label.startswith("CALL"): return "CALL"
    return "AGGR"


def pick(obs, *kinds):
    """First option whose kind is in `kinds` (in that priority), else None."""
    for kind in kinds:
        for i, o in enumerate(obs.options):
            if label_kind(o.label) == kind:
                return i
    return None


# ---------------------------------------------------------------------------
# Choosers: (obs, env) -> option index
# ---------------------------------------------------------------------------

def random_chooser(rng):
    return lambda obs, env: rng.randrange(len(obs.options))

def always_fold(obs, env):
    i = pick(obs, "FOLD", "CHECK"); return i if i is not None else 0

def calling_station(obs, env):
    i = pick(obs, "CALL", "CHECK"); return i if i is not None else 0

def always_bet(obs, env):
    i = pick(obs, "AGGR", "CALL", "CHECK"); return i if i is not None else 0


def trained_chooser(actor_critic, adapter, deterministic=False):
    import torch

    def choose(obs, env):
        rep = adapter.adapt(obs)
        with torch.no_grad():
            r = actor_critic.act(obs.seat, rep, obs.legal_mask, deterministic=deterministic)
        return r.action_type
    return choose


# ---------------------------------------------------------------------------
# Hand runner
# ---------------------------------------------------------------------------

def play_hand(env, chooser):
    from poker_rl_lab.representation.card_utils import mask_to_card_ids

    obs = env.reset()
    st = env.state
    holes = [mask_to_card_ids(p.hand_mask) for p in st.players]
    board = [c for c in st.node_cards if c is not None]
    snap = {
        "pot": st.pot,
        "stacks": [p.stack for p in st.players],
        "contrib": [p.total_contribution for p in st.players],
        "starting": tuple(env._starting_stacks),
        "first_seat": obs.seat,
        "hist0": len(env.history),
        "n_options": len(obs.options),
    }
    decisions = []
    while True:
        hist_len = len(env.history)
        idx = chooser(obs, env)
        decisions.append({
            "seat": obs.seat, "labels": [o.label for o in obs.options],
            "idx": idx, "hist": hist_len,
        })
        res = env.step(idx)
        if res.done:
            return {"holes": holes, "board": board, "snap": snap, "decisions": decisions,
                    "rewards": res.rewards, "outcome": res.info.get("outcome")}
        obs = res.obs


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / max(n - 1, 1)
    return m, math.sqrt(var / n)


# ---------------------------------------------------------------------------
# PART A
# ---------------------------------------------------------------------------

def part_a(env, config, args, rng):
    print("=" * 78)
    print(f"PART A: engine wiring ({args.hands} random-policy hands, rule={args.rule})")
    print("=" * 78)
    bb = config.stack.big_blind_size
    exp_pot = int(round(config.start.pot_bb * bb)) if config.start else None
    exp_behind = int(round(config.start.stack_behind_bb * bb)) if config.start else None

    chooser = random_chooser(rng)
    problems = Counter()
    examples = []
    first_seats = Counter()
    option_sets = Counter()
    showdowns = mismatches = discriminating = folds = 0
    rewards_by_seat = defaultdict(list)
    reward_by_first = []

    def flag(kind, msg):
        problems[kind] += 1
        if len(examples) < 8:
            examples.append(f"{kind}: {msg}")

    for _ in range(args.hands):
        h = play_hand(env, chooser)
        s, holes, board = h["snap"], h["holes"], h["board"]
        first_seats[s["first_seat"]] += 1
        option_sets[tuple(h["decisions"][0]["labels"])] += 1

        if exp_pot is not None and s["pot"] != exp_pot: flag("pot", f"{s['pot']} != {exp_pot}")
        if exp_behind is not None and any(x != exp_behind for x in s["stacks"]):
            flag("stacks", f"{s['stacks']} != {exp_behind}")
        if exp_pot is not None and any(c != exp_pot // len(s["contrib"]) for c in s["contrib"]):
            flag("contribution", f"{s['contrib']}")
        if any(len(x) != args.hole_cards for x in holes): flag("hole_cards", f"{[len(x) for x in holes]}")
        if len(board) != args.board_cards: flag("board_cards", f"{len(board)}")
        if len(set(sum(holes, []) + board)) != sum(len(x) for x in holes) + len(board):
            flag("duplicate_cards", hstr(sum(holes, []) + board))
        if s["hist0"] != 0: flag("history_not_empty", f"{s['hist0']}")
        if s["starting"] != tuple(st + c for st, c in zip(s["stacks"], s["contrib"])):
            flag("starting_baseline", f"{s['starting']}")
        if abs(sum(h["rewards"].values())) > 1e-6: flag("not_zero_sum", f"{h['rewards']}")

        for seat, r in h["rewards"].items():
            rewards_by_seat[seat].append(r)
        reward_by_first.append(h["rewards"][s["first_seat"]])

        if h["outcome"] == "showdown":
            showdowns += 1
            want = verdict(args.rule, holes[0], holes[1], board)
            r0 = h["rewards"][0]
            got = 0 if r0 > 1e-9 else (2 if r0 < -1e-9 else 1)
            other = "holdem" if args.rule == "omaha" else "omaha"
            if len(holes[0]) >= 2 and verdict(other, holes[0], holes[1], board) != want:
                discriminating += 1
            if want != got:
                mismatches += 1
                flag("showdown_mismatch",
                     f"A={hstr(holes[0])} B={hstr(holes[1])} board={hstr(board)} "
                     f"reference={('seat0 wins','tie','seat1 wins')[want]} engine rewards={h['rewards']}")
        else:
            folds += 1

    print(f"first actor (seat) counts : {dict(first_seats)}")
    print(f"first-decision option sets: {dict(option_sets)}")
    print(f"hands: {args.hands}  showdowns: {showdowns}  folds: {folds}")
    for seat in sorted(rewards_by_seat):
        m, se = mean_se(rewards_by_seat[seat])
        print(f"random-play mean reward seat {seat}: {m:+.2f} +- {se:.2f} chips")
    m, se = mean_se(reward_by_first)
    print(f"random-play mean reward of FIRST actor: {m:+.2f} +- {se:.2f} chips")
    print(f"showdowns where the OTHER rule would give a different verdict: {discriminating} "
          f"(engine matched {args.rule} in all of them unless listed below)")

    ok = not problems
    print("\nPART A:", "PASS" if ok else f"FAIL {dict(problems)}")
    for e in examples:
        print("   ", e)
    return ok, first_seats


# ---------------------------------------------------------------------------
# PART B
# ---------------------------------------------------------------------------

def part_b(env, config, trained, args, rng):
    print("\n" + "=" * 78)
    print(f"PART B: trained policy vs baselines ({args.matchup_hands} hands per seat assignment)")
    print("=" * 78)
    bb = config.stack.big_blind_size
    baselines = {
        "always_fold": always_fold,
        "calling_station": calling_station,
        "random": random_chooser(rng),
        "always_bet": always_bet,
    }
    all_ok = True
    for name, base in baselines.items():
        rewards = []
        for trained_seat in range(config.num_seats):
            def chooser(obs, env_, ts=trained_seat, b=base):
                return trained(obs, env_) if obs.seat == ts else b(obs, env_)
            for _ in range(args.matchup_hands):
                rewards.append(play_hand(env, chooser)["rewards"][trained_seat])
        m, se = mean_se(rewards)
        if m - 2 * se > 0: tag = "PASS (trained wins)"
        elif m + 2 * se < 0: tag = "FAIL (trained LOSES)"; all_ok = False
        else: tag = "inconclusive"
        print(f"  vs {name:16s}: {m:+8.2f} +- {se:5.2f} chips/hand  ({m / bb:+.2f} bb/hand)  {tag}")
    print("\nPART B:", "PASS" if all_ok else "FAIL")
    return all_ok


# ---------------------------------------------------------------------------
# PART C
# ---------------------------------------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)


def part_c(env, config, trained, args, rng):
    print("\n" + "=" * 78)
    print(f"PART C: strategy by hand strength ({args.analysis_hands} self-play hands)")
    print("=" * 78)
    rows = defaultdict(list)    # (seat, situation) -> [(strength, acted_flag)]
    t0 = time.time()
    for _ in range(args.analysis_hands):
        h = play_hand(env, trained)
        for d in h["decisions"]:
            kinds = [label_kind(l) for l in d["labels"]]
            chosen = label_kind(d["labels"][d["idx"]])
            if "FOLD" in kinds:
                situation, acted = "facing_bet (call)", chosen == "CALL"
            else:
                situation = "first_to_act (bet)" if d["hist"] == 0 else "after_check (bet)"
                acted = chosen == "AGGR"
            strength = hand_strength(args.rule, h["holes"][d["seat"]], h["board"], rng, args.strength_samples)
            rows[(d["seat"], situation)].append((strength, acted))
    print(f"(collected in {time.time() - t0:.0f}s)\n")

    warnings = []
    for key in sorted(rows):
        seat, situation = key
        data = sorted(rows[key])
        n = len(data)
        freq = sum(a for _, a in data) / n
        corr = pearson([s for s, _ in data], [1.0 if a else 0.0 for _, a in data])
        print(f"seat {seat} | {situation:20s} | n={n:5d} | overall freq = {freq:.3f} | corr(strength, action) = {corr:+.2f}")
        bins = 10 if n >= 200 else (5 if n >= 50 else 0)
        if bins:
            print("    bin  hands  mean-equity  freq")
            for b in range(bins):
                chunk = data[b * n // bins:(b + 1) * n // bins]
                if not chunk:
                    continue
                print(f"    {b + 1:3d}  {len(chunk):5d}  {sum(s for s, _ in chunk) / len(chunk):10.3f}  "
                      f"{sum(a for _, a in chunk) / len(chunk):.3f}")
            if situation.startswith("facing_bet"):
                top = data[(bins - 1) * n // bins:]
                bot = data[:n // bins]
                if sum(a for _, a in top) / len(top) < sum(a for _, a in bot) / len(bot):
                    warnings.append(f"seat {seat} {situation}: weakest bin calls MORE than strongest bin")
        if situation.startswith("facing_bet"):
            if corr < 0.3:
                warnings.append(f"seat {seat} {situation}: weak strength->call correlation ({corr:+.2f})")
            if not 0.3 <= freq <= 0.7:
                warnings.append(f"seat {seat} {situation}: call frequency {freq:.2f} far from ~0.50 (min. defence)")
        print()

    print("Indicative theory (pot-sized bet into pot at SPR 1): defender calls ~50% of the time")
    print("and calls with the STRONGER hands; a balanced betting range is ~1/3 bluffs, so bet")
    print("frequency should rise with strength at the top and show some betting at the bottom.")
    print("These are guides for a trained-but-imperfect policy, not strict pass/fail criteria.\n")
    if warnings:
        print("PART C: WARNINGS")
        for w in warnings:
            print("   -", w)
    else:
        print("PART C: no warnings")
    return not warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Validate a river-only bet-or-check scenario.")
    p.add_argument("--scenario-name", required=True)
    p.add_argument("--scenario-path", default=None)
    p.add_argument("--rule", choices=["omaha", "holdem"], default="omaha")
    p.add_argument("--hole-cards", type=int, default=None, help="Expected hole cards (default: 4 omaha / 2 holdem)")
    p.add_argument("--board-cards", type=int, default=5)
    p.add_argument("--hands", type=int, default=2000, help="Part A random-play hands")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--made-hand-max-subsets", type=int, default=12)
    p.add_argument("--board-max-subsets", type=int, default=12)
    p.add_argument("--matchup-hands", type=int, default=300, help="Part B hands per seat assignment per baseline")
    p.add_argument("--analysis-hands", type=int, default=1500, help="Part C self-play hands")
    p.add_argument("--strength-samples", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.hole_cards is None:
        args.hole_cards = 4 if args.rule == "omaha" else 2

    from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario
    from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv

    config = load_graph_scenario(args.scenario_name, path=args.scenario_path)
    env = GraphScenarioEnv(config, seed=args.seed)
    rng = random.Random(args.seed)
    print(f"scenario={config.name} engine_variant={config.engine_variant} rule={args.rule} "
          f"start={config.start}\n")

    ok_a, _ = part_a(env, config, args, rng)
    results = {"A": ok_a}

    if args.checkpoint:
        from poker_rl_lab.models.actor_critic_factory import ActorCriticConfig
        from poker_rl_lab.representation.graph_observation_adapter import GraphObservationAdapter
        from poker_rl_lab.trainers.graph_training_loop import load_dual_seat_actor_critic_for_graph

        ac_cfg = ActorCriticConfig(
            action_dim=config.max_action_options,
            made_hand_max_subsets=args.made_hand_max_subsets,
            board_max_subsets=args.board_max_subsets,
        )
        actor_critic = load_dual_seat_actor_critic_for_graph(args.checkpoint, actor_critic_config=ac_cfg)
        adapter = GraphObservationAdapter(env)
        trained = trained_chooser(actor_critic, adapter)
        results["B"] = part_b(env, config, trained, args, rng)
        results["C"] = part_c(env, config, trained, args, rng)
    else:
        print("\n(no --checkpoint given: skipping Parts B and C)")

    print("\n" + "=" * 78)
    print("SUMMARY: " + "   ".join(f"Part {k}: {'PASS' if v else 'CHECK'}" for k, v in results.items()))
    print("=" * 78)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
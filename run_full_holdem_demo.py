"""
run_full_holdem_demo.py — Trains a PPO agent on the full-table hold'em
curriculum stage (Engine-backed via envs.full_holdem_env.FullHoldemEnv),
logs per-iteration training stats, and benchmarks the final policy against
fixed baselines (per RL_lab_implementation_guide.md §10).

This is the Engine-backed sibling of run_push_fold_demo.py — same
structure (build -> train -> checkpoint -> evaluate), but every hand is
played out street-by-street by the real poker_engine.state.poker_state
.PokerState rather than self-contained env logic. Expect this to run
noticeably slower per hand than push-fold/river for the same reason:
FullHoldemEnv drives the Engine through DEAL_BOARD/BETTING phases and
runs every non-hero seat's policy in-line (see full_holdem_env.py's
module docstring) — there is real work happening per decision, not just
a dict lookup. Defaults below are deliberately smaller than
run_push_fold_demo.py's for a first smoke test; scale up once timing is
confirmed acceptable.

Setup notes:
  - Requires `triton` NOT installed alongside tensorflow (torch.optim.Adam's
    lazy torch._dynamo->triton import segfaults against TF's bundled
    oneDNN libs). `pip uninstall triton`.
  - Run with KMP_DUPLICATE_LIB_OK=TRUE in the environment.
  - poker_eval shim installed below (mock — see mock_poker_eval.py).
    full_holdem_experiment.py's own import of envs.full_holdem_env needs
    this shim in place BEFORE that import happens (PokerState is imported
    at module level there) — see that module's docstring.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_full_holdem_demo.py
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_full_holdem_demo.py \
        --iterations 10 --player-count 6 --checkpoint-out ckpt_full.pt
"""
import sys

try:
    import poker_eval  # noqa: F401
except ImportError:
    import mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse
import time

from actions.abstract_action import ActionType
from envs.base_env import RandomHeroPolicy
from evaluation.baseline_policies import always_fold_policy, calling_station_policy
from evaluation.policy_evaluator import compare_policies, policy_from_actor_critic
from experiments.full_holdem_experiment import (
    build_full_holdem_experiment,
    FullHoldemExperimentConfig,
    save_checkpoint,
)

_ACTION_ORDER = [a.name for a in ActionType]


def _fmt_action_freqs(freqs: dict) -> str:
    """One compact line: FOLD=0.42 CALL=0.58 (omits actions never taken)."""
    parts = [f"{name}={freqs[name]:.2f}" for name in _ACTION_ORDER if name in freqs]
    return " ".join(parts) if parts else "(no actions recorded)"


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description="Full-table hold'em PPO training demo.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--episodes-per-iteration", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--player-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=500,
                         help="Hands per policy in the final baseline comparison. "
                              "Kept far lower than push-fold's default since every "
                              "hand here runs through the real Engine.")
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=1,
                         help="Print training stats every N iterations.")
    args = parser.parse_args()

    cfg = FullHoldemExperimentConfig(
        player_count=args.player_count,
        big_blind=args.big_blind,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    _print_header("BUILDING EXPERIMENT")
    print("NOTE: untrained hand encoder + mock poker_eval + mock scoring engine —")
    print("      this run exercises the full Engine-backed pipeline end-to-end")
    print("      but carries no real poker skill.")
    print(f"config: {cfg}")
    exp = build_full_holdem_experiment(cfg)
    print(f"hero seat is re-derived from scenario.current_player at each reset() "
          f"(UTG for a fresh preflop deal — see full_holdem_generator.py); "
          f"table size = {args.player_count}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    _print_header("TRAINING")
    header = (
        f"{'iter':>4}  {'reward':>8}  {'policy_loss':>12}  {'value_loss':>10}  "
        f"{'entropy':>8}  {'clip_frac':>9}  {'time/it':>8}  {'hands/s':>8}"
    )
    print(header)
    print("-" * len(header))

    start = time.time()
    for i in range(cfg.num_iterations):
        t0 = time.time()
        buffer = exp.trainer.collect_rollouts(
            exp.env, exp.scenario_gen, cfg.episodes_per_iteration
        )
        mean_reward = exp.trainer._iteration_reward(buffer)
        stats = exp.trainer.update(
            buffer, epochs=cfg.epochs, minibatch_size=cfg.minibatch_size
        )
        buffer.clear()
        dt = time.time() - t0
        hands_per_sec = cfg.episodes_per_iteration / dt if dt > 0 else float("inf")

        if i % args.log_every == 0 or i == cfg.num_iterations - 1:
            print(
                f"{i:>4}  {mean_reward:>8.3f}  {stats['policy_loss']:>12.5f}  "
                f"{stats['value_loss']:>10.3f}  {stats['entropy']:>8.4f}  "
                f"{stats['clip_fraction']:>9.3f}  {dt:>7.2f}s  {hands_per_sec:>7.1f}"
            )

    total_time = time.time() - start
    print(f"\nTotal training time: {total_time:.1f}s "
          f"({total_time / cfg.num_iterations:.2f}s/iter)")

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    if args.checkpoint_out:
        save_checkpoint(exp, args.checkpoint_out)
        print(f"\nSaved checkpoint to {args.checkpoint_out}")

    # ------------------------------------------------------------------
    # Evaluation vs fixed baselines (guide §10)
    # ------------------------------------------------------------------
    _print_header(
        f"EVALUATION  ({args.eval_episodes} hands/policy, "
        f"{args.player_count}-max, paired seed={args.seed}0)"
    )
    print("NOTE: baselines (always_fold / calling_station / random) only ever see")
    print("      the HERO seat's legal_mask each hand — non-hero seats always run")
    print("      config.default_villain_policy (CheckFoldVillainPolicy by default).")

    trained_policy = policy_from_actor_critic(
        exp.actor_critic, exp.adapter, deterministic=True
    )

    policies = {
        "trained":         trained_policy,
        "random":          RandomHeroPolicy(seed=args.seed),
        "always_fold":     always_fold_policy,
        "calling_station": calling_station_policy,
    }

    eval_start = time.time()
    results = compare_policies(
        exp.env,
        exp.scenario_gen,
        policies,
        num_episodes=args.eval_episodes,
        big_blind=args.big_blind,
        seed=args.seed * 10,  # same hand sequence for every policy
    )
    eval_time = time.time() - eval_start

    name_w = max(len(n) for n in results)
    print(f"\n{'policy':<{name_w}}  {'reward':>8}  {'95% CI':>18}  "
          f"{'bb/100':>8}  {'win%':>6}  action_freq")
    print("-" * 100)
    for name, r in results.items():
        lo, hi = r.confidence_interval_95()
        print(
            f"{name:<{name_w}}  {r.mean_reward:>+8.3f}  "
            f"[{lo:>+7.3f}, {hi:>+7.3f}]  {r.bb_per_100:>+8.2f}  "
            f"{r.win_rate:>5.1%}  {_fmt_action_freqs(r.action_frequencies)}"
        )
    print(f"\nEvaluation wall time: {eval_time:.1f}s "
          f"({4 * args.eval_episodes / eval_time:.1f} hands/s across 4 policies)")

    print()
    trained_bb100 = results["trained"].bb_per_100
    best_baseline = max(
        (r.bb_per_100 for name, r in results.items() if name != "trained"),
    )
    margin = trained_bb100 - best_baseline
    verdict = "beats" if margin > 0 else "does NOT beat"
    print(f"Trained policy {verdict} the best baseline by {margin:+.2f} bb/100 "
          f"(expect near-zero with an untrained hand encoder + mock evaluator).")


if __name__ == "__main__":
    main()
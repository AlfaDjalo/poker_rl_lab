"""
run_push_fold_demo.py — Trains a PPO agent on the push-fold curriculum
stage, logs per-iteration training stats, and benchmarks the final
policy against fixed baselines (per RL_lab_implementation_guide.md §10).

Setup notes:
  - Requires `triton` NOT installed alongside tensorflow (torch.optim.Adam's
    lazy torch._dynamo->triton import segfaults against TF's bundled
    oneDNN libs). `pip uninstall triton`.
  - Run with KMP_DUPLICATE_LIB_OK=TRUE in the environment.
  - poker_eval shim installed below (mock — see mock_poker_eval.py).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_push_fold_demo.py
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_push_fold_demo.py --iterations 50 --checkpoint-out ckpt.pt
"""
import sys

try:
    import poker_engine.poker_eval  # noqa: F401
except ImportError:
    import mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse
import time

from actions.abstract_action import ActionType
from envs.base_env import RandomHeroPolicy
from evaluation.baseline_policies import always_fold_policy, calling_station_policy
from evaluation.policy_evaluator import compare_policies, policy_from_actor_critic
from experiments.push_fold_experiment import (
    build_push_fold_experiment,
    PushFoldExperimentConfig,
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
    parser = argparse.ArgumentParser(description="Push-fold PPO training demo.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes-per-iteration", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=2000,
                         help="Hands per policy in the final baseline comparison.")
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=1,
                         help="Print training stats every N iterations.")
    args = parser.parse_args()

    cfg = PushFoldExperimentConfig(
        big_blind=args.big_blind,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    _print_header("BUILDING EXPERIMENT")
    print("NOTE: untrained hand encoder + mock poker_eval — this run exercises")
    print("      the full pipeline end-to-end but carries no real poker skill.")
    print(f"config: {cfg}")
    exp = build_push_fold_experiment(cfg)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    _print_header("TRAINING")
    header = (
        f"{'iter':>4}  {'reward':>8}  {'policy_loss':>12}  {'value_loss':>10}  "
        f"{'entropy':>8}  {'clip_frac':>9}  {'time/it':>8}"
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

        if i % args.log_every == 0 or i == cfg.num_iterations - 1:
            print(
                f"{i:>4}  {mean_reward:>8.3f}  {stats['policy_loss']:>12.5f}  "
                f"{stats['value_loss']:>10.3f}  {stats['entropy']:>8.4f}  "
                f"{stats['clip_fraction']:>9.3f}  {dt:>7.2f}s"
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
    _print_header(f"EVALUATION  ({args.eval_episodes} hands/policy, paired seed={args.seed}0)")

    trained_policy = policy_from_actor_critic(
        exp.actor_critic, exp.adapter, deterministic=True
    )

    policies = {
        "trained":         trained_policy,
        "random":          RandomHeroPolicy(seed=args.seed),
        "always_fold":     always_fold_policy,
        "calling_station": calling_station_policy,
    }

    results = compare_policies(
        exp.env,
        exp.scenario_gen,
        policies,
        num_episodes=args.eval_episodes,
        big_blind=args.big_blind,
        seed=args.seed * 10,  # same hand sequence for every policy
    )

    name_w = max(len(n) for n in results)
    print(f"{'policy':<{name_w}}  {'reward':>8}  {'95% CI':>18}  "
          f"{'bb/100':>8}  {'win%':>6}  action_freq")
    print("-" * 100)
    for name, r in results.items():
        lo, hi = r.confidence_interval_95()
        print(
            f"{name:<{name_w}}  {r.mean_reward:>+8.3f}  "
            f"[{lo:>+7.3f}, {hi:>+7.3f}]  {r.bb_per_100:>+8.2f}  "
            f"{r.win_rate:>5.1%}  {_fmt_action_freqs(r.action_frequencies)}"
        )

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
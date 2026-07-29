"""
run_river_demo.py — Trains a PPO agent on the heads-up river curriculum
stage (envs.river_env.RiverEnv), logs per-iteration training stats, and
benchmarks the final policy against fixed baselines (per
RL_lab_implementation_guide.md §10).

Same structure as run_push_fold_demo.py (build -> train -> checkpoint ->
evaluate), but river hands have more than one hero decision point (a
raise/reraise can bounce action back to the hero — see river_env.py's
step() docstring), so GAE is doing real work here rather than reducing to
advantage = reward - value as it does for push-fold's single-decision
hands (see trainers/rollout_buffer.py's module docstring).

Villain policy note
--------------------
Unlike push-fold, RiverEnv's opponent acts every episode via a pluggable
villain_policy (see envs/river_env.py: CheckFoldVillainPolicy /
CallAnyVillainPolicy / FoldAnyVillainPolicy / RandomVillainPolicy). The
policy the agent is TRAINED against is a single fixed choice
(`--villain` below, default check_fold — matching RiverExperimentConfig's
own default). The final evaluation section then re-benchmarks the
trained policy against baselines under that SAME fixed villain, so a high
bb/100 here specifically means "beats these baselines when the opponent
plays --villain's style" — it does not by itself say anything about
generalisation to a different opponent. Swap --villain and rerun to
check that.

Setup notes:
  - Requires `triton` NOT installed alongside tensorflow (torch.optim.Adam's
    lazy torch._dynamo->triton import segfaults against TF's bundled
    oneDNN libs). `pip uninstall triton`.
  - Run with KMP_DUPLICATE_LIB_OK=TRUE in the environment.
  - poker_eval shim installed below (mock — see mock_poker_eval.py).

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_river_demo.py
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_river_demo.py \
        --villain random --iterations 50 --checkpoint-out ckpt_river.pt
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
from envs.river_env import (
    CallAnyVillainPolicy,
    CheckFoldVillainPolicy,
    FoldAnyVillainPolicy,
    RandomVillainPolicy,
)
from envs.poker_eval_evaluator import PokerEvalEvaluator
from evaluation.baseline_policies import always_fold_policy, calling_station_policy
from evaluation.policy_evaluator import compare_policies, policy_from_actor_critic
from experiments.river_experiment import (
    build_river_experiment,
    RiverExperimentConfig,
    save_checkpoint,
)
from representation.pretrained_hand_encoder import (
    HandEncoderArchitectureMismatch,
    PretrainedHandEncoder,
)

_ACTION_ORDER = [a.name for a in ActionType]

_VILLAIN_CHOICES = {
    "check_fold":  CheckFoldVillainPolicy,
    "call_any":    CallAnyVillainPolicy,
    "fold_any":    FoldAnyVillainPolicy,
    "random":      RandomVillainPolicy,
}


def _fmt_action_freqs(freqs: dict) -> str:
    """One compact line: FOLD=0.42 CALL=0.58 (omits actions never taken)."""
    parts = [f"{name}={freqs[name]:.2f}" for name in _ACTION_ORDER if name in freqs]
    return " ".join(parts) if parts else "(no actions recorded)"


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _build_villain(name: str, seed: int):
    cls = _VILLAIN_CHOICES[name]
    if cls is RandomVillainPolicy:
        return cls(seed=seed)
    return cls()


def _load_hand_encoder(path: str, expected_embedding_dim=None):
    """
    Load the real pretrained hand encoder from `path`, or fall back to an
    untrained copy (with a loud warning) if the file isn't there / fails
    to load. Identical loader to run_push_fold_demo.py's own — see that
    script for the full rationale.
    """
    if not path:
        print("No --encoder-path given — using an UNTRAINED hand encoder "
              "(no poker knowledge). Pass --encoder-path to use real weights.")
        return PretrainedHandEncoder.untrained()

    try:
        encoder = PretrainedHandEncoder.from_file(
            path, expected_embedding_dim=expected_embedding_dim
        )
        print(f"Loaded pretrained hand encoder from {path} "
              f"(embedding_dim={encoder.embedding_dim}).")
        return encoder
    except FileNotFoundError:
        print(f"WARNING: encoder file not found at {path!r} — falling back "
              f"to an UNTRAINED hand encoder (no poker knowledge).")
        return PretrainedHandEncoder.untrained()
    except HandEncoderArchitectureMismatch as exc:
        print(f"WARNING: {path!r} failed architecture validation ({exc}) — "
              f"falling back to an UNTRAINED hand encoder.")
        return PretrainedHandEncoder.untrained()


def main():
    parser = argparse.ArgumentParser(description="Heads-up river PPO training demo.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes-per-iteration", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-raises", type=int, default=4)
    parser.add_argument("--villain", type=str, default="check_fold",
                         choices=sorted(_VILLAIN_CHOICES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-episodes", type=int, default=2000,
                         help="Hands per policy in the final baseline comparison.")
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=1,
                         help="Print training stats every N iterations.")
    parser.add_argument(
        "--encoder-path", type=str, default="encoders/encoder_model.keras",
        help="Path to the pretrained hand encoder .keras file. Pass an "
             "empty string ('') to force an untrained encoder.",
    )
    parser.add_argument(
        "--encoder-embedding-dim", type=int, default=32,
        help="Expected embedding_dim of the loaded encoder; loading fails "
             "loudly if it doesn't match. Pass 0 to skip this check.",
    )
    parser.add_argument(
        "--use-mock-evaluator", action="store_true",
        help="Use river_env's MockEvaluator (mask-integer comparison, no "
             "real poker knowledge) instead of the real poker_eval-backed "
             "PokerEvalEvaluator. Default is the real evaluator.",
    )
    args = parser.parse_args()

    villain_policy = _build_villain(args.villain, args.seed)

    evaluator = None if args.use_mock_evaluator else PokerEvalEvaluator(seed=args.seed)

    cfg = RiverExperimentConfig(
        big_blind=args.big_blind,
        max_raises=args.max_raises,
        villain_policy=villain_policy,
        evaluator=evaluator,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    _print_header("BUILDING EXPERIMENT")
    print(f"showdown evaluator: "
          f"{'MockEvaluator (no real poker knowledge)' if args.use_mock_evaluator else 'PokerEvalEvaluator (real poker_eval)'}")
    print(f"villain policy : {args.villain}  (fixed opponent style for both "
          f"training and evaluation below)")
    hand_encoder = _load_hand_encoder(
        args.encoder_path,
        expected_embedding_dim=args.encoder_embedding_dim or None,
    )
    print(f"config: {cfg}")
    exp = build_river_experiment(cfg, hand_encoder=hand_encoder)

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
    _print_header(
        f"EVALUATION  ({args.eval_episodes} hands/policy, villain={args.villain}, "
        f"paired seed={args.seed}0)"
    )

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
          f"vs a {args.villain} villain "
          f"(expect near-zero with an untrained hand encoder + mock evaluator).")


if __name__ == "__main__":
    main()
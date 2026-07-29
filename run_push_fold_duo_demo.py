"""
run_push_fold_duo_demo.py — Trains a push-fold-duo PPO agent pair, logs
per-iteration training stats, checkpoints, and reports a basic
self-play evaluation (per RL_lab_implementation_guide.md §10).

Mirrors run_push_fold_demo.py's structure (build -> train -> checkpoint
-> evaluate), but there are two seats being trained simultaneously (or
alternately/league — see --mode), so there is no single "trained vs
baselines" table here yet: fixed rule-based baselines for push-fold-duo
(push_any / fold_any / call_any per seat) are a follow-up module
(evaluation/push_fold_duo_baselines.py) not yet built. In the meantime
this script's "EVALUATION" section plays the trained pair against
ITSELF deterministically and reports each seat's mean reward / bb-per-100
— useful for confirming the pipeline learns something sane (SB should
push a positive-EV range, BB should call it correctly) even without a
baseline comparison table. Re-run with --checkpoint-in against an older
checkpoint once evaluation/push_fold_duo_baselines.py exists for a real
comparison.

Setup notes
-----------
  - Requires `triton` NOT installed alongside tensorflow (see
    run_push_fold_demo.py's identical setup note).
  - Run with KMP_DUPLICATE_LIB_OK=TRUE in the environment.
  - poker_eval shim installed below (mock — see mock_poker_eval.py).

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_push_fold_duo_demo.py
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_push_fold_duo_demo.py \
        --variant shared_single_head --mode alternating --iterations 50 \
        --checkpoint-out ckpt_duo.pt
"""
import sys

try:
    import poker_engine.poker_eval  # noqa: F401
except ImportError:
    import mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse
import statistics
import time

import torch

from actions.abstract_action import ActionType
from experiments.push_fold_duo_experiment import (
    PolicyVariant,
    PushFoldDuoExperimentConfig,
    build_push_fold_duo_experiment,
    load_checkpoint,
    save_checkpoint,
)
from scenarios.push_fold_duo_generator import SB_SEAT, BB_SEAT
from trainers.duo_training_loop import DuoTrainingMode, collect_duo_rollouts
from representation.pretrained_hand_encoder import (
    HandEncoderArchitectureMismatch,
    PretrainedHandEncoder,
)

_ACTION_ORDER = [a.name for a in ActionType]


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _fmt_loss_dict(d: dict) -> str:
    return " ".join(f"seat{k}={v:+.4f}" for k, v in d.items())


def _bb_per_100(mean_reward: float, big_blind: int) -> float:
    return mean_reward / big_blind * 100.0


def _load_hand_encoder(path: str, expected_embedding_dim=None):
    """
    Load the real pretrained hand encoder from `path`, or fall back to an
    untrained copy (with a loud warning) if the file isn't there / fails
    to load. Identical loader to run_push_fold_demo.py's own — CLI-only
    convenience, distinct from
    experiments.push_fold_duo_experiment._default_hand_encoder(), which
    is hardcoded to that module's own default path for non-CLI callers
    of build_push_fold_duo_experiment().
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


def _evaluate_self_play(exp, num_episodes: int, seed: int):
    """
    Play the trained pair against itself deterministically for
    `num_episodes` hands and report per-seat reward stats.

    Returns
    -------
    dict keyed "sb"/"bb" -> {"mean": float, "std": float, "n": int}
    """
    exp.scenario_gen.seed(seed)
    exp.actor_critic.eval()

    results = {"sb": [], "bb": []}

    for _ in range(num_episodes):
        scenario = exp.scenario_gen.generate()
        obs_sb = exp.env.reset(scenario)
        rep_sb = exp.adapter.adapt(obs_sb)

        with torch.no_grad():
            result_sb = exp.actor_critic.act(
                SB_SEAT, rep_sb, obs_sb.legal_mask, deterministic=True
            )
        from actions.abstract_action import AbstractAction
        action_sb = AbstractAction(ActionType(result_sb.action_type))
        step1 = exp.env.step(action_sb)

        if step1.done:
            results["sb"].append(step1.rewards[SB_SEAT])
            results["bb"].append(step1.rewards[BB_SEAT])
            continue

        obs_bb = step1.obs
        rep_bb = exp.adapter.adapt(obs_bb)
        with torch.no_grad():
            result_bb = exp.actor_critic.act(
                BB_SEAT, rep_bb, obs_bb.legal_mask, deterministic=True
            )
        action_bb = AbstractAction(ActionType(result_bb.action_type))
        step2 = exp.env.step(action_bb)

        results["sb"].append(step2.rewards[SB_SEAT])
        results["bb"].append(step2.rewards[BB_SEAT])

    exp.actor_critic.train()

    out = {}
    for seat_name, rewards in results.items():
        out[seat_name] = {
            "mean": statistics.mean(rewards),
            "std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "n": len(rewards),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Push-fold-duo PPO training demo.")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes-per-iteration", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--variant", type=str, default="shared_trunk_two_heads",
        choices=[v.value for v in PolicyVariant],
    )
    parser.add_argument(
        "--mode", type=str, default="simultaneous",
        choices=[m.value for m in DuoTrainingMode],
    )
    parser.add_argument("--iters-per-seat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--checkpoint-in", type=str, default=None,
                         help="Resume training / evaluate from an existing checkpoint.")
    parser.add_argument("--log-every", type=int, default=1)
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
    args = parser.parse_args()

    cfg = PushFoldDuoExperimentConfig(
        big_blind=args.big_blind,
        seed=args.seed,
        variant=PolicyVariant(args.variant),
        mode=DuoTrainingMode(args.mode),
        iters_per_seat=args.iters_per_seat,
        learning_rate_sb=args.learning_rate,
        learning_rate_bb=args.learning_rate,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
    )

    _print_header("BUILDING EXPERIMENT")
    print(f"variant: {args.variant}   mode: {args.mode}")
    hand_encoder = _load_hand_encoder(
        args.encoder_path,
        expected_embedding_dim=args.encoder_embedding_dim or None,
    )
    print(f"config: {cfg}")
    exp = build_push_fold_duo_experiment(cfg, hand_encoder=hand_encoder)

    if args.checkpoint_in:
        load_checkpoint(exp, args.checkpoint_in)
        print(f"Resumed from checkpoint {args.checkpoint_in}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    _print_header("TRAINING")
    header = (
        f"{'iter':>4}  {'active':>8}  {'reward_sb':>10}  {'reward_bb':>10}  "
        f"{'policy_loss':>28}  {'time/it':>8}"
    )
    print(header)
    print("-" * len(header))

    start = time.time()
    for i in range(cfg.num_iterations):
        t0 = time.time()

        if cfg.mode == DuoTrainingMode.SIMULTANEOUS:
            stats = exp.loop._iteration_simultaneous(
                i, cfg.episodes_per_iteration, cfg.epochs, cfg.minibatch_size
            )
        elif cfg.mode == DuoTrainingMode.ALTERNATING:
            stats = exp.loop._iteration_alternating(
                i, cfg.episodes_per_iteration, cfg.epochs, cfg.minibatch_size
            )
        else:
            stats = exp.loop._iteration_league(
                i, cfg.episodes_per_iteration, cfg.epochs, cfg.minibatch_size
            )

        dt = time.time() - t0

        if i % args.log_every == 0 or i == cfg.num_iterations - 1:
            active = stats.active_seat if stats.active_seat is not None else "both"
            print(
                f"{i:>4}  {str(active):>8}  {stats.mean_reward_sb:>+10.3f}  "
                f"{stats.mean_reward_bb:>+10.3f}  "
                f"{_fmt_loss_dict(stats.policy_loss):>28}  {dt:>7.2f}s"
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
    # Self-play evaluation
    # ------------------------------------------------------------------
    _print_header(
        f"SELF-PLAY EVALUATION  ({args.eval_episodes} hands, seed={args.seed}0)"
    )
    print("NOTE: no fixed-baseline comparison table yet — see module docstring.")

    eval_results = _evaluate_self_play(exp, args.eval_episodes, seed=args.seed * 10)

    for seat_name in ("sb", "bb"):
        r = eval_results[seat_name]
        bb100 = _bb_per_100(r["mean"], args.big_blind)
        print(
            f"  seat {seat_name.upper()}: reward={r['mean']:+.3f} "
            f"(std={r['std']:.3f})  bb/100={bb100:+.2f}  n={r['n']}"
        )

    zero_sum_check = eval_results["sb"]["mean"] + eval_results["bb"]["mean"]
    print(f"\nSB+BB mean reward (should be ~0, heads-up zero-sum): "
          f"{zero_sum_check:+.4f}")


if __name__ == "__main__":
    main()
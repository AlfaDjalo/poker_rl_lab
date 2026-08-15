"""
run_generic_tree_demo.py — Trains a two-seat PPO agent pair against any
scenarios/duo_games/<config_name>.yaml config, logs per-iteration
training stats, checkpoints, and reports a self-play evaluation.

Generalizes run_push_fold_duo_demo.py (hardcoded to push-fold-duo,
SB_SEAT/BB_SEAT) into one script that trains EITHER push-fold-duo OR
river-duo (or any future scenarios/duo_games/*.yaml) by name, via
experiments.generic_tree_experiment.build_generic_tree_experiment().

No fixed-baseline comparison table (same caveat as
run_push_fold_duo_demo.py's own docstring)
--------------------------------------------------------------------------
evaluation/push_fold_duo_baselines.py and evaluation/river_duo_baselines.py
already provide rule-based baselines + a paired-seed evaluation harness
for their respective games, but both are hardcoded to their own env
class (PushFoldDuoEnv / RiverDuoEnv) and seat constants
(SB_SEAT/BB_SEAT, OOP_SEAT/IP_SEAT) — not GenericTreeEnv-generic. Rather
than force those through this script (or duplicate them generically
before there's a second use case), this demo's "EVALUATION" section
plays the trained pair against ITSELF deterministically through
GenericTreeEnv/GenericTreeScenarioGenerator (config-agnostic, since both
are already N-seat-generic — see their own docstrings) and reports each
seat's mean reward / bb-per-100. That is enough to sanity-check the
pipeline learns something sane (e.g. push-fold SB should push a
positive-EV range) without needing a baseline table wired through
GenericTreeEnv specifically. A generic-tree baseline module is a
natural follow-up once a second config beyond push-fold/river exists to
validate the abstraction against (mirrors
RL_Lab_Scenario_Framework - Addendum.md §3.3's own "prove it against
something real before generalizing further" sequencing).

Setup notes
-----------
  - poker_eval shim installed below (mock — see mock_poker_eval.py),
    same pattern as every other run_*_demo.py entry point.
  - Requires `triton` NOT installed alongside tensorflow (see
    run_push_fold_duo_demo.py's identical setup note).
  - Run with KMP_DUPLICATE_LIB_OK=TRUE in the environment if you hit
    the usual OpenMP duplicate-runtime crash from mixing TF and PyTorch.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_generic_tree_demo.py \
        --config-name push_fold_duo --iterations 20
    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python3 run_generic_tree_demo.py \
        --config-name river_duo --variant shared_single_head \
        --mode alternating --iterations 50 --checkpoint-out ckpt_river.pt \
        --device cuda
"""
import sys

try:
    import poker_engine.poker_eval  # noqa: F401
    print("Running true poker eval.")
except ImportError:
    import mock_poker_eval
    print("Running true poker eval.")
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse
import statistics
import time

import torch

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.experiments.generic_tree_experiment import (
    GenericTreeExperimentConfig,
    PolicyVariant,
    build_generic_tree_experiment,
    load_checkpoint,
    save_checkpoint,
)
from poker_rl_lab.trainers.duo_training_loop import DuoTrainingMode

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


def _evaluate_self_play(exp, num_episodes: int, seed: int):
    """
    Play the trained pair against itself deterministically through
    GenericTreeEnv for `num_episodes` hands and report per-seat reward
    stats. Config- and seat-count-generic — walks env.current_seat
    exactly like evaluation/*_duo_baselines.py's evaluate_duo_matchup()
    does, just querying the same actor_critic for every seat instead of
    per-seat baseline callables.

    Returns
    -------
    Dict[int, dict] keyed by seat index -> {"mean": float, "std": float, "n": int}
    """
    exp.scenario_gen.seed(seed)
    exp.actor_critic.eval()

    seat_indices = [s.index for s in exp.game_config.seats]
    results = {s: [] for s in seat_indices}

    for _ in range(num_episodes):
        scenario = exp.scenario_gen.generate()
        obs = exp.env.reset(scenario)

        while True:
            seat = exp.env.current_seat
            rep_input = exp.adapter.adapt(obs)
            with torch.no_grad():
                result = exp.actor_critic.act(
                    seat, rep_input, obs.legal_mask, deterministic=True
                )
            action = AbstractAction(ActionType(result.action_type))
            step = exp.env.step(action)
            if step.done:
                for s in seat_indices:
                    results[s].append(step.rewards[s])
                break
            obs = step.obs

    exp.actor_critic.train()

    out = {}
    for seat, rewards in results.items():
        out[seat] = {
            "mean": statistics.mean(rewards),
            "std": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
            "n": len(rewards),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Generic-tree PPO training demo.")
    parser.add_argument("--config-name", type=str, default="push_fold_duo",
                         help="Name of scenarios/duo_games/<config_name>.yaml.")
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
        "--device", type=str, default="cpu",
        help="PyTorch device for the actor_critic, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    args = parser.parse_args()

    cfg = GenericTreeExperimentConfig(
        config_name=args.config_name,
        seed=args.seed,
        variant=PolicyVariant(args.variant),
        mode=DuoTrainingMode(args.mode),
        iters_per_seat=args.iters_per_seat,
        learning_rate=args.learning_rate,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )

    _print_header("BUILDING EXPERIMENT")
    print(f"config-name: {args.config_name}   variant: {args.variant}   "
          f"mode: {args.mode}   device: {args.device}")
    print(f"config: {cfg}")
    exp = build_generic_tree_experiment(cfg)
    print(f"seats: {[(s.id, s.index) for s in exp.game_config.seats]}")

    if args.checkpoint_in:
        load_checkpoint(exp, args.checkpoint_in)
        print(f"Resumed from checkpoint {args.checkpoint_in}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    _print_header("TRAINING")
    header = (
        f"{'iter':>4}  {'active':>8}  {'mean_reward_by_seat':>30}  "
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
            rewards_str = " ".join(
                f"s{seat}={r:+.3f}" for seat, r in stats.mean_reward_by_seat.items()
            )
            print(
                f"{i:>4}  {str(active):>8}  {rewards_str:>30}  "
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
    print("NOTE: no fixed-baseline comparison table yet for GenericTreeEnv "
          "— see module docstring.")

    eval_results = _evaluate_self_play(exp, args.eval_episodes, seed=args.seed * 10)

    total_mean = 0.0
    for seat in sorted(eval_results):
        r = eval_results[seat]
        bb100 = _bb_per_100(r["mean"], args.big_blind)
        seat_id = exp.game_config.seat_id_for_index(seat)
        print(
            f"  seat {seat} ({seat_id}): reward={r['mean']:+.3f} "
            f"(std={r['std']:.3f})  bb/100={bb100:+.2f}  n={r['n']}"
        )
        total_mean += r["mean"]

    print(f"\nSum of seat mean rewards (should equal the config's dead-pot "
          f"contribution, 0 for 'blinds'-mode games like push-fold-duo): "
          f"{total_mean:+.4f}")


if __name__ == "__main__":
    main()
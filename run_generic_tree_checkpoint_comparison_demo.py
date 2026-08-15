"""
run_generic_tree_checkpoint_comparison_demo.py — Compares two saved
GenericTreeExperiment checkpoints head-to-head, on whichever
scenarios/duo_games/<config_name>.yaml config they were trained on.

Config-generic analog of run_checkpoint_comparison_demo.py (which
compares two single-hero PPOTrainer checkpoints via
evaluation.policy_evaluator.compare_policies). That approach doesn't
apply here — GenericTreeEnv has no single hero, every seat is being
optimised, and env.step()'s reward is per-seat (see
run_generic_tree_demo.py's own "no fixed-baseline comparison table
yet" docstring note). So this script plays the SAME two checkpoints
against each other with seats SWAPPED across two matchups (checkpoint
A as seat 0 / checkpoint B as seat 1, then the reverse), reseeding the
scenario generator identically before each matchup — the same
paired-samples variance-reduction trick
evaluation.policy_evaluator.compare_policies uses, just applied across
two matchups instead of across N policies.

KNOWN LIMITATION — 2-seat configs only
------------------------------------------
models.dual_seat_policy.DualSeatActorCritic (and both its concrete
implementations) validate `seat in (0, 1)` via `_check_seat` — see that
module's own docstring. This script raises a clear error up front for
any config declaring seats != 2, rather than letting a 3+ seat config
fail confusingly three layers down inside actor_critic.act(). Every
config shipped today (push_fold_duo, river_duo, turn_river_duo) is
2-seat, so this is not a practical restriction yet.

Design
------
Both checkpoints must come from the SAME config_name and the SAME
PolicyVariant/ActorCriticConfig — save_checkpoint() only stores
actor_critic.state_dict() (+ optimizer state), not the architecture
that produced it (see experiments/generic_tree_experiment.py's
save_checkpoint), so build_generic_tree_experiment() must be called
with matching hyperparameters before load_state_dict() can succeed.
This script builds two full experiments with the SAME
GenericTreeExperimentConfig (only --checkpoint-a/--checkpoint-b/
--label-a/--label-b differ) purely to get a correctly-shaped
actor_critic + adapter per checkpoint — the trainer/optimizer each
build produces is unused and discarded here (no gradient step is ever
taken).

Setup notes
-----------
  - poker_eval shim installed below (mock — see mock_poker_eval.py).
  - Requires `triton` NOT installed alongside tensorflow (see the other
    run_*_demo.py scripts' setup notes).

Usage
-----
    PYTHONPATH=. python3 run_generic_tree_checkpoint_comparison_demo.py \
        --config-name push_fold_duo --checkpoint-a ckpt_old.pt --checkpoint-b ckpt_new.pt
    PYTHONPATH=. python3 run_generic_tree_checkpoint_comparison_demo.py \
        --config-name river_duo --variant shared_single_head \
        --checkpoint-a a.pt --checkpoint-b b.pt --label-a iter20 --label-b iter50 \
        --eval-episodes 5000
"""
import sys

try:
    import poker_engine.poker_eval  # noqa: F401
except ImportError:
    import mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse
import math
import statistics
from typing import Dict, List, Tuple

import torch

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.experiments.generic_tree_experiment import (
    GenericTreeExperimentConfig,
    PolicyVariant,
    build_generic_tree_experiment,
    load_checkpoint,
)
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _load_actor_critic_only(cfg: GenericTreeExperimentConfig, path: str):
    """
    Build a fresh (env, scenario_gen, adapter, actor_critic) bundle for
    `cfg` and load `path`'s actor_critic weights into it. The
    trainer/optimizer build_generic_tree_experiment() also constructs
    is discarded — this script never takes a gradient step, only runs
    inference.
    """
    exp = build_generic_tree_experiment(cfg)
    load_checkpoint(exp, path)
    exp.actor_critic.eval()
    return exp


def play_matchup(
    env,
    scenario_gen,
    adapter,
    policy_by_seat: Dict[int, DualSeatActorCritic],
    num_episodes: int,
    seed: int,
) -> Dict[int, List[float]]:
    """
    Play `num_episodes` hands with each seat's action chosen by
    whichever DualSeatActorCritic policy_by_seat[seat] names, walking
    env.current_seat exactly like run_generic_tree_demo.py's own
    _evaluate_self_play — generalized here to two DIFFERENT checkpoints
    instead of one shared actor_critic.

    Returns
    -------
    Dict[int, List[float]]
        Per-episode reward list for each seat index.
    """
    scenario_gen.seed(seed)
    for policy in policy_by_seat.values():
        policy.eval()

    results: Dict[int, List[float]] = {s: [] for s in policy_by_seat}

    for _ in range(num_episodes):
        scenario = scenario_gen.generate()
        obs = env.reset(scenario)

        while True:
            seat = env.current_seat
            rep_input = adapter.adapt(obs)
            with torch.no_grad():
                result = policy_by_seat[seat].act(
                    seat, rep_input, obs.legal_mask, deterministic=True
                )
            action = AbstractAction(ActionType(result.action_type))
            step = env.step(action)
            if step.done:
                for s in policy_by_seat:
                    results[s].append(step.rewards[s])
                break
            obs = step.obs

    return results


def _stats(rewards: List[float], big_blind: int) -> Tuple[float, float, float, float]:
    """Returns (mean, std_error, bb_per_100, win_rate)."""
    n = len(rewards)
    mean = statistics.mean(rewards)
    std = statistics.pstdev(rewards) if n > 1 else 0.0
    stderr = std / math.sqrt(n) if n > 1 else 0.0
    win_rate = sum(1 for r in rewards if r > 0) / n
    return mean, stderr, mean / big_blind * 100.0, win_rate


def main():
    parser = argparse.ArgumentParser(
        description="Compare two GenericTreeEnv checkpoints head-to-head, seats swapped."
    )
    parser.add_argument("--config-name", type=str, default="push_fold_duo",
                         help="Name of scenarios/duo_games/<config_name>.yaml.")
    parser.add_argument("--checkpoint-a", type=str, required=True)
    parser.add_argument("--checkpoint-b", type=str, required=True)
    parser.add_argument("--label-a", type=str, default="checkpoint_a")
    parser.add_argument("--label-b", type=str, default="checkpoint_b")
    parser.add_argument(
        "--variant", type=str, default="shared_trunk_two_heads",
        choices=[v.value for v in PolicyVariant],
        help="Must match the architecture BOTH checkpoints were trained with.",
    )
    parser.add_argument("--seed", type=int, default=7,
                         help="Config seed for env/experiment construction defaults "
                              "(not the evaluation deal sequence — see --eval-seed).")
    parser.add_argument("--eval-episodes", type=int, default=2000,
                         help="Hands per matchup. Two matchups are played (seats swapped).")
    parser.add_argument("--eval-seed", type=int, default=99,
                         help="Deal sequence seed, shared by BOTH matchups (paired samples).")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for both actor_critics, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    args = parser.parse_args()

    _print_header("BUILDING")
    cfg = GenericTreeExperimentConfig(
        config_name=args.config_name,
        seed=args.seed,
        variant=PolicyVariant(args.variant),
        device=args.device,
    )
    print(f"config-name: {args.config_name}   variant: {args.variant}   device: {args.device}")

    exp_a = _load_actor_critic_only(cfg, args.checkpoint_a)
    exp_b = _load_actor_critic_only(cfg, args.checkpoint_b)
    print(f"loaded {args.label_a!r} from {args.checkpoint_a}")
    print(f"loaded {args.label_b!r} from {args.checkpoint_b}")

    seat_indices = [s.index for s in exp_a.game_config.seats]
    if len(seat_indices) != 2:
        raise ValueError(
            f"This script only supports 2-seat configs today (see module "
            f"docstring's KNOWN LIMITATION) — {args.config_name!r} declares "
            f"{len(seat_indices)} seats."
        )
    seat0, seat1 = sorted(seat_indices)
    big_blind = exp_a.game_config.big_blind

    # Both checkpoints' envs/scenario_gens are architecturally identical
    # builds of the same config — exp_a's env/scenario_gen/adapter drive
    # every matchup here (same object -> same seeded RNG stream when
    # reseeded, matching evaluation.policy_evaluator.compare_policies'
    # own paired-seed convention).
    env = exp_a.env
    scenario_gen = exp_a.scenario_gen
    adapter = exp_a.adapter

    _print_header(
        f"EVALUATION  ({args.eval_episodes} hands/matchup x 2 matchups, "
        f"paired eval-seed={args.eval_seed})"
    )

    print(f"Matchup 1: {args.label_a} as seat {seat0}, {args.label_b} as seat {seat1}")
    matchup1 = play_matchup(
        env, scenario_gen, adapter,
        policy_by_seat={seat0: exp_a.actor_critic, seat1: exp_b.actor_critic},
        num_episodes=args.eval_episodes, seed=args.eval_seed,
    )

    print(f"Matchup 2: {args.label_b} as seat {seat0}, {args.label_a} as seat {seat1}")
    matchup2 = play_matchup(
        env, scenario_gen, adapter,
        policy_by_seat={seat0: exp_b.actor_critic, seat1: exp_a.actor_critic},
        num_episodes=args.eval_episodes, seed=args.eval_seed,
    )

    # Pool each checkpoint's per-hand rewards across both matchups,
    # regardless of which physical seat it occupied that matchup.
    rewards_a = matchup1[seat0] + matchup2[seat1]
    rewards_b = matchup1[seat1] + matchup2[seat0]

    mean_a, stderr_a, bb100_a, win_a = _stats(rewards_a, big_blind)
    mean_b, stderr_b, bb100_b, win_b = _stats(rewards_b, big_blind)

    _print_header("RESULTS (pooled over both seat assignments)")
    name_w = max(len(args.label_a), len(args.label_b))
    print(f"{'checkpoint':<{name_w}}  {'reward':>8}  {'95% CI':>18}  "
          f"{'bb/100':>8}  {'win%':>6}  n")
    print("-" * (name_w + 60))
    for label, mean, stderr, bb100, win in (
        (args.label_a, mean_a, stderr_a, bb100_a, win_a),
        (args.label_b, mean_b, stderr_b, bb100_b, win_b),
    ):
        margin = 1.96 * stderr
        print(
            f"{label:<{name_w}}  {mean:>+8.3f}  "
            f"[{mean - margin:>+7.3f}, {mean + margin:>+7.3f}]  {bb100:>+8.2f}  "
            f"{win:>5.1%}  {len(rewards_a)}"
        )

    zero_sum_check = mean_a + mean_b
    print(f"\nMean(A) + Mean(B) (should be ~0, heads-up zero-sum): "
          f"{zero_sum_check:+.4f}")

    print()
    margin = bb100_b - bb100_a
    verdict = (
        f"{args.label_b} beats {args.label_a}"
        if margin > 0
        else f"{args.label_b} does NOT beat {args.label_a}"
    )
    print(f"{verdict} by {margin:+.2f} bb/100 (pooled over both seat assignments).")


if __name__ == "__main__":
    main()
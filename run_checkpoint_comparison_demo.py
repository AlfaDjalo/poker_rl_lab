"""
run_checkpoint_comparison_demo.py — Compares two saved PPO checkpoints
against each other (and against the fixed baselines), on whichever
curriculum stage they were trained on.

Per RL_lab_implementation_guide.md §10 ("Evaluation"):

    Policies should be benchmarked against:
        Random, Rule Based, Previous Checkpoints, Current Best

The other run_*_demo.py scripts already cover Random/Rule Based (via
evaluation.baseline_policies) against a freshly-trained policy. This
script is the "Previous Checkpoints" / "Current Best" leg: two
checkpoint files, same architecture, evaluated head-to-head under
identical dealt hands (evaluation.policy_evaluator.compare_policies'
paired-seed variance reduction — see that module's docstring).

Design
------
Both checkpoints must come from the SAME curriculum stage (push_fold /
river / full_holdem) and the SAME ActorCriticConfig — save_checkpoint()
only stores actor_critic.state_dict()/optimizer.state_dict(), not the
architecture that produced it (see experiments/push_fold_experiment.py's
save_checkpoint docstring), so build_actor_critic() must be called with
matching hyperparameters before load_state_dict() can succeed. This
script reuses each stage's own build_*_experiment() (with default
hyperparameters) purely to get a correctly-shaped actor_critic + adapter
per checkpoint — the trainer/optimizer that build produces is unused and
discarded.

Setup notes
-----------
  - poker_eval shim installed below (mock — see mock_poker_eval.py). Must
    be installed before envs.full_holdem_env is imported (module-level
    PokerState import) when --stage full_holdem is selected.
  - Requires `triton` NOT installed alongside tensorflow — see the other
    run_*_demo.py scripts' setup notes.

Usage
-----
    PYTHONPATH=. python3 run_checkpoint_comparison_demo.py \
        --stage push_fold --checkpoint-a ckpt_old.pt --checkpoint-b ckpt_new.pt
    PYTHONPATH=. python3 run_checkpoint_comparison_demo.py \
        --stage river --checkpoint-a ckpt_iter20.pt --checkpoint-b ckpt_iter50.pt \
        --label-a iter20 --label-b iter50
    PYTHONPATH=. python3 run_checkpoint_comparison_demo.py \
        --stage full_holdem --checkpoint-a a.pt --checkpoint-b b.pt --player-count 6
"""
import sys

try:
    import poker_eval  # noqa: F401
except ImportError:
    import poker_rl_lab.mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse

import torch

from poker_rl_lab.actions.abstract_action import ActionType
from poker_rl_lab.envs.base_env import RandomHeroPolicy
from poker_rl_lab.evaluation.baseline_policies import always_fold_policy, calling_station_policy
from poker_rl_lab.evaluation.policy_evaluator import compare_policies, policy_from_actor_critic

_ACTION_ORDER = [a.name for a in ActionType]


def _fmt_action_freqs(freqs: dict) -> str:
    parts = [f"{name}={freqs[name]:.2f}" for name in _ACTION_ORDER if name in freqs]
    return " ".join(parts) if parts else "(no actions recorded)"


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _load_actor_critic_only(path: str, build_fn, cfg):
    """
    Build a fresh (env, scenario_gen, adapter, actor_critic) bundle for
    `cfg`'s stage and load `path`'s actor_critic weights into it. The
    trainer/optimizer build_fn also constructs is discarded — this
    script never takes a gradient step, only runs inference.
    """
    exp = build_fn(cfg)
    checkpoint = torch.load(path, map_location="cpu")
    exp.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    exp.actor_critic.eval()
    return exp


def _build_stage(args):
    """
    Returns (build_fn, cfg, env, scenario_gen) for the chosen --stage,
    using each stage's own default experiment config (only the knobs
    this script exposes are overridden). env/scenario_gen come from
    checkpoint A's build (arbitrary choice — both checkpoints must share
    architecture/env config for a fair comparison, so either build's env
    is equally valid; see module docstring).
    """
    if args.stage == "push_fold":
        from poker_rl_lab.experiments.push_fold_experiment import (
            build_push_fold_experiment as build_fn,
            PushFoldExperimentConfig as ConfigCls,
        )
        cfg = ConfigCls(big_blind=args.big_blind, seed=args.seed)

    elif args.stage == "river":
        from poker_rl_lab.experiments.river_experiment import (
            build_river_experiment as build_fn,
            RiverExperimentConfig as ConfigCls,
        )
        cfg = ConfigCls(big_blind=args.big_blind, seed=args.seed)

    elif args.stage == "full_holdem":
        from poker_rl_lab.experiments.full_holdem_experiment import (
            build_full_holdem_experiment as build_fn,
            FullHoldemExperimentConfig as ConfigCls,
        )
        cfg = ConfigCls(
            player_count=args.player_count,
            big_blind=args.big_blind,
            seed=args.seed,
        )

    else:
        raise ValueError(f"Unknown stage {args.stage!r}")

    return build_fn, cfg


def main():
    parser = argparse.ArgumentParser(
        description="Compare two PPO checkpoints head-to-head, plus baselines."
    )
    parser.add_argument("--stage", type=str, default="push_fold",
                         choices=["push_fold", "river", "full_holdem"])
    parser.add_argument("--checkpoint-a", type=str, required=True)
    parser.add_argument("--checkpoint-b", type=str, required=True)
    parser.add_argument("--label-a", type=str, default="checkpoint_a")
    parser.add_argument("--label-b", type=str, default="checkpoint_b")
    parser.add_argument("--player-count", type=int, default=6,
                         help="Only used when --stage full_holdem.")
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7,
                         help="Config seed for env construction defaults "
                              "(not the evaluation deal sequence — see --eval-seed).")
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--eval-seed", type=int, default=99,
                         help="Deal sequence seed shared by every policy in "
                              "the comparison (paired samples).")
    parser.add_argument("--include-baselines", action="store_true",
                         default=True)
    parser.add_argument("--no-baselines", dest="include_baselines",
                         action="store_false")
    args = parser.parse_args()

    _print_header("BUILDING")
    build_fn, cfg = _build_stage(args)
    print(f"stage: {args.stage}")
    print(f"config: {cfg}")

    exp_a = _load_actor_critic_only(args.checkpoint_a, build_fn, cfg)
    exp_b = _load_actor_critic_only(args.checkpoint_b, build_fn, cfg)
    print(f"loaded {args.label_a!r} from {args.checkpoint_a}")
    print(f"loaded {args.label_b!r} from {args.checkpoint_b}")

    policy_a = policy_from_actor_critic(exp_a.actor_critic, exp_a.adapter, deterministic=True)
    policy_b = policy_from_actor_critic(exp_b.actor_critic, exp_b.adapter, deterministic=True)

    policies = {
        args.label_a: policy_a,
        args.label_b: policy_b,
    }
    if args.include_baselines:
        policies["random"] = RandomHeroPolicy(seed=args.eval_seed)
        policies["always_fold"] = always_fold_policy
        policies["calling_station"] = calling_station_policy

    _print_header(
        f"EVALUATION  ({args.eval_episodes} hands/policy, "
        f"paired eval-seed={args.eval_seed})"
    )
    # Both checkpoints' envs are architecturally identical builds of the
    # same stage/config; env A drives every policy's episodes here so
    # the deal sequence is genuinely shared (same RiverEnv/PushFoldEnv/
    # FullHoldemEnv instance, same scenario_gen instance -> same seeded
    # RNG stream) across every entry in `policies`.
    results = compare_policies(
        exp_a.env,
        exp_a.scenario_gen,
        policies,
        num_episodes=args.eval_episodes,
        big_blind=args.big_blind,
        seed=args.eval_seed,
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
    a_bb100 = results[args.label_a].bb_per_100
    b_bb100 = results[args.label_b].bb_per_100
    margin = b_bb100 - a_bb100
    verdict = f"{args.label_b} beats {args.label_a}" if margin > 0 else f"{args.label_b} does NOT beat {args.label_a}"
    print(f"{verdict} by {margin:+.2f} bb/100.")


if __name__ == "__main__":
    main()
"""
experiments/full_holdem_experiment.py — End-to-end training entry point for
the RL Lab's third curriculum stage: full-table hold'em, played out by the
real Engine.

Per RL_lab_implementation_guide.md's "Curriculum Learning" progression
(§1, "Core Principles"):

    1. Push/Fold
    2. River Hold'em
    3. Full Hold'em      <-- this module
    ...

Mirrors experiments/push_fold_experiment.py and experiments/river_experiment.py
in structure — FullHoldemScenarioGenerator -> FullHoldemEnv ->
ObservationAdapter -> build_actor_critic() -> PPOTrainer.train() — differing
only in:

  - the scenario generator / environment used (envs.full_holdem_env.FullHoldemEnv
    is Engine-backed; see that module's docstring for why it is structurally
    different from PushFoldEnv/RiverEnv), and
  - villain configuration: FullHoldemEnv supports a `default_villain_policy`
    (applied to every non-hero seat) plus optional per-seat overrides via
    `villain_policies`. This config only exposes the uniform default —
    per-seat overrides are a construction-time knob for callers who need
    them (e.g. mixed-difficulty self-play), not something the config
    dataclass needs to parameterise yet.
  - one extra generator knob relative to push_fold/river: `player_count`
    here drives a genuinely variable-size full table (2-9+ seats), not just
    "how many already-folded seats to pad the scenario with".

See push_fold_experiment.py's own docstring for the full rationale behind
the poker_eval shim and the untrained-hand-encoder caveat below — identical
here, not repeated in full.

Device placement
------------------
`FullHoldemExperimentConfig.device` (default "cpu") selects where the
PyTorch actor_critic lives — see push_fold_experiment.py's identical
"Device placement" section for the full rationale.

Dependencies
------------
    PyTorch. scenarios.full_holdem_generator, envs.full_holdem_env,
    envs.river_env (CheckFoldVillainPolicy default only — no other
    river-specific coupling), representation.observation_adapter,
    models.actor_critic_factory, models.policy_value_network,
    trainers.ppo_trainer, actions.abstract_action. loader / rules.* /
    mock_poker_eval — see push_fold_experiment.py's "poker_eval shim"
    section. Unlike push_fold_experiment.py / river_experiment.py, this
    module ALSO needs the poker_eval shim installed before importing
    envs.full_holdem_env itself (that module imports PokerState at
    module level — see its own docstring's "poker_eval shim" section),
    not just before importing loader/rules.
"""

from __future__ import annotations

import sys

try:
    import poker_eval  # noqa: F401  (prefer the real extension if present)
except ImportError:
    import mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from envs.full_holdem_env import FullHoldemEnv, VillainPolicyLike
from envs.river_env import CheckFoldVillainPolicy
from models.actor_critic_factory import ActorCriticConfig, build_actor_critic
from models.policy_value_network import PokerActorCritic
from representation.base_hand_encoder import BaseHandEncoder
from representation.observation_adapter import ObservationAdapter
from scenarios.full_holdem_generator import FullHoldemScenarioGenerator
from trainers.ppo_trainer import IterationStats, PPOTrainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FullHoldemExperimentConfig:
    """All knobs for one full-table hold'em training run, grouped by the
    piece of the pipeline they configure."""

    # FullHoldemScenarioGenerator
    player_count: int = 6
    small_blind: int = 1
    big_blind: int = 2
    ante: int = 0
    hole_cards_per_player: int = 2
    node_count: int = 5
    min_stack_bb: float = 20.0
    max_stack_bb: float = 100.0
    game_name: str = "holdem"
    dealer_position: int = 0
    seed: Optional[int] = 42

    # FullHoldemEnv
    default_villain_policy: object = field(default_factory=CheckFoldVillainPolicy)
    scoring_engine: object = None  # None -> FullHoldemEnv's own MockScoringEngine default

    # Device placement — see module docstring's "Device placement" section.
    device: str = "cpu"

    # build_actor_critic()
    actor_critic_config: ActorCriticConfig = field(default_factory=ActorCriticConfig)

    # PPOTrainer
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: Optional[float] = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95

    # PPOTrainer.train()
    num_iterations: int = 100
    episodes_per_iteration: int = 256
    epochs: int = 4
    minibatch_size: int = 64


# ---------------------------------------------------------------------------
# Wired-up experiment bundle
# ---------------------------------------------------------------------------

@dataclass
class FullHoldemExperiment:
    """Every piece of one training/evaluation run, already wired together."""
    scenario_gen: FullHoldemScenarioGenerator
    env: FullHoldemEnv
    adapter: ObservationAdapter
    actor_critic: PokerActorCritic
    trainer: PPOTrainer
    config: FullHoldemExperimentConfig


def build_full_holdem_experiment(
    config: Optional[FullHoldemExperimentConfig] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
    villain_policies: Optional[Dict[int, VillainPolicyLike]] = None,
) -> FullHoldemExperiment:
    """
    Construct every piece of the full-holdem training pipeline, wired
    together but not yet run.

    Parameters
    ----------
    config : FullHoldemExperimentConfig, optional
        Defaults to FullHoldemExperimentConfig() if omitted.
        `config.device` selects where the PyTorch actor_critic is placed.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to models.actor_critic_factory.build_actor_critic() —
        see that function's docstring. Omit to get an UNTRAINED
        PretrainedHandEncoder (pipeline testing only); pass a real
        PretrainedHandEncoder.from_file(...) for an actual training run.
    villain_policies : Dict[int, VillainPolicyLike], optional
        Per-seat policy overrides forwarded straight to FullHoldemEnv —
        see that class's own docstring. Seats not present here use
        config.default_villain_policy. Kept as a separate build-time
        argument rather than a config field since per-seat overrides
        are keyed by seat index, which is only meaningful once
        config.player_count is fixed.

    Returns
    -------
    FullHoldemExperiment
    """
    cfg = config or FullHoldemExperimentConfig()

    scenario_gen = FullHoldemScenarioGenerator(
        player_count=cfg.player_count,
        small_blind=cfg.small_blind,
        big_blind=cfg.big_blind,
        ante=cfg.ante,
        hole_cards_per_player=cfg.hole_cards_per_player,
        node_count=cfg.node_count,
        min_stack_bb=cfg.min_stack_bb,
        max_stack_bb=cfg.max_stack_bb,
        game_name=cfg.game_name,
        dealer_position=cfg.dealer_position,
    )
    if cfg.seed is not None:
        scenario_gen.seed(cfg.seed)

    env_kwargs = dict(
        default_villain_policy=cfg.default_villain_policy,
        villain_policies=villain_policies,
    )
    if cfg.scoring_engine is not None:
        env_kwargs["scoring_engine"] = cfg.scoring_engine
    env = FullHoldemEnv(**env_kwargs)

    adapter = ObservationAdapter()

    actor_critic = build_actor_critic(
        hand_encoder=hand_encoder, config=cfg.actor_critic_config
    )

    # Move to the target device BEFORE constructing the optimizer — see
    # push_fold_experiment.py's identical note for the full rationale.
    actor_critic = actor_critic.to(cfg.device)

    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=cfg.learning_rate)

    trainer = PPOTrainer(
        actor_critic=actor_critic,
        adapter=adapter,
        optimizer=optimizer,
        clip_epsilon=cfg.clip_epsilon,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    return FullHoldemExperiment(
        scenario_gen=scenario_gen,
        env=env,
        adapter=adapter,
        actor_critic=actor_critic,
        trainer=trainer,
        config=cfg,
    )


def run_full_holdem_experiment(experiment: FullHoldemExperiment) -> List[IterationStats]:
    """
    Run experiment.config's configured number of PPO iterations.

    Returns
    -------
    List[IterationStats]
        One entry per iteration — see trainers.ppo_trainer.PPOTrainer.train().
    """
    cfg = experiment.config
    return experiment.trainer.train(
        experiment.env,
        experiment.scenario_gen,
        num_iterations=cfg.num_iterations,
        episodes_per_iteration=cfg.episodes_per_iteration,
        epochs=cfg.epochs,
        minibatch_size=cfg.minibatch_size,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(experiment: FullHoldemExperiment, path: Union[str, Path]) -> None:
    """
    Save actor_critic + optimizer state to `path`.

    See push_fold_experiment.save_checkpoint's docstring — identical
    TF/PyTorch boundary caveat applies (hand_encoder weights are not
    included; checkpoint those separately via Keras if ever fine-tuned).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_critic_state_dict": experiment.actor_critic.state_dict(),
            "optimizer_state_dict": experiment.trainer.optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(experiment: FullHoldemExperiment, path: Union[str, Path]) -> None:
    """
    Load actor_critic + optimizer state from `path` into `experiment`,
    in place (mutates experiment.actor_critic and
    experiment.trainer.optimizer).

    map_location is set to experiment.config.device so a checkpoint
    saved from a CUDA run can still be loaded on a CPU-only machine (or
    vice versa) without raising.
    """
    checkpoint = torch.load(Path(path), map_location=experiment.config.device)
    experiment.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    experiment.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a full-table hold'em PPO agent.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--player-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for the actor_critic, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    args = parser.parse_args()

    cfg = FullHoldemExperimentConfig(
        player_count=args.player_count,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )

    experiment = build_full_holdem_experiment(cfg)
    history = run_full_holdem_experiment(experiment)

    for i, stats in enumerate(history):
        print(
            f"iter {i:4d}  "
            f"reward={stats.mean_episode_reward:+.3f}  "
            f"policy_loss={stats.policy_loss:+.4f}  "
            f"value_loss={stats.value_loss:.4f}  "
            f"entropy={stats.entropy:.4f}  "
            f"clip_frac={stats.clip_fraction:.3f}"
        )

    if args.checkpoint_out:
        save_checkpoint(experiment, args.checkpoint_out)
        print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    _main()
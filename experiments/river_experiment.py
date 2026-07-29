"""
experiments/river_experiment.py — End-to-end training entry point for the
RL Lab's second curriculum stage: heads-up river betting.

Per RL_lab_implementation_guide.md's "Curriculum Learning" progression
(§1, "Core Principles"):

    1. Push/Fold
    2. River Hold'em     <-- this module
    3. Full Hold'em
    ...

Mirrors experiments/push_fold_experiment.py exactly in structure —
RiverScenarioGenerator -> RiverEnv -> ObservationAdapter ->
build_actor_critic() -> PPOTrainer.train() — differing only in:

  - the scenario generator / environment used, and
  - one extra knob: `villain_policy`, since RiverEnv (unlike
    PushFoldEnv) has a villain that acts every episode via a pluggable
    policy (see river_env.py's CheckFoldVillainPolicy /
    CallAnyVillainPolicy / FoldAnyVillainPolicy / RandomVillainPolicy).
    Defaults to CheckFoldVillainPolicy, matching RiverEnv's own default.

See push_fold_experiment.py's own docstring for the full rationale
behind the poker_eval shim and the untrained-hand-encoder caveat below
— identical here, not repeated in full.

Device placement
------------------
`RiverExperimentConfig.device` (default "cpu") selects where the
PyTorch actor_critic lives — see push_fold_experiment.py's identical
"Device placement" section for the full rationale (hand encoder stays
TF-native regardless; move happens before optimizer construction).

Dependencies
------------
    PyTorch. scenarios.river_generator, envs.river_env,
    representation.observation_adapter, models.actor_critic_factory,
    models.policy_value_network, trainers.ppo_trainer,
    actions.abstract_action. loader / rules.* / mock_poker_eval — see
    push_fold_experiment.py's "poker_eval shim" section.
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
from typing import List, Optional, Union

import torch

from envs.river_env import (
    CheckFoldVillainPolicy,
    RiverEnv,
)
from models.actor_critic_factory import ActorCriticConfig, build_actor_critic
from models.policy_value_network import PokerActorCritic
from representation.base_hand_encoder import BaseHandEncoder
from representation.observation_adapter import ObservationAdapter
from scenarios.river_generator import RiverScenarioGenerator
from trainers.ppo_trainer import IterationStats, PPOTrainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RiverExperimentConfig:
    """All knobs for one river-holdem training run, grouped by the piece
    of the pipeline they configure."""

    # RiverScenarioGenerator
    player_count: int = 2
    small_blind: int = 1
    big_blind: int = 2
    hole_cards_per_player: int = 2
    min_pot_bb: float = 2.0
    max_pot_bb: float = 20.0
    min_stack_bb: float = 2.0
    max_stack_bb: float = 50.0
    seed: Optional[int] = 42

    # RiverEnv
    max_raises: int = 4
    villain_policy: object = field(default_factory=CheckFoldVillainPolicy)
    evaluator: object = None  # None -> RiverEnv's own MockEvaluator default

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
class RiverExperiment:
    """Every piece of one training/evaluation run, already wired together."""
    scenario_gen: RiverScenarioGenerator
    env: RiverEnv
    adapter: ObservationAdapter
    actor_critic: PokerActorCritic
    trainer: PPOTrainer
    config: RiverExperimentConfig


def build_river_experiment(
    config: Optional[RiverExperimentConfig] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
) -> RiverExperiment:
    """
    Construct every piece of the river-holdem training pipeline, wired
    together but not yet run.

    Parameters
    ----------
    config : RiverExperimentConfig, optional
        Defaults to RiverExperimentConfig() if omitted. `config.device`
        selects where the PyTorch actor_critic is placed.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to models.actor_critic_factory.build_actor_critic() —
        see that function's docstring. Omit to get an UNTRAINED
        PretrainedHandEncoder (pipeline testing only); pass a real
        PretrainedHandEncoder.from_file(...) for an actual training run.

    Returns
    -------
    RiverExperiment
    """
    cfg = config or RiverExperimentConfig()

    scenario_gen = RiverScenarioGenerator(
        player_count=cfg.player_count,
        small_blind=cfg.small_blind,
        big_blind=cfg.big_blind,
        hole_cards_per_player=cfg.hole_cards_per_player,
        min_pot_bb=cfg.min_pot_bb,
        max_pot_bb=cfg.max_pot_bb,
        min_stack_bb=cfg.min_stack_bb,
        max_stack_bb=cfg.max_stack_bb,
    )
    if cfg.seed is not None:
        scenario_gen.seed(cfg.seed)

    env_kwargs = dict(
        villain_policy=cfg.villain_policy,
        max_raises=cfg.max_raises,
    )
    if cfg.evaluator is not None:
        env_kwargs["evaluator"] = cfg.evaluator
    env = RiverEnv(**env_kwargs)
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

    return RiverExperiment(
        scenario_gen=scenario_gen,
        env=env,
        adapter=adapter,
        actor_critic=actor_critic,
        trainer=trainer,
        config=cfg,
    )


def run_river_experiment(experiment: RiverExperiment) -> List[IterationStats]:
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

def save_checkpoint(experiment: RiverExperiment, path: Union[str, Path]) -> None:
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


def load_checkpoint(experiment: RiverExperiment, path: Union[str, Path]) -> None:
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

    parser = argparse.ArgumentParser(description="Train a river-holdem PPO agent.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-raises", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for the actor_critic, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    args = parser.parse_args()

    cfg = RiverExperimentConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        max_raises=args.max_raises,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )

    experiment = build_river_experiment(cfg)
    history = run_river_experiment(experiment)

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
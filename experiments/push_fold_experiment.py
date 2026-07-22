"""
experiments/push_fold_experiment.py — End-to-end training entry point for
the RL Lab's first curriculum stage.

Per RL_lab_implementation_guide.md's "Curriculum Learning" progression
(§1, "Core Principles"):

    1. Push/Fold
    2. River Hold'em
    3. Full Hold'em
    ...

This module wires everything built so far into one runnable pipeline:

    PushFoldScenarioGenerator -> PushFoldEnv -> ObservationAdapter
        -> build_actor_critic() -> PPOTrainer.train()

matching the guide's §2 architecture diagram end to end for this one
curriculum stage. It is intentionally the only place in the RL Lab that
imports all of scenarios / envs / representation / models / trainers
together — everything else stays isolated per-layer (see each of those
modules' own docstrings on why).

poker_eval shim
----------------
ObservationAdapter needs the real Engine's game rules for the scenario's
game_name (rules.game_rules.GameRules, loaded via loader.load_game ->
poker_eval). The compiled poker_eval C++ extension is not part of this
checkout (RL Lab is a separate component from the Engine). Following
mock_poker_eval.py's own documented usage pattern, this module installs
that shim into sys.modules BEFORE importing anything that transitively
imports poker_eval (loader.py, rules.game_rules, rules.point_definition),
so the full pipeline is runnable and testable today.

Showdown resolution (win/loss/chop) is handled separately from the
poker_eval shim above: PushFoldEnv is constructed here with
envs.poker_eval_evaluator.PokerEvalEvaluator, which itself prefers the
real poker_eval extension and falls back to mock_poker_eval — see that
module's docstring. When only the mock is importable, showdown
outcomes carry no real poker knowledge (mock evaluation always returns
"player at the higher list index wins"); drop the real extension on
PYTHONPATH so `import poker_eval` succeeds and both this module and
PokerEvalEvaluator upgrade automatically, no code changes needed.
Note this is a genuinely different evaluator instance from the one
ObservationAdapter's rules/loader machinery may end up using — see
PokerEvalEvaluator's own docstring for why it deliberately avoids a
poker_engine dependency.

Similarly, the default (omitted) hand_encoder is an UNTRAINED
PretrainedHandEncoder — see build_actor_critic()'s docstring — so a run
using every default in this module exercises the full pipeline
end-to-end but does not yet produce a meaningfully-trained agent. Two
prerequisites for a real training run, in order of how much they
change results: (1) the real poker_eval extension, (2) the pretrained
encoder_model.keras file.

Dependencies
------------
    PyTorch. scenarios.push_fold_generator, envs.push_fold_env,
    representation.observation_adapter, models.actor_critic_factory,
    models.policy_value_network, trainers.ppo_trainer,
    actions.abstract_action. loader / rules.* / mock_poker_eval — see
    "poker_eval shim" above.
"""

from __future__ import annotations

import sys

try:
    import poker_engine.poker_eval  # noqa: F401  (prefer the real extension if present)
except ImportError:
    import mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch

from envs.push_fold_env import PushFoldEnv
from envs.poker_eval_evaluator import PokerEvalEvaluator
from models.actor_critic_factory import ActorCriticConfig, build_actor_critic
from models.policy_value_network import PokerActorCritic
from representation.base_hand_encoder import BaseHandEncoder
from representation.observation_adapter import ObservationAdapter
from scenarios.push_fold_generator import PushFoldScenarioGenerator
from trainers.ppo_trainer import IterationStats, PPOTrainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PushFoldExperimentConfig:
    """All knobs for one push-fold training run, grouped by the piece
    of the pipeline they configure."""

    # PushFoldScenarioGenerator
    player_count: int = 2
    small_blind: int = 1
    big_blind: int = 2
    min_stack_bb: float = 2.0
    max_stack_bb: float = 20.0
    starting_stack: int = 200
    seed: Optional[int] = 42

    # PokerEvalEvaluator — board-runout RNG for real showdown resolution.
    # Kept independent of `seed` (scenario dealing) so re-seeding one
    # doesn't silently change the other. Defaults to `seed` if omitted.
    evaluator_seed: Optional[int] = None

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
class PushFoldExperiment:
    """Every piece of one training/evaluation run, already wired together."""
    scenario_gen: PushFoldScenarioGenerator
    env: PushFoldEnv
    adapter: ObservationAdapter
    actor_critic: PokerActorCritic
    trainer: PPOTrainer
    config: PushFoldExperimentConfig


def build_push_fold_experiment(
    config: Optional[PushFoldExperimentConfig] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
) -> PushFoldExperiment:
    """
    Construct every piece of the push-fold training pipeline, wired
    together but not yet run.

    Parameters
    ----------
    config : PushFoldExperimentConfig, optional
        Defaults to PushFoldExperimentConfig() if omitted.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to models.actor_critic_factory.build_actor_critic() —
        see that function's docstring. Omit to get an UNTRAINED
        PretrainedHandEncoder (pipeline testing only); pass a real
        PretrainedHandEncoder.from_file(...) for an actual training run.

    Returns
    -------
    PushFoldExperiment
    """
    cfg = config or PushFoldExperimentConfig()

    scenario_gen = PushFoldScenarioGenerator(
        player_count=cfg.player_count,
        small_blind=cfg.small_blind,
        big_blind=cfg.big_blind,
        min_stack_bb=cfg.min_stack_bb,
        max_stack_bb=cfg.max_stack_bb,
        starting_stack=cfg.starting_stack,
    )
    if cfg.seed is not None:
        scenario_gen.seed(cfg.seed)

    env = PushFoldEnv(
        evaluator=PokerEvalEvaluator(
            seed=cfg.evaluator_seed if cfg.evaluator_seed is not None else cfg.seed
        )
    )
    adapter = ObservationAdapter()

    actor_critic = build_actor_critic(
        hand_encoder=hand_encoder, config=cfg.actor_critic_config
    )

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

    return PushFoldExperiment(
        scenario_gen=scenario_gen,
        env=env,
        adapter=adapter,
        actor_critic=actor_critic,
        trainer=trainer,
        config=cfg,
    )


def run_push_fold_experiment(experiment: PushFoldExperiment) -> List[IterationStats]:
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

def save_checkpoint(experiment: PushFoldExperiment, path: Union[str, Path]) -> None:
    """
    Save actor_critic + optimizer state to `path`.

    Only PyTorch state is saved. Per representation_model.py's
    documented TF/PyTorch boundary (Option B: the hand encoder is
    frozen and TF-native), actor_critic.state_dict() does NOT include
    the hand_encoder's weights — those live in a separate .keras file
    (see representation/pretrained_hand_encoder.py) and, if ever
    fine-tuned, must be checkpointed independently via Keras's own
    model.save().
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


def load_checkpoint(experiment: PushFoldExperiment, path: Union[str, Path]) -> None:
    """
    Load actor_critic + optimizer state from `path` into `experiment`,
    in place (mutates experiment.actor_critic and
    experiment.trainer.optimizer).
    """
    checkpoint = torch.load(Path(path), map_location="cpu")
    experiment.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    experiment.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a push-fold PPO agent.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    args = parser.parse_args()

    cfg = PushFoldExperimentConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
    )

    experiment = build_push_fold_experiment(cfg)
    history = run_push_fold_experiment(experiment)

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
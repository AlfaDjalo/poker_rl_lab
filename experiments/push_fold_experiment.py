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

Showdown resolution (win/loss/chop) and pretrained embeddings — DEFAULTS
--------------------------------------------------------------------------
PushFoldEnv is now constructed here with
envs.engine_scoring_evaluator.EngineScoringEvaluator by default, NOT
envs.poker_eval_evaluator.PokerEvalEvaluator. The difference: PokerEvalEvaluator
calls poker_eval.evaluate_hands(...) directly; EngineScoringEvaluator
instead routes through poker_engine.scoring.scoring_engine.CppScoringEngine
— the SAME scoring_engine surface FullHoldemEnv uses and the same one
ShowdownResolver uses for real games — so a push-fold hand ruled a
win/loss/chop during training is guaranteed consistent with what the
real Engine would rule for identical cards. If the poker_eval C++
extension isn't importable, EngineScoringEvaluator itself falls back to
poker_engine.scoring.mock_scoring_engine.MockScoringEngine (see that
module's own docstring) — no code here needs to change either way.

Likewise, build_push_fold_experiment() now defaults `hand_encoder` to
the REAL pretrained embeddings loaded from encoders/encoder_model.keras
(via representation.pretrained_hand_encoder.PretrainedHandEncoder.from_file),
rather than silently falling through to build_actor_critic()'s own
untrained() fallback. If that file is missing or fails architecture
validation, this module falls back to an untrained encoder with a loud
warning printed to stderr — it will not fail silently, but it also
will not raise, so pipeline/shape testing without the real weights
still works exactly as before. Pass `hand_encoder=` explicitly to
override this (e.g. a different path, or force `.untrained()`).

Device placement
------------------
`PushFoldExperimentConfig.device` (default "cpu") selects where the
PyTorch actor_critic lives — "cpu", "cuda", "cuda:0", etc. The hand
encoder stays on whatever device TensorFlow/Keras itself decides (this
codebase treats it as a frozen, non-PyTorch feature source — see
representation_model.py's "TensorFlow / PyTorch boundary" docstring),
so setting device="cuda" here does NOT move TF work to GPU; it only
affects the PyTorch RepresentationModel/PolicyValueNetwork/PPO update.
build_push_fold_experiment() calls actor_critic.to(cfg.device) BEFORE
constructing the Adam optimizer — the optimizer must be built against
parameter tensors on their final device, not moved after the fact
(standard PyTorch gotcha: an optimizer holds references to the specific
tensor objects passed to it at construction time).

Dependencies
------------
    PyTorch. scenarios.push_fold_generator, envs.push_fold_env,
    envs.engine_scoring_evaluator, representation.observation_adapter,
    representation.pretrained_hand_encoder, models.actor_critic_factory,
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
from envs.engine_scoring_evaluator import EngineScoringEvaluator
from models.actor_critic_factory import ActorCriticConfig, build_actor_critic
from models.policy_value_network import PokerActorCritic
from representation.base_hand_encoder import BaseHandEncoder
from representation.observation_adapter import ObservationAdapter
from scenarios.push_fold_generator import PushFoldScenarioGenerator
from trainers.ppo_trainer import IterationStats, PPOTrainer

# Default location of the pretrained hand-encoder weights. Matches
# run_push_fold_demo.py's own --encoder-path default, kept as one
# constant here so build_push_fold_experiment()'s default and the CLI's
# default can never silently drift apart.
_DEFAULT_ENCODER_PATH = "encoders/encoder_model.keras"
_DEFAULT_ENCODER_EMBEDDING_DIM = 32


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

    # EngineScoringEvaluator — board-runout RNG for real showdown resolution.
    # Kept independent of `seed` (scenario dealing) so re-seeding one
    # doesn't silently change the other. Defaults to `seed` if omitted.
    evaluator_seed: Optional[int] = None

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
class PushFoldExperiment:
    """Every piece of one training/evaluation run, already wired together."""
    scenario_gen: PushFoldScenarioGenerator
    env: PushFoldEnv
    adapter: ObservationAdapter
    actor_critic: PokerActorCritic
    trainer: PPOTrainer
    config: PushFoldExperimentConfig


def _default_hand_encoder() -> BaseHandEncoder:
    """
    Load the real pretrained hand encoder from _DEFAULT_ENCODER_PATH.
    Falls back to an untrained copy (with a loud stderr warning) if the
    file is missing or fails architecture validation — mirrors
    run_push_fold_demo.py's own _load_hand_encoder(), duplicated here
    (rather than imported from that CLI-only script) so this function
    stays usable from non-CLI callers of build_push_fold_experiment().
    """
    from representation.pretrained_hand_encoder import (
        HandEncoderArchitectureMismatch,
        PretrainedHandEncoder,
    )

    try:
        encoder = PretrainedHandEncoder.from_file(
            _DEFAULT_ENCODER_PATH,
            expected_embedding_dim=_DEFAULT_ENCODER_EMBEDDING_DIM,
        )
        print(
            f"Loaded pretrained hand encoder from {_DEFAULT_ENCODER_PATH} "
            f"(embedding_dim={encoder.embedding_dim}).",
            file=sys.stderr,
        )
        return encoder
    except FileNotFoundError:
        print(
            f"WARNING: encoder file not found at {_DEFAULT_ENCODER_PATH!r} — "
            f"falling back to an UNTRAINED hand encoder (no poker knowledge). "
            f"Pass hand_encoder=PretrainedHandEncoder.from_file(...) explicitly "
            f"to use real weights from a different path.",
            file=sys.stderr,
        )
        return PretrainedHandEncoder.untrained()
    except HandEncoderArchitectureMismatch as exc:
        print(
            f"WARNING: {_DEFAULT_ENCODER_PATH!r} failed architecture "
            f"validation ({exc}) — falling back to an UNTRAINED hand encoder.",
            file=sys.stderr,
        )
        return PretrainedHandEncoder.untrained()


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
        Defaults to PushFoldExperimentConfig() if omitted. `config.device`
        selects where the PyTorch actor_critic is placed — see module
        docstring's "Device placement" section.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to models.actor_critic_factory.build_actor_critic().
        If omitted, this now defaults to the REAL pretrained embeddings
        loaded from encoders/encoder_model.keras (see
        _default_hand_encoder() above) rather than an untrained encoder.
        Pass PretrainedHandEncoder.untrained() explicitly if you
        actually want the untrained pipeline-testing behaviour.

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
        evaluator=EngineScoringEvaluator(
            seed=cfg.evaluator_seed if cfg.evaluator_seed is not None else cfg.seed
        )
    )
    adapter = ObservationAdapter()

    if hand_encoder is None:
        hand_encoder = _default_hand_encoder()

    actor_critic = build_actor_critic(
        hand_encoder=hand_encoder, config=cfg.actor_critic_config
    )

    # Move to the target device BEFORE constructing the optimizer — the
    # optimizer holds references to the specific parameter tensor objects
    # passed to it at construction time, so moving the model afterward
    # would leave the optimizer tracking stale (now-orphaned) CPU tensors.
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

def _load_hand_encoder(path: str, expected_embedding_dim: Optional[int] = None):
    """
    Load the real pretrained hand encoder from `path`, or fall back to an
    untrained copy (with a loud warning) if the file isn't there / fails
    to load. Mirrors run_push_fold_demo.py's own loader — kept duplicated
    rather than shared since this is CLI-only convenience code, not part
    of the importable experiment-building API above. (Distinct from
    _default_hand_encoder() above, which is hardcoded to
    _DEFAULT_ENCODER_PATH for the non-CLI build_push_fold_experiment()
    default; this version takes an explicit --encoder-path.)
    """
    from representation.pretrained_hand_encoder import (
        HandEncoderArchitectureMismatch,
        PretrainedHandEncoder,
    )

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
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for the actor_critic, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    parser.add_argument(
        "--encoder-path", type=str, default=_DEFAULT_ENCODER_PATH,
        help="Path to the pretrained hand encoder .keras file. Pass an "
             "empty string ('') to force an untrained encoder.",
    )
    parser.add_argument(
        "--encoder-embedding-dim", type=int, default=_DEFAULT_ENCODER_EMBEDDING_DIM,
        help="Expected embedding_dim of the loaded encoder; loading fails "
             "loudly if it doesn't match. Pass 0 to skip this check.",
    )
    args = parser.parse_args()

    cfg = PushFoldExperimentConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )

    hand_encoder = _load_hand_encoder(
        args.encoder_path,
        expected_embedding_dim=args.encoder_embedding_dim or None,
    )

    experiment = build_push_fold_experiment(cfg, hand_encoder=hand_encoder)
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
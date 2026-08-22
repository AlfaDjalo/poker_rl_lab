"""
experiments/multi_street_experiment.py — End-to-end training entry point
for envs.multi_street_env.MultiStreetEnv (Phase D: full multi-street,
re-raise-capable self-play), via
scenarios.preflop_scenario_generator.PreflopScenarioGenerator /
trainers.multi_street_training_loop.MultiStreetTrainingLoop.

Mirrors experiments/generic_tree_experiment.py's build/run/checkpoint
shape deliberately, so anyone already familiar with that entry point
recognises this one immediately. Network construction
(build_dual_seat_actor_critic, PolicyVariant, ActorCriticConfig,
_default_hand_encoder) is IMPORTED from generic_tree_experiment.py, not
duplicated — that module already builds a fresh, untrained-unless-given
hand_encoder two-seat DualSeatActorCritic with zero DuoGameConfig
dependency (see its own "Inference-only builders" section), which is
exactly what this entry point needs too.

KNOWN LIMITATION — 2 seats only
------------------------------------
Same limitation trainers/multi_street_training_loop.py already states:
DualSeatActorCritic only supports seat in (0, 1). This entry point is
therefore heads-up only (SB vs BB) — PreflopScenarioGenerator itself
supports player_count > 2, but nothing downstream of it here does yet.

poker_eval shim
----------------
Same pattern as every other experiment entry point in this codebase.

Dependencies
------------
    PyTorch. envs.multi_street_env, scenarios.preflop_scenario_generator,
    trainers.multi_street_training_loop, representation.observation_adapter,
    experiments.generic_tree_experiment (PolicyVariant, ActorCriticConfig,
    build_dual_seat_actor_critic, _default_hand_encoder — reused).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import poker_rl_lab.poker_eval as poker_eval  # noqa: F401  (prefer the real extension if present)
except ImportError:
    import poker_rl_lab.mock_poker_eval as mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

import torch

from poker_rl_lab.envs.multi_street_env import MultiStreetEnv
from poker_rl_lab.scenarios.preflop_scenario_generator import PreflopScenarioGenerator
from poker_rl_lab.representation.observation_adapter import ObservationAdapter
from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.models.actor_critic_factory import ActorCriticConfig
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.trainers.multi_street_training_loop import (
    MultiStreetIterationStats,
    MultiStreetTrainingLoop,
    SB_SEAT,
    BB_SEAT,
)
from poker_rl_lab.experiments.generic_tree_experiment import (
    PolicyVariant,
    build_dual_seat_actor_critic,
    _default_hand_encoder,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiStreetExperimentConfig:
    """All knobs for one Phase D training run, grouped by the piece of
    the pipeline they configure."""

    # Which Engine game to play (poker_engine.games.loader.load_game name)
    game_name: str = "holdem"

    seed: Optional[int] = 42

    # Policy architecture
    variant: PolicyVariant = PolicyVariant.SHARED_TRUNK_TWO_HEADS
    actor_critic_config: ActorCriticConfig = field(default_factory=ActorCriticConfig)

    # PreflopScenarioGenerator
    small_blind: int = 1
    big_blind: int = 2
    stack_mode: str = "fixed_bb"
    stack_bb: float = 100.0
    min_stack_bb: float = 20.0
    max_stack_bb: float = 100.0
    randomize_dealer: bool = True

    # Device placement — same rationale as generic_tree_experiment.py:
    # move BEFORE constructing optimizers.
    device: str = "cpu"

    # Optimizers — one learning rate for every seat unless overridden.
    learning_rate: float = 3e-4
    learning_rates: Dict[int, float] = field(default_factory=dict)

    # MultiStreetTrainingLoop
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: Optional[float] = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95

    # train()
    num_iterations: int = 100
    episodes_per_iteration: int = 256
    epochs: int = 4
    minibatch_size: int = 64


# ---------------------------------------------------------------------------
# Wired-up experiment bundle
# ---------------------------------------------------------------------------

@dataclass
class MultiStreetExperiment:
    """Every piece of one training/evaluation run, already wired
    together."""
    env: MultiStreetEnv
    scenario_gen: PreflopScenarioGenerator
    adapter: ObservationAdapter
    actor_critic: DualSeatActorCritic
    loop: MultiStreetTrainingLoop
    config: MultiStreetExperimentConfig


def build_multi_street_experiment(
    config: Optional[MultiStreetExperimentConfig] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
) -> MultiStreetExperiment:
    """
    Construct every piece of a Phase D self-play training pipeline for
    `config.game_name`, wired together but not yet run.

    Parameters
    ----------
    config : MultiStreetExperimentConfig, optional
        Defaults to MultiStreetExperimentConfig() (holdem,
        shared_trunk_two_heads) if omitted.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to build_dual_seat_actor_critic() (via
        models.actor_critic_factory.build_actor_critic() internally) —
        see that function's docstring. Omit to get an UNTRAINED
        PretrainedHandEncoder (pipeline testing only) via
        generic_tree_experiment._default_hand_encoder().

    Returns
    -------
    MultiStreetExperiment
    """
    cfg = config or MultiStreetExperimentConfig()

    if hand_encoder is None:
        hand_encoder = _default_hand_encoder()

    env = MultiStreetEnv(cfg.game_name)

    scenario_gen = PreflopScenarioGenerator(
        player_count=2,
        game_name=cfg.game_name,
        small_blind=cfg.small_blind,
        big_blind=cfg.big_blind,
        stack_mode=cfg.stack_mode,
        stack_bb=cfg.stack_bb,
        min_stack_bb=cfg.min_stack_bb,
        max_stack_bb=cfg.max_stack_bb,
        randomize_dealer=cfg.randomize_dealer,
    )
    if cfg.seed is not None:
        scenario_gen.seed(cfg.seed)

    adapter = ObservationAdapter()

    actor_critic = build_dual_seat_actor_critic(
        cfg.variant, hand_encoder=hand_encoder, actor_critic_config=cfg.actor_critic_config
    )

    # Move to the target device BEFORE constructing optimizers — see
    # generic_tree_experiment.py's own "Device placement" note.
    actor_critic = actor_critic.to(cfg.device)

    optimizer_sb = torch.optim.Adam(
        actor_critic.seat_parameters(SB_SEAT),
        lr=cfg.learning_rates.get(SB_SEAT, cfg.learning_rate),
    )
    optimizer_bb = torch.optim.Adam(
        actor_critic.seat_parameters(BB_SEAT),
        lr=cfg.learning_rates.get(BB_SEAT, cfg.learning_rate),
    )

    loop = MultiStreetTrainingLoop(
        env=env,
        scenario_gen=scenario_gen,
        actor_critic=actor_critic,
        adapter=adapter,
        optimizer_sb=optimizer_sb,
        optimizer_bb=optimizer_bb,
        clip_epsilon=cfg.clip_epsilon,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
    )

    return MultiStreetExperiment(
        env=env, scenario_gen=scenario_gen, adapter=adapter,
        actor_critic=actor_critic, loop=loop, config=cfg,
    )


def run_multi_street_experiment(
    experiment: MultiStreetExperiment,
) -> List[MultiStreetIterationStats]:
    """Run experiment.config's configured number of training iterations."""
    cfg = experiment.config
    return experiment.loop.train(
        num_iterations=cfg.num_iterations,
        episodes_per_iteration=cfg.episodes_per_iteration,
        epochs=cfg.epochs,
        minibatch_size=cfg.minibatch_size,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(experiment: MultiStreetExperiment, path: Union[str, Path]) -> None:
    """Save actor_critic + both seats' optimizer state to `path`. Does
    not persist experiment.config — same convention as
    generic_tree_experiment.save_checkpoint (see that function's
    docstring for why the caller must rebuild with matching config
    before load_checkpoint() can succeed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_critic_state_dict": experiment.actor_critic.state_dict(),
            "optimizer_state_dicts": {
                seat: opt.state_dict()
                for seat, opt in experiment.loop.optimizers.items()
            },
        },
        path,
    )


def load_checkpoint(experiment: MultiStreetExperiment, path: Union[str, Path]) -> None:
    """Load actor_critic + both seats' optimizer state from `path` into
    `experiment`, in place. See save_checkpoint()'s docstring for the
    matching-config requirement."""
    checkpoint = torch.load(Path(path), map_location=experiment.config.device)
    experiment.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    for seat, state_dict in checkpoint["optimizer_state_dicts"].items():
        experiment.loop.optimizers[seat].load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train a two-seat PPO agent pair via full multi-street, "
            "re-raise-capable self-play (Phase D)."
        )
    )
    parser.add_argument("--game-name", type=str, default="holdem")
    parser.add_argument(
        "--variant", type=str, default="shared_trunk_two_heads",
        choices=[v.value for v in PolicyVariant],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument(
        "--stack-mode", type=str, default="fixed_bb",
        choices=["fixed_bb", "randomized_bb"],
    )
    parser.add_argument("--stack-bb", type=float, default=100.0)
    parser.add_argument("--min-stack-bb", type=float, default=20.0)
    parser.add_argument("--max-stack-bb", type=float, default=100.0)
    parser.add_argument(
        "--fixed-dealer-rotation", action="store_true",
        help="Rotate the button deterministically instead of randomizing "
             "it each hand.",
    )
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--checkpoint-in", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for the actor_critic, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    args = parser.parse_args()

    cfg = MultiStreetExperimentConfig(
        game_name=args.game_name,
        seed=args.seed,
        variant=PolicyVariant(args.variant),
        learning_rate=args.learning_rate,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        stack_mode=args.stack_mode,
        stack_bb=args.stack_bb,
        min_stack_bb=args.min_stack_bb,
        max_stack_bb=args.max_stack_bb,
        randomize_dealer=not args.fixed_dealer_rotation,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )

    experiment = build_multi_street_experiment(cfg)

    if args.checkpoint_in:
        load_checkpoint(experiment, args.checkpoint_in)
        print(f"Resumed from checkpoint {args.checkpoint_in}")

    print(f"config: {cfg}")
    history = run_multi_street_experiment(experiment)

    for i, stats in enumerate(history):
        rewards = " ".join(
            f"seat{seat}={r:+.3f}" for seat, r in stats.mean_reward_by_seat.items()
        )
        decisions = " ".join(
            f"seat{seat}={d:.2f}" for seat, d in stats.mean_decisions_by_seat.items()
        )
        print(f"iter {i:4d}  {rewards}  decisions/hand: {decisions}")

    if args.checkpoint_out:
        save_checkpoint(experiment, args.checkpoint_out)
        print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    _main()
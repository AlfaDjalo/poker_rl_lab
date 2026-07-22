"""
experiments/double_board_river_experiment.py — End-to-end training entry
point for the RL Lab's "Double Board" curriculum stage (§1, §8):

    1. Push/Fold
    2. River Hold'em
    3. Full Hold'em
    4. Omaha
    5. Double Board      <-- this module
    ...

Why this is FullHoldemEnv, not a new self-contained env
------------------------------------------------------------
scenarios.double_board_river_generator.DoubleBoardRiverScenarioGenerator
produces already-river, both-boards-dealt BaseScenarios (game_name=
"double_board") exactly like river_generator.py does for a single board
— but a double-board hand's payout needs real points/node_sets scoring
(two independent HIGH points, one per board — see
poker_engine/games/double_board.yaml), which is genuine Engine logic
(rules.game_rules / showdown_resolver.py), not something a hand-rolled
MockEvaluator like river_env.py's should reimplement (guide §3: "The
Environment must never implement... hand evaluation, showdown logic").
envs.full_holdem_env.FullHoldemEnv is already built variant-agnostic —
it drives ANY (game_def, rules) pair through the real
poker_engine.state.poker_state.PokerState, and PokerState.load_state()
already supports resuming mid-hand (street_index=3, all node_cards
filled — the exact shape DoubleBoardRiverScenarioGenerator produces).
So this stage needs zero new Environment code: only a game YAML
(poker_engine/games/double_board.yaml) plus this wiring module, mirroring
river_experiment.py's structure but swapping in FullHoldemEnv +
DoubleBoardRiverScenarioGenerator in place of RiverEnv + RiverGenerator.

Mirrors experiments/full_holdem_experiment.py almost exactly (same
FullHoldemEnv, same villain-policy knobs) — differing only in the
scenario generator (double-board river deal, not a fresh preflop table)
and generator-specific knobs (min/max pot & stack in BB, hole cards
per player defaulting to 4 for Omaha-style boards, matching the game's
`showdown: type: OMAHA` in double_board.yaml).

See push_fold_experiment.py's own docstring for the full rationale
behind the poker_eval shim and the untrained-hand-encoder caveat below
— identical here, not repeated in full.

Dependencies
------------
    PyTorch. scenarios.double_board_river_generator, envs.full_holdem_env,
    envs.river_env (CheckFoldVillainPolicy default only), representation.
    observation_adapter, models.actor_critic_factory,
    models.policy_value_network, trainers.ppo_trainer,
    actions.abstract_action. loader / rules.* / mock_poker_eval — see
    push_fold_experiment.py's "poker_eval shim" section. Like
    full_holdem_experiment.py, this module ALSO needs the poker_eval
    shim installed before importing envs.full_holdem_env itself (module-
    level PokerState import — see that module's docstring).
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
from typing import Dict, List, Optional, Union

import torch

from envs.full_holdem_env import FullHoldemEnv, VillainPolicyLike
from envs.river_env import CheckFoldVillainPolicy
from models.actor_critic_factory import ActorCriticConfig, build_actor_critic
from models.policy_value_network import PokerActorCritic
from representation.base_hand_encoder import BaseHandEncoder
from representation.observation_adapter import ObservationAdapter
from scenarios.double_board_river_generator import DoubleBoardRiverScenarioGenerator
from trainers.ppo_trainer import IterationStats, PPOTrainer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DoubleBoardRiverExperimentConfig:
    """All knobs for one double-board-river training run, grouped by the
    piece of the pipeline they configure."""

    # DoubleBoardRiverScenarioGenerator
    player_count: int = 2
    small_blind: int = 1
    big_blind: int = 2
    hole_cards_per_player: int = 4          # Omaha-style, matches double_board.yaml
    min_pot_bb: float = 2.0
    max_pot_bb: float = 20.0
    min_stack_bb: float = 2.0
    max_stack_bb: float = 50.0
    game_name: str = "double_board"
    seed: Optional[int] = 42

    # FullHoldemEnv
    default_villain_policy: object = field(default_factory=CheckFoldVillainPolicy)
    scoring_engine: object = None  # None -> FullHoldemEnv's own MockScoringEngine default

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
class DoubleBoardRiverExperiment:
    """Every piece of one training/evaluation run, already wired together."""
    scenario_gen: DoubleBoardRiverScenarioGenerator
    env: FullHoldemEnv
    adapter: ObservationAdapter
    actor_critic: PokerActorCritic
    trainer: PPOTrainer
    config: DoubleBoardRiverExperimentConfig


def build_double_board_river_experiment(
    config: Optional[DoubleBoardRiverExperimentConfig] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
    villain_policies: Optional[Dict[int, VillainPolicyLike]] = None,
) -> DoubleBoardRiverExperiment:
    """
    Construct every piece of the double-board-river training pipeline,
    wired together but not yet run.

    Parameters
    ----------
    config : DoubleBoardRiverExperimentConfig, optional
        Defaults to DoubleBoardRiverExperimentConfig() if omitted.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to models.actor_critic_factory.build_actor_critic() —
        see that function's docstring. Omit to get an UNTRAINED
        PretrainedHandEncoder (pipeline testing only).
    villain_policies : Dict[int, VillainPolicyLike], optional
        Per-seat policy overrides forwarded straight to FullHoldemEnv —
        see envs/full_holdem_env.py's own docstring. Only seat 0 (the
        villain in this heads-up generator) is ever actually consulted
        at player_count=2.

    Returns
    -------
    DoubleBoardRiverExperiment
    """
    cfg = config or DoubleBoardRiverExperimentConfig()

    scenario_gen = DoubleBoardRiverScenarioGenerator(
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

    return DoubleBoardRiverExperiment(
        scenario_gen=scenario_gen,
        env=env,
        adapter=adapter,
        actor_critic=actor_critic,
        trainer=trainer,
        config=cfg,
    )


def run_double_board_river_experiment(
    experiment: DoubleBoardRiverExperiment,
) -> List[IterationStats]:
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

def save_checkpoint(experiment: DoubleBoardRiverExperiment, path: Union[str, Path]) -> None:
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


def load_checkpoint(experiment: DoubleBoardRiverExperiment, path: Union[str, Path]) -> None:
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

    parser = argparse.ArgumentParser(description="Train a double-board-river PPO agent.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    args = parser.parse_args()

    cfg = DoubleBoardRiverExperimentConfig(
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
    )

    experiment = build_double_board_river_experiment(cfg)
    history = run_double_board_river_experiment(experiment)

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
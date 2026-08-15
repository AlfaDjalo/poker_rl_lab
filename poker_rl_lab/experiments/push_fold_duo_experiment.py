"""
experiments/push_fold_duo_experiment.py — End-to-end training entry point
for the two-decision heads-up push-fold-duo curriculum stage.

Unlike push_fold_experiment.py / river_experiment.py (one hero, one
BasePokerEnv), this stage trains BOTH seats via
envs.push_fold_duo_env.PushFoldDuoEnv + trainers.duo_training_loop
.DuoTrainingLoop — see those modules' own docstrings for why a
single-hero BasePokerEnv contract doesn't fit here.

Policy variant
--------------
`PolicyVariant.SHARED_TRUNK_TWO_HEADS` and `PolicyVariant.SHARED_SINGLE_HEAD`
select between models.shared_trunk_two_heads_policy
.SharedTrunkTwoHeadsActorCritic and models.shared_single_head_policy
.SharedSingleHeadActorCritic (both implement models.dual_seat_policy
.DualSeatActorCritic). One `ActorCriticConfig` (models.actor_critic_factory)
supplies every component-encoder hyperparameter for the shared trunk, AND
— for the two-heads variant — is reused identically to build BOTH heads
(head_sb, head_bb), so there's exactly one place to change e.g.
policy_hidden_dim for a run rather than two that could drift apart.

Trunk-building duplication note
--------------------------------
models.actor_critic_factory.build_actor_critic() builds a trunk
(RepresentationModel) and ONE PolicyValueNetwork together, returning a
single PokerActorCritic — there's no seam to get the trunk back out
separately for the two-heads variant. `_build_representation_model()`
below duplicates that module's component-wiring (CardEncoder shared
into BoardEncoder, StackEncoder, etc.) so a trunk can be built once and
fed into either one or two heads. If actor_critic_factory.py's wiring
ever changes, this function needs to be re-synced by hand — same
maintenance tradeoff already accepted by representation/encoders.py for
the vendored Keras architecture (see that file's own docstring). This
includes StackEncoder (representation/stack_encoder.py): both seats
need real stack/pot signal in their trunk input — SB's push/fold
decision in particular is a function of effective stack depth almost
by definition (see stack_encoder.py's own docstring on why this was
previously missing entirely), so wiring StackEncoder here is not
optional the way, say, a cosmetic hyperparameter tweak would be.

Device placement
------------------
`PushFoldDuoExperimentConfig.device` (default "cpu") selects where the
PyTorch DualSeatActorCritic lives — "cpu", "cuda", "cuda:0", etc. The
hand encoder stays TF-native regardless (see push_fold_experiment.py's
identical note). build_push_fold_duo_experiment() calls
actor_critic.to(cfg.device) BEFORE constructing EITHER seat's Adam
optimizer — both optimizer_sb and optimizer_bb are built from
actor_critic.seat_parameters(seat), so the move must happen first or
each optimizer would hold references to now-orphaned CPU tensors (same
gotcha as every other experiment file's optimizer construction).

Dependencies
------------
    PyTorch. scenarios.push_fold_duo_generator, envs.push_fold_duo_env,
    envs.engine_scoring_evaluator, representation.*, models.*,
    trainers.duo_training_loop, actions.abstract_action.
"""

from __future__ import annotations

import sys

try:
    import poker_engine.poker_eval  # noqa: F401
except ImportError:
    import poker_rl_lab.mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import torch

from poker_rl_lab.actions.abstract_action import NUM_ACTIONS
from poker_rl_lab.envs.engine_scoring_evaluator import EngineScoringEvaluator
from poker_rl_lab.envs.push_fold_duo_env import PushFoldDuoEnv
from poker_rl_lab.models.actor_critic_factory import ActorCriticConfig
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.models.policy_value_network import PokerActorCritic, PolicyValueNetwork
from poker_rl_lab.models.shared_single_head_policy import SharedSingleHeadActorCritic
from poker_rl_lab.models.shared_trunk_two_heads_policy import SharedTrunkTwoHeadsActorCritic
from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.board_encoder import BoardEncoder
from poker_rl_lab.representation.card_encoder import CardEncoder
from poker_rl_lab.representation.game_encoder import GameEncoder
from poker_rl_lab.representation.history_encoder import HistoryEncoder
from poker_rl_lab.representation.observation_adapter import ObservationAdapter
from poker_rl_lab.representation.point_encoder import PointEncoder
from poker_rl_lab.representation.position_encoder import PositionEncoder
from poker_rl_lab.representation.stack_encoder import StackEncoder
from poker_rl_lab.representation.representation_model import RepresentationModel
from poker_rl_lab.scenarios.push_fold_duo_generator import BB_SEAT, PushFoldDuoScenarioGenerator, SB_SEAT
from poker_rl_lab.trainers.duo_training_loop import DuoTrainingLoop, DuoTrainingMode

# Default location of the pretrained hand-encoder weights. Matches
# experiments/push_fold_experiment.py's own _DEFAULT_ENCODER_PATH — kept
# as one constant per module (rather than importing the other module's
# private constant) so each experiment file's default can't silently
# drift by editing only one of them, while still resolving to the same
# path in practice.
_DEFAULT_ENCODER_PATH = "poker_rl_lab/encoders/encoder_model.keras"
_DEFAULT_ENCODER_EMBEDDING_DIM = 32


# ---------------------------------------------------------------------------
# Policy variant selection
# ---------------------------------------------------------------------------

class PolicyVariant(Enum):
    SHARED_TRUNK_TWO_HEADS = "shared_trunk_two_heads"
    SHARED_SINGLE_HEAD = "shared_single_head"


# ---------------------------------------------------------------------------
# Trunk building (duplicated from models.actor_critic_factory.build_actor_critic
# — see module docstring)
# ---------------------------------------------------------------------------

def _build_representation_model(
    hand_encoder: BaseHandEncoder, cfg: ActorCriticConfig
) -> RepresentationModel:
    card_encoder = CardEncoder(rank_dim=cfg.rank_dim, suit_dim=cfg.suit_dim)

    board_encoder = BoardEncoder(
        card_encoder=card_encoder,
        output_dim=cfg.board_output_dim,
        num_heads=cfg.board_num_heads,
        num_layers=cfg.board_num_layers,
        dropout=cfg.board_dropout,
    )

    point_encoder = PointEncoder(
        score_type_dim=cfg.point_score_type_dim,
        showdown_type_dim=cfg.point_showdown_type_dim,
        scalar_hidden_dim=cfg.point_scalar_hidden_dim,
        output_dim=cfg.point_output_dim,
    )

    game_encoder = GameEncoder(
        betting_type_dim=cfg.game_betting_type_dim,
        payout_type_dim=cfg.game_payout_type_dim,
        showdown_type_dim=cfg.game_showdown_type_dim,
        scalar_hidden_dim=cfg.game_scalar_hidden_dim,
        output_dim=cfg.game_output_dim,
    )

    position_encoder = PositionEncoder(
        hidden_dim=cfg.position_hidden_dim,
        output_dim=cfg.position_output_dim,
    )

    history_encoder = HistoryEncoder(
        action_type_dim=cfg.history_action_type_dim,
        scalar_hidden_dim=cfg.history_scalar_hidden_dim,
        gru_hidden_dim=cfg.history_gru_hidden_dim,
        output_dim=cfg.history_output_dim,
    )

    stack_encoder = StackEncoder(
        hidden_dim=cfg.stack_hidden_dim,
        output_dim=cfg.stack_output_dim,
    )

    return RepresentationModel(
        hand_encoder=hand_encoder,
        board_encoder=board_encoder,
        point_encoder=point_encoder,
        game_encoder=game_encoder,
        position_encoder=position_encoder,
        history_encoder=history_encoder,
        stack_encoder=stack_encoder,
        hidden_dim=cfg.representation_hidden_dim,
        output_dim=cfg.representation_output_dim,
        dropout=cfg.representation_dropout,
    )


def _build_head(cfg: ActorCriticConfig, latent_dim: int) -> PolicyValueNetwork:
    return PolicyValueNetwork(
        latent_dim=latent_dim,
        action_dim=NUM_ACTIONS,
        hidden_dim=cfg.policy_hidden_dim,
        num_shared_layers=cfg.policy_num_shared_layers,
    )


def build_dual_seat_actor_critic(
    variant: PolicyVariant,
    config: Optional[ActorCriticConfig] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
) -> DualSeatActorCritic:
    """
    Build a DualSeatActorCritic of the requested variant.

    Parameters
    ----------
    variant : PolicyVariant
    config : ActorCriticConfig, optional
        Hyperparameters for the trunk and (for SHARED_TRUNK_TWO_HEADS)
        BOTH heads identically. Defaults to ActorCriticConfig().
    hand_encoder : BaseHandEncoder, optional
        Omit for an UNTRAINED PretrainedHandEncoder (pipeline testing
        only — same caveat as build_actor_critic()).

    Returns
    -------
    DualSeatActorCritic
        Built on CPU. Callers that want it on a different device should
        call .to(device) on the result themselves — see
        build_push_fold_duo_experiment()'s own device handling for the
        "move before optimizer construction" ordering this implies.
    """
    cfg = config or ActorCriticConfig()

    if hand_encoder is None:
        from poker_rl_lab.representation.pretrained_hand_encoder import PretrainedHandEncoder

        hand_encoder = PretrainedHandEncoder.untrained()

    trunk = _build_representation_model(hand_encoder, cfg)

    if variant == PolicyVariant.SHARED_SINGLE_HEAD:
        head = _build_head(cfg, trunk.output_dim)
        actor_critic = PokerActorCritic(trunk, head)
        return SharedSingleHeadActorCritic(actor_critic)

    if variant == PolicyVariant.SHARED_TRUNK_TWO_HEADS:
        head_sb = _build_head(cfg, trunk.output_dim)
        head_bb = _build_head(cfg, trunk.output_dim)
        return SharedTrunkTwoHeadsActorCritic(trunk, head_sb, head_bb)

    raise ValueError(f"Unknown variant {variant!r}")


def _default_hand_encoder() -> BaseHandEncoder:
    """
    Load the real pretrained hand encoder from _DEFAULT_ENCODER_PATH.

    Falls back to an untrained copy (with a loud stderr warning) if the
    file is missing or fails architecture validation — mirrors
    experiments/push_fold_experiment.py's own _default_hand_encoder().
    Without this, build_dual_seat_actor_critic() would silently default
    to .untrained() every time hand_encoder is omitted, which is exactly
    the bug this function exists to close: the resulting frozen,
    meaningless hole-card embedding gives the policy nothing
    hand-strength-correlated to condition on (see representation_model
    .py's forward() — hole_embed is detached regardless of which
    encoder produced it, so an untrained encoder can never be fixed by
    training longer).
    """
    from poker_rl_lab.representation.pretrained_hand_encoder import (
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PushFoldDuoExperimentConfig:
    """All knobs for one push-fold-duo training run, grouped by the
    piece of the pipeline they configure."""

    # PushFoldDuoScenarioGenerator
    small_blind: int = 1
    big_blind: int = 2
    min_stack_bb: float = 4.0
    max_stack_bb: float = 20.0
    seed: Optional[int] = 42

    # EngineScoringEvaluator — board-runout RNG for showdown resolution.
    evaluator_seed: Optional[int] = None

    # Device placement — see module docstring's "Device placement" section.
    device: str = "cpu"

    # Policy
    variant: PolicyVariant = PolicyVariant.SHARED_TRUNK_TWO_HEADS
    actor_critic_config: ActorCriticConfig = field(default_factory=ActorCriticConfig)

    # Optimizers (one per seat — see build_dual_seat_actor_critic /
    # DualSeatActorCritic.seat_parameters)
    learning_rate_sb: float = 3e-4
    learning_rate_bb: float = 3e-4

    # PPO
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: Optional[float] = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95

    # DuoTrainingLoop mode
    mode: DuoTrainingMode = DuoTrainingMode.SIMULTANEOUS
    iters_per_seat: int = 5
    snapshot_pool_size: int = 8
    snapshot_every: int = 5

    # train()
    num_iterations: int = 100
    episodes_per_iteration: int = 256
    epochs: int = 4
    minibatch_size: int = 64


# ---------------------------------------------------------------------------
# Wired-up experiment bundle
# ---------------------------------------------------------------------------

@dataclass
class PushFoldDuoExperiment:
    """Every piece of one training/evaluation run, already wired together."""
    scenario_gen: PushFoldDuoScenarioGenerator
    env: PushFoldDuoEnv
    adapter: ObservationAdapter
    actor_critic: DualSeatActorCritic
    loop: DuoTrainingLoop
    config: PushFoldDuoExperimentConfig


def build_push_fold_duo_experiment(
    config: Optional[PushFoldDuoExperimentConfig] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
) -> PushFoldDuoExperiment:
    """
    Construct every piece of the push-fold-duo training pipeline, wired
    together but not yet run.

    Parameters
    ----------
    config : PushFoldDuoExperimentConfig, optional
        `config.device` selects where the PyTorch DualSeatActorCritic is
        placed — see module docstring's "Device placement" section.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to build_dual_seat_actor_critic(). If omitted, this
        now defaults to the REAL pretrained embeddings loaded from
        _DEFAULT_ENCODER_PATH (see _default_hand_encoder() above)
        rather than an untrained encoder — matching
        experiments/push_fold_experiment.py's own default. Pass
        PretrainedHandEncoder.untrained() explicitly if you actually
        want the untrained pipeline-testing behaviour.
    """
    cfg = config or PushFoldDuoExperimentConfig()

    scenario_gen = PushFoldDuoScenarioGenerator(
        small_blind=cfg.small_blind,
        big_blind=cfg.big_blind,
        min_stack_bb=cfg.min_stack_bb,
        max_stack_bb=cfg.max_stack_bb,
    )
    if cfg.seed is not None:
        scenario_gen.seed(cfg.seed)

    env = PushFoldDuoEnv(
        evaluator=EngineScoringEvaluator(
            seed=cfg.evaluator_seed if cfg.evaluator_seed is not None else cfg.seed
        )
    )
    adapter = ObservationAdapter()

    if hand_encoder is None:
        hand_encoder = _default_hand_encoder()

    actor_critic = build_dual_seat_actor_critic(
        cfg.variant, config=cfg.actor_critic_config, hand_encoder=hand_encoder
    )

    # Move to the target device BEFORE constructing either seat's
    # optimizer — both optimizer_sb and optimizer_bb are built from
    # actor_critic.seat_parameters(seat), and an optimizer holds
    # references to the specific parameter tensor objects passed to it
    # at construction time. Moving the model afterward would leave both
    # optimizers tracking stale, now-orphaned CPU tensors.
    actor_critic = actor_critic.to(cfg.device)

    optimizer_sb = torch.optim.Adam(
        actor_critic.seat_parameters(SB_SEAT), lr=cfg.learning_rate_sb
    )
    optimizer_bb = torch.optim.Adam(
        actor_critic.seat_parameters(BB_SEAT), lr=cfg.learning_rate_bb
    )

    loop = DuoTrainingLoop(
        env=env,
        scenario_gen=scenario_gen,
        actor_critic=actor_critic,
        adapter=adapter,
        optimizer_sb=optimizer_sb,
        optimizer_bb=optimizer_bb,
        mode=cfg.mode,
        clip_epsilon=cfg.clip_epsilon,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        iters_per_seat=cfg.iters_per_seat,
        snapshot_pool_size=cfg.snapshot_pool_size,
        snapshot_every=cfg.snapshot_every,
    )

    return PushFoldDuoExperiment(
        scenario_gen=scenario_gen,
        env=env,
        adapter=adapter,
        actor_critic=actor_critic,
        loop=loop,
        config=cfg,
    )


def run_push_fold_duo_experiment(experiment: PushFoldDuoExperiment):
    """
    Run experiment.config's configured number of DuoTrainingLoop
    iterations.

    Returns
    -------
    List[trainers.duo_training_loop.DuoIterationStats]
    """
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

def save_checkpoint(experiment: PushFoldDuoExperiment, path: Union[str, Path]) -> None:
    """
    Save actor_critic (both seats' weights — a single state_dict covers
    the whole DualSeatActorCritic, including a shared trunk if present)
    plus both optimizers.

    Also embeds `variant` and `actor_critic_config` (as a plain dict via
    dataclasses.asdict) so the checkpoint is self-describing: any
    consumer — including outside this codebase, e.g. the CAP backend —
    can reconstruct the EXACT architecture via
    build_dual_seat_actor_critic(variant, config) without needing to
    separately know or guess which hyperparameters this run used. See
    load_dual_seat_actor_critic() below, the intended external-consumer
    entry point.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": experiment.config.variant.value,
            "actor_critic_config": asdict(experiment.config.actor_critic_config),
            "small_blind": experiment.config.small_blind,
            "big_blind": experiment.config.big_blind,
            "actor_critic_state_dict": experiment.actor_critic.state_dict(),
            "optimizer_sb_state_dict": experiment.loop.optimizers[SB_SEAT].state_dict(),
            "optimizer_bb_state_dict": experiment.loop.optimizers[BB_SEAT].state_dict(),
        },
        path,
    )


def load_dual_seat_actor_critic(
    path: Union[str, Path],
    hand_encoder: Optional[BaseHandEncoder] = None,
    map_location: str = "cpu",
    device: Optional[str] = None,
) -> DualSeatActorCritic:
    """
    Minimal inference-only checkpoint loader.

    Builds a DualSeatActorCritic that exactly matches the architecture a
    checkpoint was saved with (reading `variant` and `actor_critic_config`
    back out of the file — see save_checkpoint()), loads its weights,
    and returns it in eval() mode. Does NOT construct a PushFoldDuoEnv,
    scenario generator, optimizers, or DuoTrainingLoop — those are
    training-only concerns build_push_fold_duo_experiment() exists for.

    This is the intended entry point for external consumers (e.g. the
    CAP backend, to run a trained agent against a human) that only want
    to call .act(seat, rep_input, legal_mask, deterministic=True) and
    have no need for the rest of the training pipeline.

    Parameters
    ----------
    path : str or Path
        A checkpoint produced by save_checkpoint(). Must contain the
        'variant' and 'actor_critic_config' keys — checkpoints saved
        before those were added must be re-saved once from a training
        run that still has its own PushFoldDuoExperimentConfig in
        memory (call save_checkpoint() again there), or built manually
        via build_dual_seat_actor_critic(variant, config) + a direct
        .load_state_dict() call using the exact config the run used.

        NOTE: checkpoints saved before StackEncoder was added to the
        trunk (see representation/stack_encoder.py) are ALSO
        incompatible — actor_critic_config alone is not enough to tell
        the difference (StackEncoder's hyperparameters just weren't
        part of the dataclass yet), and load_state_dict() will raise a
        shape/missing-key error naming stack_encoder.* parameters.
        Re-train from that checkpoint's optimizer state is not
        recoverable across this change; start a fresh run.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to build_dual_seat_actor_critic(). Its own weights
        are never part of `actor_critic_state_dict` regardless (see
        representation_model.py's TF/PyTorch boundary note — the hand
        encoder is always treated as a separate, frozen artifact), so
        this must be supplied independently and consistently with
        whatever encoder the checkpoint was trained against — pass the
        same PretrainedHandEncoder.from_file(...) call used at training
        time, or omit only if training itself used .untrained().
    map_location : str
        Forwarded to torch.load(). Default "cpu" — safe on a machine
        with no GPU. If `device` is also given, map_location defaults
        to matching it instead (see below) unless explicitly overridden.
    device : str, optional
        If given, the rebuilt actor_critic is moved here via .to(device)
        AFTER weights are loaded (inference-only — no optimizer exists
        for this loader to worry about ordering against, unlike
        build_push_fold_duo_experiment()). Also used as the default for
        map_location when map_location is left at its default "cpu" and
        device is not "cpu" itself, so a checkpoint trained on CUDA and
        being reloaded for CUDA inference doesn't unnecessarily round-trip
        through CPU. Omit to leave the model on whatever map_location
        placed it on (CPU by default).

    Returns
    -------
    DualSeatActorCritic
        In eval() mode.
    """
    effective_map_location = map_location
    if device is not None and map_location == "cpu" and device != "cpu":
        effective_map_location = device

    checkpoint = torch.load(Path(path), map_location=effective_map_location)

    if "variant" not in checkpoint or "actor_critic_config" not in checkpoint:
        raise ValueError(
            f"{path} is missing 'variant'/'actor_critic_config' metadata — "
            f"this checkpoint predates that being saved by save_checkpoint(). "
            f"See this function's docstring for how to load it anyway."
        )

    variant = PolicyVariant(checkpoint["variant"])
    cfg = ActorCriticConfig(**checkpoint["actor_critic_config"])

    actor_critic = build_dual_seat_actor_critic(
        variant, config=cfg, hand_encoder=hand_encoder
    )
    actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    if device is not None:
        actor_critic = actor_critic.to(device)
    actor_critic.eval()
    return actor_critic


def load_checkpoint(experiment: PushFoldDuoExperiment, path: Union[str, Path]) -> None:
    """
    Load actor_critic + both optimizers from `path` into `experiment`,
    in place.

    map_location is set to experiment.config.device so a checkpoint
    saved from a CUDA run can still be loaded on a CPU-only machine (or
    vice versa) without raising.

    Raises
    ------
    ValueError
        If the checkpoint's variant doesn't match `experiment`'s — the
        state_dict shapes (two heads vs. one) are incompatible across
        variants and loading would fail confusingly otherwise.
    """
    checkpoint = torch.load(Path(path), map_location=experiment.config.device)
    saved_variant = checkpoint.get("variant")
    if saved_variant is not None and saved_variant != experiment.config.variant.value:
        raise ValueError(
            f"Checkpoint variant {saved_variant!r} does not match "
            f"experiment variant {experiment.config.variant.value!r}."
        )
    experiment.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    experiment.loop.optimizers[SB_SEAT].load_state_dict(checkpoint["optimizer_sb_state_dict"])
    experiment.loop.optimizers[BB_SEAT].load_state_dict(checkpoint["optimizer_bb_state_dict"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a push-fold-duo PPO agent pair.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for the actor_critic, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    args = parser.parse_args()

    cfg = PushFoldDuoExperimentConfig(
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
        device=args.device,
    )

    experiment = build_push_fold_duo_experiment(cfg)
    history = run_push_fold_duo_experiment(experiment)

    for stats in history:
        active = stats.active_seat if stats.active_seat is not None else "both"
        print(
            f"iter {stats.iteration:4d}  active={active}  "
            f"reward_sb={stats.mean_reward_sb:+.3f}  "
            f"reward_bb={stats.mean_reward_bb:+.3f}  "
            f"policy_loss={stats.policy_loss}  "
            f"value_loss={stats.value_loss}"
        )

    if args.checkpoint_out:
        save_checkpoint(experiment, args.checkpoint_out)
        print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    _main()
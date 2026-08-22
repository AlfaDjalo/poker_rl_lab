"""
experiments/generic_tree_experiment.py — End-to-end training entry point
for ANY scenarios.duo_game_config.DuoGameConfig, via
envs.generic_tree_env.GenericTreeEnv /
scenarios.generic_tree_scenario_generator.GenericTreeScenarioGenerator /
trainers.generic_tree_training_loop.GenericTreeTrainingLoop.

Why this file exists instead of one bespoke *_experiment.py per game
------------------------------------------------------------------------
push_fold_experiment.py / river_experiment.py (single-hero BasePokerEnv
stages) and the earlier push_fold_duo_experiment.py (hardcoded SB_SEAT/
BB_SEAT two-agent stage) each wire up their own
scenario-generator/env/actor-critic/trainer stack by hand. Per
RL_Lab_Scenario_Framework - Addendum.md §3.3 step 4 ("generalize
PolicyValueNetwork/RepresentationModel... informed by what step 3
actually required"), the generic-tree path (steps 1-3 of that sequencing)
is now done: any push-fold-duo/river-duo-shaped game is fully described
by a DuoGameConfig YAML, executed by GenericTreeEnv, dealt by
GenericTreeScenarioGenerator, and trained by GenericTreeTrainingLoop —
all seat-count- and tree-shape-agnostic already (see those modules' own
docstrings). This file is the one remaining piece: turning a config NAME
("push_fold_duo", "river_duo", or any future scenarios/duo_games/*.yaml)
into a fully wired, ready-to-train bundle, the same role
push_fold_experiment.build_push_fold_experiment() plays for the
single-hero stages.

Building a two-head DualSeatActorCritic without touching
models/actor_critic_factory.py
--------------------------------------------------------------------------
models.actor_critic_factory.build_actor_critic() already does all the
encoder wiring (CardEncoder sharing, hand_encoder TF/Keras handling,
etc.) but only ever returns a single-head PokerActorCritic. Rather than
duplicating that wiring here (risking the two copies drifting), this
module builds one PokerActorCritic via build_actor_critic() as normal,
then — only for the "shared_trunk_two_heads" variant — reuses its
already-constructed `.representation_model` (a public attribute of
PokerActorCritic, models/policy_value_network.py) as the shared trunk
for a fresh pair of models.policy_value_network.PolicyValueNetwork heads,
and discards the single-head PokerActorCritic's own head. This is
exactly the "trunk shared, heads separate" composition
SharedTrunkTwoHeadsActorCritic (models/shared_trunk_two_heads_policy.py)
already expects as constructor arguments — no factory changes needed.
For the "shared_single_head" variant, the freshly-built PokerActorCritic
is used as-is, wrapped by SharedSingleHeadActorCritic
(models/shared_single_head_policy.py).

KNOWN LIMITATION (inherited, not introduced here)
------------------------------------------------------
Per generic_tree_training_loop.py's own docstring: both concrete
DualSeatActorCritic implementations validate `seat in (0, 1)` via
`_check_seat`. This module will happily build an experiment for a
config declaring 3+ seats (GenericTreeEnv / GenericTreeScenarioGenerator
/ GenericTreeTrainingLoop all support that), but the actor-critic
construction below will raise the moment training actually calls
act()/evaluate_actions() for seat index >= 2. That's a pre-existing
network-side gap (see RL_Lab_Scenario_Framework - Addendum.md §3.3 step
4), not something this file works around.

poker_eval shim
----------------
Same pattern as every other experiment/demo entry point in this
codebase (push_fold_experiment.py, river_experiment.py,
run_generic_tree_smoke_test.py): prefer the real compiled extension,
fall back to mock_poker_eval.py so this file — and everything it
imports transitively (loader.load_game via duo_game_config's Engine
game_name lookups, GenericTreeEnv's default CppScoringEngine ->
MockScoringEngine fallback) — stays importable and runnable without the
C++ extension.

Device placement
------------------
Same rationale as push_fold_experiment.py / river_experiment.py: the
hand encoder stays TF-native regardless of `device` (see
representation_model.py's "TensorFlow / PyTorch boundary" docstring);
only the PyTorch actor_critic is moved, and that move happens BEFORE
constructing any optimizer (moving a module after its optimizer is
built can silently leave the optimizer referencing stale parameter
objects).

Dependencies
------------
    PyTorch. scenarios.duo_game_config, scenarios.generic_tree_scenario_generator,
    envs.generic_tree_env, models.actor_critic_factory,
    models.policy_value_network, models.dual_seat_policy,
    models.shared_trunk_two_heads_policy, models.shared_single_head_policy,
    representation.observation_adapter, trainers.generic_tree_training_loop,
    trainers.duo_training_loop (DuoTrainingMode only).
"""

from __future__ import annotations

import sys
import logging
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import poker_rl_lab.poker_eval as poker_eval # noqa: F401  (prefer the real extension if present)
except ImportError:
    import poker_rl_lab.mock_poker_eval as mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from poker_rl_lab.scenarios.duo_game_config import DuoGameConfig, load_duo_game
from poker_rl_lab.scenarios.generic_tree_scenario_generator import GenericTreeScenarioGenerator
from poker_rl_lab.envs.generic_tree_env import GenericTreeEnv
from poker_rl_lab.models.actor_critic_factory import ActorCriticConfig, build_actor_critic
# from poker_rl_lab.models.policy_value_network import PolicyValueNetwork
from poker_rl_lab.models.policy_value_network import PokerActorCritic, PolicyValueNetwork
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.models.shared_trunk_two_heads_policy import SharedTrunkTwoHeadsActorCritic
from poker_rl_lab.models.shared_single_head_policy import SharedSingleHeadActorCritic
from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.observation_adapter import ObservationAdapter
from poker_rl_lab.trainers.duo_training_loop import DuoTrainingMode
from poker_rl_lab.trainers.generic_tree_training_loop import (
    GenericTreeIterationStats,
    GenericTreeTrainingLoop,
)

# _DEFAULT_ENCODER_PATH = "poker_rl_lab/encoders/encoder_model.keras"
# _DEFAULT_ENCODER_EMBEDDING_DIM = 32

_DEFAULT_ENCODER_PATH = (
    Path(__file__).resolve().parent.parent / "encoders" / "encoder_model.keras"
)
_DEFAULT_ENCODER_EMBEDDING_DIM = 32

# ---------------------------------------------------------------------------
# Policy variant selection
# ---------------------------------------------------------------------------

class PolicyVariant(Enum):
    """Which concrete DualSeatActorCritic implementation to build.
    Matches run_push_fold_duo_demo.py's --variant choices exactly."""
    SHARED_TRUNK_TWO_HEADS = "shared_trunk_two_heads"
    SHARED_SINGLE_HEAD = "shared_single_head"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GenericTreeExperimentConfig:
    """All knobs for one generic-tree training run, grouped by the piece
    of the pipeline they configure."""

    # Which scenarios/duo_games/<config_name>.yaml to load
    config_name: str = "push_fold_duo"
    config_path: Optional[Path] = None   # override load_duo_game's default lookup

    seed: Optional[int] = 42

    # Policy architecture
    variant: PolicyVariant = PolicyVariant.SHARED_TRUNK_TWO_HEADS
    actor_critic_config: ActorCriticConfig = field(default_factory=ActorCriticConfig)

    # GenericTreeEnv
    scoring_engine: object = None  # None -> GenericTreeEnv's own default chain
    num_showdown_samples: int = 30
    """Number of independent board runouts averaged per showdown when the
    board is incomplete at terminal time (e.g. push-fold-duo, which deals
    no board at all) — see envs.generic_tree_env.GenericTreeEnv's own
    docstring ("Monte Carlo showdown averaging"). Higher values trade
    more per-hand scoring cost for lower-variance reward signal, which
    matters most for exactly this game shape. Has no effect on configs
    whose board is already fully dealt (river_duo and similar)."""

    # # GenericTreeEnv
    # scoring_engine: object = None  # None -> GenericTreeEnv's own default chain

    # Device placement — see module docstring's "Device placement" section.
    device: str = "cpu"

    # Optimizers — one learning rate for every seat unless overridden.
    learning_rate: float = 3e-4
    learning_rates: Dict[int, float] = field(default_factory=dict)
    """Per-seat-index override; a seat missing here uses `learning_rate`."""

    # DuoTrainingLoop-style knobs (generic_tree_training_loop.GenericTreeTrainingLoop)
    mode: DuoTrainingMode = DuoTrainingMode.SIMULTANEOUS
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: Optional[float] = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95
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
class GenericTreeExperiment:
    """Every piece of one training/evaluation run, already wired together."""
    game_config: DuoGameConfig
    scenario_gen: GenericTreeScenarioGenerator
    env: GenericTreeEnv
    adapter: ObservationAdapter
    actor_critic: DualSeatActorCritic
    loop: GenericTreeTrainingLoop
    config: GenericTreeExperimentConfig


def _build_actor_critic(
    game_config: DuoGameConfig,
    cfg: GenericTreeExperimentConfig,
    hand_encoder: Optional[BaseHandEncoder],
) -> DualSeatActorCritic:
    """See module docstring's "Building a two-head DualSeatActorCritic"
    section for why this doesn't touch models/actor_critic_factory.py."""
    if hand_encoder is None:
        hand_encoder = _default_hand_encoder()

    inner = build_actor_critic(hand_encoder=hand_encoder, config=cfg.actor_critic_config)

    if cfg.variant == PolicyVariant.SHARED_SINGLE_HEAD:
        return SharedSingleHeadActorCritic(inner)

    if cfg.variant == PolicyVariant.SHARED_TRUNK_TWO_HEADS:
        if len(game_config.seats) != 2:
            raise ValueError(
                f"PolicyVariant.SHARED_TRUNK_TWO_HEADS names two heads "
                f"(seat 0 / seat 1) but config {game_config.name!r} "
                f"declares {len(game_config.seats)} seats. Use "
                f"SHARED_SINGLE_HEAD for non-2-seat configs, or extend "
                f"models/shared_trunk_two_heads_policy.py for N heads."
            )
        representation_model = inner.representation_model
        head_a = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            hidden_dim=cfg.actor_critic_config.policy_hidden_dim,
            num_shared_layers=cfg.actor_critic_config.policy_num_shared_layers,
        )
        head_b = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            hidden_dim=cfg.actor_critic_config.policy_hidden_dim,
            num_shared_layers=cfg.actor_critic_config.policy_num_shared_layers,
        )
        return SharedTrunkTwoHeadsActorCritic(representation_model, head_a, head_b)

    raise ValueError(f"Unknown PolicyVariant {cfg.variant!r}")


def build_generic_tree_experiment(
    config: Optional[GenericTreeExperimentConfig] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
) -> GenericTreeExperiment:
    """
    Construct every piece of a generic-tree training pipeline for
    `config.config_name`, wired together but not yet run.

    Parameters
    ----------
    config : GenericTreeExperimentConfig, optional
        Defaults to GenericTreeExperimentConfig() (push_fold_duo,
        shared_trunk_two_heads) if omitted.
    hand_encoder : BaseHandEncoder, optional
        Forwarded to models.actor_critic_factory.build_actor_critic() —
        see that function's docstring. Omit to get an UNTRAINED
        PretrainedHandEncoder (pipeline testing only).

    Returns
    -------
    GenericTreeExperiment
    """
    cfg = config or GenericTreeExperimentConfig()

    game_config = load_duo_game(cfg.config_name, path=cfg.config_path)

    scenario_gen = GenericTreeScenarioGenerator(game_config)
    if cfg.seed is not None:
        scenario_gen.seed(cfg.seed)

    env_kwargs = {}
    if cfg.scoring_engine is not None:
        env_kwargs["scoring_engine"] = cfg.scoring_engine
    env_kwargs["num_showdown_samples"] = cfg.num_showdown_samples
    env = GenericTreeEnv(game_config, **env_kwargs)

    # env_kwargs = {}
    # if cfg.scoring_engine is not None:
    #     env_kwargs["scoring_engine"] = cfg.scoring_engine
    # env = GenericTreeEnv(game_config, **env_kwargs)

    adapter = ObservationAdapter()

    actor_critic = _build_actor_critic(game_config, cfg, hand_encoder)

    # Move to the target device BEFORE constructing optimizers — see
    # module docstring's "Device placement" section.
    actor_critic = actor_critic.to(cfg.device)

    optimizers: Dict[int, torch.optim.Optimizer] = {}
    for seat in game_config.seats:
        lr = cfg.learning_rates.get(seat.index, cfg.learning_rate)
        optimizers[seat.index] = torch.optim.Adam(
            actor_critic.seat_parameters(seat.index), lr=lr
        )

    loop = GenericTreeTrainingLoop(
        env=env,
        scenario_gen=scenario_gen,
        actor_critic=actor_critic,
        adapter=adapter,
        optimizers=optimizers,
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

    return GenericTreeExperiment(
        game_config=game_config,
        scenario_gen=scenario_gen,
        env=env,
        adapter=adapter,
        actor_critic=actor_critic,
        loop=loop,
        config=cfg,
    )


def run_generic_tree_experiment(
    experiment: GenericTreeExperiment,
) -> List[GenericTreeIterationStats]:
    """
    Run experiment.config's configured number of training iterations.

    Returns
    -------
    List[GenericTreeIterationStats]
        One entry per iteration — see
        trainers.generic_tree_training_loop.GenericTreeTrainingLoop.train().
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

def save_checkpoint(experiment: GenericTreeExperiment, path: Union[str, Path]) -> None:
    """
    Save actor_critic + every seat's optimizer state to `path`.

    Note: does not persist `experiment.config` (architecture/hyperparameters)
    — see load_checkpoint()'s docstring for why the caller must rebuild
    with matching config before load_checkpoint() can succeed, mirroring
    push_fold_experiment.save_checkpoint / river_experiment.save_checkpoint's
    identical convention.
    """
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


def load_checkpoint(experiment: GenericTreeExperiment, path: Union[str, Path]) -> None:
    """
    Load actor_critic + every seat's optimizer state from `path` into
    `experiment`, in place.

    `experiment` must have been built with build_generic_tree_experiment()
    using the SAME config_name / variant / actor_critic_config that
    produced the checkpoint (save_checkpoint() does not store
    architecture — see that function's docstring), or
    load_state_dict() will raise a shape/key mismatch.

    map_location is set to experiment.config.device so a checkpoint
    saved from a CUDA run can still be loaded on a CPU-only machine (or
    vice versa) without raising.
    """
    checkpoint = torch.load(Path(path), map_location=experiment.config.device)
    experiment.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    for seat, state_dict in checkpoint["optimizer_state_dicts"].items():
        experiment.loop.optimizers[seat].load_state_dict(state_dict)

# ---------------------------------------------------------------------------
# Inference-only builders (NEW)
# ---------------------------------------------------------------------------
# push_fold_service.py / trainer_service.py only need a DualSeatActorCritic
# to call .act()/.act_batch() against — they don't need a full
# GenericTreeExperiment (env, scenario_gen, optimizers, training loop).
# Building a full experiment would also force them to pick a specific
# DuoGameConfig name whose fields (stack/pot/deal) have zero bearing on
# the network's own architecture — only seat count matters, and every
# duo game shipped so far is 2 seats. So these two functions build/load
# just the actor-critic directly, reusing _build_actor_critic() above.

def build_dual_seat_actor_critic(
    variant: PolicyVariant,
    hand_encoder: Optional[BaseHandEncoder] = None,
    actor_critic_config: Optional[ActorCriticConfig] = None,
) -> DualSeatActorCritic:
    """
    Build a fresh (untrained unless hand_encoder carries pretrained
    weights) two-seat policy network, with no env/scenario_gen/optimizer
    wiring attached.

    Mirrors _build_actor_critic()'s SHARED_TRUNK_TWO_HEADS /
    SHARED_SINGLE_HEAD branching exactly, but doesn't require a
    DuoGameConfig (the seat-count==2 check that function performs
    against game_config.seats is simply assumed here, since every
    caller of this function is heads-up).
    """
    cfg = actor_critic_config or ActorCriticConfig()

    if hand_encoder is None:
        hand_encoder = _default_hand_encoder()

    inner = build_actor_critic(hand_encoder=hand_encoder, config=cfg)

    if variant == PolicyVariant.SHARED_SINGLE_HEAD:
        return SharedSingleHeadActorCritic(inner)

    if variant == PolicyVariant.SHARED_TRUNK_TWO_HEADS:
        representation_model = inner.representation_model
        head_a = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            hidden_dim=cfg.policy_hidden_dim,
            num_shared_layers=cfg.policy_num_shared_layers,
        )
        head_b = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            hidden_dim=cfg.policy_hidden_dim,
            num_shared_layers=cfg.policy_num_shared_layers,
        )
        return SharedTrunkTwoHeadsActorCritic(representation_model, head_a, head_b)

    raise ValueError(f"Unknown PolicyVariant {variant!r}")


def load_dual_seat_actor_critic(
    path: Union[str, Path],
    variant: Optional[PolicyVariant] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
    actor_critic_config: Optional[ActorCriticConfig] = None,
) -> DualSeatActorCritic:
    """
    Build a DualSeatActorCritic and load weights from a checkpoint saved
    by save_checkpoint() (or torch.save({"actor_critic_state_dict": ...})
    in the same shape).

    variant defaults to SHARED_TRUNK_TWO_HEADS — matches every current
    caller's expectation (push_fold_duo's training_config.yaml) unless
    a scenario's own policy.variant says otherwise, in which case the
    caller should pass it explicitly (see trainer_service.py's
    _try_build_real_agent, which does).
    """
    variant = variant or PolicyVariant.SHARED_TRUNK_TWO_HEADS
    agent = build_dual_seat_actor_critic(
        variant, hand_encoder=hand_encoder, actor_critic_config=actor_critic_config
    )
    checkpoint = torch.load(Path(path), map_location="cpu")
    state_dict = checkpoint.get("actor_critic_state_dict", checkpoint)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent

def _default_hand_encoder() -> BaseHandEncoder:
    """
    Load the real pretrained hand encoder from _DEFAULT_ENCODER_PATH.
    Falls back to an untrained copy (with a loud stderr warning) if the
    file is missing or fails architecture validation. Mirrors
    experiments/push_fold_experiment.py's own _default_hand_encoder() —
    duplicated here because the generic-tree path had no equivalent at
    all, which was silently producing an untrained hand encoder on
    every build_generic_tree_experiment() call that didn't pass
    hand_encoder= explicitly.
    """
    from poker_rl_lab.representation.pretrained_hand_encoder import (
        HandEncoderArchitectureMismatch,
        PretrainedHandEncoder,
    )
    resolved_path = str(_DEFAULT_ENCODER_PATH)

    try:
        encoder = PretrainedHandEncoder.from_file(
            resolved_path,
            expected_embedding_dim=_DEFAULT_ENCODER_EMBEDDING_DIM,
        )
        assert encoder.is_pretrained  # from_file() always sets this True
        msg = (
            f"[generic_tree_experiment] Loaded PRETRAINED hand encoder from "
            f"{resolved_path} (embedding_dim={encoder.embedding_dim}, "
            f"is_pretrained={encoder.is_pretrained}). This encoder is shared "
            f"by both the hole-card and board CompositeEncoders."
        )
        print(msg, file=sys.stderr)
        logger.info(msg)
        return encoder

    except FileNotFoundError:
        msg = (
            f"[generic_tree_experiment] hand encoder file NOT FOUND at "
            f"{resolved_path!r} — falling back to an UNTRAINED hand encoder "
            f"(is_pretrained=False, NO poker knowledge). This affects BOTH "
            f"the hole-card encoder AND the board encoder, since "
            f"build_actor_critic() shares one hand_encoder instance between "
            f"them. Pass hand_encoder=PretrainedHandEncoder.from_file(...) "
            f"explicitly with the correct path to use real weights."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
        return PretrainedHandEncoder.untrained()
    
    except HandEncoderArchitectureMismatch as exc:
        msg = (
            f"[generic_tree_experiment] {resolved_path!r} FAILED architecture "
            f"validation ({exc}) — falling back to an UNTRAINED hand encoder "
            f"(is_pretrained=False). Affects both hole-card and board "
            f"encoders — see above."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        logger.warning(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
        return PretrainedHandEncoder.untrained()
    


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a two-seat PPO agent pair against any DuoGameConfig."
    )
    parser.add_argument("--config-name", type=str, default="push_fold_duo")
    parser.add_argument(
        "--variant", type=str, default="shared_trunk_two_heads",
        choices=[v.value for v in PolicyVariant],
    )
    parser.add_argument(
        "--mode", type=str, default="simultaneous",
        choices=[m.value for m in DuoTrainingMode],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes-per-iteration", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--iters-per-seat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--checkpoint-in", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
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
        learning_rate=args.learning_rate,
        iters_per_seat=args.iters_per_seat,
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        device=args.device,
    )

    experiment = build_generic_tree_experiment(cfg)

    if args.checkpoint_in:
        load_checkpoint(experiment, args.checkpoint_in)
        print(f"Resumed from checkpoint {args.checkpoint_in}")

    print(f"config: {cfg}")
    history = run_generic_tree_experiment(experiment)

    for i, stats in enumerate(history):
        active = stats.active_seat if stats.active_seat is not None else "both"
        rewards = " ".join(
            f"seat{seat}={r:+.3f}" for seat, r in stats.mean_reward_by_seat.items()
        )
        print(
            f"iter {i:4d}  active={active!s:>4}  {rewards}  "
            f"policy_loss={stats.policy_loss}  value_loss={stats.value_loss}"
        )

    if args.checkpoint_out:
        save_checkpoint(experiment, args.checkpoint_out)
        print(f"Saved checkpoint to {args.checkpoint_out}")

# ============================================================================
# ADD-ON: append everything below into poker_rl_lab/experiments/generic_tree_experiment.py
#
# ALSO EDIT the existing import line near the top of that file:
#
#     from poker_rl_lab.models.policy_value_network import PolicyValueNetwork
#
# to:
#
#     from poker_rl_lab.models.policy_value_network import PokerActorCritic, PolicyValueNetwork
#
# (PokerActorCritic is needed by build_dual_seat_actor_critic_legacy's
# SHARED_SINGLE_HEAD branch below, to wrap the legacy RepresentationModel
# exactly the way build_dual_seat_actor_critic's non-legacy version does
# via actor_critic_factory.build_actor_critic() internally.)
# ============================================================================

# ---------------------------------------------------------------------------
# Legacy-architecture builders/loaders (pre-composite-encoder checkpoints)
# ---------------------------------------------------------------------------
#
# Why this exists
# ------------------
# actor_critic_factory.build_actor_critic() wraps hole cards in
# CompositeHandEncoder and boards in CompositeBoardEncoder (see that
# module's own docstring — "decompose-and-pool" adapters built on top
# of the frozen pretrained CardSetEncoder). Any checkpoint saved BEFORE
# that refactor used a materially different RepresentationModel:
#
#   - hand_encoder was the RAW BaseHandEncoder (e.g. PretrainedHandEncoder)
#     passed straight through, never wrapped, never registered as an
#     nn.Module (see representation_model.py: "NOT an nn.Module
#     (TF-backed); stored, not registered") — so it contributes ZERO
#     keys to state_dict() at all.
#   - board_encoder was the from-scratch representation.board_encoder.
#     BoardEncoder (its own trainable CardEncoder + a small Transformer
#     encoder layer over dealt board cards), not CompositeBoardEncoder.
#
# Loading such a checkpoint against the CURRENT architecture fails with
# a state_dict key mismatch (missing CompositeHandEncoder/
# CompositeBoardEncoder keys, unexpected legacy BoardEncoder keys) —
# NOT a corrupt file, NOT a wrong variant, just an architecture that
# has since changed. build_dual_seat_actor_critic_legacy() reconstructs
# the OLD shape so such a checkpoint can still be loaded and played
# against, without requiring a retrain.
#
# Best-effort caveat
# ---------------------
# This reconstructs the legacy architecture using ActorCriticConfig's
# CURRENT default hyperparameters (board_output_dim, point_output_dim,
# etc.) for every component that ISN'T the hand/board encoder split
# itself, since those defaults are the only source of truth available
# here — there is no record of what hyperparameters actually produced
# a given legacy checkpoint. If the checkpoint was trained with
# non-default sizes, load_state_dict() will raise a SHAPE mismatch
# (caught and reported by the caller, e.g. trainer_service.py's
# pipeline_trace) rather than silently loading wrong-shaped weights —
# torch.nn.Module.load_state_dict() checks tensor shapes, not just key
# names, so this fails loudly rather than corrupting inference.
# ---------------------------------------------------------------------------

def _build_legacy_representation_model(hand_encoder, cfg: ActorCriticConfig):
    """
    Reconstructs the pre-composite-encoder RepresentationModel shape:
    raw (unwrapped) hand_encoder + a from-scratch board_encoder.BoardEncoder
    instead of CompositeBoardEncoder. Every other component (point/game/
    position/history/stack encoders, the final combination MLP) is
    unchanged from the current architecture — the mismatch reported by
    a legacy checkpoint load is only ever in hand_encoder.*/board_encoder.*
    keys, never these.
    """
    from poker_rl_lab.representation.card_encoder import CardEncoder
    from poker_rl_lab.representation.board_encoder import BoardEncoder
    from poker_rl_lab.representation.point_encoder import PointEncoder
    from poker_rl_lab.representation.game_encoder import GameEncoder
    from poker_rl_lab.representation.position_encoder import PositionEncoder
    from poker_rl_lab.representation.history_encoder import HistoryEncoder
    from poker_rl_lab.representation.stack_encoder import StackEncoder
    from poker_rl_lab.representation.representation_model import RepresentationModel

    # BoardEncoder owns its own CardEncoder for embedding board cards —
    # deliberately separate from the frozen hand_encoder pipeline (see
    # board_encoder.py's own docstring: "deliberately separate from
    # PretrainedHandEncoder.encode_board()"). Defaults (rank_dim=16,
    # suit_dim=8) match what CardEncoder itself defaults to.
    card_encoder = CardEncoder()

    board_encoder = BoardEncoder(
        card_encoder=card_encoder,
        output_dim=cfg.board_output_dim,
        # num_heads=4, num_layers=1, dropout=0.0 — BoardEncoder's own
        # defaults; the reported key mismatch only ever shows
        # "attention.layers.0", consistent with num_layers=1.
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
        hand_encoder=hand_encoder,  # RAW — not wrapped in CompositeHandEncoder
        board_encoder=board_encoder,  # legacy BoardEncoder — not CompositeBoardEncoder
        point_encoder=point_encoder,
        game_encoder=game_encoder,
        position_encoder=position_encoder,
        history_encoder=history_encoder,
        stack_encoder=stack_encoder,
        hidden_dim=cfg.representation_hidden_dim,
        output_dim=cfg.representation_output_dim,
        dropout=cfg.representation_dropout,
    )


def build_dual_seat_actor_critic_legacy(
    variant: PolicyVariant,
    hand_encoder: Optional[BaseHandEncoder] = None,
    actor_critic_config: Optional[ActorCriticConfig] = None,
) -> DualSeatActorCritic:
    """
    Legacy-architecture sibling of build_dual_seat_actor_critic() — see
    module-level "Legacy-architecture builders/loaders" section above
    for what's different and why.
    """
    cfg = actor_critic_config or ActorCriticConfig()

    if hand_encoder is None:
        hand_encoder = _default_hand_encoder()

    representation_model = _build_legacy_representation_model(hand_encoder, cfg)

    if variant == PolicyVariant.SHARED_SINGLE_HEAD:
        policy_value_network = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            action_dim=NUM_ACTIONS,
            hidden_dim=cfg.policy_hidden_dim,
            num_shared_layers=cfg.policy_num_shared_layers,
        )
        inner = PokerActorCritic(representation_model, policy_value_network)
        return SharedSingleHeadActorCritic(inner)

    if variant == PolicyVariant.SHARED_TRUNK_TWO_HEADS:
        head_a = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            hidden_dim=cfg.policy_hidden_dim,
            num_shared_layers=cfg.policy_num_shared_layers,
        )
        head_b = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            hidden_dim=cfg.policy_hidden_dim,
            num_shared_layers=cfg.policy_num_shared_layers,
        )
        return SharedTrunkTwoHeadsActorCritic(representation_model, head_a, head_b)

    raise ValueError(f"Unknown PolicyVariant {variant!r}")


def load_dual_seat_actor_critic_legacy(
    path: Union[str, Path],
    variant: Optional[PolicyVariant] = None,
    hand_encoder: Optional[BaseHandEncoder] = None,
    actor_critic_config: Optional[ActorCriticConfig] = None,
) -> DualSeatActorCritic:
    """
    Legacy-architecture sibling of load_dual_seat_actor_critic() — same
    contract, built against build_dual_seat_actor_critic_legacy()
    instead. Intended to be tried automatically as a fallback when
    load_dual_seat_actor_critic() raises a state_dict key-mismatch
    RuntimeError (see trainer_service.py's _try_build_real_agent).
    """
    variant = variant or PolicyVariant.SHARED_TRUNK_TWO_HEADS
    agent = build_dual_seat_actor_critic_legacy(
        variant, hand_encoder=hand_encoder, actor_critic_config=actor_critic_config
    )
    checkpoint = torch.load(Path(path), map_location="cpu")
    state_dict = checkpoint.get("actor_critic_state_dict", checkpoint)
    agent.load_state_dict(state_dict)
    agent.eval()
    return agent



if __name__ == "__main__":
    _main()
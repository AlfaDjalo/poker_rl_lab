"""
models/actor_critic_factory.py — Builds a fully-wired PokerActorCritic.

representation_model.py's RepresentationModel takes seven already-
constructed component encoders as constructor arguments (by design —
see that module's docstring: "Pass already-constructed instances so
callers control hyperparameters and, critically, so the SAME
CardEncoder instance can be shared between board_encoder and anywhere
else per-card embeddings are needed"). Wiring those seven encoders plus
the hand encoder plus PolicyValueNetwork by hand in every training
script/notebook/test is repetitive and easy to get subtly wrong (e.g.
forgetting to share the CardEncoder instance, or mismatching latent_dim
between RepresentationModel and PolicyValueNetwork — both of which
PokerActorCritic's own __init__ already guards against, but only once
you've gotten this far). This module is that wiring, done once, with
sensible defaults matching each encoder's own defaults.

Not a general-purpose DI framework — deliberately just one function
(build_actor_critic) plus one dataclass of hyperparameters
(ActorCriticConfig), per the guide's "prioritise correctness,
simplicity" (§11).

Hand encoder note
--------------------
build_actor_critic does NOT import representation.pretrained_hand_encoder
(TF/Keras, per that module's docstring) at module level — only inside
the function body, and only if the caller didn't supply their own
hand_encoder. This keeps this module importable (and cheaply testable
with a stub BaseHandEncoder) without a hard TensorFlow dependency for
callers who don't need the real one yet.

Composite (pretrained-encoder-backed) hole/board encoders
--------------------------------------------------------------
Both the hand encoder and the board encoder are now wrapped in a
decompose-and-pool adapter on top of the pretrained CardSetEncoder,
rather than using it directly (hole cards) or a from-scratch
CardEncoder-based module (board):

    CompositeHandEncoder  (representation/composite_hand_encoder.py)
        Wraps the raw pretrained BaseHandEncoder so hole_card_ids of
        ANY length — 2 (hold'em), 4 (Omaha), or a hole+board composite
        of any size — is safe to feed through
        RepresentationModel.forward_batch()'s existing
        hand_encoder.encode_batch(...) call, by decomposing
        out-of-distribution sizes into native {2,3,5}-card subsets and
        learned-pooling them. Sizes already native (2, 3, 5) pass
        straight through with no decomposition overhead.

    CompositeBoardEncoder  (representation/composite_board_encoder.py)
        Same decompose-and-pool strategy, applied to
        RepresentationInput.board_card_ids. Drop-in replacement for the
        from-scratch BoardEncoder — identical forward()/forward_batch()
        contract, so RepresentationModel and observation_adapter.py
        needed zero changes.

Both wrap the SAME underlying pretrained hand_encoder instance (one
frozen Keras model, per the existing "Option B: hand encoder is a
frozen, non-PyTorch feature source" boundary — representation_model.py)
— so hole and board embeddings are built from a single shared card-set
representation, with two SEPARATE small trainable adapters (attention
pooling + projection) on top, one per role. This mirrors CardEncoder's
own existing sharing pattern (one embedding table, reused by
board_encoder and anywhere else per-card embeddings are needed) at the
composite-set level instead of the per-card level.

Frozen base + trainable adapter (why not fine-tune the base)
------------------------------------------------------------------
The pretrained Keras weights are never updated by PPO — every
composite encoder's own forward pass detaches the base_encoder's output
before pooling. Only each encoder's attn_scorer + project layers are
trainable. This avoids catastrophic forgetting of the pretrained
hand-strength structure under RL's noisier, sparser gradient signal,
and keeps the whole pipeline on the documented TF/PyTorch boundary
(no backprop into TF). See representation/composite_board_encoder.py's
and representation/composite_hand_encoder.py's own docstrings for the
fuller rationale. `project_hidden_dim` on both classes controls whether
that trainable adapter is a single Linear or a small 2-layer MLP —
default now uses the MLP (more capacity to reshape the frozen
embedding) rather than a bare Linear.

Stack/Pot encoder (added)
--------------------------
RepresentationModel now takes a seventh component encoder,
representation.stack_encoder.StackEncoder — see that module's and
representation_model.py's own docstrings for why (the Representation
Model previously carried no stack/pot/bet-to-call signal at all,
starving every downstream policy of stack-depth awareness). Wired here
identically to every other encoder: sensible config-driven defaults,
built fresh per build_actor_critic() call, no sharing required (unlike
CardEncoder, StackEncoder owns no embedding table another encoder needs
to reuse).

Dependencies
------------
    PyTorch. representation.* (composite hand/board encoders, point/
    game/position/history/stack encoders, representation_model),
    models.policy_value_network, actions.abstract_action.NUM_ACTIONS.
    TensorFlow only if hand_encoder is omitted (lazy import — see
    above).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from poker_rl_lab.actions.abstract_action import NUM_ACTIONS
from poker_rl_lab.models.policy_value_network import PokerActorCritic, PolicyValueNetwork
from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.composite_hand_encoder import CompositeHandEncoder
from poker_rl_lab.representation.composite_board_encoder import CompositeBoardEncoder
from poker_rl_lab.representation.game_encoder import GameEncoder
from poker_rl_lab.representation.history_encoder import HistoryEncoder
from poker_rl_lab.representation.point_encoder import PointEncoder
from poker_rl_lab.representation.position_encoder import PositionEncoder
from poker_rl_lab.representation.stack_encoder import StackEncoder
from poker_rl_lab.representation.representation_model import RepresentationModel


@dataclass
class ActorCriticConfig:
    """
    Hyperparameters for build_actor_critic(). Every field mirrors a
    constructor argument of the corresponding component encoder or
    network — defaults here match each encoder's own default (see
    representation/*.py, models/policy_value_network.py) rather than
    introducing a second, possibly-drifting set of defaults.
    """
    # CompositeHandEncoder (hole cards / hole+board composites)
    hand_output_dim: Optional[int] = None   # None -> base_encoder.embedding_dim, no projection
    hand_attn_hidden_dim: int = 32
    hand_project_hidden_dim: Optional[int] = 64   # MLP adapter by default — see module docstring
    hand_max_subsets: int = 12

    # CompositeBoardEncoder
    board_output_dim: int = 32
    board_attn_hidden_dim: int = 32
    board_project_hidden_dim: Optional[int] = 64  # MLP adapter by default — see module docstring
    board_max_subsets: int = 12

    # PointEncoder
    point_score_type_dim: int = 8
    point_showdown_type_dim: int = 8
    point_scalar_hidden_dim: int = 8
    point_output_dim: int = 16

    # GameEncoder
    game_betting_type_dim: int = 8
    game_payout_type_dim: int = 8
    game_showdown_type_dim: int = 8
    game_scalar_hidden_dim: int = 16
    game_output_dim: int = 16

    # PositionEncoder
    position_hidden_dim: int = 16
    position_output_dim: int = 16

    # HistoryEncoder
    history_action_type_dim: int = 8
    history_scalar_hidden_dim: int = 8
    history_gru_hidden_dim: int = 32
    history_output_dim: int = 16

    # StackEncoder
    stack_hidden_dim: int = 16
    stack_output_dim: int = 16

    # RepresentationModel combination MLP
    representation_hidden_dim: int = 256
    representation_output_dim: int = 128
    representation_dropout: float = 0.0

    # PolicyValueNetwork
    policy_hidden_dim: int = 128
    policy_num_shared_layers: int = 1


def build_actor_critic(
    hand_encoder: Optional[BaseHandEncoder] = None,
    config: Optional[ActorCriticConfig] = None,
) -> PokerActorCritic:
    """
    Construct a fully-wired PokerActorCritic from an ActorCriticConfig.

    Parameters
    ----------
    hand_encoder : BaseHandEncoder, optional
        The RAW (base) pretrained hole-card embedding source
        (base_hand_encoder.py) — NOT yet wrapped in CompositeHandEncoder;
        this function does that wrapping internally, twice (once for
        hole cards, once for boards), sharing this same base instance
        between both wrappers. If omitted, this function lazily imports
        representation.pretrained_hand_encoder.PretrainedHandEncoder and
        builds an UNTRAINED copy via .untrained() — sufficient for
        pipeline/shape testing (see that module's own docstring) but
        carrying NO poker knowledge; do not use the omitted-argument
        path for real training. Once the pretrained .keras file is
        available, pass
        ``PretrainedHandEncoder.from_file("path/to/encoder_model.keras")``
        explicitly instead.
    config : ActorCriticConfig, optional
        Hyperparameters. Defaults to ``ActorCriticConfig()`` (every
        encoder's own defaults) if omitted.

    Returns
    -------
    PokerActorCritic
        Ready for PPOTrainer.collect_rollouts() / .update(), or direct
        .act() / .evaluate_actions() calls.
    """
    cfg = config or ActorCriticConfig()

    if hand_encoder is None:
        # Lazy import: keeps this module (and anything that only wants
        # ActorCriticConfig / the wiring logic) importable without a
        # hard TensorFlow dependency. See module + function docstrings.
        from poker_rl_lab.representation.pretrained_hand_encoder import PretrainedHandEncoder

        hand_encoder = PretrainedHandEncoder.untrained()

    # Both composite encoders wrap the SAME base pretrained instance —
    # one frozen Keras model, two separate small trainable adapters on
    # top (hole-role pooling/projection vs. board-role pooling/
    # projection). See module docstring's "Composite encoders" section.
    composite_hand_encoder = CompositeHandEncoder(
        base_encoder=hand_encoder,
        output_dim=cfg.hand_output_dim,
        attn_hidden_dim=cfg.hand_attn_hidden_dim,
        project_hidden_dim=cfg.hand_project_hidden_dim,
        max_subsets=cfg.hand_max_subsets,
    )

    board_encoder = CompositeBoardEncoder(
        hand_encoder=hand_encoder,
        output_dim=cfg.board_output_dim,
        attn_hidden_dim=cfg.board_attn_hidden_dim,
        project_hidden_dim=cfg.board_project_hidden_dim,
        max_subsets=cfg.board_max_subsets,
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

    representation_model = RepresentationModel(
        hand_encoder=composite_hand_encoder,
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

    policy_value_network = PolicyValueNetwork(
        latent_dim=representation_model.output_dim,
        action_dim=NUM_ACTIONS,
        hidden_dim=cfg.policy_hidden_dim,
        num_shared_layers=cfg.policy_num_shared_layers,
    )

    return PokerActorCritic(representation_model, policy_value_network)
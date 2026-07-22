"""
models/actor_critic_factory.py — Builds a fully-wired PokerActorCritic.

representation_model.py's RepresentationModel takes six already-
constructed component encoders as constructor arguments (by design —
see that module's docstring: "Pass already-constructed instances so
callers control hyperparameters and, critically, so the SAME
CardEncoder instance can be shared between board_encoder and anywhere
else per-card embeddings are needed"). Wiring those six encoders plus
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

Dependencies
------------
    PyTorch. representation.* (card/board/point/game/position/history
    encoders, representation_model), models.policy_value_network,
    actions.abstract_action.NUM_ACTIONS. TensorFlow only if hand_encoder
    is omitted (lazy import — see above).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from actions.abstract_action import NUM_ACTIONS
from models.policy_value_network import PokerActorCritic, PolicyValueNetwork
from representation.base_hand_encoder import BaseHandEncoder
from representation.board_encoder import BoardEncoder
from representation.card_encoder import CardEncoder
from representation.game_encoder import GameEncoder
from representation.history_encoder import HistoryEncoder
from representation.point_encoder import PointEncoder
from representation.position_encoder import PositionEncoder
from representation.representation_model import RepresentationModel


@dataclass
class ActorCriticConfig:
    """
    Hyperparameters for build_actor_critic(). Every field mirrors a
    constructor argument of the corresponding component encoder or
    network — defaults here match each encoder's own default (see
    representation/*.py, models/policy_value_network.py) rather than
    introducing a second, possibly-drifting set of defaults.
    """
    # CardEncoder (shared between BoardEncoder and, eventually, any
    # other per-card embedding consumer).
    rank_dim: int = 16
    suit_dim: int = 8

    # BoardEncoder
    board_output_dim: int = 32
    board_num_heads: int = 4
    board_num_layers: int = 1
    board_dropout: float = 0.0

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
        Hole-card embedding source (base_hand_encoder.py). If omitted,
        this function lazily imports
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
        from representation.pretrained_hand_encoder import PretrainedHandEncoder

        hand_encoder = PretrainedHandEncoder.untrained()

    # CardEncoder is shared: BoardEncoder consumes it today, and per
    # board_encoder.py's docstring, any future per-card embedding
    # consumer added to RepresentationModel should reuse this same
    # instance rather than constructing its own table.
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

    representation_model = RepresentationModel(
        hand_encoder=hand_encoder,
        board_encoder=board_encoder,
        point_encoder=point_encoder,
        game_encoder=game_encoder,
        position_encoder=position_encoder,
        history_encoder=history_encoder,
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
"""
models/actor_critic_factory.py — Builds a fully-wired PokerActorCritic.

Three-role pretrained encoder wiring
--------------------------------------
The pretrained base_encoder (representation/pretrained_hand_encoder.py)
is a poker-SEMANTIC card-set encoder (trained on 2/3/5-card sets for
hand type + strength), not exclusively a "hole cards" encoder. This
function now builds THREE separate trainable adapters over the SAME
frozen base, each targeting the native size that actually matches its
poker concept (see representation/composite_board_encoder.py's module
docstring):

    hand_encoder            hole cards only, 2-card native (or Omaha's
                             C(4,2) via CompositeHandEncoder — unchanged)
    board_texture_encoder   board alone, decomposed into 3-card windows
                             (CompositeBoardEncoder, window=3)
    made_hand_encoder       hole ∪ board, decomposed into legal 5-card
                             candidate hands per the game's showdown_type
                             (CompositeBoardEncoder, window=5, via
                             forward_batch_made_hand() — see
                             representation/showdown_constraints.py for
                             Hold'em/Omaha/Make-5 legality bounds)

Each adapter has its own attn_scorer/project — separately trainable,
despite sharing the same frozen base_encoder — since each is solving a
different pooling problem (texture vs. made-hand strength).

Configurable action_dim (for graph-native scenarios)
---------------------------------------------------------
ActorCriticConfig.action_dim now defaults to
actions.abstract_action.NUM_ACTIONS (the old fixed 9-value abstract
BETTING-only action space) purely for backward compatibility with
anything not yet migrated. A graph-native scenario
(scenarios.graph_scenario_config.GraphScenarioConfig) has its own,
generally DIFFERENT action-space width — see
GraphScenarioConfig.max_action_options's own docstring: it covers
whichever DecisionDomain(s) the scenario is actually in scope for
(BETTING and/or CARD_SELECT/CARD_PASS/BOOLEAN/CHOICE), sized from that
scenario's own action_abstraction, not a single game-wide constant.
Callers building a policy for a graph scenario should pass
ActorCriticConfig(action_dim=graph_scenario_config.max_action_options,
...) explicitly.

Dependencies
------------
    PyTorch. representation.composite_hand_encoder,
    representation.composite_board_encoder, representation.point_encoder,
    representation.game_encoder, representation.position_encoder,
    representation.history_encoder, representation.stack_encoder,
    representation.representation_model, models.policy_value_network,
    actions.abstract_action.NUM_ACTIONS. TensorFlow only if hand_encoder
    is omitted (lazy import).
"""

from __future__ import annotations

import logging
import sys
import warnings
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

logger = logging.getLogger(__name__)


@dataclass
class ActorCriticConfig:
    """
    Hyperparameters for build_actor_critic(). Defaults mirror each
    encoder's own defaults.
    """
    # CompositeHandEncoder (hole cards only)
    hand_output_dim: Optional[int] = None
    hand_attn_hidden_dim: int = 32
    hand_project_hidden_dim: Optional[int] = 64
    hand_max_subsets: int = 12

    # CompositeBoardEncoder — board texture role (window=3)
    board_output_dim: int = 32
    board_attn_hidden_dim: int = 32
    board_project_hidden_dim: Optional[int] = 64
    board_max_subsets: int = 12
    board_window: int = 3

    # CompositeBoardEncoder — made-hand role (window=5, constraint-aware)
    made_hand_output_dim: int = 32
    made_hand_attn_hidden_dim: int = 32
    made_hand_project_hidden_dim: Optional[int] = 64
    made_hand_max_subsets: int = 12
    made_hand_window: int = 5

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
    action_dim: int = NUM_ACTIONS
    """
    Width of the policy's categorical action head. Defaults to the old
    fixed 9-value abstract action space (actions.abstract_action.
    NUM_ACTIONS) for backward compatibility with any caller not yet
    migrated to the graph-native path. For a graph-native scenario, pass
    scenarios.graph_scenario_config.GraphScenarioConfig.max_action_options
    instead — the action space is no longer one fixed game-wide
    constant now that different domains (BETTING/CARD_SELECT/CARD_PASS/
    BOOLEAN/CHOICE) and different scenarios' own bet-sizing abstractions
    can each produce a different option count.
    envs.graph_scenario_env.GraphScenarioEnv always pads
    GraphDecisionObs.legal_mask out to exactly the scenario's
    max_action_options, so passing a narrower action_dim here would
    surface immediately as a legal_mask/logits shape mismatch at the
    first act() call — not a silent bug.
    """

    # Debugging — see composite_board_encoder.py's `debug` param.
    debug_encoders: bool = False


def build_actor_critic(
    hand_encoder: Optional[BaseHandEncoder] = None,
    config: Optional[ActorCriticConfig] = None,
) -> PokerActorCritic:
    """
    Construct a fully-wired PokerActorCritic from an ActorCriticConfig.

    See module docstring for the three-role pretrained encoder wiring
    (hand / board_texture / made_hand), all sharing `hand_encoder` as
    their frozen base with separate trainable adapters, and for
    `config.action_dim`'s role in sizing the action head for graph-
    native scenarios.
    """
    cfg = config or ActorCriticConfig()

    if hand_encoder is None:
        from poker_rl_lab.representation.pretrained_hand_encoder import PretrainedHandEncoder

        hand_encoder = PretrainedHandEncoder.untrained()
        _warn_untrained_encoder(caller="build_actor_critic (hand_encoder=None)")

    _log_encoder_provenance(hand_encoder)

    composite_hand_encoder = CompositeHandEncoder(
        base_encoder=hand_encoder,
        output_dim=cfg.hand_output_dim,
        attn_hidden_dim=cfg.hand_attn_hidden_dim,
        project_hidden_dim=cfg.hand_project_hidden_dim,
        max_subsets=cfg.hand_max_subsets,
    )

    board_texture_encoder = CompositeBoardEncoder(
        hand_encoder=hand_encoder,
        output_dim=cfg.board_output_dim,
        attn_hidden_dim=cfg.board_attn_hidden_dim,
        project_hidden_dim=cfg.board_project_hidden_dim,
        max_subsets=cfg.board_max_subsets,
        window=cfg.board_window,
        debug=cfg.debug_encoders,
    )

    made_hand_encoder = CompositeBoardEncoder(
        hand_encoder=hand_encoder,
        output_dim=cfg.made_hand_output_dim,
        attn_hidden_dim=cfg.made_hand_attn_hidden_dim,
        project_hidden_dim=cfg.made_hand_project_hidden_dim,
        max_subsets=cfg.made_hand_max_subsets,
        window=cfg.made_hand_window,
        debug=cfg.debug_encoders,
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
        board_encoder=board_texture_encoder,
        made_hand_encoder=made_hand_encoder,
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
        action_dim=cfg.action_dim,
        hidden_dim=cfg.policy_hidden_dim,
        num_shared_layers=cfg.policy_num_shared_layers,
    )

    return PokerActorCritic(representation_model, policy_value_network)


def _warn_untrained_encoder(caller: str) -> None:
    message = (
        f"[build_actor_critic] hand_encoder=None passed in from {caller!r} — "
        f"falling back to PretrainedHandEncoder.untrained(). This ONE instance "
        f"backs ALL THREE roles (hole, board-texture, made-hand) — this policy "
        f"is training with NO pretrained poker-hand knowledge anywhere. Pass "
        f"hand_encoder=PretrainedHandEncoder.from_file(...) explicitly if this "
        f"is not a pipeline/shape-testing run."
    )
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    logger.warning(message)
    print(f"WARNING: {message}", file=sys.stderr)


def _log_encoder_provenance(hand_encoder: BaseHandEncoder) -> None:
    is_pretrained = getattr(hand_encoder, "is_pretrained", None)
    embedding_dim = getattr(hand_encoder, "embedding_dim", "?")

    if is_pretrained is True:
        logger.info(
            f"[build_actor_critic] hand_encoder is PRETRAINED "
            f"(embedding_dim={embedding_dim}). Shared as the frozen base "
            f"for hole / board-texture / made-hand adapters."
        )
    elif is_pretrained is False:
        message = (
            f"[build_actor_critic] hand_encoder is UNTRAINED "
            f"(embedding_dim={embedding_dim}) — shared as the frozen base "
            f"for ALL THREE roles (hole / board-texture / made-hand)."
        )
        logger.warning(message)
        print(f"WARNING: {message}", file=sys.stderr)
    else:
        logger.warning(
            f"[build_actor_critic] hand_encoder of type "
            f"{type(hand_encoder).__name__} does not expose `is_pretrained` — "
            f"cannot verify whether pretrained weights are actually loaded."
        )
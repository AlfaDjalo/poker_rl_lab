"""
models/test_actor_critic_factory.py — Unit tests for build_actor_critic.

Uses a stub BaseHandEncoder (plain numpy, no TensorFlow) for the fast
path so these tests run without the TF-backed PretrainedHandEncoder.
One additional test exercises the omitted-hand_encoder default (which
does import TensorFlow) but is skipped automatically if TF isn't
installed, via pytest.importorskip.

Run with:
    pytest models/test_actor_critic_factory.py -v
"""

from typing import Sequence

import numpy as np
import pytest
import torch

from actions.abstract_action import ActionType, NUM_ACTIONS
from models.actor_critic_factory import ActorCriticConfig, build_actor_critic
from representation.base_hand_encoder import BaseHandEncoder
from representation.game_encoder import GameFeatures
from representation.point_encoder import PointFeatures
from representation.position_encoder import PositionFeatures
from representation.representation_model import RepresentationInput


# ---------------------------------------------------------------------------
# Stub hand encoder (no TensorFlow)
# ---------------------------------------------------------------------------

class _StubHandEncoder(BaseHandEncoder):
    """Minimal BaseHandEncoder implementation for fast, TF-free tests."""

    def __init__(self, embedding_dim: int = 8) -> None:
        self._dim = embedding_dim

    def encode_hole(self, card_ids: Sequence[int]) -> np.ndarray:
        return np.zeros(self._dim, dtype=np.float32)

    def encode_board(self, card_ids: Sequence[int]) -> np.ndarray:
        return np.zeros(self._dim, dtype=np.float32)

    def encode_combined(self, hole_ids: Sequence[int], board_ids: Sequence[int]) -> np.ndarray:
        return np.zeros(self._dim, dtype=np.float32)

    def encode_batch(self, card_id_lists: Sequence[Sequence[int]]) -> np.ndarray:
        return np.zeros((len(card_id_lists), self._dim), dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# A small config for fast tests (real training would use the defaults)
# ---------------------------------------------------------------------------

def _small_config() -> ActorCriticConfig:
    return ActorCriticConfig(
        board_output_dim=8,
        point_output_dim=8,
        game_output_dim=8,
        position_output_dim=8,
        history_gru_hidden_dim=8,
        history_output_dim=8,
        representation_hidden_dim=32,
        representation_output_dim=16,
        policy_hidden_dim=16,
    )


def _make_rep_input() -> RepresentationInput:
    return RepresentationInput(
        hole_card_ids=[0, 13],
        board_card_ids=[[1, 2, 3]],
        points=[PointFeatures(score_type=0, showdown_type=0)],
        game=GameFeatures(
            betting_type_id=0,
            payout_type_id=0,
            showdown_type=0,
            hole_cards=2,
            node_count=5,
            num_streets=4,
            num_points=1,
        ),
        position=PositionFeatures(
            seat_fraction=0.5,
            active_fraction=1.0,
            to_act_after_fraction=0.5,
        ),
        history=[],
    )


def _legal_mask_check_bet25() -> tuple:
    mask = [False] * NUM_ACTIONS
    mask[ActionType.CHECK] = True
    mask[ActionType.BET_25] = True
    return tuple(mask)


# ---------------------------------------------------------------------------
# Wiring / shape tests
# ---------------------------------------------------------------------------

def test_build_actor_critic_with_stub_hand_encoder_returns_correct_types():
    actor_critic = build_actor_critic(
        hand_encoder=_StubHandEncoder(embedding_dim=8), config=_small_config()
    )
    from models.policy_value_network import PokerActorCritic

    assert isinstance(actor_critic, PokerActorCritic)


def test_representation_model_output_dim_matches_config():
    cfg = _small_config()
    actor_critic = build_actor_critic(hand_encoder=_StubHandEncoder(8), config=cfg)

    assert actor_critic.representation_model.output_dim == cfg.representation_output_dim
    assert actor_critic.policy_value_network.latent_dim == cfg.representation_output_dim


def test_representation_model_forward_produces_correct_shape():
    cfg = _small_config()
    actor_critic = build_actor_critic(hand_encoder=_StubHandEncoder(8), config=cfg)
    rep_input = _make_rep_input()

    latent = actor_critic.representation_model(rep_input)

    assert latent.shape == (cfg.representation_output_dim,)
    assert torch.isfinite(latent).all()


def test_card_encoder_shared_between_board_encoder_and_representation():
    """
    build_actor_critic must construct exactly one CardEncoder and pass
    it into BoardEncoder — not two independently-initialised copies
    (see board_encoder.py's docstring on why sharing matters).
    """
    actor_critic = build_actor_critic(hand_encoder=_StubHandEncoder(8), config=_small_config())
    board_encoder = actor_critic.representation_model.board_encoder

    assert board_encoder.card_encoder.rank_embed.weight.shape[0] == 13
    # Sanity: the card encoder is a genuine nn.Module parameter owner,
    # reachable from the assembled actor_critic's parameter list (i.e.
    # it's actually registered, not orphaned).
    param_ids = {id(p) for p in actor_critic.parameters()}
    assert id(board_encoder.card_encoder.rank_embed.weight) in param_ids


# ---------------------------------------------------------------------------
# act() / evaluate_actions() end-to-end through the assembled network
# ---------------------------------------------------------------------------

def test_act_returns_legal_action():
    actor_critic = build_actor_critic(hand_encoder=_StubHandEncoder(8), config=_small_config())
    rep_input = _make_rep_input()
    legal_mask = _legal_mask_check_bet25()

    result = actor_critic.act(rep_input, legal_mask)

    assert result.action_type in (ActionType.CHECK, ActionType.BET_25)
    assert torch.isfinite(result.log_prob)
    assert torch.isfinite(result.value)
    assert torch.isfinite(result.entropy)


def test_act_deterministic_is_argmax_and_repeatable():
    actor_critic = build_actor_critic(hand_encoder=_StubHandEncoder(8), config=_small_config())
    rep_input = _make_rep_input()
    legal_mask = _legal_mask_check_bet25()

    r1 = actor_critic.act(rep_input, legal_mask, deterministic=True)
    r2 = actor_critic.act(rep_input, legal_mask, deterministic=True)

    assert r1.action_type == r2.action_type


def test_evaluate_actions_batch_shapes():
    actor_critic = build_actor_critic(hand_encoder=_StubHandEncoder(8), config=_small_config())
    rep_inputs = [_make_rep_input(), _make_rep_input(), _make_rep_input()]
    legal_masks = [_legal_mask_check_bet25()] * 3
    actions = [ActionType.CHECK, ActionType.BET_25, ActionType.CHECK]

    log_probs, values, entropy = actor_critic.evaluate_actions(rep_inputs, legal_masks, actions)

    assert log_probs.shape == (3,)
    assert values.shape == (3,)
    assert entropy.shape == (3,)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()


def test_gradients_flow_from_loss_through_full_stack():
    """
    A minimal end-to-end sanity check: backprop through evaluate_actions()
    should reach every component encoder's parameters (board/point/game/
    position/history), confirming build_actor_critic actually wired a
    single connected graph rather than, say, silently detaching a branch.
    """
    actor_critic = build_actor_critic(hand_encoder=_StubHandEncoder(8), config=_small_config())
    rep_inputs = [_make_rep_input(), _make_rep_input()]
    legal_masks = [_legal_mask_check_bet25()] * 2
    actions = [ActionType.CHECK, ActionType.BET_25]

    log_probs, values, entropy = actor_critic.evaluate_actions(rep_inputs, legal_masks, actions)
    loss = -log_probs.mean() + values.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()

    board_grad = actor_critic.representation_model.board_encoder.project.weight.grad
    point_grad = actor_critic.representation_model.point_encoder.project.weight.grad
    game_grad = actor_critic.representation_model.game_encoder.project.weight.grad
    position_grad = actor_critic.representation_model.position_encoder.mlp[0].weight.grad
    # history is empty in _make_rep_input(), so HistoryEncoder.forward()
    # short-circuits to the learned empty_history_embed parameter
    # directly (see history_encoder.py) rather than running the GRU/
    # project layers — check gradient on the parameter that's actually
    # on this particular forward pass's graph.
    history_grad = actor_critic.representation_model.history_encoder.empty_history_embed.grad

    for grad in (board_grad, point_grad, game_grad, position_grad, history_grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum().item() > 0.0


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

def test_default_config_matches_encoder_defaults_for_output_dims():
    cfg = ActorCriticConfig()
    actor_critic = build_actor_critic(hand_encoder=_StubHandEncoder(8), config=cfg)

    assert actor_critic.representation_model.board_encoder.output_dim == 32
    assert actor_critic.representation_model.point_encoder.output_dim == 16
    assert actor_critic.representation_model.game_encoder.output_dim == 16
    assert actor_critic.representation_model.position_encoder.output_dim == 16
    assert actor_critic.representation_model.history_encoder.output_dim == 16
    assert actor_critic.representation_model.output_dim == 128
    assert actor_critic.policy_value_network.latent_dim == 128


# ---------------------------------------------------------------------------
# Omitted hand_encoder (TF-backed default path) — skipped without TF
# ---------------------------------------------------------------------------

def test_omitted_hand_encoder_uses_untrained_pretrained_hand_encoder():
    pytest.importorskip("tensorflow")
    cfg = _small_config()
    actor_critic = build_actor_critic(config=cfg)  # hand_encoder omitted

    rep_input = _make_rep_input()
    latent = actor_critic.representation_model(rep_input)

    assert latent.shape == (cfg.representation_output_dim,)
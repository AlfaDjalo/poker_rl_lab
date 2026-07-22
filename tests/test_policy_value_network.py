"""
test_policy_value_network.py — Standalone smoke test for
PolicyValueNetwork and PokerActorCritic. Reuses the mock-encoder setup
from test_representation_model.py so no poker_eval / TF weights needed.

Run directly:
    python test_policy_value_network.py
"""

from __future__ import annotations

import torch

from actions.abstract_action import NUM_ACTIONS, ActionType
from models.policy_value_network import (
    PolicyValueNetwork,
    PokerActorCritic,
    apply_legal_mask,
)
from representation.representation_model import RepresentationInput

from test_representation_model import (
    build_model as build_representation_model,
    holdem_game_features,
    dummy_position,
)


def build_actor_critic() -> PokerActorCritic:
    rep_model = build_representation_model()  # output_dim = 64
    pv_net = PolicyValueNetwork(latent_dim=rep_model.output_dim, hidden_dim=64)
    return PokerActorCritic(rep_model, pv_net)


def sample_obs() -> RepresentationInput:
    from representation.point_encoder import PointFeatures

    return RepresentationInput(
        hole_card_ids=[0, 13],
        board_card_ids=[[]],
        points=[PointFeatures(score_type=0, showdown_type=0)],
        game=holdem_game_features(),
        position=dummy_position(),
        history=(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_policy_value_network_shapes():
    net = PolicyValueNetwork(latent_dim=64, hidden_dim=32)
    latent = torch.randn(64)
    logits, value = net(latent)
    assert logits.shape == (NUM_ACTIONS,)
    assert value.shape == ()
    print("test_policy_value_network_shapes: PASS")


def test_policy_value_network_batch_shapes():
    net = PolicyValueNetwork(latent_dim=64, hidden_dim=32)
    latent = torch.randn(5, 64)
    logits, value = net(latent)
    assert logits.shape == (5, NUM_ACTIONS)
    assert value.shape == (5,)
    print("test_policy_value_network_batch_shapes: PASS")


def test_apply_legal_mask_zeroes_illegal_probability():
    logits = torch.randn(NUM_ACTIONS)
    mask = torch.zeros(NUM_ACTIONS, dtype=torch.bool)
    mask[ActionType.FOLD] = True
    mask[ActionType.CALL] = True

    masked = apply_legal_mask(logits, mask)
    probs = torch.softmax(masked, dim=-1)

    assert probs[ActionType.FOLD] > 0
    assert probs[ActionType.CALL] > 0
    for i in range(NUM_ACTIONS):
        if i not in (ActionType.FOLD, ActionType.CALL):
            assert probs[i].item() < 1e-6, f"action {i} leaked probability {probs[i].item()}"
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    print("test_apply_legal_mask_zeroes_illegal_probability: PASS")


def test_apply_legal_mask_rejects_all_illegal_row():
    logits = torch.randn(2, NUM_ACTIONS)
    mask = torch.zeros(2, NUM_ACTIONS, dtype=torch.bool)
    mask[0, ActionType.FOLD] = True
    # row 1 has no legal actions -> should raise
    try:
        apply_legal_mask(logits, mask)
        raised = False
    except ValueError:
        raised = True
    assert raised, "expected ValueError for a row with no legal actions"
    print("test_apply_legal_mask_rejects_all_illegal_row: PASS")


def test_actor_critic_act_respects_mask():
    ac = build_actor_critic()
    obs = sample_obs()

    legal_mask = [False] * NUM_ACTIONS
    legal_mask[ActionType.FOLD] = True
    legal_mask[ActionType.CALL] = True

    for _ in range(20):
        result = ac.act(obs, legal_mask, deterministic=False)
        assert result.action_type in (int(ActionType.FOLD), int(ActionType.CALL)), (
            f"sampled illegal action {result.action_type}"
        )
        assert torch.isfinite(result.log_prob)
        assert torch.isfinite(result.value)
        assert torch.isfinite(result.entropy)
    print("test_actor_critic_act_respects_mask: PASS")


def test_actor_critic_deterministic_is_reproducible():
    ac = build_actor_critic()
    obs = sample_obs()
    legal_mask = [True] * NUM_ACTIONS

    a1 = ac.act(obs, legal_mask, deterministic=True)
    a2 = ac.act(obs, legal_mask, deterministic=True)
    assert a1.action_type == a2.action_type
    print("test_actor_critic_deterministic_is_reproducible: PASS")


def test_actor_critic_evaluate_actions_matches_batch():
    ac = build_actor_critic()
    obs1 = sample_obs()
    obs2 = sample_obs()

    legal_mask = [True] * NUM_ACTIONS
    actions = [int(ActionType.FOLD), int(ActionType.CALL)]

    log_probs, values, entropy = ac.evaluate_actions(
        [obs1, obs2], [legal_mask, legal_mask], actions
    )
    assert log_probs.shape == (2,)
    assert values.shape == (2,)
    assert entropy.shape == (2,)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(entropy).all()
    print("test_actor_critic_evaluate_actions_matches_batch: PASS")


def test_gradients_flow_through_full_stack():
    """
    Uses a NON-empty board for both examples. An empty board (as in
    sample_obs()) short-circuits BoardEncoder.forward straight to its
    learned empty_board_embed parameter, bypassing the `project` layer
    entirely — that's correct behaviour (see board_encoder.py), not
    something to grad-check here, so this test exercises the dealt-board
    path instead to confirm `project` itself is trainable.
    """
    from representation.point_encoder import PointFeatures

    ac = build_actor_critic()

    def river_obs():
        return RepresentationInput(
            hole_card_ids=[0, 13],
            board_card_ids=[[1, 14, 27, 40, 51]],
            points=[PointFeatures(score_type=0, showdown_type=0)],
            game=holdem_game_features(),
            position=dummy_position(),
            history=(),
        )

    legal_mask = [True] * NUM_ACTIONS
    actions = [int(ActionType.FOLD), int(ActionType.CALL)]

    log_probs, values, entropy = ac.evaluate_actions(
        [river_obs(), river_obs()], [legal_mask, legal_mask], actions
    )
    loss = -(log_probs.mean()) + values.pow(2).mean() - 0.01 * entropy.mean()
    loss.backward()

    grad = ac.policy_value_network.policy_head.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()

    rep_grad = ac.representation_model.board_encoder.project.weight.grad
    assert rep_grad is not None
    assert torch.isfinite(rep_grad).all()
    print("test_gradients_flow_through_full_stack: PASS")


if __name__ == "__main__":
    test_policy_value_network_shapes()
    test_policy_value_network_batch_shapes()
    test_apply_legal_mask_zeroes_illegal_probability()
    test_apply_legal_mask_rejects_all_illegal_row()
    test_actor_critic_act_respects_mask()
    test_actor_critic_deterministic_is_reproducible()
    test_actor_critic_evaluate_actions_matches_batch()
    test_gradients_flow_through_full_stack()
    print("\nAll PolicyValueNetwork/PokerActorCritic smoke tests passed.")
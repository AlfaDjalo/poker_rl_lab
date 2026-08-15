"""
test_representation_model.py — Standalone smoke test for RepresentationModel.

No dependency on poker_eval or a real .keras checkpoint: hole-card
embedding is provided by MockHandEncoder (implements BaseHandEncoder with
random-but-deterministic output), matching how mock_scoring_engine.py /
mock_poker_eval.py let the rest of the engine's tests run without the C++
extension.

Run directly:
    python test_representation_model.py
"""

from __future__ import annotations

import numpy as np
import torch

from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.card_encoder import CardEncoder
from poker_rl_lab.representation.board_encoder import BoardEncoder
from poker_rl_lab.representation.point_encoder import PointEncoder, PointFeatures
from poker_rl_lab.representation.game_encoder import GameEncoder, GameFeatures
from poker_rl_lab.representation.position_encoder import PositionEncoder, PositionFeatures
from poker_rl_lab.representation.history_encoder import HistoryEncoder, HistoryEvent
from poker_rl_lab.representation.stack_encoder import StackEncoder, StackFeatures
from poker_rl_lab.representation.representation_model import RepresentationModel, RepresentationInput


# ---------------------------------------------------------------------------
# Mock hand encoder (no TF / no real weights required)
# ---------------------------------------------------------------------------

class MockHandEncoder(BaseHandEncoder):
    """Deterministic per-card-set embedding, seeded by the sorted card ids."""

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    def _embed(self, card_ids) -> np.ndarray:
        seed = hash(tuple(sorted(card_ids))) % (2 ** 31)
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self._dim).astype(np.float32)

    def encode_hole(self, card_ids) -> np.ndarray:
        return self._embed(card_ids)

    def encode_board(self, card_ids) -> np.ndarray:
        return self._embed(card_ids)

    def encode_combined(self, hole_ids, board_ids) -> np.ndarray:
        return self._embed(list(hole_ids) + list(board_ids))

    def encode_batch(self, card_id_lists) -> np.ndarray:
        return np.stack([self._embed(ids) for ids in card_id_lists], axis=0)

    @property
    def embedding_dim(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_model() -> RepresentationModel:
    card_encoder = CardEncoder(rank_dim=16, suit_dim=8)  # output_dim = 24

    hand_encoder = MockHandEncoder(dim=32)
    board_encoder = BoardEncoder(card_encoder, output_dim=32, num_heads=4, num_layers=1)
    point_encoder = PointEncoder(output_dim=16)
    game_encoder = GameEncoder(output_dim=16)
    position_encoder = PositionEncoder(output_dim=16)
    history_encoder = HistoryEncoder(output_dim=16)
    stack_encoder = StackEncoder(output_dim=16)  # <-- Add this line (adjust class name/params as needed)

    return RepresentationModel(
        hand_encoder=hand_encoder,
        board_encoder=board_encoder,
        point_encoder=point_encoder,
        game_encoder=game_encoder,
        position_encoder=position_encoder,
        history_encoder=history_encoder,
        stack_encoder=stack_encoder,           # <-- Pass it here
        hidden_dim=128,
        output_dim=64,
    )


def holdem_game_features() -> GameFeatures:
    """Matches holdem.yaml: no_limit, points payout, 1 point, no ante."""
    return GameFeatures(
        betting_type_id=0,   # no_limit
        payout_type_id=0,    # points
        showdown_type=0,     # HOLDEM
        hole_cards=2,
        node_count=5,
        num_streets=4,
        num_points=1,
        has_ante=False,
        ante_bb=0.0,
        has_low=False,
    )


def double_board_game_features() -> GameFeatures:
    return GameFeatures(
        betting_type_id=1,   # pot_limit
        payout_type_id=0,    # points
        showdown_type=1,     # OMAHA
        hole_cards=4,
        node_count=10,
        num_streets=4,
        num_points=3,
        has_ante=False,
        ante_bb=0.0,
        has_low=False,
    )


def dummy_position() -> PositionFeatures:
    return PositionFeatures(
        seat_fraction=0.5,
        active_fraction=1.0,
        to_act_after_fraction=0.0,
        is_button=False,
        is_small_blind=False,
        is_big_blind=True,
        is_next_to_act=True,
        table_size=2,
    )


def dummy_stack() -> StackFeatures:
    return StackFeatures(
        stack_bb=100.0,
        effective_stack_bb=100.0,
        pot_bb=1.5,
        bet_to_call_bb=1.0,
        spr=66.67,
        pot_odds=0.4,
        sb_bb=0.5,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_board_single_point_no_history():
    model = build_model()

    x = RepresentationInput(
        hole_card_ids=[0, 13],           # 2c, 2d
        board_card_ids=[[]],             # preflop: single empty board
        points=[PointFeatures(score_type=0, showdown_type=0)],  # HIGH/HOLDEM
        game=holdem_game_features(),
        position=dummy_position(),
        stack=dummy_stack(),
        history=(),
    )

    out = model(x)
    assert out.shape == (64,), f"expected (64,), got {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "output contains NaN/Inf"
    print("test_single_board_single_point_no_history: PASS", out.shape)


def test_river_with_full_board_and_history():
    model = build_model()

    history = (
        HistoryEvent(action_type=1, street_index=3, seat_fraction=0.0,
                     pot_fraction=0.0, is_hero=False, is_aggressor=False),  # villain checks
        HistoryEvent(action_type=4, street_index=3, seat_fraction=0.5,
                     pot_fraction=0.5, is_hero=True, is_aggressor=True),    # hero bets 50%
    )

    x = RepresentationInput(
        hole_card_ids=[0, 13],
        board_card_ids=[[1, 14, 27, 40, 51]],   # 5-card river board
        points=[PointFeatures(score_type=0, showdown_type=0)],
        game=holdem_game_features(),
        position=dummy_position(),
        stack=dummy_stack(),
        history=history,
    )

    out = model(x)
    assert out.shape == (64,)
    assert torch.isfinite(out).all()
    print("test_river_with_full_board_and_history: PASS", out.shape)


def test_double_board_multi_point_pooling():
    model = build_model()

    x = RepresentationInput(
        hole_card_ids=[0, 13, 26, 39],           # 4-card Omaha-style hole
        board_card_ids=[
            [1, 14, 27, 40, 51],                  # board 1, 5 cards
            [2, 15, 28, 41, 50],                  # board 2, 5 cards
        ],
        points=[
            PointFeatures(score_type=0, showdown_type=1),  # board 1 high
            PointFeatures(score_type=0, showdown_type=1),  # board 2 high
            PointFeatures(score_type=0, showdown_type=1),  # hand point
        ],
        game=double_board_game_features(),
        position=dummy_position(),
        stack=dummy_stack(),
        history=(),
    )

    out = model(x)
    assert out.shape == (64,)
    assert torch.isfinite(out).all()
    print("test_double_board_multi_point_pooling: PASS", out.shape)


def test_forward_batch():
    model = build_model()

    x1 = RepresentationInput(
        hole_card_ids=[0, 13],
        board_card_ids=[[]],
        points=[PointFeatures(score_type=0, showdown_type=0)],
        game=holdem_game_features(),
        position=dummy_position(),
        stack=dummy_stack(),
        history=(),
    )
    x2 = RepresentationInput(
        hole_card_ids=[5, 18],
        board_card_ids=[[1, 14, 27, 40, 51]],
        points=[PointFeatures(score_type=0, showdown_type=0)],
        game=holdem_game_features(),
        position=dummy_position(),
        stack=dummy_stack(),
        history=(HistoryEvent(action_type=1, street_index=3, seat_fraction=0.0),),
    )

    out = model.forward_batch([x1, x2])
    assert out.shape == (2, 64), f"expected (2, 64), got {tuple(out.shape)}"
    assert torch.isfinite(out).all()
    print("test_forward_batch: PASS", out.shape)


def test_gradients_flow_to_pytorch_encoders():
    """Confirm the PyTorch sub-encoders receive gradients (hole embed stays frozen)."""
    model = build_model()

    x = RepresentationInput(
        hole_card_ids=[0, 13],
        board_card_ids=[[1, 14, 27]],
        points=[PointFeatures(score_type=0, showdown_type=0)],
        game=holdem_game_features(),
        position=dummy_position(),
        stack=dummy_stack(),
        history=(HistoryEvent(action_type=3, street_index=1, seat_fraction=0.5, is_aggressor=True),),
    )

    out = model(x)
    loss = out.sum()
    loss.backward()

    board_grad = model.board_encoder.project.weight.grad
    point_grad = model.point_encoder.project.weight.grad
    game_grad = model.game_encoder.project.weight.grad
    position_grad = list(model.position_encoder.mlp.parameters())[-1].grad
    history_grad = model.history_encoder.project.weight.grad

    for name, grad in [
        ("board_encoder", board_grad),
        ("point_encoder", point_grad),
        ("game_encoder", game_grad),
        ("position_encoder", position_grad),
        ("history_encoder", history_grad),
    ]:
        assert grad is not None, f"{name} received no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient contains NaN/Inf"

    print("test_gradients_flow_to_pytorch_encoders: PASS")


if __name__ == "__main__":
    test_single_board_single_point_no_history()
    test_river_with_full_board_and_history()
    test_double_board_multi_point_pooling()
    test_forward_batch()
    test_gradients_flow_to_pytorch_encoders()
    print("\nAll RepresentationModel smoke tests passed.")
"""
tests/test_history_encoder.py — Unit tests for representation/history_encoder.py
"""

from __future__ import annotations

import pytest
import torch

from actions.abstract_action import ActionType, NUM_ACTIONS
from representation.history_encoder import HistoryEncoder, HistoryEvent


def ev(action_type=ActionType.CHECK, street=0, seat_fraction=0.0,
       pot_fraction=0.0, is_hero=False, is_aggressor=False):
    return HistoryEvent(
        action_type=int(action_type),
        street_index=street,
        seat_fraction=seat_fraction,
        pot_fraction=pot_fraction,
        is_hero=is_hero,
        is_aggressor=is_aggressor,
    )


# ---------------------------------------------------------------------------
# HistoryEvent validation
# ---------------------------------------------------------------------------

class TestHistoryEventValidation:
    def test_valid_construction(self):
        e = ev(ActionType.BET_50, street=1, seat_fraction=0.5, pot_fraction=0.5,
               is_hero=True, is_aggressor=True)
        assert e.action_type == ActionType.BET_50
        assert e.is_aggressor is True

    @pytest.mark.parametrize("bad", [-1, NUM_ACTIONS])
    def test_invalid_action_type(self, bad):
        with pytest.raises(ValueError):
            HistoryEvent(action_type=bad, street_index=0, seat_fraction=0.0)

    def test_negative_street_rejected(self):
        with pytest.raises(ValueError):
            HistoryEvent(action_type=0, street_index=-1, seat_fraction=0.0)

    @pytest.mark.parametrize("bad", [-0.01, 1.0, 1.5])
    def test_seat_fraction_out_of_range(self, bad):
        with pytest.raises(ValueError):
            HistoryEvent(action_type=0, street_index=0, seat_fraction=bad)

    def test_negative_pot_fraction_rejected(self):
        with pytest.raises(ValueError):
            HistoryEvent(action_type=0, street_index=0, seat_fraction=0.0, pot_fraction=-1.0)


# ---------------------------------------------------------------------------
# HistoryEncoder forward passes
# ---------------------------------------------------------------------------

class TestHistoryEncoderForward:
    def test_empty_sequence_returns_learned_placeholder(self):
        enc = HistoryEncoder(output_dim=16)
        out = enc([])
        assert out.shape == (16,)
        assert torch.allclose(out, enc.empty_history_embed)

    def test_single_event_shape(self):
        enc = HistoryEncoder(output_dim=16)
        out = enc([ev(ActionType.CHECK)])
        assert out.shape == (16,)

    def test_multi_event_sequence_shape(self):
        enc = HistoryEncoder(output_dim=16)
        events = [
            ev(ActionType.CALL, street=0, seat_fraction=0.5, pot_fraction=0.5, is_hero=False),
            ev(ActionType.CHECK, street=1, seat_fraction=0.0, is_hero=True),
            ev(ActionType.BET_75, street=1, seat_fraction=0.5, pot_fraction=0.75,
               is_hero=False, is_aggressor=True),
            ev(ActionType.CALL, street=1, seat_fraction=0.0, pot_fraction=0.75, is_hero=True),
        ]
        out = enc(events)
        assert out.shape == (16,)
        assert torch.isfinite(out).all()

    def test_order_matters(self):
        """bet-then-call should differ from call-then-bet embeddings."""
        torch.manual_seed(0)
        enc = HistoryEncoder(output_dim=16)
        a = ev(ActionType.BET_50, pot_fraction=0.5, is_aggressor=True)
        b = ev(ActionType.CALL, pot_fraction=0.5)
        out1 = enc([a, b])
        out2 = enc([b, a])
        assert not torch.allclose(out1, out2)

    def test_batch_forward_shape(self):
        enc = HistoryEncoder(output_dim=16)
        batch = [
            [],
            [ev(ActionType.CHECK)],
            [ev(ActionType.BET_50, pot_fraction=0.5, is_aggressor=True),
             ev(ActionType.CALL, pot_fraction=0.5)],
        ]
        out = enc.forward_batch(batch)
        assert out.shape == (3, 16)

    def test_batch_empty_row_matches_placeholder(self):
        enc = HistoryEncoder(output_dim=16)
        batch = [[], [ev(ActionType.CHECK)]]
        out = enc.forward_batch(batch)
        assert torch.allclose(out[0], enc.empty_history_embed)

    def test_forward_matches_forward_batch_single(self):
        enc = HistoryEncoder(output_dim=16)
        events = [ev(ActionType.CALL, pot_fraction=0.3), ev(ActionType.CHECK, street=1)]
        single = enc(events)
        batched = enc.forward_batch([events])[0]
        assert torch.allclose(single, batched, atol=1e-6)

    def test_forward_batch_padding_does_not_leak_into_shorter_sequences(self):
        """
        Critical correctness check for the padding scheme: a short
        sequence's embedding must be identical whether or not it is
        batched alongside a much longer sequence (GRU causality).
        """
        torch.manual_seed(0)
        enc = HistoryEncoder(output_dim=16)
        short = [ev(ActionType.CHECK), ev(ActionType.BET_50, pot_fraction=0.5, is_aggressor=True)]
        long_seq = [ev(ActionType.CALL, pot_fraction=x) for x in
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]

        alone = enc.forward_batch([short])[0]
        with_long = enc.forward_batch([short, long_seq])[0]
        assert torch.allclose(alone, with_long, atol=1e-6)

    def test_all_empty_batch(self):
        enc = HistoryEncoder(output_dim=16)
        out = enc.forward_batch([[], [], []])
        assert out.shape == (3, 16)
        for row in out:
            assert torch.allclose(row, enc.empty_history_embed)

    def test_empty_batch_list_raises(self):
        enc = HistoryEncoder(output_dim=16)
        with pytest.raises(ValueError):
            enc.forward_batch([])

    def test_gradients_flow(self):
        enc = HistoryEncoder(output_dim=16)
        batch = [
            [ev(ActionType.BET_50, pot_fraction=0.5, is_aggressor=True), ev(ActionType.CALL)],
            [ev(ActionType.FOLD)],
        ]
        out = enc.forward_batch(batch)
        out.sum().backward()
        assert enc.action_type_embed.weight.grad is not None
        assert not torch.all(enc.action_type_embed.weight.grad == 0)
        assert enc.project.weight.grad is not None

    def test_custom_dims_respected(self):
        enc = HistoryEncoder(
            action_type_dim=4, scalar_hidden_dim=4, gru_hidden_dim=12, output_dim=24
        )
        out = enc([ev(ActionType.FOLD)])
        assert out.shape == (24,)

    def test_long_hand_all_streets_and_actions(self):
        """Smoke test resembling a full multi-street hand history."""
        enc = HistoryEncoder(output_dim=16)
        events = []
        for street in range(4):
            events.append(ev(ActionType.CHECK, street=street, seat_fraction=0.0, is_hero=True))
            events.append(
                ev(ActionType.BET_100, street=street, seat_fraction=0.5,
                   pot_fraction=1.0, is_aggressor=True)
            )
            events.append(
                ev(ActionType.CALL, street=street, seat_fraction=0.0,
                   pot_fraction=1.0, is_hero=True)
            )
        out = enc(events)
        assert out.shape == (16,)
        assert torch.isfinite(out).all()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
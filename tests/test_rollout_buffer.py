"""
trainers/test_rollout_buffer.py — Unit tests for RolloutBuffer.

These tests only exercise the GAE math and bookkeeping, so `obs` is
passed as None throughout — RolloutBuffer never inspects .obs, it only
stores and returns it (the real RepresentationInput payload is opaque
to this module by design; see rollout_buffer.py's docstring).

Run with:
    pytest trainers/test_rollout_buffer.py -v
"""

import math

import pytest
import torch

from trainers.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add(buf, reward, value, done, action=0, log_prob=-0.1):
    buf.add(
        obs=None,
        legal_mask=(True, True),
        action=action,
        log_prob=log_prob,
        value=value,
        reward=reward,
        done=done,
    )


# ---------------------------------------------------------------------------
# Single-step episode (push-fold shape: one hero decision, reward at done)
# ---------------------------------------------------------------------------

def test_single_step_episode_advantage_is_reward_minus_value():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=0.95)
    buf.start_episode()
    _add(buf, reward=10.0, value=2.0, done=True)
    buf.finish_episode()

    assert buf.advantages == pytest.approx([8.0])
    assert buf.returns == pytest.approx([10.0])


def test_single_step_episode_zero_reward_loss():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=0.95)
    buf.start_episode()
    _add(buf, reward=-4.0, value=-1.0, done=True)
    buf.finish_episode()

    assert buf.advantages == pytest.approx([-3.0])
    assert buf.returns == pytest.approx([-4.0])


# ---------------------------------------------------------------------------
# Multi-step episode (river shape: intermediate reward=0, final has chips)
# ---------------------------------------------------------------------------

def test_two_step_episode_matches_hand_computed_gae():
    gamma, lam = 1.0, 0.9
    buf = RolloutBuffer(gamma=gamma, gae_lambda=lam)
    buf.start_episode()
    _add(buf, reward=0.0, value=1.0, done=False)   # hero bets, waits for villain
    _add(buf, reward=6.0, value=3.0, done=True)     # showdown, hero nets +6
    buf.finish_episode()

    # Hand-computed GAE, gamma=1, lambda=0.9:
    #   delta_1 = reward_1 + gamma*V_2 - V_1 = 6 + 0 - 3 = 3   (terminal: next_value=0)
    #   gae_1   = delta_1 = 3
    #   delta_0 = reward_0 + gamma*V_1 - V_0 = 0 + 3 - 1 = 2
    #   gae_0   = delta_0 + gamma*lam*gae_1 = 2 + 1*0.9*3 = 4.7
    assert buf.advantages == pytest.approx([4.7, 3.0])
    assert buf.returns == pytest.approx([4.7 + 1.0, 3.0 + 3.0])


def test_three_step_episode_gae_recursion():
    gamma, lam = 0.99, 0.95
    buf = RolloutBuffer(gamma=gamma, gae_lambda=lam)
    buf.start_episode()
    _add(buf, reward=0.0, value=0.5, done=False)
    _add(buf, reward=0.0, value=1.5, done=False)
    _add(buf, reward=8.0, value=4.0, done=True)
    buf.finish_episode()

    delta2 = 8.0 + gamma * 0.0 - 4.0
    gae2 = delta2
    delta1 = 0.0 + gamma * 4.0 - 1.5
    gae1 = delta1 + gamma * lam * gae2
    delta0 = 0.0 + gamma * 1.5 - 0.5
    gae0 = delta0 + gamma * lam * gae1

    assert buf.advantages == pytest.approx([gae0, gae1, gae2])
    assert buf.returns == pytest.approx([gae0 + 0.5, gae1 + 1.5, gae2 + 4.0])


# ---------------------------------------------------------------------------
# Multiple episodes accumulate independently (no cross-episode leakage)
# ---------------------------------------------------------------------------

def test_multiple_episodes_do_not_leak_across_boundary():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=0.95)

    buf.start_episode()
    _add(buf, reward=5.0, value=0.0, done=True)
    buf.finish_episode()

    buf.start_episode()
    _add(buf, reward=-5.0, value=0.0, done=True)
    buf.finish_episode()

    assert len(buf) == 2
    assert buf.advantages == pytest.approx([5.0, -5.0])
    assert buf.returns == pytest.approx([5.0, -5.0])


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_finish_episode_without_any_transitions_raises():
    buf = RolloutBuffer()
    buf.start_episode()
    with pytest.raises(RuntimeError):
        buf.finish_episode()


def test_finish_episode_without_terminal_last_step_raises():
    buf = RolloutBuffer()
    buf.start_episode()
    _add(buf, reward=0.0, value=1.0, done=False)
    with pytest.raises(RuntimeError):
        buf.finish_episode()


def test_get_before_finish_episode_raises():
    buf = RolloutBuffer()
    buf.start_episode()
    _add(buf, reward=1.0, value=0.0, done=True)
    # forgot buf.finish_episode()
    with pytest.raises(RuntimeError):
        buf.get()


def test_invalid_gamma_raises():
    with pytest.raises(ValueError):
        RolloutBuffer(gamma=1.5)


def test_invalid_gae_lambda_raises():
    with pytest.raises(ValueError):
        RolloutBuffer(gae_lambda=-0.1)


# ---------------------------------------------------------------------------
# get() shape / dtype / normalisation
# ---------------------------------------------------------------------------

def test_get_returns_correctly_shaped_tensors():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=0.95)
    buf.start_episode()
    _add(buf, reward=0.0, value=1.0, done=False, action=1, log_prob=-0.2)
    _add(buf, reward=4.0, value=2.0, done=True, action=0, log_prob=-0.3)
    buf.finish_episode()

    batch = buf.get(normalize_advantages=False)

    assert len(batch["obs"]) == 2
    assert len(batch["legal_masks"]) == 2
    assert batch["actions"].tolist() == [1, 0]
    assert batch["old_log_probs"].dtype == torch.float32
    assert batch["advantages"].shape == (2,)
    assert batch["returns"].shape == (2,)


def test_get_normalizes_advantages_by_default():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=0.95)
    buf.start_episode()
    _add(buf, reward=0.0, value=0.0, done=False)
    _add(buf, reward=10.0, value=0.0, done=True)
    buf.finish_episode()

    buf.start_episode()
    _add(buf, reward=0.0, value=0.0, done=False)
    _add(buf, reward=-10.0, value=0.0, done=True)
    buf.finish_episode()

    batch = buf.get(normalize_advantages=True)
    adv = batch["advantages"]

    assert adv.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert adv.std().item() == pytest.approx(1.0, abs=1e-4)


def test_get_skips_normalization_for_single_transition():
    buf = RolloutBuffer(gamma=1.0, gae_lambda=0.95)
    buf.start_episode()
    _add(buf, reward=3.0, value=1.0, done=True)
    buf.finish_episode()

    batch = buf.get(normalize_advantages=True)
    # With only one transition, std is degenerate — raw GAE value (2.0)
    # should pass through unchanged rather than dividing by ~0.
    assert batch["advantages"].item() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

def test_clear_empties_buffer():
    buf = RolloutBuffer()
    buf.start_episode()
    _add(buf, reward=1.0, value=0.0, done=True)
    buf.finish_episode()

    buf.clear()

    assert len(buf) == 0
    assert buf.advantages == []
    assert buf.returns == []
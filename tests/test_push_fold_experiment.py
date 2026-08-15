"""
experiments/test_push_fold_experiment.py — Integration tests for
push_fold_experiment.py.

These are the first tests in the RL Lab that exercise the FULL pipeline
(ScenarioGenerator -> Env -> ObservationAdapter -> RepresentationModel
-> PolicyValueNetwork -> PPOTrainer) together, rather than one layer in
isolation with stubs above/below it. Two things are still stubbed to
keep this fast and dependency-light:

    - poker_eval: mock_poker_eval.py's shim, installed automatically by
      push_fold_experiment.py on import (see that module's docstring).
    - hand_encoder: a stub BaseHandEncoder (no TensorFlow), passed
      explicitly into build_push_fold_experiment() everywhere below.
      Carries no poker knowledge, so these tests check that training
      RUNS correctly end-to-end (shapes, checkpointing, determinism, no
      crashes) — NOT that a push-fold strategy is actually learned.
      Learning-quality checks belong in a slower, TF+poker_eval-backed
      test once those dependencies are available.

Run with:
    pytest experiments/test_push_fold_experiment.py -v
"""

from typing import Sequence

import numpy as np
import pytest
import torch

from poker_rl_lab.actions.abstract_action import ActionType
from poker_rl_lab.experiments.push_fold_experiment import (
    PushFoldExperimentConfig,
    build_push_fold_experiment,
    load_checkpoint,
    run_push_fold_experiment,
    save_checkpoint,
)
from poker_rl_lab.models.actor_critic_factory import ActorCriticConfig
from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder


# ---------------------------------------------------------------------------
# Stub hand encoder (no TensorFlow) — see module docstring
# ---------------------------------------------------------------------------

class _StubHandEncoder(BaseHandEncoder):
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


def _fast_config(**overrides) -> PushFoldExperimentConfig:
    """A tiny, fast-to-run config suitable for unit tests."""
    defaults = dict(
        seed=7,
        actor_critic_config=ActorCriticConfig(
            board_output_dim=8,
            point_output_dim=8,
            game_output_dim=8,
            position_output_dim=8,
            history_gru_hidden_dim=8,
            history_output_dim=8,
            representation_hidden_dim=32,
            representation_output_dim=16,
            policy_hidden_dim=16,
        ),
        num_iterations=2,
        episodes_per_iteration=6,
        epochs=2,
        minibatch_size=3,
    )
    defaults.update(overrides)
    return PushFoldExperimentConfig(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_build_push_fold_experiment_wires_all_pieces():
    experiment = build_push_fold_experiment(
        _fast_config(), hand_encoder=_StubHandEncoder(8)
    )

    assert experiment.scenario_gen.player_count == 2
    assert experiment.trainer.actor_critic is experiment.actor_critic
    assert experiment.trainer.adapter is experiment.adapter


def test_scenario_generator_seeded_reproducibly():
    exp_a = build_push_fold_experiment(_fast_config(seed=123), hand_encoder=_StubHandEncoder(8))
    exp_b = build_push_fold_experiment(_fast_config(seed=123), hand_encoder=_StubHandEncoder(8))

    scenario_a = exp_a.scenario_gen.generate()
    scenario_b = exp_b.scenario_gen.generate()

    assert scenario_a.players[0].hand_mask == scenario_b.players[0].hand_mask
    assert scenario_a.players[1].hand_mask == scenario_b.players[1].hand_mask
    assert scenario_a.bet_to_call == scenario_b.bet_to_call


# ---------------------------------------------------------------------------
# Single episode through the real adapter (no PPO update)
# ---------------------------------------------------------------------------

def test_single_episode_runs_through_full_adapter_pipeline():
    """
    Exercises env.reset() -> ObservationAdapter.adapt() (touching the
    REAL loader.load_game("holdem") + mock_poker_eval-backed rules) ->
    actor_critic.act() -> env.step(), without going through PPOTrainer,
    to isolate failures in the adapter layer from failures in the PPO
    loop.
    """
    experiment = build_push_fold_experiment(_fast_config(), hand_encoder=_StubHandEncoder(8))

    scenario = experiment.scenario_gen.generate()
    obs = experiment.env.reset(scenario)
    assert not obs.is_terminal

    rep_input = experiment.adapter.adapt(obs)
    with torch.no_grad():
        result = experiment.actor_critic.act(rep_input, obs.legal_mask)

    assert result.action_type in (ActionType.FOLD, ActionType.CALL)

    from poker_rl_lab.actions.abstract_action import AbstractAction

    next_obs, reward, done, info = experiment.env.step(
        AbstractAction(ActionType(result.action_type))
    )
    assert done is True
    assert next_obs.is_terminal
    assert isinstance(reward, float)
    assert info["action"] in ("FOLD", "CALL")


def test_adapter_caches_game_lookup_across_episodes():
    experiment = build_push_fold_experiment(_fast_config(), hand_encoder=_StubHandEncoder(8))

    obs1 = experiment.env.reset(experiment.scenario_gen.generate())
    experiment.adapter.adapt(obs1)
    assert "holdem" in experiment.adapter._cache

    obs2 = experiment.env.reset(experiment.scenario_gen.generate())
    experiment.adapter.adapt(obs2)
    # Still the same cached (game_def, rules) tuple, not re-loaded.
    assert len(experiment.adapter._cache) == 1


# ---------------------------------------------------------------------------
# Full run_push_fold_experiment()
# ---------------------------------------------------------------------------

def test_run_push_fold_experiment_end_to_end():
    cfg = _fast_config()
    experiment = build_push_fold_experiment(cfg, hand_encoder=_StubHandEncoder(8))

    history = run_push_fold_experiment(experiment)

    assert len(history) == cfg.num_iterations
    for stats in history:
        assert stats.num_episodes == cfg.episodes_per_iteration
        assert -cfg.starting_stack <= stats.mean_episode_reward <= cfg.max_stack_bb * cfg.big_blind + cfg.starting_stack
        assert torch.isfinite(torch.tensor(stats.policy_loss))
        assert torch.isfinite(torch.tensor(stats.value_loss))


def test_run_push_fold_experiment_parameters_change():
    """A very loose sanity check: PPO updates should actually move
    the network's parameters, not leave them at their random init."""
    experiment = build_push_fold_experiment(_fast_config(), hand_encoder=_StubHandEncoder(8))
    before = {
        name: p.clone().detach()
        for name, p in experiment.actor_critic.named_parameters()
    }

    run_push_fold_experiment(experiment)

    after = dict(experiment.actor_critic.named_parameters())
    changed = any(
        not torch.equal(before[name], after[name].detach()) for name in before
    )
    assert changed


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_save_and_load_checkpoint_roundtrip(tmp_path):
    cfg = _fast_config()
    experiment = build_push_fold_experiment(cfg, hand_encoder=_StubHandEncoder(8))
    run_push_fold_experiment(experiment)

    ckpt_path = tmp_path / "checkpoints" / "push_fold.pt"
    save_checkpoint(experiment, ckpt_path)
    assert ckpt_path.exists()

    trained_params = {
        name: p.clone().detach()
        for name, p in experiment.actor_critic.named_parameters()
    }

    fresh_experiment = build_push_fold_experiment(cfg, hand_encoder=_StubHandEncoder(8))
    fresh_params_before = {
        name: p.clone().detach()
        for name, p in fresh_experiment.actor_critic.named_parameters()
    }
    # Fresh init should differ from the trained checkpoint (extremely
    # unlikely to coincide by chance).
    assert any(
        not torch.equal(trained_params[name], fresh_params_before[name])
        for name in trained_params
    )

    load_checkpoint(fresh_experiment, ckpt_path)

    for name, p in fresh_experiment.actor_critic.named_parameters():
        assert torch.equal(p.detach(), trained_params[name])


def test_load_checkpoint_restores_optimizer_state(tmp_path):
    cfg = _fast_config()
    experiment = build_push_fold_experiment(cfg, hand_encoder=_StubHandEncoder(8))
    run_push_fold_experiment(experiment)

    ckpt_path = tmp_path / "push_fold.pt"
    save_checkpoint(experiment, ckpt_path)

    original_optimizer_state = experiment.trainer.optimizer.state_dict()

    fresh_experiment = build_push_fold_experiment(cfg, hand_encoder=_StubHandEncoder(8))
    load_checkpoint(fresh_experiment, ckpt_path)

    restored_state = fresh_experiment.trainer.optimizer.state_dict()
    assert restored_state["param_groups"] == original_optimizer_state["param_groups"]


# ---------------------------------------------------------------------------
# poker_eval shim actually installed
# ---------------------------------------------------------------------------

def test_poker_eval_shim_installed():
    import sys

    assert "poker_eval" in sys.modules
    assert hasattr(sys.modules["poker_eval"], "ScoreType")
    assert hasattr(sys.modules["poker_eval"], "ShowdownType")
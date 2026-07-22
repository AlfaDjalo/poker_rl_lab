"""
tests/test_river_experiment.py — Smoke test for experiments/river_experiment.py.

Not a correctness test of PPO math (that belongs in test_ppo_trainer.py /
test_rollout_buffer.py) or of poker rules (mock_poker_eval + MockEvaluator
are used throughout). This just proves the wiring is correct end-to-end:

    RiverScenarioGenerator -> RiverEnv -> ObservationAdapter
        -> build_actor_critic() -> PPOTrainer.train()

runs without shape errors, produces IterationStats, and that checkpoint
save/load round-trips. Uses tiny dimensions and 1-2 iterations so it runs
in a few seconds even with the untrained (TF-backed) hand encoder.

Run with:
    python -m pytest tests/test_river_experiment.py -v
or:
    python tests/test_river_experiment.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# poker_eval shim — see push_fold_experiment.py's own docstring for why.
try:
    import poker_eval  # noqa: F401
except ImportError:
    import mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

from models.actor_critic_factory import ActorCriticConfig
from experiments.river_experiment import (
    RiverExperimentConfig,
    build_river_experiment,
    run_river_experiment,
    save_checkpoint,
    load_checkpoint,
)
from envs.river_env import CallAnyVillainPolicy


def _tiny_actor_critic_config() -> ActorCriticConfig:
    """Small dims everywhere so the smoke test is fast."""
    return ActorCriticConfig(
        rank_dim=4,
        suit_dim=2,
        board_output_dim=8,
        board_num_heads=2,
        board_num_layers=1,
        point_score_type_dim=4,
        point_showdown_type_dim=4,
        point_scalar_hidden_dim=4,
        point_output_dim=8,
        game_betting_type_dim=4,
        game_payout_type_dim=4,
        game_showdown_type_dim=4,
        game_scalar_hidden_dim=8,
        game_output_dim=8,
        position_hidden_dim=8,
        position_output_dim=8,
        history_action_type_dim=4,
        history_scalar_hidden_dim=4,
        history_gru_hidden_dim=8,
        history_output_dim=8,
        representation_hidden_dim=32,
        representation_output_dim=16,
        policy_hidden_dim=16,
        policy_num_shared_layers=1,
    )


def _tiny_experiment_config() -> RiverExperimentConfig:
    return RiverExperimentConfig(
        seed=7,
        villain_policy=CallAnyVillainPolicy(),
        actor_critic_config=_tiny_actor_critic_config(),
        num_iterations=2,
        episodes_per_iteration=4,
        epochs=1,
        minibatch_size=4,
        max_raises=2,
    )


class TestRiverExperimentWiring(unittest.TestCase):
    def test_build_produces_all_components(self):
        experiment = build_river_experiment(_tiny_experiment_config())

        self.assertIsNotNone(experiment.scenario_gen)
        self.assertIsNotNone(experiment.env)
        self.assertIsNotNone(experiment.adapter)
        self.assertIsNotNone(experiment.actor_critic)
        self.assertIsNotNone(experiment.trainer)

    def test_run_produces_iteration_stats(self):
        experiment = build_river_experiment(_tiny_experiment_config())
        history = run_river_experiment(experiment)

        self.assertEqual(len(history), 2)
        for stats in history:
            self.assertEqual(stats.num_episodes, 4)
            self.assertGreater(stats.num_transitions, 0)
            self.assertTrue(
                all(
                    isinstance(getattr(stats, f), float)
                    for f in (
                        "mean_episode_reward",
                        "policy_loss",
                        "value_loss",
                        "entropy",
                        "clip_fraction",
                    )
                )
            )

    def test_checkpoint_round_trip(self):
        experiment = build_river_experiment(_tiny_experiment_config())
        run_river_experiment(experiment)

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            ckpt_path = tmp_dir / "river_ckpt.pt"
            save_checkpoint(experiment, ckpt_path)
            self.assertTrue(ckpt_path.exists())

            # Fresh experiment, then load into it.
            experiment2 = build_river_experiment(_tiny_experiment_config())
            load_checkpoint(experiment2, ckpt_path)

            # Parameters should now match exactly.
            params1 = list(experiment.actor_critic.parameters())
            params2 = list(experiment2.actor_critic.parameters())
            self.assertEqual(len(params1), len(params2))
            for p1, p2 in zip(params1, params2):
                self.assertTrue((p1 == p2).all())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_seed_is_reproducible(self):
        """Same seed -> same sequence of scenarios generated."""
        cfg = _tiny_experiment_config()

        exp_a = build_river_experiment(cfg)
        exp_b = build_river_experiment(cfg)

        scenarios_a = [exp_a.scenario_gen.generate() for _ in range(5)]
        scenarios_b = [exp_b.scenario_gen.generate() for _ in range(5)]

        for sa, sb in zip(scenarios_a, scenarios_b):
            self.assertEqual(sa.pot, sb.pot)
            self.assertEqual(sa.node_cards, sb.node_cards)
            self.assertEqual(
                [p.hand_mask for p in sa.players],
                [p.hand_mask for p in sb.players],
            )


if __name__ == "__main__":
    unittest.main()
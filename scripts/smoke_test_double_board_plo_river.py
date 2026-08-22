"""
scripts/smoke_test_double_board_plo_river.py — Standalone smoke test for
the double_board_plo_river GenericTreeConfig.

Plays random-action hands through GenericTreeEnv / GenericTreeScenarioGenerator
against duo_games/double_board_plo_river.yaml, asserting the deal shape
(10 board node cards, 4 hole cards/seat) and printing each hand's outcome.

KNOWN GAP — see the accompanying conversation / GenericTreeEnv patch:
GenericTreeEnv._resolve_showdown() currently evaluates self._board_mask
(every node card dealt so far) as ONE flat board via a single
scoring_engine.evaluate() call. It never consults the Engine's
rules.points, so for double_board_plo it treats both 5-card boards'
cards as a single 10-card board instead of two independent boards paid
out split_pot. This script will RUN without error either way — it
exercises tree traversal, dealing, and legal-action masking correctly —
but the printed payouts are not yet trustworthy for this config until
that gap is fixed.
"""

from __future__ import annotations

import random
import sys

try:
    import poker_rl_lab.poker_eval as poker_eval  # noqa: F401  (prefer the real extension if present)
except ImportError:
    import poker_rl_lab.mock_poker_eval as mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.scenarios.duo_game_config import load_duo_game
from poker_rl_lab.scenarios.generic_tree_scenario_generator import GenericTreeScenarioGenerator
from poker_rl_lab.envs.generic_tree_env import GenericTreeEnv


def main(num_hands: int = 20, seed: int = 0) -> None:
    config = load_duo_game("double_board_plo_river")
    scenario_gen = GenericTreeScenarioGenerator(config)
    scenario_gen.seed(seed)

    env = GenericTreeEnv(config, num_showdown_samples=1)
    rng = random.Random(seed)

    print(
        f"Loaded config {config.name!r}: game_name={config.game_name!r}, "
        f"seats={[s.id for s in config.seats]}, "
        f"hole_cards_per_player={config.deal.hole_cards_per_player}, "
        f"node_count={config.deal.node_count}"
    )

    outcomes = {"showdown": 0, "fold": 0}

    for hand_idx in range(num_hands):
        scenario = scenario_gen.generate()

        assert len(scenario.node_cards) == 10, (
            f"expected 10 board node cards (two independent 5-card "
            f"boards), got {len(scenario.node_cards)}"
        )
        for seat, hole_mask in enumerate(scenario.hole_masks):
            n_hole = bin(hole_mask).count("1")
            assert n_hole == 4, (
                f"seat {seat}: expected 4 hole cards, got {n_hole} "
                f"(mask={hole_mask:#x})"
            )

        obs = env.reset(scenario)
        done = False
        rewards = {}
        info = {}

        while not done:
            legal = [i for i, m in enumerate(obs.legal_mask) if m]
            action = AbstractAction(ActionType(rng.choice(legal)))
            result = env.step(action)
            done = result.done
            if done:
                rewards, info = result.rewards, result.info
            else:
                obs = result.obs

        outcomes[info["outcome"]] += 1
        rounded = {k: round(v, 2) for k, v in rewards.items()}
        print(f"hand {hand_idx:3d}: outcome={info['outcome']:8s} rewards={rounded}")

    print(f"\n{num_hands} hands complete: {outcomes}")


if __name__ == "__main__":
    main()
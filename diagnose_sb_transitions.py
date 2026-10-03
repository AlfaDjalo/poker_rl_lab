"""
diagnose_sb_transitions.py — re-implements collect_graph_rollouts()'s
lockstep loop for a SMALL number of hands (verbose), printing exactly
what happened in each one: SB's action, BB's action (if any), the
single-sample reward, the resampled/averaged reward, and the stack
depth that hand. This is meant to catch a reward-attribution bug that
only shows up in the REAL batched multi-hand loop, not in the isolated
single-hand scripted diagnostics already run.

Usage
-----
    python diagnose_sb_transitions.py --hands 30 --variant separate_networks
"""

from __future__ import annotations

import argparse
import copy
import random

import torch

from poker_rl_lab.envs.graph_scenario_env import GraphScenarioEnv
from poker_rl_lab.representation.graph_observation_adapter import GraphObservationAdapter
from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario
from poker_rl_lab.trainers.graph_training_loop import (
    PolicyVariant,
    build_dual_seat_actor_critic_for_graph,
    _clone_env,
    _resample_showdown_rewards,
)

_SB_SEAT = 0
_BB_SEAT = 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-name", type=str, default="push_fold_duo")
    parser.add_argument("--scenario-path", type=str, default=None)
    parser.add_argument("--variant", type=str, default="separate_networks",
                         choices=[v.value for v in PolicyVariant])
    parser.add_argument("--hands", type=int, default=30)
    parser.add_argument("--num-showdown-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_graph_scenario(args.scenario_name, args.scenario_path)
    template_env = GraphScenarioEnv(config, seed=args.seed)
    actor_critic = build_dual_seat_actor_critic_for_graph(
        PolicyVariant(args.variant), action_dim=config.max_action_options,
    )

    envs = [_clone_env(template_env, seed=random.Random(args.seed).randrange(2**31) + i)
            for i in range(args.hands)]
    adapters = [GraphObservationAdapter(e) for e in envs]

    for i in range(args.hands):
        env = envs[i]
        adapter = adapters[i]
        obs = env.reset()
        starting_stacks = list(env._starting_stacks)

        action_log = []
        sb_stack_before_action = None
        pre_request = pre_response = snap_state = snap_deck = None
        need_snapshot = False

        while True:
            rep_input = adapter.adapt(obs)
            with torch.no_grad():
                result = actor_critic.act(obs.seat, rep_input, obs.legal_mask, deterministic=False)
            chosen = obs.options[result.action_type]
            action_log.append(f"seat{obs.seat}:{chosen.label}(amt={chosen.amount})")

            need_snapshot = (
                args.num_showdown_samples > 1 and obs.domain in ("BETTING", "BOOLEAN", "CHOICE")
            )
            if need_snapshot:
                pre_request = env._engine.pending_request
                pre_response = env._finalize_single_shot(pre_request, result.action_type)
                snap_state = copy.deepcopy(env._engine.state)
                snap_deck = copy.deepcopy(env._engine.deck)

            step = env.step(result.action_type)
            if step.done:
                single_sample = step.rewards
                if need_snapshot and step.info.get("outcome") == "showdown":
                    averaged = _resample_showdown_rewards(
                        env, snap_state, snap_deck, pre_request, pre_response,
                        first_sample_rewards=single_sample, num_samples=args.num_showdown_samples,
                    )
                else:
                    averaged = single_sample
                break
            obs = step.obs

        print(
            f"hand {i:3d}  stacks={starting_stacks}  outcome={step.info.get('outcome')}  "
            f"actions={' -> '.join(action_log)}"
        )
        print(f"          single_sample_rewards={single_sample}  averaged_rewards={averaged}")


if __name__ == "__main__":
    main()
"""
run_river_human_demo.py — Play the hero seat yourself, against a
Controller-driven villain, on the heads-up river curriculum stage
(envs.river_env.RiverEnv). Exercises the controllers/ package built per
RL_lab_implementation_guide.md §9 ("Each seat in the Environment owns a
Controller").

Why the hero uses HumanController directly (not through an env slot)
------------------------------------------------------------------------
RiverEnv.step() already hands the hero a full StructuredObservation
(hole cards, board, betting, history) — that's the existing hero_policy
contract from envs/base_env.py's run_episode(). HumanController.act(obs)
consumes exactly that, so the hero side needs no adapter: this script
calls human.act(obs) directly on what env.reset()/env.step() return.

Why the villain goes through LegalMaskControllerAdapter
------------------------------------------------------------------------
RiverEnv's villain_policy slot only ever calls `act(legal_mask)` (see
river_env.py's CheckFoldVillainPolicy docstring) — there is no seam
today that hands the villain seat its own real observation (see
controllers/base_controller.py's module docstring, "KNOWN GAP"). So the
villain Controller must be wrapped in LegalMaskControllerAdapter, which
means only Controllers that decide from legal_mask ALONE are honestly
usable here:

    random          -> controllers.RandomController
    always_fold     -> controllers.RuleBasedController(always_fold_policy)
    calling_station -> controllers.RuleBasedController(calling_station_policy)

A FrozenPolicyController (a real trained/checkpointed agent) is
deliberately NOT offered as a villain choice below: wrapped through this
same adapter it would be forced to decide with fabricated hole
cards/board (see LegalMaskControllerAdapter's own docstring) — playing
worse than random, not a meaningful demo of the trained policy. Facing a
real trained villain requires an env-level fix (a per-seat observation
seam in RiverEnv), not something this script can paper over.

Setup notes
-----------
  - poker_eval shim installed below (mock — see mock_poker_eval.py).
  - No PyTorch/TensorFlow dependency at all: every villain option here
    is a HeroPolicy-shaped rule (evaluation.baseline_policies) or pure
    randomness, so this demo runs even without the heavier ML stack
    installed.

Usage
-----
    PYTHONPATH=. python3 run_river_human_demo.py
    PYTHONPATH=. python3 run_river_human_demo.py --villain calling_station --hands 5
"""
import sys

try:
    import poker_eval  # noqa: F401
except ImportError:
    import poker_rl_lab.mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse

from poker_rl_lab.actions.abstract_action import ActionType
from poker_rl_lab.controllers.base_controller import (
    LegalMaskControllerAdapter,
    RandomController,
    RuleBasedController,
)
from poker_rl_lab.controllers.human_controller import HumanController
from poker_rl_lab.evaluation.baseline_policies import always_fold_policy, calling_station_policy
from poker_rl_lab.envs.river_env import RiverEnv, VILLAIN_SEAT
from poker_rl_lab.representation.card_utils import card_id_to_str
from poker_rl_lab.scenarios.river_generator import RiverScenarioGenerator

_VILLAIN_BUILDERS = {
    "random": lambda seed: RandomController(seed=seed),
    "always_fold": lambda seed: RuleBasedController(always_fold_policy),
    "calling_station": lambda seed: RuleBasedController(calling_station_policy),
}


def _build_villain_policy(name: str, seed: int, big_blind: int):
    """Build the mask-only villain_policy RiverEnv actually calls."""
    controller = _VILLAIN_BUILDERS[name](seed)
    return LegalMaskControllerAdapter(
        controller, hero_seat=VILLAIN_SEAT, game_name="holdem"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play river hold'em against a Controller-driven villain."
    )
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument(
        "--villain", type=str, default="random", choices=sorted(_VILLAIN_BUILDERS)
    )
    parser.add_argument("--max-raises", type=int, default=4)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None,
                         help="Deal sequence seed. Omit for a fresh shuffle each run.")
    args = parser.parse_args()

    villain_policy = _build_villain_policy(args.villain, args.seed, args.big_blind)
    env = RiverEnv(villain_policy=villain_policy, max_raises=args.max_raises)

    scenario_gen = RiverScenarioGenerator(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
    )
    if args.seed is not None:
        scenario_gen.seed(args.seed)

    human = HumanController(name="You")

    print("=" * 60)
    print(f"RIVER HOLD'EM — you vs. villain={args.villain!r}")
    print(f"{args.hands} hand(s), max_raises={args.max_raises}, "
          f"blinds={args.small_blind}/{args.big_blind}")
    print("=" * 60)

    total_reward = 0.0

    for hand_num in range(1, args.hands + 1):
        print(f"\n########## HAND {hand_num}/{args.hands} ##########")

        scenario = scenario_gen.generate()
        obs = env.reset(scenario)

        while not obs.is_terminal:
            action = human.act(obs)
            obs, reward, done, info = env.step(action)

        total_reward += reward
        outcome = info.get("outcome", "?")
        print(f"\n>>> Hand result: {outcome}   reward={reward:+.0f}   "
              f"pot={info.get('pot')}   running total={total_reward:+.0f}")

    print("\n" + "=" * 60)
    print(f"Final result over {args.hands} hand(s): {total_reward:+.0f} chips "
          f"({total_reward / args.big_blind:+.2f} bb)")
    print("=" * 60)


if __name__ == "__main__":
    main()
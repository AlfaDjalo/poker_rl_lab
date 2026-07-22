"""
run_human_double_board_river_demo.py — Play the hero seat yourself,
against a Controller-driven villain, on the double-board river
curriculum stage (envs.full_holdem_env.FullHoldemEnv +
scenarios.double_board_river_generator.DoubleBoardRiverScenarioGenerator).
Sibling of run_human_river_demo.py — see that file's docstring for the
general rationale (HumanController used directly for the hero;
LegalMaskControllerAdapter for the villain).

Why FullHoldemEnv (not RiverEnv) here
------------------------------------------
Same reasoning as experiments/double_board_river_experiment.py: a
double-board hand's payout needs real two-board points scoring
(poker_engine/games/double_board.yaml's board1/board2 HIGH points),
which is genuine Engine/showdown_resolver logic, not something a
hand-rolled evaluator should reimplement. FullHoldemEnv already drives
any (game_def, rules) pair through the real PokerState, including
resuming mid-hand at street_index=3 with every node_card already dealt
— exactly DoubleBoardRiverScenarioGenerator's output shape.

Hero/villain seating
------------------------
DoubleBoardRiverScenarioGenerator is fixed heads-up: seat 1 is
out-of-position and acts first (current_player=1), seat 0 is
in-position. FullHoldemEnv reads hero_seat off scenario.current_player
at reset() time, so the human always plays seat 1 here, matching the
generator's own convention (see that module's docstring) — there is no
--player-count flag, unlike run_human_full_holdem_demo.py.

Villain options
------------------
Only Controllers that decide from legal_mask ALONE are honestly usable
behind LegalMaskControllerAdapter (see controllers/base_controller.py's
"KNOWN GAP" docstring) — a FrozenPolicyController would be forced to
act with fabricated (empty) hole cards/boards through this adapter, so
it is deliberately not offered here, matching every other
run_human_*_demo.py script:

    random          -> controllers.RandomController
    always_fold     -> controllers.RuleBasedController(always_fold_policy)
    calling_station -> controllers.RuleBasedController(calling_station_policy)

Setup notes
-----------
  - poker_eval shim installed below (mock — see mock_poker_eval.py).
    Must be installed BEFORE envs.full_holdem_env is imported (module-
    level PokerState import — see that module's docstring).
  - Requires poker_engine/games/double_board.yaml to be present.
  - No PyTorch/TensorFlow dependency: every villain option here is a
    HeroPolicy-shaped rule (evaluation.baseline_policies) or pure
    randomness.

Usage
-----
    PYTHONPATH=. python3 run_human_double_board_river_demo.py
    PYTHONPATH=. python3 run_human_double_board_river_demo.py --villain calling_station --hands 5
"""
import sys

try:
    import poker_engine.poker_eval  # noqa: F401
except ImportError:
    import mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse

from controllers.base_controller import (
    LegalMaskControllerAdapter,
    RandomController,
    RuleBasedController,
)
from controllers.human_controller import HumanController
from evaluation.baseline_policies import always_fold_policy, calling_station_policy
from envs.full_holdem_env import FullHoldemEnv
from scenarios.double_board_river_generator import DoubleBoardRiverScenarioGenerator

_VILLAIN_SEAT = 0   # in-position; matches DoubleBoardRiverScenarioGenerator's seating

_VILLAIN_BUILDERS = {
    "random": lambda seed: RandomController(seed=seed),
    "always_fold": lambda seed: RuleBasedController(always_fold_policy),
    "calling_station": lambda seed: RuleBasedController(calling_station_policy),
}


def _build_villain_policy(name: str, seed: int, game_name: str):
    """Build the mask-only villain_policy FullHoldemEnv actually calls
    for seat 0 (see envs/full_holdem_env.py's `villain_policies` dict)."""
    controller = _VILLAIN_BUILDERS[name](seed)
    return LegalMaskControllerAdapter(
        controller, hero_seat=_VILLAIN_SEAT, game_name=game_name
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play double-board river hold'em against a Controller-driven villain."
    )
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument(
        "--villain", type=str, default="random", choices=sorted(_VILLAIN_BUILDERS)
    )
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--game-name", type=str, default="double_board")
    parser.add_argument("--seed", type=int, default=None,
                         help="Deal sequence seed. Omit for a fresh shuffle each run.")
    args = parser.parse_args()

    villain_policy = _build_villain_policy(args.villain, args.seed, args.game_name)
    env = FullHoldemEnv(villain_policies={_VILLAIN_SEAT: villain_policy})

    scenario_gen = DoubleBoardRiverScenarioGenerator(
        small_blind=args.small_blind,
        big_blind=args.big_blind,
    )
    if args.seed is not None:
        scenario_gen.seed(args.seed)

    human = HumanController(name="You")

    print("=" * 60)
    print(f"DOUBLE-BOARD RIVER HOLD'EM — you vs. villain={args.villain!r}")
    print(f"{args.hands} hand(s), blinds={args.small_blind}/{args.big_blind}, "
          f"game={args.game_name!r}")
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
        print(f"\n>>> Hand result: pot={info.get('pot')}   reward={reward:+.0f}   "
              f"running total={total_reward:+.0f}")

    print("\n" + "=" * 60)
    print(f"Final result over {args.hands} hand(s): {total_reward:+.0f} chips "
          f"({total_reward / args.big_blind:+.2f} bb)")
    print("=" * 60)


if __name__ == "__main__":
    main()
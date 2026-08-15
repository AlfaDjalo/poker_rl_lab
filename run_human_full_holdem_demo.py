"""
run_human_full_holdem_demo.py — Play the hero seat yourself, against a
table of Controller-driven villains, on the full-table hold'em
curriculum stage (envs.full_holdem_env.FullHoldemEnv). Sibling of
run_human_river_demo.py — see that file's docstring for the general
rationale (HumanController used directly for the hero;
LegalMaskControllerAdapter for every villain seat).

Why per-seat villain policies here (unlike run_human_river_demo.py)
------------------------------------------------------------------------
RiverEnv is heads-up (exactly one villain seat). FullHoldemEnv has
player_count - 1 villain seats, each independently driven via
FullHoldemEnv's `default_villain_policy` / `villain_policies` dict (see
envs/full_holdem_env.py's constructor). This script builds one
LegalMaskControllerAdapter-wrapped Controller per non-hero seat so the
table can be seeded with a genuine MIX of styles (e.g. seat 2 always
folds, seat 4 is a calling station) rather than one uniform villain
policy — closer to how the KNOWN GAP note in
controllers/base_controller.py imagines this Protocol being exercised.

Same restriction as run_human_river_demo.py applies: only Controllers
that decide from legal_mask ALONE are honestly usable behind the
adapter (RandomController, RuleBasedController(always_fold_policy /
calling_station_policy)). FrozenPolicyController is not offered — see
that file's docstring for why.

Hero seat note
------------------
FullHoldemEnv's hero seat is whatever scenario.current_player names at
reset() (UTG for a fresh preflop deal from FullHoldemScenarioGenerator
— see that generator's docstring), NOT a fixed seat index. This script
reads env's hero seat back off the first observation each hand
(obs.hero_seat) so the human's prompt and the villain-policy table stay
correctly aligned regardless of which physical seat UTG lands on
(dealer_position is fixed at 0 by the generator, so in practice this is
seat 3 every hand here — but the script does not assume that).

Setup notes
-----------
  - poker_eval shim installed below (mock — see mock_poker_eval.py).
    Must be installed BEFORE envs.full_holdem_env is imported, since
    that module imports PokerState at module level (see its docstring).
  - No PyTorch/TensorFlow dependency: every villain option is a
    HeroPolicy-shaped rule (evaluation.baseline_policies) or pure
    randomness.

Usage
-----
    PYTHONPATH=. python3 run_human_full_holdem_demo.py
    PYTHONPATH=. python3 run_human_full_holdem_demo.py --player-count 6 \
        --villains random,always_fold,calling_station,random,always_fold --hands 5
"""
import sys

try:
    import poker_eval  # noqa: F401
except ImportError:
    import poker_rl_lab.mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import argparse

from poker_rl_lab.controllers.base_controller import (
    LegalMaskControllerAdapter,
    RandomController,
    RuleBasedController,
)
from poker_rl_lab.controllers.human_controller import HumanController
from poker_rl_lab.evaluation.baseline_policies import always_fold_policy, calling_station_policy
from poker_rl_lab.envs.full_holdem_env import FullHoldemEnv
from full_holdem_generator import FullHoldemScenarioGenerator

_VILLAIN_BUILDERS = {
    "random": lambda seed: RandomController(seed=seed),
    "always_fold": lambda seed: RuleBasedController(always_fold_policy),
    "calling_station": lambda seed: RuleBasedController(calling_station_policy),
}


def _build_villain_policies(names, seed, game_name):
    """
    Build FullHoldemEnv's `villain_policies` dict, keyed by every seat
    index EXCEPT the hero. Seat->name assignment is fixed at construction
    time in seat order (skipping nothing, since the hero seat is decided
    later at reset() time — see module docstring's "Hero seat note").

    Returns
    -------
    Dict[int, LegalMaskControllerAdapter]
        Keyed 0..player_count-1. FullHoldemEnv only ever consults the
        entry for a seat it's actually driving as a villain (see
        _apply_villain_action) — the hero's own entry, if present, is
        simply never read.
    """
    policies = {}
    for seat, name in enumerate(names):
        controller = _VILLAIN_BUILDERS[name](seed)
        policies[seat] = LegalMaskControllerAdapter(
            controller, hero_seat=seat, game_name=game_name
        )
    return policies


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play full-table hold'em against Controller-driven villains."
    )
    parser.add_argument("--hands", type=int, default=5)
    parser.add_argument("--player-count", type=int, default=6)
    parser.add_argument(
        "--villains",
        type=str,
        default="random,always_fold,calling_station,random,always_fold",
        help=(
            "Comma-separated villain style per non-hero seat, in seat "
            "order (wraps if shorter than player_count - 1). Choices: "
            + ", ".join(sorted(_VILLAIN_BUILDERS))
        ),
    )
    parser.add_argument("--small-blind", type=int, default=1)
    parser.add_argument("--big-blind", type=int, default=2)
    parser.add_argument("--ante", type=int, default=0)
    parser.add_argument("--game-name", type=str, default="holdem")
    parser.add_argument("--seed", type=int, default=None,
                         help="Deal sequence seed. Omit for a fresh shuffle each run.")
    args = parser.parse_args()

    villain_names_cycle = [s.strip() for s in args.villains.split(",") if s.strip()]
    for name in villain_names_cycle:
        if name not in _VILLAIN_BUILDERS:
            parser.error(
                f"Unknown villain style {name!r}; choices are "
                f"{sorted(_VILLAIN_BUILDERS)}"
            )
    # One name per seat (0..player_count-1), cycling if the user gave fewer.
    seat_names = [
        villain_names_cycle[i % len(villain_names_cycle)]
        for i in range(args.player_count)
    ]

    villain_policies = _build_villain_policies(seat_names, args.seed, args.game_name)
    env = FullHoldemEnv(villain_policies=villain_policies)

    scenario_gen = FullHoldemScenarioGenerator(
        player_count=args.player_count,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        ante=args.ante,
        game_name=args.game_name,
    )
    if args.seed is not None:
        scenario_gen.seed(args.seed)

    human = HumanController(name="You")

    print("=" * 60)
    print(f"FULL HOLD'EM — {args.player_count}-max, you vs. "
          f"{list(zip(range(args.player_count), seat_names))}")
    print(f"{args.hands} hand(s), blinds={args.small_blind}/{args.big_blind}, "
          f"ante={args.ante}")
    print("=" * 60)

    total_reward = 0.0

    for hand_num in range(1, args.hands + 1):
        print(f"\n########## HAND {hand_num}/{args.hands} ##########")

        scenario = scenario_gen.generate()
        obs = env.reset(scenario)
        print(f"(hero this hand is seat {obs.hero_seat} — "
              f"style {seat_names[obs.hero_seat]!r} is ignored for this seat)")

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
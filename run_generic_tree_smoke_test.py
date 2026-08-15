"""
run_generic_tree_smoke_test.py — Smoke test for the config-driven
GenericTreeEnv / GenericTreeScenarioGenerator / GenericTreeTrainingLoop
path (envs/generic_tree_env.py, scenarios/generic_tree_scenario_generator.py,
trainers/generic_tree_training_loop.py) against push_fold_duo.yaml.

Why this doesn't compare against PushFoldDuoScenarioGenerator directly
--------------------------------------------------------------------------
PushFoldDuoScenarioGenerator's stack-sizing floor is
`max(big_blind + 1, int(min_stack_bb * big_blind))`; GenericTreeScenarioGenerator's
is `max(1, int(min_stack_bb * big_blind))`. Same seed, different randint
bounds, different draws — bit-for-bit parity between the two generators
isn't meaningful to test. What actually matters is that GenericTreeEnv's
PAYOUT LOGIC (fold / push-fold / showdown resolution) is correct, which
this file checks directly with hand-constructed GenericTreeScenarios and
hand-computed expected rewards, independent of either generator's RNG.

Three manual scenarios (small_blind=1, big_blind=2, stack=20 both seats)
--------------------------------------------------------------------------
  1. SB folds                    -> BB wins SB's posted blind (+1/-1)
  2. SB pushes, BB folds         -> SB wins BB's posted blind (+2/-2)
  3. SB pushes, BB calls         -> showdown; MockScoringEngine always
                                     ranks the HIGHEST seat index as the
                                     winner (see mock_scoring_engine.py:
                                     score = 1_500_000 + i), so BB (seat
                                     index 1) always wins here regardless
                                     of actual cards -> (+20/-20)
All three are independently checked for the zero-sum invariant
(rewards must sum to 0 — heads-up, no rake) in addition to the exact
expected numbers.

Setup notes
-----------
  - poker_eval shim installed below (mock — see mock_poker_eval.py),
    same pattern as every other run_*_demo.py entry point.
  - Requires `triton` NOT installed alongside tensorflow (see
    run_push_fold_duo_demo.py's identical setup note) for the PPO
    smoke section only.

Usage
-----
    PYTHONPATH=. python3 run_generic_tree_smoke_test.py
"""
import sys

try:
    import poker_engine.poker_eval  # noqa: F401
except ImportError:
    print("Importing mock_poker_eval")
    import mock_poker_eval
    sys.modules.setdefault("poker_eval", mock_poker_eval)

import math
import random

import torch

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType
from poker_rl_lab.envs.generic_tree_env import GenericTreeEnv, GenericTreeScenario
from poker_engine.scoring.mock_scoring_engine import MockScoringEngine
from poker_rl_lab.scenarios.duo_game_config import load_duo_game
from poker_rl_lab.scenarios.generic_tree_scenario_generator import GenericTreeScenarioGenerator

SMALL_BLIND = 1
BIG_BLIND = 2
STACK = 20


def _print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, abs_tol=1e-6):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _make_env(config):
    # MockScoringEngine explicitly (not the default CppScoringEngine
    # fallback chain) so scenario 3's winner is deterministic regardless
    # of what's installed in this environment.
    return GenericTreeEnv(config, scoring_engine=MockScoringEngine())


def _manual_scenario(config) -> GenericTreeScenario:
    sb = config.seat_index("sb")
    bb = config.seat_index("bb")
    hole_masks = [0, 0]
    hole_masks[sb] = 0b0011          # arbitrary, non-overlapping — mock
    hole_masks[bb] = 0b1100          # evaluator ignores actual strength
    posted = [0, 0]
    posted[sb] = SMALL_BLIND
    posted[bb] = BIG_BLIND
    starting = [STACK, STACK]
    return GenericTreeScenario(
        config_name=config.name,
        hole_masks=tuple(hole_masks),
        board_mask=0,
        node_cards=(),
        starting_stacks=tuple(starting),
        posted=tuple(posted),
        pot=0,
    )


# ---------------------------------------------------------------------------
# Manual payout checks
# ---------------------------------------------------------------------------

def test_sb_fold(config) -> None:
    env = _make_env(config)
    obs = env.reset(_manual_scenario(config))
    assert obs.legal_mask[ActionType.FOLD]
    assert obs.legal_mask[ActionType.ALL_IN]

    step = env.step(AbstractAction(ActionType.FOLD))
    assert step.done
    assert step.info["outcome"] == "fold"

    sb, bb = config.seat_index("sb"), config.seat_index("bb")
    _assert_close(sum(step.rewards.values()), 0.0, "sb_fold zero-sum")
    _assert_close(step.rewards[sb], -1.0, "sb_fold sb reward")
    _assert_close(step.rewards[bb], +1.0, "sb_fold bb reward")
    print("PASS  sb_fold:", step.rewards)


def test_sb_push_bb_fold(config) -> None:
    env = _make_env(config)
    env.reset(_manual_scenario(config))

    step1 = env.step(AbstractAction(ActionType.ALL_IN))
    assert not step1.done
    assert step1.obs.legal_mask[ActionType.FOLD]
    assert step1.obs.legal_mask[ActionType.CALL]

    step2 = env.step(AbstractAction(ActionType.FOLD))
    assert step2.done
    assert step2.info["outcome"] == "fold"

    sb, bb = config.seat_index("sb"), config.seat_index("bb")
    _assert_close(sum(step2.rewards.values()), 0.0, "sb_push_bb_fold zero-sum")
    _assert_close(step2.rewards[sb], +2.0, "sb_push_bb_fold sb reward")
    _assert_close(step2.rewards[bb], -2.0, "sb_push_bb_fold bb reward")
    print("PASS  sb_push_bb_fold:", step2.rewards)


def test_sb_push_bb_call_showdown(config) -> None:
    env = _make_env(config)
    env.reset(_manual_scenario(config))

    env.step(AbstractAction(ActionType.ALL_IN))
    step = env.step(AbstractAction(ActionType.CALL))
    assert step.done
    assert step.info["outcome"] == "showdown"

    sb, bb = config.seat_index("sb"), config.seat_index("bb")
    _assert_close(sum(step.rewards.values()), 0.0, "showdown zero-sum")
    # MockScoringEngine: highest seat INDEX always wins (see docstring).
    _assert_close(step.rewards[bb], +20.0, "showdown bb reward")
    _assert_close(step.rewards[sb], -20.0, "showdown sb reward")
    print("PASS  sb_push_bb_call_showdown:", step.rewards)


# ---------------------------------------------------------------------------
# Random-policy invariant sweep
# ---------------------------------------------------------------------------

def test_random_policy_invariants(config, num_episodes: int = 500) -> None:
    """
    Plays num_episodes hands with a uniform-random legal action at every
    decision, through the REAL GenericTreeScenarioGenerator (not the
    manual scenario above), and checks structural invariants that must
    hold regardless of cards/RNG:
      - rewards always sum to exactly scenario.pot — see note below
      - no reward magnitude ever exceeds either seat's starting stack
        plus the dead-money pot
      - every action taken was actually legal per obs.legal_mask

    Why "sum to scenario.pot", not "sum to zero"
    ---------------------------------------------
    A GenericTreeScenario's `pot` field is DEAD MONEY — chips already
    in the middle from streets this scenario doesn't model (see
    duo_game_config.py's PotConfig docstring: "randomized_bb" pot mode
    represents already-in-the-middle chips with no per-seat
    attribution). It was never subtracted from any seat's
    starting_stacks, so it isn't part of the zero-sum accounting
    between the two players' reward deltas — it's exogenous funding
    that flows entirely to whoever wins it. `pot.mode: "blinds"` (e.g.
    push-fold-duo) is the zero-sum special case where scenario.pot == 0
    and every committed chip instead lives in `posted` (which DOES come
    out of a seat's own starting_stacks) — so this single formula
    correctly reduces to the old "sum to zero" check for that game
    while also being correct for river-duo's dead-pot case. The manual
    payout checks above already assert this exact behaviour per-outcome
    (e.g. "check_check pot-sum" == RIVER_POT, not 0).
    """
    env = _make_env(config)
    scenario_gen = GenericTreeScenarioGenerator(config)
    scenario_gen.seed(12345)
    rng = random.Random(999)

    for ep in range(num_episodes):
        scenario = scenario_gen.generate()
        obs = env.reset(scenario)
        starting_stacks = scenario.starting_stacks

        while True:
            legal = [i for i, m in enumerate(obs.legal_mask) if m]
            assert legal, f"episode {ep}: no legal actions at a non-terminal obs"
            choice = ActionType(rng.choice(legal))
            step = env.step(AbstractAction(choice))
            if step.done:
                total = sum(step.rewards.values())
                if not math.isclose(total, scenario.pot, abs_tol=1e-6):
                    raise AssertionError(
                        f"episode {ep}: rewards do not sum to scenario.pot "
                        f"({scenario.pot}): {step.rewards} (sum={total})"
                    )
                for seat, r in step.rewards.items():
                    # Ceiling is the seat's own stack PLUS the dead pot —
                    # a winner can gain scenario.pot on top of their own
                    # stack change (see docstring above); a loser can
                    # never lose more than they personally started with.
                    ceiling = starting_stacks[seat] + scenario.pot
                    if abs(r) > ceiling + 1e-6:
                        raise AssertionError(
                            f"episode {ep}: seat {seat} reward {r} exceeds "
                            f"ceiling {ceiling} (stack {starting_stacks[seat]} "
                            f"+ pot {scenario.pot})"
                        )
                break
            obs = step.obs

    print(f"PASS  random_policy_invariants over {num_episodes} episodes")


# ---------------------------------------------------------------------------
# Tiny end-to-end PPO smoke run
# ---------------------------------------------------------------------------

def test_ppo_smoke(config) -> None:
    """
    A handful of GenericTreeTrainingLoop iterations against push_fold_duo,
    just checking the whole pipeline (env -> adapter -> actor_critic ->
    rollout buffer -> PPO update) runs without error and produces finite
    losses — NOT a convergence test.
    """
    from poker_rl_lab.models.actor_critic_factory import build_actor_critic
    from poker_rl_lab.models.shared_single_head_policy import SharedSingleHeadActorCritic
    from poker_rl_lab.representation.observation_adapter import ObservationAdapter
    from poker_rl_lab.trainers.generic_tree_training_loop import GenericTreeTrainingLoop

    env = _make_env(config)
    scenario_gen = GenericTreeScenarioGenerator(config)
    scenario_gen.seed(7)
    adapter = ObservationAdapter()

    inner = build_actor_critic()  # untrained hand encoder — pipeline test only
    actor_critic = SharedSingleHeadActorCritic(inner)

    optimizers = {
        s.index: torch.optim.Adam(
            actor_critic.seat_parameters(s.index), lr=3e-4
        )
        for s in config.seats
    }

    loop = GenericTreeTrainingLoop(
        env=env, scenario_gen=scenario_gen, actor_critic=actor_critic,
        adapter=adapter, optimizers=optimizers,
    )

    history = loop.train(
        num_iterations=2, episodes_per_iteration=8,
        epochs=1, minibatch_size=8, log_every=0, verbose=False,
    )

    assert len(history) == 2
    for stats in history:
        for seat, loss in stats.policy_loss.items():
            assert math.isfinite(loss), f"seat {seat} policy_loss is non-finite: {loss}"
        for seat, loss in stats.value_loss.items():
            assert math.isfinite(loss), f"seat {seat} value_loss is non-finite: {loss}"

    print("PASS  ppo_smoke (2 iterations, finite losses)")

RIVER_STACK = 200        # 100bb @ big_blind=2
RIVER_POT = 10           # arbitrary fixed pot for hand-checkable expectations


def _river_manual_scenario(config) -> GenericTreeScenario:
    oop = config.seat_index("oop")
    ip = config.seat_index("ip")
    hole_masks = [0, 0]
    hole_masks[oop] = 0b0011
    hole_masks[ip] = 0b1100
    board = [4, 5, 6, 7, 8]  # 5 arbitrary distinct card ids, no overlap w/ holes
    starting = [RIVER_STACK, RIVER_STACK]
    return GenericTreeScenario(
        config_name=config.name,
        hole_masks=tuple(hole_masks),
        board_mask=sum(1 << c for c in board),
        node_cards=tuple(board),
        starting_stacks=tuple(starting),
        posted=(0, 0),
        pot=RIVER_POT,
    )


def test_river_check_check(config) -> None:
    env = _make_env(config)
    obs = env.reset(_river_manual_scenario(config))
    assert obs.legal_mask[ActionType.CHECK]
    assert obs.legal_mask[ActionType.BET_100]

    step1 = env.step(AbstractAction(ActionType.CHECK))
    assert not step1.done
    assert step1.obs.legal_mask[ActionType.CHECK]
    assert step1.obs.legal_mask[ActionType.BET_100]

    step2 = env.step(AbstractAction(ActionType.CHECK))
    assert step2.done
    assert step2.info["outcome"] == "showdown"

    oop, ip = config.seat_index("oop"), config.seat_index("ip")
    _assert_close(sum(step2.rewards.values()), RIVER_POT, "check_check pot-sum")
    # MockScoringEngine: highest seat index (ip) always wins.
    _assert_close(step2.rewards[ip], +float(RIVER_POT), "check_check ip reward")
    _assert_close(step2.rewards[oop], 0.0, "check_check oop reward")
    print("PASS  river_check_check:", step2.rewards)


def test_river_check_bet_fold(config) -> None:
    env = _make_env(config)
    env.reset(_river_manual_scenario(config))

    env.step(AbstractAction(ActionType.CHECK))
    step2 = env.step(AbstractAction(ActionType.BET_100))
    assert not step2.done
    assert step2.obs.legal_mask[ActionType.CALL]
    assert step2.obs.legal_mask[ActionType.FOLD]

    step3 = env.step(AbstractAction(ActionType.FOLD))
    assert step3.done
    assert step3.info["outcome"] == "fold"

    oop, ip = config.seat_index("oop"), config.seat_index("ip")
    # IP bet pot (=10) on top of the 10 already carried -> pot at fold = 20.
    _assert_close(sum(step3.rewards.values()), RIVER_POT, "check_bet_fold pot-sum")
    _assert_close(step3.rewards[ip], +float(RIVER_POT), "check_bet_fold ip reward")
    _assert_close(step3.rewards[oop], 0.0, "check_bet_fold oop reward")
    print("PASS  river_check_bet_fold:", step3.rewards)


def test_river_check_bet_call_showdown(config) -> None:
    env = _make_env(config)
    env.reset(_river_manual_scenario(config))

    env.step(AbstractAction(ActionType.CHECK))
    env.step(AbstractAction(ActionType.BET_100))
    step = env.step(AbstractAction(ActionType.CALL))
    assert step.done
    assert step.info["outcome"] == "showdown"

    oop, ip = config.seat_index("oop"), config.seat_index("ip")
    _assert_close(sum(step.rewards.values()), RIVER_POT, "check_bet_call pot-sum")
    _assert_close(step.rewards[ip], +2.0 * RIVER_POT, "check_bet_call ip reward")
    _assert_close(step.rewards[oop], -1.0 * RIVER_POT, "check_bet_call oop reward")
    print("PASS  river_check_bet_call_showdown:", step.rewards)


def test_river_bet_fold(config) -> None:
    env = _make_env(config)
    env.reset(_river_manual_scenario(config))

    step1 = env.step(AbstractAction(ActionType.BET_100))
    assert not step1.done
    assert step1.obs.legal_mask[ActionType.CALL]
    assert step1.obs.legal_mask[ActionType.FOLD]

    step2 = env.step(AbstractAction(ActionType.FOLD))
    assert step2.done
    assert step2.info["outcome"] == "fold"

    oop, ip = config.seat_index("oop"), config.seat_index("ip")
    _assert_close(sum(step2.rewards.values()), RIVER_POT, "bet_fold pot-sum")
    _assert_close(step2.rewards[oop], +float(RIVER_POT), "bet_fold oop reward")
    _assert_close(step2.rewards[ip], 0.0, "bet_fold ip reward")
    print("PASS  river_bet_fold:", step2.rewards)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    push_fold_config = load_duo_game("push_fold_duo")
    river_config = load_duo_game("river_duo")

    _print_header("PUSH-FOLD-DUO: MANUAL PAYOUT CHECKS")
    test_sb_fold(push_fold_config)
    test_sb_push_bb_fold(push_fold_config)
    test_sb_push_bb_call_showdown(push_fold_config)

    _print_header("PUSH-FOLD-DUO: RANDOM POLICY INVARIANTS")
    test_random_policy_invariants(push_fold_config)

    _print_header("RIVER-DUO: MANUAL PAYOUT CHECKS")
    test_river_check_check(river_config)
    test_river_check_bet_fold(river_config)
    test_river_check_bet_call_showdown(river_config)
    test_river_bet_fold(river_config)

    _print_header("RIVER-DUO: RANDOM POLICY INVARIANTS")
    test_random_policy_invariants(river_config)

    _print_header("PPO SMOKE RUN (push-fold-duo)")
    test_ppo_smoke(push_fold_config)

    _print_header("PPO SMOKE RUN (river-duo)")
    test_ppo_smoke(river_config)

    _print_header("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
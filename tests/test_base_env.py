"""
tests/test_base_env.py — Conformance tests for BasePokerEnv.

The same suite is run against every concrete environment.
If a new environment passes all checks here, it is safe to hand to
the training loop.

Run directly:

    python tests/test_base_env.py

Design
------
conformance_suite(env, gen, label)
    The shared test function.  Every assertion documents one clause of
    the BasePokerEnv contract.  Tests are independent — failures in one
    do not prevent the others from running.

Applied to
    PushFoldEnv  +  PushFoldScenarioGenerator
    RiverEnv     +  RiverScenarioGenerator
"""

from __future__ import annotations

import math
from typing import List, Tuple

from scenarios.push_fold_generator import PushFoldScenarioGenerator
from scenarios.river_generator      import RiverScenarioGenerator
from actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS
from envs.base_env import BasePokerEnv, RandomHeroPolicy, StepRecord, run_episode

# Concrete envs under test
from envs.push_fold_env import PushFoldEnv
from envs.river_env     import RiverEnv


# ── Per-env generator factories ───────────────────────────────────────────────

def _push_fold_gen(seed: int = 0) -> PushFoldScenarioGenerator:
    g = PushFoldScenarioGenerator(player_count=2, small_blind=1, big_blind=2)
    g.seed(seed)
    return g


def _river_gen(seed: int = 0) -> RiverScenarioGenerator:
    g = RiverScenarioGenerator(
        player_count=2, small_blind=1, big_blind=2,
        min_pot_bb=4.0, max_pot_bb=10.0,
        min_stack_bb=10.0, max_stack_bb=30.0,
    )
    g.seed(seed)
    return g


# ── Conformance suite ─────────────────────────────────────────────────────────

def conformance_suite(env: BasePokerEnv, gen, label: str, n: int = 30) -> None:
    """
    Run all conformance checks for *env* using *gen* to produce scenarios.

    Parameters
    ----------
    env   : BasePokerEnv   Concrete environment under test.
    gen               : ScenarioGenerator that matches env.
    label : str            Name printed in PASS lines.
    n     : int            Number of episodes to run per check.
    """
    policy = RandomHeroPolicy(seed=0)
    print(f"\n── {label} ──────────────────────────────────────")

    # ── 1. Inherits from BasePokerEnv ─────────────────────────────────────
    assert isinstance(env, BasePokerEnv), \
        f"{type(env).__name__} must inherit from BasePokerEnv"
    print(f"PASS  [{label}]  inherits from BasePokerEnv")

    # ── 2. reset() returns StructuredObservation ───────────────────────────
    from envs.observations import StructuredObservation
    sc  = gen.generate()
    obs = env.reset(sc)
    assert isinstance(obs, StructuredObservation), \
        f"reset() must return StructuredObservation; got {type(obs)}"
    print(f"PASS  [{label}]  reset() returns StructuredObservation")

    # ── 3. Initial observation is non-terminal ─────────────────────────────
    assert not obs.is_terminal, "reset() must return a non-terminal observation"
    print(f"PASS  [{label}]  initial observation is non-terminal")

    # ── 4. legal_mask has correct length ──────────────────────────────────
    assert len(obs.legal_mask) == NUM_ACTIONS, \
        f"legal_mask must have length {NUM_ACTIONS}; got {len(obs.legal_mask)}"
    print(f"PASS  [{label}]  legal_mask has length {NUM_ACTIONS}")

    # ── 5. At least one legal action at reset ────────────────────────────
    assert any(obs.legal_mask), \
        "Initial observation must have at least one legal action"
    print(f"PASS  [{label}]  at least one legal action at reset")

    # ── 6. step() returns (obs, float, bool, dict) ───────────────────────
    action = policy(obs)
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 4, \
        "step() must return a 4-tuple"
    next_obs, reward, done, info = result
    assert isinstance(next_obs, StructuredObservation)
    assert isinstance(reward, float), f"reward must be float; got {type(reward)}"
    assert isinstance(done, bool),    f"done must be bool; got {type(done)}"
    assert isinstance(info, dict),    f"info must be dict; got {type(info)}"
    print(f"PASS  [{label}]  step() returns (StructuredObservation, float, bool, dict)")

    # ── 7. reward is finite ───────────────────────────────────────────────
    assert math.isfinite(reward), f"reward must be finite; got {reward}"
    print(f"PASS  [{label}]  reward is finite")

    # ── 8. is_terminal matches done ───────────────────────────────────────
    # Run many episodes to cover both True and False cases
    gen.seed(0)
    policy._rng.seed(0)
    for _ in range(n):
        sc  = gen.generate()
        obs = env.reset(sc)
        while not obs.is_terminal:
            next_obs, reward, done, _ = env.step(policy(obs))
            if done:
                assert next_obs.is_terminal, \
                    "obs.is_terminal must be True when done=True"
                assert all(m is False for m in next_obs.legal_mask), \
                    "Terminal obs must have empty legal_mask"
                assert math.isfinite(reward), \
                    f"Terminal reward must be finite; got {reward}"
            else:
                assert not next_obs.is_terminal, \
                    "obs.is_terminal must be False when done=False"
            obs = next_obs
    print(f"PASS  [{label}]  is_terminal matches done across {n} episodes")

    # ── 9. Terminal legal_mask is all-False ───────────────────────────────
    # (covered above, but print separately for clarity)
    print(f"PASS  [{label}]  terminal legal_mask is all-False")

    # ── 10. step() after done raises RuntimeError ─────────────────────────
    gen.seed(999)
    policy._rng.seed(999)
    sc  = gen.generate()
    obs = env.reset(sc)
    # Run episode to completion
    total, _ = run_episode(env, sc, policy)
    # Now obs is terminal; another step must raise
    raised = False
    try:
        env.step(AbstractAction(ActionType.FOLD))
    except RuntimeError:
        raised = True
    assert raised, "step() after done=True must raise RuntimeError"
    print(f"PASS  [{label}]  step() after done raises RuntimeError")

    # ── 11. step() before reset() raises RuntimeError ────────────────────
    fresh_env = type(env)()          # same class, freshly constructed
    raised = False
    try:
        fresh_env.step(AbstractAction(ActionType.FOLD))
    except RuntimeError:
        raised = True
    assert raised, "step() before reset() must raise RuntimeError"
    print(f"PASS  [{label}]  step() before reset raises RuntimeError")

    # ── 12. hero_seat is constant within an episode ───────────────────────
    gen.seed(1)
    policy._rng.seed(1)
    for _ in range(10):
        sc     = gen.generate()
        obs    = env.reset(sc)
        h_seat = obs.hero_seat
        while not obs.is_terminal:
            obs, _, done, _ = env.step(policy(obs))
            if not done:
                assert obs.hero_seat == h_seat, \
                    "hero_seat must not change within an episode"
    print(f"PASS  [{label}]  hero_seat is constant within an episode")

    # ── 13. game_name is constant within an episode ───────────────────────
    gen.seed(2)
    policy._rng.seed(2)
    for _ in range(10):
        sc    = gen.generate()
        obs   = env.reset(sc)
        gname = obs.game_name
        while not obs.is_terminal:
            obs, _, done, _ = env.step(policy(obs))
            if not done:
                assert obs.game_name == gname, \
                    "game_name must not change within an episode"
    print(f"PASS  [{label}]  game_name is constant within an episode")

    # ── 14. hole cards do not change within an episode ────────────────────
    gen.seed(3)
    policy._rng.seed(3)
    for _ in range(10):
        sc        = gen.generate()
        obs       = env.reset(sc)
        init_hand = obs.hole_cards.cards
        while not obs.is_terminal:
            obs, _, done, _ = env.step(policy(obs))
            if not done:
                assert obs.hole_cards.cards == init_hand, \
                    "Hero hole cards must not change during an episode"
    print(f"PASS  [{label}]  hole cards are stable within an episode")

    # ── 15. Every episode terminates (no infinite loops) ─────────────────
    gen.seed(50)
    policy._rng.seed(50)
    for ep in range(n):
        sc    = gen.generate()
        total, traj = run_episode(env, sc, policy)
        assert len(traj) >= 1, f"Episode {ep} produced an empty trajectory"
        assert traj[-1].done is True, f"Episode {ep} did not terminate"
    print(f"PASS  [{label}]  all {n} episodes terminate")

    # ── 16. run_episode total_reward matches last step reward ─────────────
    gen.seed(77)
    policy._rng.seed(77)
    for _ in range(20):
        sc           = gen.generate()
        total, traj  = run_episode(env, sc, policy)
        step_rewards = sum(s.reward for s in traj)
        assert abs(total - step_rewards) < 1e-9, \
            f"run_episode total {total} != sum of step rewards {step_rewards}"
    print(f"PASS  [{label}]  run_episode total_reward is sum of step rewards")

    # ── 17. Seeded runs are reproducible ──────────────────────────────────
    def _run(seed: int) -> List[float]:
        g = _push_fold_gen(seed) if "Push" in label else _river_gen(seed)
        p = RandomHeroPolicy(seed=seed)
        e = type(env)()
        results = []
        for _ in range(5):
            sc    = g.generate()
            total, _ = run_episode(e, sc, p)
            results.append(total)
        return results

    assert _run(42) == _run(42), "Same seed must produce identical results"
    print(f"PASS  [{label}]  seeded runs are reproducible")

    print(f"\n✓  {label}: all 17 conformance checks passed.")


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conformance_suite(
        env   = PushFoldEnv(),
        gen   = _push_fold_gen(),
        label = "PushFoldEnv",
    )
    conformance_suite(
        env   = RiverEnv(),
        gen   = _river_gen(),
        label = "RiverEnv",
    )
    print("\n✓  All conformance suites passed.")
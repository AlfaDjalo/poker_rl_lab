"""
benchmarks/benchmark_rollout_collection.py — Throughput and variance
benchmarks for the two changes made in this pass:

  1. VectorizedPokerEnv + PokerActorCritic.act_batch() vs the original
     one-hand-at-a-time, single-example collect_rollouts().
  2. MonteCarloEquityEvaluator's K-sample-averaged equity() vs a single
     hero_wins() sample, for push-fold's terminal reward.

Uses a lightweight stub ActorCritic (pure torch, no TF hand encoder, no
poker_eval extension) so this runs anywhere torch is installed — same
"duck-typed stub" pattern already used in this codebase's own test
suite (see trainers/ppo_trainer.py's ActorCriticLike docstring). Swap
in a real models.actor_critic_factory.build_actor_critic() output to
benchmark the real network; the relative speedup from batching should
be AT LEAST as large there, since the real RepresentationModel makes
several more per-call sub-encoder dispatches per example than this stub
does (see representation_model.py's own "batched forward_batch()"
docstring on why the TF hand encoder in particular dominates per-call
dispatch cost).

Usage
-----
    PYTHONPATH=. python3 benchmarks/benchmark_rollout_collection.py
"""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn

from actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS
from envs.push_fold_env import PushFoldEnv, MockEvaluator
from envs.vectorized_env import VectorizedPokerEnv
from envs.monte_carlo_evaluator import MonteCarloEquityEvaluator
from scenarios.base_scenario import BaseScenario, PlayerScenario
from scenarios.scenario_generator import ScenarioGenerator
from trainers.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Minimal self-contained push-fold scenario generator (avoids depending
# on scenarios.push_fold_generator's exact constructor signature, which
# isn't part of this change set).
# ---------------------------------------------------------------------------

class _BenchmarkPushFoldGenerator(ScenarioGenerator):
    def __init__(self, small_blind: int = 1, big_blind: int = 2,
                 min_stack_bb: float = 4.0, max_stack_bb: float = 20.0) -> None:
        super().__init__(player_count=2)
        self._sb, self._bb = small_blind, big_blind
        self._min_bb, self._max_bb = min_stack_bb, max_stack_bb

    def generate(self) -> BaseScenario:
        stack = self._rng.randint(int(self._min_bb * self._bb), int(self._max_bb * self._bb))
        villain_cards = self._deal_unique_cards(2)
        villain_mask = self._cards_to_mask(villain_cards)
        hero_cards = self._deal_unique_cards(2, excluded_mask=villain_mask)
        hero_mask = self._cards_to_mask(hero_cards)

        villain = PlayerScenario(stack=0, hand_mask=villain_mask, current_bet=stack,
                                  total_contribution=stack, has_folded=False, is_all_in=True)
        hero_posted = min(self._bb, stack)
        hero = PlayerScenario(stack=stack - hero_posted, hand_mask=hero_mask,
                               current_bet=hero_posted, total_contribution=hero_posted,
                               has_folded=False, is_all_in=False)

        return BaseScenario(
            game_name="holdem", street_index=0, pot=stack + hero_posted,
            dealer_position=0, current_player=1,
            bet_to_call=stack - hero_posted, min_raise=self._bb,
            players=(villain, hero), node_cards=tuple(),
            small_blind=self._sb, big_blind=self._bb,
        )


# ---------------------------------------------------------------------------
# Minimal stub actor-critic (pure torch, no representation model / TF)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StubActionResult:
    action_type: int
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class _StubFeaturizer:
    """obs -> fixed-size float feature vector, no representation model."""
    DIM = 8

    def adapt(self, obs) -> List[float]:
        hole = obs.hole_cards.cards
        return [
            (hole[0] if len(hole) > 0 else 0) / 51.0,
            (hole[1] if len(hole) > 1 else 0) / 51.0,
            obs.betting.pot / 1000.0,
            obs.betting.bet_to_call / 1000.0,
            obs.betting.min_raise / 1000.0,
            float(obs.hero_seat),
            float(obs.dealer_position),
            1.0,
        ]


class _StubActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_StubFeaturizer.DIM, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.policy_head = nn.Linear(64, NUM_ACTIONS)
        self.value_head = nn.Linear(64, 1)

    def _forward(self, feats: torch.Tensor):
        h = self.net(feats)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(self, obs: Sequence[float], legal_mask: Sequence[bool], deterministic: bool = False):
        return self.act_batch([obs], [legal_mask], deterministic)[0]

    def act_batch(self, obs_batch: Sequence[Sequence[float]], legal_masks: Sequence[Sequence[bool]],
                   deterministic: bool = False) -> List[_StubActionResult]:
        feats = torch.tensor(obs_batch, dtype=torch.float32)
        logits, values = self._forward(feats)
        mask = torch.as_tensor(legal_masks, dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1.0e9)
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_probs, entropies = dist.log_prob(actions), dist.entropy()
        return [
            _StubActionResult(int(actions[i].item()), log_probs[i], values[i], entropies[i])
            for i in range(len(obs_batch))
        ]

    def evaluate_actions(self, obs_batch, legal_masks, actions):
        feats = torch.tensor(obs_batch, dtype=torch.float32)
        logits, values = self._forward(feats)
        mask = torch.as_tensor(legal_masks, dtype=torch.bool)
        logits = logits.masked_fill(~mask, -1.0e9)
        dist = torch.distributions.Categorical(logits=logits)
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        return dist.log_prob(actions_t), values, dist.entropy()


# ---------------------------------------------------------------------------
# Benchmark 1: sequential vs vectorized rollout collection
# ---------------------------------------------------------------------------

def benchmark_rollout_collection(num_episodes: int = 2000, num_envs: int = 64) -> None:
    print("=" * 72)
    print(f"ROLLOUT COLLECTION THROUGHPUT  (num_episodes={num_episodes}, num_envs={num_envs})")
    print("=" * 72)

    actor_critic = _StubActorCritic()
    actor_critic.eval()
    adapter = _StubFeaturizer()

    # --- sequential path ---
    from trainers.ppo_trainer import PPOTrainer
    trainer = PPOTrainer(actor_critic, adapter, torch.optim.Adam(actor_critic.parameters()))

    gen_seq = _BenchmarkPushFoldGenerator()
    gen_seq.seed(0)
    env_seq = PushFoldEnv(evaluator=MockEvaluator())

    t0 = time.perf_counter()
    buf_seq = trainer.collect_rollouts(env_seq, gen_seq, num_episodes)
    t_seq = time.perf_counter() - t0

    # --- vectorized path ---
    gen_vec = _BenchmarkPushFoldGenerator()
    gen_vec.seed(0)
    venv = VectorizedPokerEnv(
        lambda: PushFoldEnv(evaluator=MockEvaluator()), gen_vec, num_envs=num_envs
    )

    t0 = time.perf_counter()
    buf_vec = trainer.collect_rollouts_vectorized(venv, num_episodes)
    t_vec = time.perf_counter() - t0

    print(f"sequential : {t_seq:6.2f}s  ({num_episodes / t_seq:8.1f} episodes/sec)  "
          f"transitions={len(buf_seq)}")
    print(f"vectorized : {t_vec:6.2f}s  ({num_episodes / t_vec:8.1f} episodes/sec)  "
          f"transitions={len(buf_vec)}")
    print(f"speedup    : {t_seq / t_vec:.2f}x")
    print()


# ---------------------------------------------------------------------------
# Benchmark 2: Monte Carlo variance reduction
# ---------------------------------------------------------------------------

def benchmark_mc_variance(num_trials: int = 3000) -> None:
    print("=" * 72)
    print("MONTE CARLO EQUITY VARIANCE REDUCTION (push-fold terminal reward)")
    print("=" * 72)

    gen = _BenchmarkPushFoldGenerator()
    gen.seed(1)
    scenario = gen.generate()
    hero, villain = scenario.players[1], scenario.players[0]
    to_call = min(scenario.bet_to_call - hero.current_bet, hero.stack)
    hero_total = hero.current_bet + to_call
    effective = min(hero_total, villain.current_bet)
    pot_plus_eff = scenario.pot + effective

    for k in (1, 5, 30, 100):
        mc = MonteCarloEquityEvaluator(num_samples=k, seed=None)
        rewards = []
        for _ in range(num_trials):
            eq = mc.equity(hero.hand_mask, villain.hand_mask, board_mask=0)
            rewards.append(eq.p_win * pot_plus_eff - eq.p_lose * hero_total)
        mean = statistics.mean(rewards)
        std = statistics.pstdev(rewards)
        print(f"num_samples={k:4d}   mean_reward={mean:+8.3f}   std={std:8.3f}")

    print("\n(mean should stay ~constant across K — unbiased; std should shrink ~ 1/sqrt(K))")
    print()


if __name__ == "__main__":
    benchmark_rollout_collection()
    benchmark_mc_variance()
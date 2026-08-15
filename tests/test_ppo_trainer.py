"""
trainers/test_ppo_trainer.py — Unit tests for PPOTrainer.

Deliberately avoids the real PokerActorCritic / RepresentationModel /
PretrainedHandEncoder stack (TF-backed, and the real envs need
loader.load_game -> poker_eval, the C++ extension). Per
ppo_trainer.py's docstring, PPOTrainer only depends on a narrow
duck-typed surface (ActorCriticLike / AdapterLike) — these tests
exercise exactly that surface with pure-Python/torch stubs, so they run
anywhere torch is installed, independent of the rest of the Engine.

Stub pieces
-----------
    _StubActionResult   : mirrors models.policy_value_network.ActionResult's
                           shape (action_type, log_prob, value, entropy)
                           without importing that module.
    _BanditEnv           : BasePokerEnv-shaped one-step "bandit": every
                           episode is a single hero decision between
                           FOLD (reward -1) and CHECK (reward +1). Simple
                           enough to hand-verify that PPO updates shift
                           probability mass toward the better action.
    _StubScenarioGen     : .generate() returns a trivial marker object
                           (the bandit env ignores it).
    _IdentityAdapter     : .adapt(obs) -> obs (no RepresentationInput
                           needed since _StubActorCritic ignores obs
                           content entirely — it has one global logits
                           vector, not a function of state).
    _StubActorCritic     : nn.Module with one learnable logit per action
                           and one learnable scalar value, state-
                           independent. Enough to verify PPO's loss math
                           actually moves parameters in the expected
                           direction and that collect/update/train run
                           end-to-end without shape errors.

Run with:
    pytest trainers/test_ppo_trainer.py -v
"""

from dataclasses import dataclass
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn
from torch.distributions import Categorical

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS
from poker_rl_lab.trainers.ppo_trainer import PPOTrainer
from poker_rl_lab.trainers.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Stub action result (mirrors models.policy_value_network.ActionResult)
# ---------------------------------------------------------------------------

@dataclass
class _StubActionResult:
    action_type: int
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


# ---------------------------------------------------------------------------
# Stub observation
# ---------------------------------------------------------------------------

@dataclass
class _StubObs:
    is_terminal: bool
    legal_mask: Tuple[bool, ...]


def _legal_mask_fold_check() -> Tuple[bool, ...]:
    mask = [False] * NUM_ACTIONS
    mask[ActionType.FOLD] = True
    mask[ActionType.CHECK] = True
    return tuple(mask)


# ---------------------------------------------------------------------------
# Stub environment: one-step bandit, CHECK is strictly better than FOLD
# ---------------------------------------------------------------------------

class _BanditEnv:
    """
    One hero decision per episode. CHECK always pays +1, FOLD always
    pays -1. Deliberately not a real poker environment — just enough
    structure (reset/step, is_terminal, legal_mask, reward/done) to
    exercise PPOTrainer's collection loop end-to-end.
    """

    def __init__(self) -> None:
        self._done = True

    def reset(self, scenario) -> _StubObs:
        self._done = False
        return _StubObs(is_terminal=False, legal_mask=_legal_mask_fold_check())

    def step(self, action: AbstractAction):
        if self._done:
            raise RuntimeError("step() called before reset() or after done.")
        self._done = True
        reward = 1.0 if action.action_type == ActionType.CHECK else -1.0
        obs = _StubObs(is_terminal=True, legal_mask=tuple(False for _ in range(NUM_ACTIONS)))
        return obs, reward, True, {}


class _StubScenarioGen:
    def generate(self):
        return object()  # _BanditEnv doesn't inspect the scenario


class _IdentityAdapter:
    def adapt(self, obs):
        return obs


# ---------------------------------------------------------------------------
# Stub actor-critic: state-independent logits + value
# ---------------------------------------------------------------------------

class _StubActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(NUM_ACTIONS))
        self.value_param = nn.Parameter(torch.zeros(1))

    def _masked_logits(self, legal_mask: Tuple[bool, ...]) -> torch.Tensor:
        mask_t = torch.tensor(legal_mask, dtype=torch.bool)
        return self.logits.masked_fill(~mask_t, -1.0e9)

    def act(self, obs, legal_mask, deterministic: bool = False) -> _StubActionResult:
        masked = self._masked_logits(legal_mask)
        dist = Categorical(logits=masked)
        action = torch.argmax(masked) if deterministic else dist.sample()
        return _StubActionResult(
            action_type=int(action.item()),
            log_prob=dist.log_prob(action),
            value=self.value_param.squeeze(0),
            entropy=dist.entropy(),
        )

    def evaluate_actions(self, obs_batch: List, legal_masks: List[Tuple[bool, ...]], actions: List[int]):
        batch_size = len(obs_batch)
        mask_t = torch.tensor(legal_masks, dtype=torch.bool)  # (batch, NUM_ACTIONS)
        logits = self.logits.unsqueeze(0).expand(batch_size, -1)
        masked_logits = logits.masked_fill(~mask_t, -1.0e9)

        dist = Categorical(logits=masked_logits)
        actions_t = torch.tensor(actions, dtype=torch.long)

        log_probs = dist.log_prob(actions_t)
        values = self.value_param.expand(batch_size)
        entropy = dist.entropy()

        return log_probs, values, entropy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trainer_and_parts():
    torch.manual_seed(0)
    actor_critic = _StubActorCritic()
    adapter = _IdentityAdapter()
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=0.1)
    trainer = PPOTrainer(
        actor_critic=actor_critic,
        adapter=adapter,
        optimizer=optimizer,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        gamma=1.0,
        gae_lambda=0.95,
    )
    env = _BanditEnv()
    scenario_gen = _StubScenarioGen()
    return trainer, actor_critic, env, scenario_gen


# ---------------------------------------------------------------------------
# collect_rollouts()
# ---------------------------------------------------------------------------

def test_collect_rollouts_produces_one_transition_per_episode(trainer_and_parts):
    trainer, _actor_critic, env, scenario_gen = trainer_and_parts
    buffer = trainer.collect_rollouts(env, scenario_gen, num_episodes=10)

    assert len(buffer) == 10
    assert len(buffer.advantages) == 10
    assert len(buffer.returns) == 10


def test_collect_rollouts_rewards_are_plus_or_minus_one(trainer_and_parts):
    trainer, _actor_critic, env, scenario_gen = trainer_and_parts
    buffer = trainer.collect_rollouts(env, scenario_gen, num_episodes=20)

    rewards = {t.reward for t in buffer._transitions}
    assert rewards <= {1.0, -1.0}


def test_collect_rollouts_appends_into_existing_buffer(trainer_and_parts):
    trainer, _actor_critic, env, scenario_gen = trainer_and_parts
    buffer = RolloutBuffer(gamma=1.0, gae_lambda=0.95)

    trainer.collect_rollouts(env, scenario_gen, num_episodes=3, buffer=buffer)
    assert len(buffer) == 3

    trainer.collect_rollouts(env, scenario_gen, num_episodes=2, buffer=buffer)
    assert len(buffer) == 5


def test_collect_rollouts_rejects_zero_episodes(trainer_and_parts):
    trainer, _actor_critic, env, scenario_gen = trainer_and_parts
    with pytest.raises(ValueError):
        trainer.collect_rollouts(env, scenario_gen, num_episodes=0)


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

def test_update_runs_without_error_and_returns_stats(trainer_and_parts):
    trainer, _actor_critic, env, scenario_gen = trainer_and_parts
    buffer = trainer.collect_rollouts(env, scenario_gen, num_episodes=16)

    stats = trainer.update(buffer, epochs=2, minibatch_size=8)

    for key in ("policy_loss", "value_loss", "entropy", "clip_fraction"):
        assert key in stats
        assert isinstance(stats[key], float)


def test_update_rejects_empty_buffer(trainer_and_parts):
    trainer, _actor_critic, _env, _scenario_gen = trainer_and_parts
    empty_buffer = RolloutBuffer()
    with pytest.raises(ValueError):
        trainer.update(empty_buffer)


def test_update_moves_policy_toward_higher_reward_action(trainer_and_parts):
    """
    CHECK (+1) strictly dominates FOLD (-1) in the bandit env. After
    enough PPO iterations the policy should assign CHECK a higher
    probability than FOLD (the value estimate should also rise toward
    the achievable average reward once mixed experience washes out the
    zero-init).
    """
    trainer, actor_critic, env, scenario_gen = trainer_and_parts

    for _ in range(15):
        buffer = trainer.collect_rollouts(env, scenario_gen, num_episodes=32)
        trainer.update(buffer, epochs=4, minibatch_size=16)
        buffer.clear()

    with torch.no_grad():
        mask = torch.tensor(_legal_mask_fold_check(), dtype=torch.bool)
        masked_logits = actor_critic.logits.masked_fill(~mask, -1.0e9)
        probs = torch.softmax(masked_logits, dim=-1)

    check_prob = probs[ActionType.CHECK].item()
    fold_prob = probs[ActionType.FOLD].item()

    assert check_prob > fold_prob


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def test_train_runs_full_cycle_and_returns_history(trainer_and_parts):
    trainer, _actor_critic, env, scenario_gen = trainer_and_parts

    history = trainer.train(
        env,
        scenario_gen,
        num_iterations=3,
        episodes_per_iteration=8,
        epochs=2,
        minibatch_size=4,
    )

    assert len(history) == 3
    for stats in history:
        assert stats.num_episodes == 8
        assert stats.num_transitions == 8
        assert isinstance(stats.mean_episode_reward, float)


def test_train_mean_episode_reward_is_within_bandit_bounds(trainer_and_parts):
    trainer, _actor_critic, env, scenario_gen = trainer_and_parts

    history = trainer.train(
        env, scenario_gen, num_iterations=2, episodes_per_iteration=10,
        epochs=1, minibatch_size=5,
    )

    for stats in history:
        assert -1.0 <= stats.mean_episode_reward <= 1.0


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_invalid_clip_epsilon_raises():
    actor_critic = _StubActorCritic()
    optimizer = torch.optim.Adam(actor_critic.parameters())
    with pytest.raises(ValueError):
        PPOTrainer(actor_critic, _IdentityAdapter(), optimizer, clip_epsilon=0.0)


def test_invalid_value_coef_raises():
    actor_critic = _StubActorCritic()
    optimizer = torch.optim.Adam(actor_critic.parameters())
    with pytest.raises(ValueError):
        PPOTrainer(actor_critic, _IdentityAdapter(), optimizer, value_coef=-1.0)


def test_invalid_entropy_coef_raises():
    actor_critic = _StubActorCritic()
    optimizer = torch.optim.Adam(actor_critic.parameters())
    with pytest.raises(ValueError):
        PPOTrainer(actor_critic, _IdentityAdapter(), optimizer, entropy_coef=-0.5)
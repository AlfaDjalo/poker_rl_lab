"""
trainers/graph_training_loop.py — Self-play PPO training loop for
envs.graph_scenario_env.GraphScenarioEnv.

ARCHITECTURE CHOICE (restored)
--------------------------------
This module previously trained ONE shared models.policy_value_network.
PokerActorCritic for every seat, with no seat argument anywhere in the
call chain. For push_fold_duo.yaml (2 seats, stack.fixed_dealer_position:
1, so seat 0 is SB / first-to-act and seat 1 is BB every single hand —
see that YAML's own comment) that gave the network zero architectural
pressure to keep "I am SB" and "I am BB" apart: both roles are trained
through literally the same weights, with only PositionFeatures telling
them apart internally. In practice this collapsed to a degenerate
self-play equilibrium (SB push% -> 0, BB continue% -> ~fixed), instead
of the differentiated, hand-strength-dependent policy the OLD
DualSeatActorCritic-based generic_tree_training_loop.py pipeline used to
reach.

This version trains a models.dual_seat_policy.DualSeatActorCritic
instead — same interface generic_tree_training_loop.py already uses
successfully — with a real choice of concrete architecture:

    PolicyVariant.SHARED_TRUNK_TWO_HEADS
        One shared RepresentationModel trunk, two SEPARATE
        PolicyValueNetwork heads (seat 0 / seat 1). Recommended default
        — gives each seat's role its own policy/value head while still
        sharing the expensive card/board/point encoders.
    PolicyVariant.SHARED_SINGLE_HEAD
        One PokerActorCritic used identically for both seats (the OLD
        behaviour this file used to hardcode, now just one selectable
        option instead of the only option) — kept for A/B comparison,
        not recommended given the observed collapse.

Physical seat, not "role"
--------------------------------
Earlier versions of collect_graph_rollouts() carefully computed a
per-hand "role" (chronological acting order) because GraphScenarioEnv
can randomize dealer_position hand-to-hand for a genuinely N-seat
scenario, so physical seat isn't always a stable proxy for "the SB-
shaped decision." A DualSeatActorCritic's two heads are keyed by
PHYSICAL seat index (0 or 1) though — seat_parameters(0) trains
"head_sb", seat_parameters(1) trains "head_bb" — so using it correctly
requires physical seat to BE stable across hands, i.e.
config.stack.fixed_dealer_position must be set. build_graph_experiment()
below raises loudly at construction time if you ask for a
DualSeatActorCritic against a config with no fixed dealer position,
rather than silently mixing SB-shaped and BB-shaped decisions into both
heads. push_fold_duo.yaml already sets fixed_dealer_position: 1, so
nothing changes for that scenario.

Per-seat rollout buffers and per-seat PPO updates
------------------------------------------------------
collect_graph_rollouts() now groups each tick's in-flight decisions by
PHYSICAL seat (env.state.current_player, equivalently obs.seat) and
issues one actor_critic.act_batch(seat, ...) call per seat per tick —
mirrors generic_tree_training_loop.collect_generic_tree_rollouts()
exactly. Returns Dict[int, RolloutBuffer] keyed by physical seat index.
GraphTrainingLoop.train() runs one _ppo_update_seat() call per seat,
each against actor_critic.seat_parameters(seat) and that seat's own
optimizer — exactly duo_training_loop.py's SIMULTANEOUS-mode pattern,
reused (not reimplemented) here.

Everything else (lockstep env cloning, Monte Carlo showdown resampling)
is UNCHANGED from the previous version — see the preserved docstring
sections below.

Equity-based showdown rewards (NEW)
------------------------------------
When the scenario is a heads-up hold'em showdown, collect_graph_rollouts()
no longer replays the engine num_showdown_samples times: it uses
trainers.showdown_equity.HoldemEquityRewards (C++ matchup counts + forced-
score ShowdownResolver). Unsupported scenarios fall back to the replay
described below.

Monte Carlo showdown averaging (legacy replay path)
------------------------------------
experiments/generic_tree_experiment.py's old pipeline never resolved a
showdown from a single random board: GenericTreeEnv averaged
`num_showdown_samples` (default 30) INDEPENDENT board runouts into one
low-variance, near-exact-equity reward whenever the board was still
incomplete at terminal time (exactly push_fold_duo's shape — no board
is ever dealt before both seats commit). `collect_graph_rollouts` takes
`num_showdown_samples` (default 30) and, whenever a hand's FINAL
decision resolves directly to a showdown with no further decisions
pending, replays the board-dealing + showdown portion
`num_showdown_samples - 1` MORE times from a snapshot taken immediately
before that final decision was submitted, and averages every sample's
reward. Set `num_showdown_samples=1` to disable this entirely.

This assumes the decision being resampled is followed ONLY by AUTO
nodes (dealing + showdown) with no further DECISION nodes before the
graph terminates — true for push_fold_duo and any "last street's
betting resolves straight to showdown" shape. `_resample_showdown_rewards`
raises loudly (RuntimeError) rather than silently producing a wrong
reward if that assumption is ever violated.

Cloning shares the compiled graph, not just config
--------------------------------------------------------
`_clone_env` builds each per-hand env WITHOUT re-running
`GraphScenarioEnv.__init__` (which recompiles the variant YAML into a
GameGraph and re-validates scope against it). Instead it shares the
already-compiled `graph`/`rules`/`game_def`/`num_streets`/
`_showdown_resolver` directly and only gives each clone its own RNG and
its own per-hand mutable `_engine`/`_pending_pick`/etc.

Dependencies
------------
    PyTorch. envs.graph_scenario_env (GraphScenarioEnv, GraphDecisionObs),
    representation.graph_observation_adapter.GraphObservationAdapter,
    models.dual_seat_policy.DualSeatActorCritic,
    models.shared_trunk_two_heads_policy.SharedTrunkTwoHeadsActorCritic,
    models.shared_single_head_policy.SharedSingleHeadActorCritic,
    models.policy_value_network (PokerActorCritic, PolicyValueNetwork),
    models.actor_critic_factory (ActorCriticConfig, build_actor_critic),
    trainers.rollout_buffer.RolloutBuffer,
    scenarios.graph_scenario_config.GraphScenarioConfig.
"""

from __future__ import annotations

import copy
import random
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from poker_engine.graph.graph_engine import GraphEngine
from poker_engine.graph.core_types import DecisionRequest, DecisionResponse

from poker_rl_lab.envs.graph_scenario_env import (
    GraphDecisionObs,
    GraphScenarioEnv,
)
from poker_rl_lab.representation.graph_observation_adapter import GraphObservationAdapter
from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.models.shared_trunk_two_heads_policy import SharedTrunkTwoHeadsActorCritic
from poker_rl_lab.models.shared_single_head_policy import SharedSingleHeadActorCritic
from poker_rl_lab.models.independent_seat_policy import IndependentSeatActorCritic
from poker_rl_lab.models.policy_value_network import PokerActorCritic, PolicyValueNetwork
from poker_rl_lab.models.actor_critic_factory import ActorCriticConfig, build_actor_critic
from poker_rl_lab.trainers.rollout_buffer import RolloutBuffer
from poker_rl_lab.trainers.showdown_equity import HoldemEquityRewards
from poker_rl_lab.scenarios.graph_scenario_config import GraphScenarioConfig
from poker_rl_lab.config.hand_encoder_config import load_default_hand_encoder


def _infer_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Architecture choice
# ---------------------------------------------------------------------------

class PolicyVariant(Enum):
    """Which concrete DualSeatActorCritic implementation to build.
    SHARED_TRUNK_TWO_HEADS / SHARED_SINGLE_HEAD match
    experiments/generic_tree_experiment.py's PolicyVariant naming
    exactly, so the two pipelines' CLI flags/config read the same way.

    SEPARATE_NETWORKS (NEW) — two fully independent PokerActorCritic
    networks, no shared trunk at all (models.independent_seat_policy.
    IndependentSeatActorCritic). Recommended when one seat's training
    data is much sparser/burstier than the other's (push-fold's SB-
    gates-BB structure: BB only ever sees data on hands where SB
    shoved) — a shared trunk lets an oversized, noisy gradient from the
    sparse seat corrupt the other seat's representation; see
    independent_seat_policy.py's own docstring for the full failure
    mode this was built to eliminate.
    """
    SHARED_TRUNK_TWO_HEADS = "shared_trunk_two_heads"
    SHARED_SINGLE_HEAD = "shared_single_head"
    SEPARATE_NETWORKS = "separate_networks"


def build_dual_seat_actor_critic_for_graph(
    variant: PolicyVariant,
    action_dim: int,
    hand_encoder: Optional[BaseHandEncoder] = None,
    actor_critic_config: Optional[ActorCriticConfig] = None,
) -> DualSeatActorCritic:
    """
    Build a fresh (untrained unless hand_encoder carries pretrained
    weights) DualSeatActorCritic sized for a graph scenario's own
    action space (`action_dim` — pass config.max_action_options; see
    GraphScenarioConfig.max_action_options's own docstring for why this
    is generally NOT actions.abstract_action.NUM_ACTIONS).

    Mirrors experiments/generic_tree_experiment.py's
    build_dual_seat_actor_critic() SHARED_TRUNK_TWO_HEADS /
    SHARED_SINGLE_HEAD branching exactly, parameterised by an explicit
    action_dim instead of assuming the old fixed 9-value abstract
    action space.
    """
    cfg = actor_critic_config or ActorCriticConfig(action_dim=action_dim)
    if cfg.action_dim != action_dim:
        raise ValueError(
            f"actor_critic_config.action_dim ({cfg.action_dim}) must "
            f"equal action_dim ({action_dim}) — GraphScenarioEnv always "
            f"pads legal_mask to the scenario's own max_action_options."
        )

    if hand_encoder is None:
        hand_encoder = load_default_hand_encoder()

    if variant == PolicyVariant.SEPARATE_NETWORKS:
        # Two INDEPENDENT calls to build_actor_critic(), each producing
        # its own trainable RepresentationModel + PolicyValueNetwork.
        # Sharing `hand_encoder` across both calls is safe and free —
        # it's a frozen, non-nn.Module, TF-backed base (see
        # representation_model.py's own "stored, not registered" note)
        # — only the per-call trainable adapter layers end up
        # independent, which is the whole point. See
        # independent_seat_policy.py's module docstring.
        net_sb = build_actor_critic(hand_encoder=hand_encoder, config=cfg)
        net_bb = build_actor_critic(hand_encoder=hand_encoder, config=cfg)
        return IndependentSeatActorCritic(net_sb, net_bb)

    inner = build_actor_critic(hand_encoder=hand_encoder, config=cfg)

    if variant == PolicyVariant.SHARED_SINGLE_HEAD:
        return SharedSingleHeadActorCritic(inner)

    if variant == PolicyVariant.SHARED_TRUNK_TWO_HEADS:
        representation_model = inner.representation_model
        head_a = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            action_dim=action_dim,
            hidden_dim=cfg.policy_hidden_dim,
            num_shared_layers=cfg.policy_num_shared_layers,
        )
        head_b = PolicyValueNetwork(
            latent_dim=representation_model.output_dim,
            action_dim=action_dim,
            hidden_dim=cfg.policy_hidden_dim,
            num_shared_layers=cfg.policy_num_shared_layers,
        )
        return SharedTrunkTwoHeadsActorCritic(representation_model, head_a, head_b)

    raise ValueError(f"Unknown PolicyVariant {variant!r}")


# ---------------------------------------------------------------------------
# Env cloning (one independent GraphScenarioEnv per in-flight hand)
# ---------------------------------------------------------------------------

def _clone_env(template: GraphScenarioEnv, seed: Optional[int] = None) -> GraphScenarioEnv:
    clone = GraphScenarioEnv.__new__(GraphScenarioEnv)
    clone.config = template.config
    clone.game_def = template.game_def
    clone.rules = template.rules
    clone.graph = template.graph
    clone.num_streets = template.num_streets
    clone.transactional = getattr(template, "transactional", True)
    clone._showdown_resolver = template._showdown_resolver
    clone._rng = random.Random(seed)
    clone._engine = None
    clone._pending_pick = None
    clone._current_options = ()
    clone._starting_stacks = ()
    clone._done = True
    return clone


# ---------------------------------------------------------------------------
# Monte Carlo showdown averaging — see module docstring
# ---------------------------------------------------------------------------

def _resample_showdown_rewards(
    env: GraphScenarioEnv,
    snapshot_state,
    snapshot_deck,
    request: DecisionRequest,
    response: DecisionResponse,
    first_sample_rewards: Dict[int, float],
    num_samples: int,
) -> Dict[int, float]:
    num_seats = len(first_sample_rewards)
    totals = dict(first_sample_rewards)

    for _ in range(num_samples - 1):
        restored_state = copy.deepcopy(snapshot_state)
        restored_deck = copy.deepcopy(snapshot_deck)
        restored_deck.reshuffle_remaining()

        replay_engine = GraphEngine(
            env.graph, restored_state, restored_deck, showdown_resolver=env._showdown_resolver
        )
        replay_engine._pending_request = request
        replay_engine.submit_decision(response)

        if not replay_engine.is_complete:
            raise RuntimeError(
                "Monte Carlo showdown resampling assumed the decision "
                "being resampled leads directly to a terminal state with "
                "no further decisions pending -- that assumption was "
                "violated. Pass num_showdown_samples=1 for scenarios "
                "where a decision can be followed by MORE decisions on a "
                "later street before showdown is reached."
            )

        final = [p.stack for p in replay_engine.state.players]
        for s in range(num_seats):
            totals[s] += final[s] - env._starting_stacks[s]

    return {s: totals[s] / num_samples for s in range(num_seats)}


def auto_reward_scale(config: GraphScenarioConfig) -> float:
    """
    Default reward divisor: the chips each seat has at the START of the
    scenario (its maximum possible loss), so scaled rewards land roughly
    in [-1, +1]. Raw chip rewards (e.g. +-150 on a 100-chip pot river)
    made the value loss ~1e4 and swamped PPO's value/policy balance.
    """
    sc = config.stack
    bb = sc.big_blind_size
    if config.start is not None:
        chips = (config.start.stack_behind_bb * bb
                 + config.start.pot_bb * bb / config.num_seats)
    elif sc.mode == "fixed_bb":
        chips = sc.stack_bb * bb
    else:
        chips = sc.max_stack_bb * bb
    return max(float(chips), 1.0)


# ---------------------------------------------------------------------------
# Rollout collection (BATCHED, grouped by PHYSICAL seat)
# ---------------------------------------------------------------------------

def collect_graph_rollouts(
    env: GraphScenarioEnv,
    actor_critic: DualSeatActorCritic,
    adapter: GraphObservationAdapter,
    num_episodes: int,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    num_showdown_samples: int = 30,
    opponent_override: Optional[Dict[int, DualSeatActorCritic]] = None,
    equity: Optional["HoldemEquityRewards"] = None,
    reward_scale: float = 1.0,
) -> "tuple[Dict[int, RolloutBuffer], List[Dict[int, float]]]":
    """
    Play `num_episodes` hands IN LOCKSTEP, filing each decision into a
    RolloutBuffer keyed by PHYSICAL seat index (0, 1, ...) — see module
    docstring's "Physical seat, not role" section for why this requires
    env.config.stack.fixed_dealer_position to be set.

    Parameters
    ----------
    opponent_override : Dict[int, DualSeatActorCritic], optional
        If seat `s` has an entry here, that seat's actions this call are
        chosen from the override policy instead of `actor_critic`.
    reward_scale : float
        Rewards are DIVIDED by this before entering the RolloutBuffers
        (PPO sees ~[-1, 1] values). reward_log keeps raw chip rewards.
    equity : HoldemEquityRewards, optional
        Low-variance showdown rewards from hand equity (see
        trainers/showdown_equity.py). Used instead of replay-based Monte
        Carlo resampling whenever it is supported for this scenario.

    Returns
    -------
    (buffers, reward_log)
    """
    if num_episodes < 1:
        raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")

    opponent_override = opponent_override or {}
    num_seats = env.config.num_seats
    use_equity = equity is not None and equity.supported

    def _policy_for(seat: int) -> DualSeatActorCritic:
        return opponent_override.get(seat, actor_critic)

    buffers: Dict[int, RolloutBuffer] = {
        seat: RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda) for seat in range(num_seats)
    }

    envs: List[GraphScenarioEnv] = [
        _clone_env(env, seed=env._rng.randrange(2 ** 31)) for _ in range(num_episodes)
    ]
    adapters: List[GraphObservationAdapter] = [GraphObservationAdapter(e) for e in envs]

    decisions: List[List[dict]] = [[] for _ in range(num_episodes)]
    obs_list: List[Optional[GraphDecisionObs]] = [None] * num_episodes
    final_rewards: List[Optional[Dict[int, float]]] = [None] * num_episodes

    for i in range(num_episodes):
        obs_list[i] = envs[i].reset()

    active = list(range(num_episodes))

    while active:
        groups: Dict[int, List[int]] = {}
        for i in active:
            groups.setdefault(obs_list[i].seat, []).append(i)

        newly_done: List[int] = []

        for seat, idxs in groups.items():
            policy = _policy_for(seat)
            rep_inputs = [adapters[i].adapt(obs_list[i]) for i in idxs]
            masks = [obs_list[i].legal_mask for i in idxs]

            with torch.no_grad():
                results = policy.act_batch(seat, rep_inputs, masks, deterministic=False)

            for rep_input, mask, i, result in zip(rep_inputs, masks, idxs, results):
                obs = obs_list[i]
                decisions[i].append({
                    "seat": obs.seat,
                    "rep": rep_input,
                    "mask": mask,
                    "action": result.action_type,
                    "log_prob": float(result.log_prob.item()),
                    "value": float(result.value.item()),
                })

                resample = (
                    num_showdown_samples > 1
                    and obs.domain in ("BETTING", "BOOLEAN", "CHOICE")
                )

                board_pre = 0
                if resample:
                    if use_equity:
                        # Cheap: just the board cards known BEFORE this
                        # decision (cards dealt after it get re-sampled).
                        board_pre = equity.board_mask(envs[i].state)
                    else:
                        # Legacy replay path: full state/deck snapshot.
                        pre_request = envs[i]._engine.pending_request
                        pre_response = envs[i]._finalize_single_shot(pre_request, result.action_type)
                        snap_state = copy.deepcopy(envs[i]._engine.state)
                        snap_deck = copy.deepcopy(envs[i]._engine.deck)

                step = envs[i].step(result.action_type)

                if step.done:
                    if resample and step.info.get("outcome") == "showdown":
                        if use_equity:
                            eq_rewards = equity.rewards(envs[i], board_pre)
                            final_rewards[i] = eq_rewards if eq_rewards is not None else step.rewards
                        else:
                            final_rewards[i] = _resample_showdown_rewards(
                                envs[i], snap_state, snap_deck, pre_request, pre_response,
                                first_sample_rewards=step.rewards, num_samples=num_showdown_samples,
                            )
                    else:
                        final_rewards[i] = step.rewards
                    newly_done.append(i)
                else:
                    obs_list[i] = step.obs

        for i in newly_done:
            active.remove(i)

    reward_log: List[Dict[int, float]] = []
    for i in range(num_episodes):
        rewards = final_rewards[i]
        assert rewards is not None, f"hand {i} never reached a terminal step"
        reward_log.append(dict(rewards))

        last_index_for_seat: Dict[int, int] = {}
        for idx, d in enumerate(decisions[i]):
            last_index_for_seat[d["seat"]] = idx

        acted_seats = set(last_index_for_seat.keys())
        for s in acted_seats:
            buffers[s].start_episode()

        for idx, d in enumerate(decisions[i]):
            seat = d["seat"]
            is_terminal_for_seat = (idx == last_index_for_seat[seat])
            buffers[seat].add(
                obs=d["rep"], legal_mask=d["mask"], action=d["action"],
                log_prob=d["log_prob"], value=d["value"],
                reward=(rewards[seat] / reward_scale if is_terminal_for_seat else 0.0),
                done=is_terminal_for_seat,
            )

        for s in acted_seats:
            buffers[s].finish_episode()

    return buffers, reward_log


# ---------------------------------------------------------------------------
# Per-seat PPO update (reused shape from trainers/duo_training_loop.py's
# _ppo_update_seat — kept local, since that module's version is scoped
# to PushFoldDuoEnv-flavoured imports we don't want here)
# ---------------------------------------------------------------------------

def _ppo_update_seat(
    actor_critic: DualSeatActorCritic,
    seat: int,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    minibatch_size: int,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: Optional[float],
) -> Dict[str, float]:
    if len(buffer) == 0:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}

    batch = buffer.get()
    obs: List = batch["obs"]
    legal_masks: List = batch["legal_masks"]
    actions: torch.Tensor = batch["actions"]

    device = _infer_device(actor_critic)
    old_log_probs: torch.Tensor = batch["old_log_probs"].to(device)
    advantages: torch.Tensor = batch["advantages"].to(device)
    returns: torch.Tensor = batch["returns"].to(device)

    n = len(obs)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0}
    num_minibatches = 0
    indices = list(range(n))

    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            mb_idx = indices[start:start + minibatch_size]

            mb_obs = [obs[i] for i in mb_idx]
            mb_masks = [legal_masks[i] for i in mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            new_log_probs, values, entropy = actor_critic.evaluate_actions(
                seat, mb_obs, mb_masks, mb_actions.tolist()
            )

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, mb_returns)
            entropy_mean = entropy.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    actor_critic.seat_parameters(seat), max_grad_norm, foreach=True
                )
            optimizer.step()

            with torch.no_grad():
                clip_fraction = (torch.abs(ratio - 1.0) > clip_epsilon).float().mean()

            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy_mean.item()
            totals["clip_fraction"] += clip_fraction.item()
            num_minibatches += 1

    return {k: v / num_minibatches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Per-iteration diagnostics
# ---------------------------------------------------------------------------

@dataclass
class GraphIterationStats:
    iteration: int
    mean_reward_by_seat: Dict[int, float] = field(default_factory=dict)
    policy_loss: Dict[int, float] = field(default_factory=dict)
    value_loss: Dict[int, float] = field(default_factory=dict)
    entropy: Dict[int, float] = field(default_factory=dict)
    clip_fraction: Dict[int, float] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class GraphTrainingLoop:
    """
    Orchestrates collect -> update cycles for a DualSeatActorCritic over
    a GraphScenarioEnv. One PPO update PER PHYSICAL SEAT, each against
    that seat's own rollout buffer and own optimizer
    (actor_critic.seat_parameters(seat)) — see module docstring.

    Parameters
    ----------
    env : GraphScenarioEnv
        Must have config.stack.fixed_dealer_position set (checked by
        build_graph_experiment() at construction time) — see module
        docstring's "Physical seat, not role" section.
    actor_critic : DualSeatActorCritic
    adapter : representation.graph_observation_adapter.GraphObservationAdapter
    optimizers : Dict[int, torch.optim.Optimizer]
        One entry per seat index in [0, env.config.num_seats), each
        pre-scoped to actor_critic.seat_parameters(that seat).
    """

    def __init__(
        self,
        env: GraphScenarioEnv,
        actor_critic: DualSeatActorCritic,
        adapter: GraphObservationAdapter,
        optimizers: Dict[int, torch.optim.Optimizer],
        variant: Optional["PolicyVariant"] = None,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        num_showdown_samples: int = 30,
        equity_samples: int = 100,
        reward_scale: Optional[float] = None,
    ) -> None:
        self.env = env
        self.actor_critic = actor_critic
        self.adapter = adapter
        self.variant = variant
        """Which PolicyVariant `actor_critic` was built as, if known —
        set by build_graph_experiment(). Threaded into save_checkpoint()
        so a saved checkpoint can self-describe its own architecture;
        see load_dual_seat_actor_critic_for_graph()'s docstring for why
        this matters (no more passing --variant by hand when loading)."""

        num_seats = env.config.num_seats
        missing = set(range(num_seats)) - set(optimizers)
        if missing:
            raise ValueError(
                f"optimizers is missing entries for seat indices "
                f"{sorted(missing)} (config {env.config.name!r} declares "
                f"{num_seats} seats)"
            )
        self.optimizers = optimizers

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_showdown_samples = num_showdown_samples

        # Rewards fed to PPO are chips / reward_scale. None = auto (the
        # chips each seat starts the scenario with); pass 1.0 to disable.
        self.reward_scale = (
            auto_reward_scale(env.config) if reward_scale is None else float(reward_scale)
        )
        if self.reward_scale <= 0:
            raise ValueError(f"reward_scale must be > 0, got {self.reward_scale}")
        print(f"[GraphTrainingLoop] reward scale: rewards / {self.reward_scale:g} "
              f"(logged rewards stay in chips)")

        self.equity = None
        if num_showdown_samples > 1:
            self.equity = HoldemEquityRewards(env.rules, n_samples=equity_samples, seed=0)
            if self.equity.supported:
                print(f"[GraphTrainingLoop] showdown rewards: equity path "
                      f"({equity_samples} MC runouts when >=3 board cards missing)")
            else:
                print(f"[GraphTrainingLoop] showdown rewards: legacy replay x{num_showdown_samples} "
                      f"(equity path unavailable: {self.equity.reason})")

    def train(
        self,
        num_iterations: int,
        episodes_per_iteration: int = 256,
        epochs: int = 4,
        minibatch_size: int = 64,
        log_every: int = 10,
        verbose: bool = True,
    ) -> List[GraphIterationStats]:
        history: List[GraphIterationStats] = []
        start_time = time.time()
        num_seats = self.env.config.num_seats

        for i in range(num_iterations):
            buffers, reward_log = collect_graph_rollouts(
                self.env, self.actor_critic, self.adapter, episodes_per_iteration,
                gamma=self.gamma, gae_lambda=self.gae_lambda,
                num_showdown_samples=self.num_showdown_samples,
                equity=self.equity,
                reward_scale=self.reward_scale,
            )

            policy_loss, value_loss, entropy, clip_fraction = {}, {}, {}, {}
            for seat in range(num_seats):
                loss = _ppo_update_seat(
                    self.actor_critic, seat, buffers[seat], self.optimizers[seat],
                    epochs, minibatch_size, self.clip_epsilon, self.value_coef,
                    self.entropy_coef, self.max_grad_norm,
                )
                policy_loss[seat] = loss["policy_loss"]
                value_loss[seat] = loss["value_loss"]
                entropy[seat] = loss["entropy"]
                clip_fraction[seat] = loss["clip_fraction"]

            mean_reward_by_seat = {
                s: statistics.mean(entry.get(s, 0.0) for entry in reward_log)
                for s in range(num_seats)
            }

            elapsed = time.time() - start_time
            stats = GraphIterationStats(
                iteration=i,
                mean_reward_by_seat=mean_reward_by_seat,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=entropy,
                clip_fraction=clip_fraction,
                elapsed_seconds=elapsed,
            )
            history.append(stats)

            iteration_num = i + 1
            if verbose and log_every > 0 and (
                iteration_num % log_every == 0 or iteration_num == num_iterations
            ):
                rewards_str = " ".join(
                    f"seat{s}={v:+.3f}" for s, v in mean_reward_by_seat.items()
                )
                loss_str = " ".join(
                    f"pl{s}={policy_loss[s]:.4f}/vl{s}={value_loss[s]:.4f}"
                    for s in range(num_seats)
                )
                print(
                    f"[GraphTrainingLoop:{self.env.config.name}] "
                    f"{iteration_num}/{num_iterations} iterations completed "
                    f"({elapsed:.1f}s elapsed, {elapsed / iteration_num:.2f}s/iter)  "
                    f"{rewards_str}  {loss_str}"
                )

        if verbose:
            total_time = time.time() - start_time
            avg = total_time / num_iterations if num_iterations else 0.0
            print(
                f"[GraphTrainingLoop:{self.env.config.name}] Training complete: "
                f"{num_iterations} iterations in {total_time:.1f}s ({avg:.2f}s/iter avg)"
            )

        if verbose and self.equity is not None and self.equity.supported:
            print(f"[GraphTrainingLoop] equity showdowns: {self.equity.used} used, "
                  f"{self.equity.fallbacks} fell back to a single sample")

        return history


# ---------------------------------------------------------------------------
# Experiment builder (config -> env/actor_critic/optimizers/loop, wired)
# ---------------------------------------------------------------------------

def build_graph_experiment(
    config: GraphScenarioConfig,
    variant: PolicyVariant = PolicyVariant.SHARED_TRUNK_TWO_HEADS,
    hand_encoder=None,
    actor_critic_config: Optional[ActorCriticConfig] = None,
    learning_rate: float = 3e-4,
    learning_rates: Optional[Dict[int, float]] = None,
    device: str = "cpu",
    scoring_engine=None,
    seed: Optional[int] = None,
    **loop_kwargs,
):
    """
    Wire a GraphScenarioConfig into a ready-to-train bundle: env,
    actor_critic (a DualSeatActorCritic of the requested `variant`),
    adapter, per-seat optimizers, GraphTrainingLoop.

    Parameters
    ----------
    config : GraphScenarioConfig
        Must have config.stack.fixed_dealer_position set — see module
        docstring's "Physical seat, not role" section for why a
        DualSeatActorCritic requires a stable physical-seat-to-role
        mapping. Raises ValueError at construction time otherwise.
    variant : PolicyVariant
        SHARED_TRUNK_TWO_HEADS, SHARED_SINGLE_HEAD, or SEPARATE_NETWORKS
        (recommended for push_fold_duo-shaped scenarios — see
        PolicyVariant's own docstring for why a shared trunk is
        vulnerable to exactly the SB-gates-BB data sparsity this
        scenario has).
    hand_encoder : optional
        Forwarded to build_actor_critic(). Defaults to
        config.hand_encoder_config.load_default_hand_encoder().
    actor_critic_config : ActorCriticConfig, optional
        Defaults to ActorCriticConfig(action_dim=config.max_action_options).
        If given explicitly, its action_dim MUST equal
        config.max_action_options.
    learning_rate : float
        Default per-seat learning rate.
    learning_rates : Dict[int, float], optional
        Per-seat-index override; a seat missing here uses `learning_rate`.
    device : str
        PyTorch device — moved BEFORE optimizers are constructed.
    scoring_engine, seed : forwarded to GraphScenarioEnv's constructor.
    **loop_kwargs : forwarded to GraphTrainingLoop's constructor
        (clip_epsilon, value_coef, entropy_coef, max_grad_norm, gamma,
        gae_lambda, num_showdown_samples).

    Returns
    -------
    (env, actor_critic, adapter, loop)
    """
    if config.stack.fixed_dealer_position is None:
        raise ValueError(
            f"scenario {config.name!r} has no stack.fixed_dealer_position "
            f"set — a DualSeatActorCritic's two seat-indexed heads require "
            f"a STABLE physical-seat-to-role mapping across hands (seat 0 "
            f"always plays the same role, e.g. SB), which only holds when "
            f"the dealer position is fixed. Either set "
            f"stack.fixed_dealer_position in this scenario's YAML (see "
            f"push_fold_duo.yaml for the pattern), or use a training setup "
            f"that doesn't assume per-seat heads for a rotating-dealer "
            f"scenario."
        )

    # transactional=False: skip GraphEngine's per-call rollback deepcopies
    # (a failing step abandons the episode anyway during training).
    env = GraphScenarioEnv(config, scoring_engine=scoring_engine, seed=seed, transactional=False)
    adapter = GraphObservationAdapter(env)

    if hand_encoder is None:
        hand_encoder = load_default_hand_encoder()

    ac_cfg = actor_critic_config or ActorCriticConfig(action_dim=config.max_action_options)
    if ac_cfg.action_dim != config.max_action_options:
        raise ValueError(
            f"actor_critic_config.action_dim ({ac_cfg.action_dim}) must "
            f"equal config.max_action_options ({config.max_action_options}) "
            f"— GraphScenarioEnv always pads legal_mask to the latter."
        )

    actor_critic = build_dual_seat_actor_critic_for_graph(
        variant, action_dim=config.max_action_options,
        hand_encoder=hand_encoder, actor_critic_config=ac_cfg,
    )
    actor_critic = actor_critic.to(device)

    learning_rates = learning_rates or {}
    optimizers: Dict[int, torch.optim.Optimizer] = {
        seat: torch.optim.Adam(
            actor_critic.seat_parameters(seat),
            lr=learning_rates.get(seat, learning_rate),
            foreach=True,   # many tiny tensors: one fused-loop update beats a Python loop
        )
        for seat in range(config.num_seats)
    }

    loop = GraphTrainingLoop(
        env=env, actor_critic=actor_critic, adapter=adapter,
        optimizers=optimizers, variant=variant, **loop_kwargs,
    )
    return env, actor_critic, adapter, loop


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    actor_critic: DualSeatActorCritic,
    optimizers: Dict[int, torch.optim.Optimizer],
    path,
    variant: Optional[PolicyVariant] = None,
    action_dim: Optional[int] = None,
    actor_critic_config: Optional[ActorCriticConfig] = None,
    scenario_name: Optional[str] = None,
) -> None:
    """
    Save actor_critic + every seat's optimizer state to `path`.

    Self-describing checkpoints (for the Backend / any consumer that only
    has the .pt path): besides `variant` and `action_dim`, this embeds the
    full ActorCriticConfig (encoder subset caps such as
    made_hand_max_subsets=60 for Omaha change inference behaviour but not
    parameter shapes, so they could NOT be recovered from the weights) and
    the `scenario_name` it was trained on. See read_checkpoint_metadata()
    and load_dual_seat_actor_critic_for_graph().

    Also embeds `variant` and `action_dim` (when given — GraphTrainingLoop.
    train() calls this with both, via loop.variant / env.config.
    max_action_options) so the checkpoint is self-describing: any
    consumer (push_fold_duo_hand_grids.py, a future eval script) can
    reconstruct the EXACT architecture via
    load_dual_seat_actor_critic_for_graph() below without needing to
    separately know or guess which --variant a given .pt file was
    trained with. Mirrors experiments/push_fold_duo_experiment.py's own
    save_checkpoint() convention.
    """
    import dataclasses
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": variant.value if variant is not None else None,
            "action_dim": action_dim,
            "actor_critic_config": (
                dataclasses.asdict(actor_critic_config) if actor_critic_config is not None else None
            ),
            "scenario_name": scenario_name,
            "actor_critic_state_dict": actor_critic.state_dict(),
            "optimizer_state_dicts": {
                seat: opt.state_dict() for seat, opt in optimizers.items()
            },
        },
        path,
    )


def load_checkpoint(
    actor_critic: DualSeatActorCritic,
    optimizers: Dict[int, torch.optim.Optimizer],
    path,
) -> None:
    """Load weights + optimizer state into an ALREADY-BUILT actor_critic
    (caller must have built it with the matching variant/action_dim —
    e.g. via build_graph_experiment(..., variant=...)). For inference-
    only loading where you don't want to specify the variant by hand,
    use load_dual_seat_actor_critic_for_graph() instead."""
    from pathlib import Path

    checkpoint = torch.load(Path(path), map_location=_infer_device(actor_critic))
    actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    for seat, state_dict in checkpoint["optimizer_state_dicts"].items():
        optimizers[seat].load_state_dict(state_dict)


def read_checkpoint_metadata(path, device: str = "cpu") -> dict:
    """
    Everything a consumer needs to know about a checkpoint WITHOUT building
    a network: {"variant": PolicyVariant|None, "action_dim": int|None,
    "scenario_name": str|None, "actor_critic_config": ActorCriticConfig|None}.
    Fields are None for checkpoints saved before they were recorded.
    """
    import dataclasses
    from pathlib import Path

    ckpt = torch.load(Path(path), map_location=device)
    raw_cfg = ckpt.get("actor_critic_config")
    cfg = None
    if raw_cfg is not None:
        known = {f.name for f in dataclasses.fields(ActorCriticConfig)}
        cfg = ActorCriticConfig(**{k: v for k, v in raw_cfg.items() if k in known})
    variant_value = ckpt.get("variant")
    return {
        "variant": PolicyVariant(variant_value) if variant_value is not None else None,
        "action_dim": ckpt.get("action_dim"),
        "scenario_name": ckpt.get("scenario_name"),
        "actor_critic_config": cfg,
    }


def load_dual_seat_actor_critic_for_graph(
    path,
    hand_encoder=None,
    actor_critic_config: Optional[ActorCriticConfig] = None,
    device: str = "cpu",
) -> DualSeatActorCritic:
    """
    Inference-only checkpoint loader: reads `variant` and `action_dim`
    back out of a checkpoint saved by save_checkpoint() (with those
    values passed — GraphTrainingLoop.train() does this automatically)
    and reconstructs the EXACT matching architecture before loading
    weights, so callers (push_fold_duo_hand_grids.py) never need to
    pass --variant by hand and get it wrong.

    Does NOT build a GraphScenarioEnv, optimizers, or a
    GraphTrainingLoop — this is the minimal path for something that
    only wants to call .act()/.act_batch() against a trained network.

    Raises
    ------
    ValueError
        If the checkpoint predates variant/action_dim being saved
        (both keys will be missing or None) — re-save it once from a
        training run that still has the original GraphScenarioConfig
        in scope (call save_checkpoint(..., variant=..., action_dim=
        config.max_action_options) there), or pass variant/action_dim
        explicitly via build_dual_seat_actor_critic_for_graph() and a
        manual actor_critic.load_state_dict() call instead.
    """
    from pathlib import Path

    checkpoint = torch.load(Path(path), map_location=device)

    variant_value = checkpoint.get("variant")
    action_dim = checkpoint.get("action_dim")
    if variant_value is None or action_dim is None:
        raise ValueError(
            f"{path} is missing 'variant'/'action_dim' metadata — this "
            f"checkpoint predates that being saved. See this function's "
            f"docstring for how to load it anyway (you'll need to know "
            f"which --variant this checkpoint was originally trained "
            f"with and pass it to build_dual_seat_actor_critic_for_graph() "
            f"directly)."
        )

    variant = PolicyVariant(variant_value)

    # Use the encoder settings recorded at training time unless the caller
    # passes an explicit override (older checkpoints have none -> defaults).
    if actor_critic_config is None:
        meta = read_checkpoint_metadata(path, device=device)
        actor_critic_config = meta["actor_critic_config"]

    actor_critic = build_dual_seat_actor_critic_for_graph(
        variant, action_dim=action_dim,
        hand_encoder=hand_encoder, actor_critic_config=actor_critic_config,
    )
    actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
    if device is not None:
        actor_critic = actor_critic.to(device)
    actor_critic.eval()
    return actor_critic


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario

    parser = argparse.ArgumentParser(
        description="Self-play PPO training for a graph-native poker scenario."
    )
    parser.add_argument("--scenario-name", type=str, required=True,
                         help="File stem under graph_scenarios/, e.g. 'push_fold_duo'.")
    parser.add_argument("--scenario-path", type=str, default=None,
                         help="Explicit path to a scenario YAML, overriding --scenario-name's lookup.")
    parser.add_argument(
        "--variant", type=str, default="shared_trunk_two_heads",
        choices=[v.value for v in PolicyVariant],
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--episodes-per-iteration", type=int, default=256,
        help="For a scenario where one seat's decisions gate the other's "
             "(e.g. push_fold_duo: BB only acts on hands where SB shoved), "
             "the gating seat's own shove rate can make the gated seat's "
             "per-iteration buffer very small/noisy at low values. Consider "
             "1024+ if the gated seat's policy/value loss looks unstable "
             "(large swings, or an early collapse to a fixed reward with "
             "pl/vl both -> 0.0000) in the training log.",
    )
    parser.add_argument("--num-showdown-samples", type=int, default=30,
                         help="Independent board runouts averaged per showdown "
                              "when a hand's final decision resolves directly to "
                              "one (e.g. push_fold_duo). 1 disables resampling.")
    parser.add_argument("--reward-scale", type=float, default=None,
                         help="Divide rewards by this before PPO (default: auto = each "
                              "seat's chips at scenario start, so rewards are ~[-1, 1]). "
                              "Use 1 to disable.")
    parser.add_argument("--made-hand-max-subsets", type=int, default=12,
                         help="Cap on candidate 5-card hands the made-hand encoder pools. "
                              "Use 60 for Omaha (C(4,2)*C(5,3) legal hands on a full board).")
    parser.add_argument("--board-max-subsets", type=int, default=12)
    parser.add_argument("--equity-samples", type=int, default=100,
                         help="Monte Carlo runouts per showdown for the equity path "
                              "(only used when >=3 board cards are missing; exact otherwise).")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                         help="Raise this (e.g. 0.03-0.05) if a seat's policy "
                              "collapses to a single action early in training "
                              "(reward pinned at a fixed value, pl/vl both "
                              "-> 0.0000 in the log) -- see this module's "
                              "docstring.")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-out", type=str, default=None)
    parser.add_argument("--checkpoint-in", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = load_graph_scenario(args.scenario_name, path=args.scenario_path)

    ac_cfg = ActorCriticConfig(
        action_dim=config.max_action_options,
        made_hand_max_subsets=args.made_hand_max_subsets,
        board_max_subsets=args.board_max_subsets,
    )

    env, actor_critic, adapter, loop = build_graph_experiment(
        config,
        variant=PolicyVariant(args.variant),
        actor_critic_config=ac_cfg,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        num_showdown_samples=args.num_showdown_samples,
        equity_samples=args.equity_samples,
        reward_scale=args.reward_scale,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        clip_epsilon=args.clip_epsilon,
        max_grad_norm=args.max_grad_norm,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )

    if args.checkpoint_in:
        load_checkpoint(actor_critic, loop.optimizers, args.checkpoint_in)
        print(f"Resumed from checkpoint {args.checkpoint_in}")

    print(f"scenario: {config.name}  engine_variant: {config.engine_variant}  "
          f"num_seats: {config.num_seats}  variant: {args.variant}  "
          f"max_action_options: {config.max_action_options}")

    loop.train(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes_per_iteration,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        log_every=args.log_every,
    )

    if args.checkpoint_out:
        save_checkpoint(
            actor_critic, loop.optimizers, args.checkpoint_out,
            variant=loop.variant, action_dim=config.max_action_options,
            actor_critic_config=ac_cfg, scenario_name=config.name,
        )
        print(f"Saved checkpoint to {args.checkpoint_out}")


if __name__ == "__main__":
    _main()
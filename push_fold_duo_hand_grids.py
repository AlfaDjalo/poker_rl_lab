"""
push_fold_duo_hand_grids.py — SB push-probability and BB call-probability
hand-range grids, graph-engine edition.

DualSeatActorCritic edition (CHANGED)
--------------------------------------
trainers/graph_training_loop.py no longer trains a single shared
models.policy_value_network.PokerActorCritic — it trains a
models.dual_seat_policy.DualSeatActorCritic (SharedTrunkTwoHeadsActorCritic
or SharedSingleHeadActorCritic — see that module's own docstring for why).
This file's action_probs() now takes an explicit `seat` argument and
resolves the (representation_model, head) pair for that seat by duck-
typing on the concrete DualSeatActorCritic subclass, instead of reaching
into a single PokerActorCritic's .representation_model/.policy_value_network
directly (which no longer exist on the object this module loads).

Seat convention
----------------
GraphScenarioEnv.reset() picks a random dealer_position and randomized
stacks (training-time behaviour). For a hand-range grid we need a FIXED
stack depth and a FIXED, known hero hand, evaluated at each seat's own
decision point -- so this module drives the compiled push_fold GameGraph
directly via a fresh GraphEngine per stack depth, instead of using
env.reset()/step().

dealer_position is fixed at seat 1 (matches push_fold_duo.yaml's
stack.fixed_dealer_position: 1 — the SAME setting
trainers/graph_training_loop.py now REQUIRES before it will build a
DualSeatActorCritic at all, since the two heads are keyed by physical
seat index and that only means anything stable if the dealer never
rotates). This makes seat 0 the small blind AND first to act preflop,
and seat 1 the big blind -- verified by asserting
pending_request.player_index == 0 immediately after engine.start()
below.

Action-type semantics (push_fold / AllInOrFoldRules)
------------------------------------------------------
This module does not have visibility into betting_rules.py's
AllInOrFoldRules implementation, only its documented behaviour: FOLD is
always offered; CALL matches bet_to_call without necessarily committing
the whole stack; RAISE is the only way to shove all-in, and is only
offered when stack > to_call. Every probability this module reports is
a SUM over whichever resolved GraphActionOption entries carry the
relevant engine ActionType(s):

    SB "push" probability   = P(action_type == RAISE)   (the shove)
    BB "continue" probability = 1 - P(action_type == FOLD)
                                 (calls OR re-shoves; whichever of
                                 CALL/RAISE the resolver actually offers,
                                 correctly summed either way)

Dependencies
------------
    PyTorch, numpy. matplotlib only inside plot_heatmap.
    poker_engine (graph.graph_engine.GraphEngine, graph.graph_state.
    GraphGameState, state.player_state.PlayerState, cards.deck.Deck,
    actions.action.Action, actions.action_type.ActionType,
    graph.core_types.DecisionDomain/DecisionResponse).
    poker_rl_lab.envs.graph_scenario_env (GraphScenarioEnv,
    GraphDecisionObs), poker_rl_lab.representation.
    graph_observation_adapter.GraphObservationAdapter,
    poker_rl_lab.representation.card_utils (str_to_card_id,
    card_ids_to_mask), poker_rl_lab.models.dual_seat_policy
    (DualSeatActorCritic), poker_rl_lab.models.shared_trunk_two_heads_policy
    (SharedTrunkTwoHeadsActorCritic), poker_rl_lab.models.shared_single_head_policy
    (SharedSingleHeadActorCritic), poker_rl_lab.models.policy_value_network
    (apply_legal_mask), poker_rl_lab.scenarios.graph_scenario_config
    (load_graph_scenario), poker_rl_lab.trainers.graph_training_loop
    (build_graph_experiment, load_checkpoint, PolicyVariant).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from poker_engine.actions.action import Action
from poker_engine.actions.action_type import ActionType as EngineActionType
from poker_engine.cards.deck import Deck
from poker_engine.graph.core_types import DecisionDomain, DecisionResponse
from poker_engine.graph.graph_engine import GraphEngine
from poker_engine.graph.graph_state import GraphGameState
from poker_engine.state.player_state import PlayerState

from poker_rl_lab.envs.graph_scenario_env import GraphDecisionObs, GraphScenarioEnv
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.models.shared_trunk_two_heads_policy import SharedTrunkTwoHeadsActorCritic
from poker_rl_lab.models.shared_single_head_policy import SharedSingleHeadActorCritic
from poker_rl_lab.models.independent_seat_policy import IndependentSeatActorCritic
from poker_rl_lab.models.policy_value_network import apply_legal_mask
from poker_rl_lab.representation.card_utils import card_ids_to_mask, str_to_card_id
from poker_rl_lab.representation.graph_observation_adapter import GraphObservationAdapter
from poker_rl_lab.scenarios.graph_scenario_config import load_graph_scenario

RANKS: Tuple[str, ...] = ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2")
SUITS: Tuple[str, ...] = ("c", "d", "h", "s")

# Fixed per module docstring's "Seat convention" section.
_DEALER_POSITION = 1
_SB_SEAT = 0
_BB_SEAT = 1


# ---------------------------------------------------------------------------
# Engine plumbing: fresh, fixed-stack GraphEngine for one stack depth
# ---------------------------------------------------------------------------

def _build_env(scenario_name: str = "push_fold_duo", scenario_path: str | None = None) -> GraphScenarioEnv:
    config = load_graph_scenario(scenario_name, path=scenario_path)
    return GraphScenarioEnv(config)


def _deal_fixed_stack_engine(env: GraphScenarioEnv, stack_bb: float) -> GraphEngine:
    """
    Build + start() a fresh GraphEngine against env's compiled graph,
    BOTH seats given the same fixed stack_bb (unlike env.reset(), which
    randomizes dealer/stack per config.stack for training). Advances
    only to the first DECISION node (SB's bet_preflop) -- callers own
    submit_decision() from there.
    """
    big_blind = env.config.stack.big_blind_size
    stack = max(big_blind, int(round(stack_bb * big_blind)))

    players = [
        PlayerState(
            stack=stack, hand_mask=0, current_bet=0, total_contribution=0,
            has_folded=False, is_all_in=False, sitting_out=False,
        )
        for _ in range(env.config.num_seats)
    ]
    state = GraphGameState(players, betting_type="no_limit")
    state.node_cards = [None] * env.game_def.node_count
    state.dealer_position = _DEALER_POSITION

    deck = Deck()
    deck.shuffle()

    engine = GraphEngine(env.graph, state, deck, showdown_resolver=env._showdown_resolver)
    env._engine = engine
    env._history = []
    engine.start()

    if engine.pending_request is None or engine.pending_request.player_index != _SB_SEAT:
        raise RuntimeError(
            f"Expected SB (seat {_SB_SEAT}) to act first preflop after "
            f"engine.start(), but pending_request is "
            f"{engine.pending_request!r}. push_fold.yaml's blind posting "
            f"or first_to_act_offset may have changed -- see this "
            f"module's 'Seat convention' docstring section."
        )
    return engine


def _decision_obs(env: GraphScenarioEnv, engine: GraphEngine) -> GraphDecisionObs:
    env._engine = engine
    return env._build_decision_obs(engine.pending_request)


# ---------------------------------------------------------------------------
# DualSeatActorCritic plumbing (NEW)
# ---------------------------------------------------------------------------

def _infer_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _representation_and_head(actor_critic: DualSeatActorCritic, seat: int):
    """
    Resolve the (representation_model, policy_value_head) pair `seat`'s
    decisions actually flow through, duck-typing on the concrete
    DualSeatActorCritic subclass. Needed because DualSeatActorCritic's
    public interface (act/act_batch/evaluate_actions) always returns a
    SAMPLED action, never the raw masked probability vector this
    module's heatmaps need — so we reach one level deeper, the same way
    trainers/graph_training_loop.py's own act_batch() implementations
    do internally.
    """
    if isinstance(actor_critic, SharedTrunkTwoHeadsActorCritic):
        head = actor_critic.head_sb if seat == 0 else actor_critic.head_bb
        return actor_critic.representation_model, head
    if isinstance(actor_critic, SharedSingleHeadActorCritic):
        inner = actor_critic.actor_critic
        return inner.representation_model, inner.policy_value_network
    if isinstance(actor_critic, IndependentSeatActorCritic):
        inner = actor_critic.actor_critic_sb if seat == 0 else actor_critic.actor_critic_bb
        return inner.representation_model, inner.policy_value_network
    raise TypeError(
        f"_representation_and_head doesn't know how to resolve a "
        f"representation_model/head pair for {type(actor_critic).__name__} "
        f"— add a branch here for any new DualSeatActorCritic subclass."
    )


def action_probs(
    actor_critic: DualSeatActorCritic,
    seat: int,
    adapter: GraphObservationAdapter,
    obs: GraphDecisionObs,
) -> torch.Tensor:
    """Full (masked, softmaxed) probability vector for one decision,
    under `seat`'s own head. Shape (config.max_action_options,)."""
    representation_model, head = _representation_and_head(actor_critic, seat)

    rep_input = adapter.adapt(obs)
    device = _infer_device(actor_critic)
    with torch.no_grad():
        latent = representation_model(rep_input)
        logits, _value = head(latent)
        mask_t = torch.as_tensor(obs.legal_mask, dtype=torch.bool, device=device)
        masked_logits = apply_legal_mask(logits, mask_t)
        probs = torch.softmax(masked_logits, dim=-1)
    return probs


def _prob_mass_for(
    probs: torch.Tensor, obs: GraphDecisionObs, action_types: Iterable[EngineActionType]
) -> float:
    """Sum probability mass over every real (non-padding) option whose
    resolved engine_action_type is in `action_types`."""
    wanted = set(action_types)
    total = 0.0
    for i, opt in enumerate(obs.options):
        if opt.engine_action_type in wanted:
            total += float(probs[i].item())
    return total


# ---------------------------------------------------------------------------
# Starting-hand combo enumeration (unchanged)
# ---------------------------------------------------------------------------

def _all_starting_hand_combos(rank1: str, rank2: str, suited: bool) -> List[Tuple[int, int]]:
    combos = []
    if rank1 == rank2:
        for s1, s2 in combinations(SUITS, 2):
            combos.append((str_to_card_id(rank1 + s1), str_to_card_id(rank2 + s2)))
    elif suited:
        for s in SUITS:
            combos.append((str_to_card_id(rank1 + s), str_to_card_id(rank2 + s)))
    else:
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2:
                    combos.append((str_to_card_id(rank1 + s1), str_to_card_id(rank2 + s2)))
    return combos


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_sb_push_grid(
    env: GraphScenarioEnv,
    actor_critic: DualSeatActorCritic,
    adapter: GraphObservationAdapter,
    stack_bb: float,
) -> np.ndarray:
    """13x13 grid of P(SB shoves), at a fixed stack depth. One fresh
    engine for the whole grid -- SB's hand is overwritten per combo and
    requeried; nothing is ever submitted, so the underlying situation
    never advances."""
    engine = _deal_fixed_stack_engine(env, stack_bb)
    obs = _decision_obs(env, engine)

    grid = np.zeros((13, 13), dtype=np.float32)
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            suited = i < j
            combos = _all_starting_hand_combos(r1, r2, suited=suited)
            probs_for_combos = []
            for c1, c2 in combos:
                engine.state.players[_SB_SEAT].hand_mask = card_ids_to_mask([c1, c2])
                probs = action_probs(actor_critic, _SB_SEAT, adapter, obs)
                probs_for_combos.append(_prob_mass_for(probs, obs, (EngineActionType.RAISE,)))
            grid[i, j] = float(np.mean(probs_for_combos))

    return grid


def generate_bb_call_grid(
    env: GraphScenarioEnv,
    actor_critic: DualSeatActorCritic,
    adapter: GraphObservationAdapter,
    stack_bb: float,
) -> np.ndarray:
    """13x13 grid of P(BB continues -- calls or re-shoves) facing an SB
    shove, at a fixed stack depth. One fresh engine per grid: SB's shove
    is submitted ONCE (its amount doesn't depend on either seat's hand),
    then BB's hand is overwritten per combo and requeried."""
    engine = _deal_fixed_stack_engine(env, stack_bb)
    sb_obs = _decision_obs(env, engine)

    shove = next(
        (o for o in sb_obs.options if o.engine_action_type == EngineActionType.RAISE), None
    )
    if shove is None:
        raise RuntimeError(
            f"SB has no legal RAISE (shove) option at stack_bb={stack_bb} -- "
            f"stack may be too small (<= to_call) for this stack depth to "
            f"produce a real push-fold decision."
        )

    env._submit(
        DecisionResponse(
            node_id=sb_obs.node_id,
            domain=DecisionDomain.BETTING,
            player_index=_SB_SEAT,
            value=Action(type=EngineActionType.RAISE, amount=shove.amount),
        )
    )
    if engine.pending_request is None or engine.pending_request.player_index != _BB_SEAT:
        raise RuntimeError(
            f"Expected BB (seat {_BB_SEAT}) to face SB's shove, but "
            f"pending_request is {engine.pending_request!r} (engine may "
            f"have resolved the hand outright, e.g. if BB started with 0 "
            f"chips -- check min_stack_bb in the scenario config)."
        )
    bb_obs = _decision_obs(env, engine)

    grid = np.zeros((13, 13), dtype=np.float32)
    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            suited = i < j
            combos = _all_starting_hand_combos(r1, r2, suited=suited)
            probs_for_combos = []
            for c1, c2 in combos:
                engine.state.players[_BB_SEAT].hand_mask = card_ids_to_mask([c1, c2])
                probs = action_probs(actor_critic, _BB_SEAT, adapter, bb_obs)
                fold_prob = _prob_mass_for(probs, bb_obs, (EngineActionType.FOLD,))
                probs_for_combos.append(1.0 - fold_prob)
            grid[i, j] = float(np.mean(probs_for_combos))

    return grid


def generate_all_grids(
    env: GraphScenarioEnv,
    actor_critic: DualSeatActorCritic,
    adapter: GraphObservationAdapter,
    stack_bb_values: Sequence[float],
) -> Dict[float, Dict[str, np.ndarray]]:
    """Returns {stack_bb: {"sb_push": grid, "bb_call": grid}} for every
    stack depth in `stack_bb_values`."""
    out: Dict[float, Dict[str, np.ndarray]] = {}
    for stack_bb in stack_bb_values:
        out[stack_bb] = {
            "sb_push": generate_sb_push_grid(env, actor_critic, adapter, stack_bb),
            "bb_call": generate_bb_call_grid(env, actor_critic, adapter, stack_bb),
        }
    return out


# ---------------------------------------------------------------------------
# Plotting (unchanged)
# ---------------------------------------------------------------------------

def plot_heatmap(grid: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Action Probability")
    plt.xticks(range(13), RANKS)
    plt.yticks(range(13), RANKS)
    plt.xlabel("Second Card")
    plt.ylabel("First Card")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from poker_rl_lab.trainers.graph_training_loop import load_dual_seat_actor_critic_for_graph

    parser = argparse.ArgumentParser(
        description="Generate SB push / BB call hand-range heatmaps from a "
                    "graph-engine push-fold-duo checkpoint."
    )
    parser.add_argument("--scenario-name", type=str, default="push_fold_duo",
                         help="File stem under graph_scenarios/.")
    parser.add_argument("--scenario-path", type=str, default=None,
                         help="Explicit path to a scenario YAML, overriding --scenario-name's lookup.")
    parser.add_argument("--checkpoint-in", type=str, required=True)
    parser.add_argument(
        "--stack-bb-values", type=str, default="5,10,15,20,25",
        help="Comma-separated stack depths (in big blinds) to grid.",
    )
    parser.add_argument("--output-dir", type=str, default="hand_grids")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config = load_graph_scenario(args.scenario_name, path=args.scenario_path)
    env = GraphScenarioEnv(config)
    # Architecture (variant + action_dim) is read straight out of the
    # checkpoint's own metadata — no --variant flag to remember/get
    # wrong. See load_dual_seat_actor_critic_for_graph()'s docstring.
    actor_critic = load_dual_seat_actor_critic_for_graph(
        args.checkpoint_in, device=args.device,
    )
    adapter = GraphObservationAdapter(env)

    stack_bb_values = [float(s) for s in args.stack_bb_values.split(",") if s.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_grids = generate_all_grids(env, actor_critic, adapter, stack_bb_values)

    for stack_bb, grids in all_grids.items():
        print(f"stack={stack_bb}bb  "
              f"SB mean push%={grids['sb_push'].mean():.3f}  "
              f"BB mean continue%={grids['bb_call'].mean():.3f}")

        plot_heatmap(
            grids["sb_push"], f"SB Push Probability at {stack_bb}bb",
            out_dir / f"sb_push_{stack_bb}bb.png",
        )
        plot_heatmap(
            grids["bb_call"], f"BB Continue Probability at {stack_bb}bb",
            out_dir / f"bb_call_{stack_bb}bb.png",
        )

    print(f"\nHeatmaps saved to {out_dir}/")


if __name__ == "__main__":
    _main()
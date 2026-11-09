"""
envs/graph_scenario_env.py — Domain-general RL environment driving the
REAL Engine GraphEngine directly, per a
scenarios.graph_scenario_config.GraphScenarioConfig.

Supersedes envs.multi_street_env.MultiStreetEnv (built on the legacy
PokerState/Phase stack) and envs.generic_tree_env.GenericTreeEnv (a
hand-authored duplicate action tree) — see
scenarios/graph_scenario_config.py's module docstring for the full
rationale. Both of those files, and scenarios/duo_game_config.py, are
obsolete once this is in place and may be deleted.

Self-play contract
--------------------
There is no hero/villain split. EVERY seat is trained by the SAME
shared policy network (self-play with full parameter sharing) — the
network tells seats apart via PositionFeatures (button/blinds/
seat_fraction/is_next_to_act — representation/position_encoder.py),
not via separate weights or a separate code path per seat:

  - Every DECISION node reached, for ANY seat, is surfaced to the
    caller as a GraphDecisionObs and step() blocks pending an action
    index — reset()/step() never resolve a decision internally.
  - A DECISION node reached OUTSIDE config.scope is a RuntimeError (the
    scenario's coverage claim was wrong), not a silent auto-resolve —
    see graph_scenario_config.py's DecisionScope docstring.
  - run_hand() (below) is a convenience loop for training code: play
    one full hand under ONE policy callable, collecting every seat's
    decisions (chronologically, per seat) plus each seat's final
    reward — see graph_training_loop.py for how this feeds PPO.

Dealer rotation is now configurable (see StackConfig.fixed_dealer_position)
------------------------------------------------------------------------------
reset() previously ALWAYS drew a fresh, random dealer_position every
hand via self._rng.randrange(num_seats). That is the right choice for a
genuinely N-seat scenario (real positional generalization is the
point), but for a 2-seat scenario it adds pure noise relative to what
the pre-graph pipeline did (scenarios/push_fold_duo_generator.py /
envs/push_fold_duo_env.py, and scenarios/duo_game_config.py's
dealer_seat / envs/generic_tree_env.py, all of which pinned SB to a
single, permanently-fixed physical seat, never re-rolled per hand) — see
scenarios/graph_scenario_config.py's StackConfig.fixed_dealer_position
docstring for the full rationale. reset() now uses that value when the
scenario sets it, and falls back to the original random-every-hand
behaviour otherwise.

Per-node betting abstraction (NEW)
------------------------------------
_betting_options() now looks up the abstraction with
config.abstraction_for("BETTING", request.node_id), so a scenario can
give different streets different bet-size menus
(GraphScenarioConfig.node_action_abstraction). It also honours
BettingAbstraction.allow_raise: when False, engine RAISE options are
skipped entirely, so a facing-a-bet decision is just FOLD/CALL.

Domain-general action space
------------------------------
BETTING / BOOLEAN / CHOICE are single-shot: one GraphDecisionObs, one
action index, one DecisionResponse submitted immediately.

CARD_SELECT / CARD_PASS are decomposed into a SEQUENCE of single-card
categorical sub-decisions: each sub-decision's options are "pick one of
the still-available candidate cards" plus a STOP option once
request.options[0].min_count is already satisfied. Only once the full
set is chosen (STOP chosen, or max_count reached) does the env actually
call GraphEngine.submit_decision(). This keeps every individual step()
call a single categorical choice, matching PolicyValueNetwork's
existing one-Categorical-per-decision contract with zero changes needed
to that class.

Open assumptions (flagged for confirmation — see accompanying chat
message)
------------------------------------------------------------------------
    - Deck: assumed importable as poker_engine.state.deck.Deck, taking
      an optional constructor arg and building/shuffling a standard
      52-card deck. Only touched in _build_deck() below — isolated on
      purpose so a wrong import path is a one-line fix.
    - PlayerState: assumed constructible as
      PlayerState(stack=..., hand_mask=0, current_bet=0,
      total_contribution=0, has_folded=False, is_all_in=False,
      sitting_out=False) — inferred from
      scenarios.hand_snapshot_adapter.PlayerSnapshot's identical field
      set plus graph_engine.py's use of player.sitting_out. Only
      touched in _build_players() below.

Dependencies
------------
    poker_engine.rules.graph_loader.load_game_graph,
    poker_engine.graph.graph_engine.GraphEngine,
    poker_engine.graph.graph_state.GraphGameState,
    poker_engine.graph.core_types (DecisionDomain, DecisionRequest,
    DecisionResponse, *Option types), poker_engine.graph.game_graph
    (GameGraph, GameGraphError), poker_engine.graph.node.NodeKind,
    poker_engine.actions.action.Action, poker_engine.state.player_state.
    PlayerState, poker_engine.showdown.showdown_resolver.ShowdownResolver.
    scenarios.graph_scenario_config (GraphScenarioConfig and friends).
    representation.card_utils.mask_to_card_ids (RL-Lab-side helper,
    already used elsewhere in this codebase for the identical purpose).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from poker_engine.rules.loader import load_game as load_game_graph
from poker_engine.graph.graph_engine import GraphEngine
from poker_engine.graph.graph_state import GraphGameState
from poker_engine.graph.game_graph import GameGraph, GameGraphError
from poker_engine.graph.node import NodeKind
from poker_engine.graph.core_types import (
    CardZone,
    DecisionDomain,
    DecisionRequest,
    DecisionResponse,
    ZoneRef,
)
from poker_engine.actions.action import Action
from poker_engine.actions.action_type import ActionType as EngineActionType
from poker_engine.state.player_state import PlayerState

from poker_rl_lab.representation.card_utils import mask_to_card_ids
from poker_rl_lab.scenarios.graph_scenario_config import (
    GraphScenarioConfig,
    BettingAbstraction,
)


# ---------------------------------------------------------------------------
# Deck / showdown-resolver construction (see module docstring's "Open
# assumptions" section)
# ---------------------------------------------------------------------------

def _build_deck():
    # from poker_engine.state.deck import Deck  # ASSUMPTION — see module docstring
    from poker_engine.cards.deck import Deck  # ASSUMPTION — see module docstring

    deck = Deck()
    deck.shuffle()
    return deck


def _default_showdown_resolver(rules):
    from poker_engine.showdown.showdown_resolver import ShowdownResolver

    try:
        from poker_engine.scoring.scoring_engine import CppScoringEngine

        scoring_engine = CppScoringEngine()
    except ImportError:
        from poker_engine.scoring.mock_scoring_engine import MockScoringEngine

        scoring_engine = MockScoringEngine()
    return ShowdownResolver(scoring_engine, rules)


# ---------------------------------------------------------------------------
# Self-play policy type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphPolicyDecision:
    """
    What a training policy returns from one GraphDecisionObs, for
    run_hand() to both act on and record.

    rep_input carries whatever opaque, already-adapted representation
    the policy used to make this decision (e.g. a
    representation.representation_model.RepresentationInput built by
    graph_observation_adapter.py) — captured HERE, at decision time,
    rather than re-derived later from the observation: by the time a
    training loop is done playing a whole hand and wants to build a
    RolloutBuffer from it, env.state has moved on to wherever the hand
    ENDED, so re-adapting a historical GraphDecisionObs after the fact
    would silently read the WRONG (final) board/pot/stacks instead of
    the state as of that particular decision. GraphScenarioEnv itself
    stays representation-agnostic (this field is typed Any, not
    RepresentationInput, so envs/ never imports representation/) —
    it's the training loop's job to build and carry rep_input, not the
    env's.
    """
    action_index: int
    log_prob: float
    value: float
    rep_input: Any = None


#: Called once per decision point (any seat, any domain) during
#: run_hand(); must adapt the observation into whatever representation
#: it needs internally and return a GraphPolicyDecision. Since one
#: shared policy plays every seat, this is a plain
#: Callable[[GraphDecisionObs], GraphPolicyDecision] — no seat argument.
GraphTrainingPolicy = Callable[["GraphDecisionObs"], GraphPolicyDecision]


@dataclass(frozen=True)
class GraphSeatDecision:
    """One recorded decision from run_hand() — pairs a GraphDecisionObs
    with the GraphPolicyDecision the policy returned for it."""
    obs: "GraphDecisionObs"
    action_index: int
    log_prob: float
    value: float
    rep_input: Any = None


# ---------------------------------------------------------------------------
# Observation / step-result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphActionOption:
    """
    One discrete action the policy may choose at a single sub-decision,
    plus enough engine-native detail for the env to translate the
    chosen index back into a DecisionResponse.

    Exactly one of {engine_action_type+amount, card_id, choice_label,
    is_stop, "boolean value implied by label"} is meaningful, depending
    on which domain produced this option — see the per-domain builder
    methods on GraphScenarioEnv below.
    """
    label: str
    amount: Optional[int] = None
    engine_action_type: Optional[Any] = None   # poker_engine.actions.action_type.ActionType, BETTING only
    card_id: Optional[int] = None              # CARD_SELECT / CARD_PASS sub-pick only
    choice_label: Optional[str] = None         # CHOICE only
    is_stop: bool = False                      # CARD_SELECT / CARD_PASS "stop selecting" only


@dataclass(frozen=True)
class GraphDecisionObs:
    """
    Domain-general observation at ONE decision point, for WHICHEVER
    seat (`seat`) the graph currently asks to decide — see module
    docstring's "Self-play contract" section: there's no hero/villain
    distinction, every seat's decisions flow through this same type.

    Raw poker context (hole cards, board, pot, position, history —
    everything RepresentationModel needs) is deliberately NOT populated
    here. representation/graph_observation_adapter.py builds a
    RepresentationInput from GraphScenarioEnv's own live `.state`
    alongside this observation.
    """
    node_id: str
    domain: str          # DecisionDomain member name — "BETTING" etc.
    seat: int
    options: Tuple[GraphActionOption, ...]
    legal_mask: Tuple[bool, ...]   # same length as options, currently always all-True
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphScenarioStepResult:
    obs: Optional[GraphDecisionObs]
    rewards: Dict[int, float]   # {} on any intermediate step; one entry per seat once done
    done: bool
    info: dict


@dataclass(frozen=True)
class GraphHistoryEvent:
    """
    One action taken by one seat during the current hand, in
    engine-native terms.

    Deliberately NOT envs.observations.HistoryEventObs: that type's
    action_type field is validated against actions.abstract_action's
    OLD, fixed 9-value abstract ActionType vocabulary (FOLD/CHECK/CALL/
    BET_25.../BET_150/ALL_IN), which has nothing to do with the ENGINE's
    own poker_engine.actions.action_type.ActionType (FOLD/CHECK/CALL/
    BET/RAISE) that a GraphEngine DecisionResponse.value actually
    carries — see envs/multi_street_env.py's own docstring on this
    exact naming collision. Forcing one into the other would either
    fail HistoryEventObs's own range check or silently misrecord the
    action. This type stores the engine's ActionType directly instead;
    a downstream adapter (graph_observation_adapter.py) is responsible
    for turning a sequence of these into whatever a given encoder
    (representation.history_encoder.HistoryEncoder) actually expects.

    Populated for BETTING decisions (engine_action_type set, draw_count
    None) and for completed CARD_SELECT decisions, i.e. a DRAW
    (engine_action_type None, draw_count = number of cards the seat
    discarded/replaced). Events are appended in the order they happen,
    so the history any seat sees at a decision is exactly what it could
    legitimately know at that moment: in a draw round where seat 0
    draws first, seat 1's observation at its own draw already contains
    seat 0's draw event, while seat 0's does not contain seat 1's until
    seat 1 has drawn. CARD_PASS/BOOLEAN/CHOICE are still not recorded.
    """
    seat: int
    engine_action_type: Any   # poker_engine.actions.action_type.ActionType; None for a DRAW event
    street_index: int
    pot_fraction: float
    is_aggressor: bool
    draw_count: Optional[int] = None   # cards drawn (CARD_SELECT); None for betting events


@dataclass
class _PendingCardPick:
    """In-progress state for a CARD_SELECT/CARD_PASS decision being
    decomposed into single-card sub-picks. See module docstring."""
    request: DecisionRequest
    remaining_candidates: List[int]
    chosen: List[int] = field(default_factory=list)
    min_count: int = 0
    max_count: int = 0


# ---------------------------------------------------------------------------
# Scope validation
# ---------------------------------------------------------------------------

def validate_against_graph(config: GraphScenarioConfig, graph: GameGraph) -> None:
    """
    Check config.scope.node_ids (if given) against the COMPILED graph:
    every listed node id must exist, must be a DECISION node, and its
    domain must be one config.scope.domains actually declares. Raises
    RuntimeError with a message pointing at the scenario/graph mismatch
    — called once, at GraphScenarioEnv construction, so a bad scenario
    file fails loudly before any hand is ever dealt.

    Also checks that every key of config.node_action_abstraction names
    a BETTING DECISION node in the compiled graph.
    """
    if config.start is not None:
        try:
            start_node = graph.node(config.start.node_id)
        except GameGraphError as exc:
            raise RuntimeError(
                f"scenario {config.name!r} start.node_id {config.start.node_id!r} "
                f"is not a node in the compiled {config.engine_variant!r} graph"
            ) from exc
        if start_node.kind != NodeKind.DECISION:
            raise RuntimeError(
                f"scenario {config.name!r} start.node_id {config.start.node_id!r} "
                f"is an AUTO node, not a DECISION node"
            )
        if not config.scope.covers_node(config.start.node_id, start_node.domain.name):
            raise RuntimeError(
                f"scenario {config.name!r} start.node_id {config.start.node_id!r} "
                f"(domain {start_node.domain.name}) is outside the scenario's scope"
            )
    for node_id in config.node_action_abstraction:
        try:
            node = graph.node(node_id)
        except GameGraphError as exc:
            raise RuntimeError(
                f"scenario {config.name!r} node_action_abstraction references "
                f"{node_id!r}, which is not a node in the compiled "
                f"{config.engine_variant!r} graph"
            ) from exc
        if node.kind != NodeKind.DECISION or node.domain.name != "BETTING":
            raise RuntimeError(
                f"scenario {config.name!r} node_action_abstraction key "
                f"{node_id!r} is not a BETTING DECISION node"
            )
    if config.scope.node_ids is None:
        return
    for node_id in config.scope.node_ids:
        try:
            node = graph.node(node_id)
        except GameGraphError as exc:
            raise RuntimeError(
                f"scenario {config.name!r} scope.node_ids references "
                f"{node_id!r}, which is not a node in the compiled "
                f"{config.engine_variant!r} graph"
            ) from exc
        if node.kind != NodeKind.DECISION:
            raise RuntimeError(
                f"scenario {config.name!r} scope.node_ids references "
                f"{node_id!r}, which is an AUTO node, not a DECISION node"
            )
        domain_name = node.domain.name
        if not config.scope.covers_domain(domain_name):
            raise RuntimeError(
                f"scenario {config.name!r} scope.node_ids includes "
                f"{node_id!r} (domain {domain_name}) but scope.domains "
                f"({list(config.scope.domains)}) doesn't include it"
            )


def _cards_in_zone(zone_ref: ZoneRef, state: GraphGameState) -> List[int]:
    if zone_ref.zone == CardZone.HAND:
        return mask_to_card_ids(state.players[zone_ref.index].hand_mask)
    if zone_ref.zone == CardZone.DISCARD_PILE:
        return list(state.discard_pile)
    if zone_ref.zone == CardZone.BOARD_NODE:
        c = state.node_cards[zone_ref.index]
        return [c] if c is not None else []
    raise NotImplementedError(
        f"CARD_SELECT/CARD_PASS source zone {zone_ref.zone.name} is not "
        f"supported by GraphScenarioEnv (DECK is never a selectable source)"
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GraphScenarioEnv:
    """
    Executes one GraphScenarioConfig against the real, compiled Engine
    GameGraph for `config.engine_variant`.

    Parameters
    ----------
    config : GraphScenarioConfig
    scoring_engine : optional
        Forwarded into the ShowdownResolver built for this env — same
        CppScoringEngine/MockScoringEngine fallback chain used
        throughout this codebase. Pass explicitly to share one instance
        across many envs.
    transactional : bool
        Forwarded to GraphEngine. Default True. See GraphEngine.__init__.
    seed : int, optional
        Seeds the env's own RNG (dealer position when not fixed via
        config.stack.fixed_dealer_position, and randomized stack
        sizing). Does NOT seed the Engine's own Deck — see reset()'s
        docstring.
    """

    def __init__(
        self,
        config: GraphScenarioConfig,
        scoring_engine=None,
        seed: Optional[int] = None,
        transactional: bool = True,
    ) -> None:
        # transactional is forwarded to GraphEngine (see its __init__): True
        # keeps per-call rollback checkpoints (safe for interactive/Backend
        # use); training code passes False to skip two deepcopies per
        # engine call.
        self.transactional = transactional
        self.config = config
        self.game_def, self.rules, self.graph = load_game_graph(config.engine_variant)
        validate_against_graph(config, self.graph)

        # Graph-native GameDefinition no longer exposes street_nodes (see
        # CAP_Technical_Quick_Reference.md §12.4) — approximate "number
        # of streets" for GameFeatures.num_streets by counting DISTINCT
        # non-negative metadata["street_index"] values across every
        # compiled node. street_index: -1 is a documented sentinel for
        # "not a real board street" (see e.g. drawmaha.yaml's
        # discard_draw step) and is excluded.
        street_indices = {
            n.metadata.get("street_index")
            for n in self.graph.nodes.values()
            if isinstance(n.metadata.get("street_index"), int) and n.metadata["street_index"] >= 0
        }
        self.num_streets = len(street_indices) if street_indices else 1

        self._showdown_resolver = (
            _default_showdown_resolver(self.rules)
            if scoring_engine is None
            else __import__(
                "poker_engine.showdown.showdown_resolver", fromlist=["ShowdownResolver"]
            ).ShowdownResolver(scoring_engine, self.rules)
        )
        self._rng = random.Random(seed)

        self._engine: Optional[GraphEngine] = None
        self._pending_pick: Optional[_PendingCardPick] = None
        self._current_options: Tuple[GraphActionOption, ...] = ()
        self._starting_stacks: Tuple[int, ...] = ()
        self._history: List[GraphHistoryEvent] = []
        self._done = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def seed(self, value: int) -> None:
        """
        Seed this env's own RNG (dealer position when not fixed,
        randomized stack sizing — see _size_stack()). Does NOT seed the
        Engine's Deck (each reset() builds a fresh, independently-
        shuffled Deck() — see _build_deck()'s own docstring note), so
        calling seed() does NOT make card dealing itself deterministic,
        only the non-card-dealing scenario parameters. This is the same
        seed()/generate()-style surface every other *_generator.py in
        this codebase exposes, kept directly on the env now that a
        separate BaseScenario-producing generator object has no work
        left to do (see this file's own module docstring — the graph's
        AUTO nodes deal the hand; there's nothing upstream of reset()
        to generate anymore).
        """
        self._rng.seed(value)

    @property
    def history(self) -> Tuple[GraphHistoryEvent, ...]:
        """Betting history recorded so far this hand, chronological."""
        return tuple(self._history)

    @property
    def state(self) -> GraphGameState:
        """Live engine state for the in-progress episode. Read-only use
        (e.g. by a representation adapter) — never mutate directly."""
        if self._engine is None:
            raise RuntimeError("reset() has not been called yet")
        return self._engine.state

    @property
    def engine(self) -> GraphEngine:
        """Read-only access to the live GraphEngine driving this episode —
        for callers (e.g. Backend's graph_state_to_dto()) that need
        pending_request/is_complete/last_showdown/last_winners, not just
        .state. Public and stable; prefer this over reaching into
        _engine directly."""
        if self._engine is None:
            raise RuntimeError("reset() has not been called yet")
        return self._engine

    def reset(self) -> GraphDecisionObs:
        """
        Deal a fresh hand and play forward (through any AUTO nodes) to
        the first DECISION node, for whichever seat the graph asks to
        act first.

        Returns
        -------
        GraphDecisionObs
            The first decision point of the new hand.
        """
        players = self._build_players()
        state = GraphGameState(players, betting_type="no_limit")
        state.node_cards = [None] * self.game_def.node_count
        # See StackConfig.fixed_dealer_position's own docstring: a
        # genuinely N-seat scenario wants the dealer rotated every hand
        # (real positional generalization); a 2-seat scenario like
        # push_fold_duo gets no benefit from that and converges far
        # faster with a fixed dealer, matching what the old, pre-graph
        # pipeline always did (SB was a literal, permanently-fixed seat
        # there — see scenarios/push_fold_duo_generator.py /
        # scenarios/duo_game_config.py's dealer_seat).
        if self.config.stack.fixed_dealer_position is not None:
            state.dealer_position = self.config.stack.fixed_dealer_position
        else:
            state.dealer_position = self._rng.randrange(self.config.num_seats)
        deck = _build_deck()

        self._engine = GraphEngine(
            self.graph, state, deck,
            showdown_resolver=self._showdown_resolver,
            transactional=getattr(self, "transactional", True),
        )
        self._pending_pick = None
        self._starting_stacks = tuple(p.stack for p in players)
        self._history = []
        self._done = False

        self._engine.start()
        if self.config.start is not None:
            self._fast_forward_to_start()
        obs = self._advance_to_decision_or_terminal()
        if obs is None:
            raise RuntimeError(
                f"scenario {self.config.name!r} resolved before any "
                f"decision was reached — check scope against the compiled "
                f"{self.config.engine_variant!r} graph"
            )
        return obs

    def step(self, action_index: int) -> GraphScenarioStepResult:
        """
        Apply the chosen action (an index into the PREVIOUSLY returned
        GraphDecisionObs.options — whichever seat that observation
        named), then play forward through any remaining sub-picks /
        AUTO nodes until the next decision or a terminal state.
        """
        if self._done or self._engine is None:
            raise RuntimeError("call reset() before step(), or the episode has already ended")

        request = self._engine.pending_request
        domain_name = request.domain.name

        if domain_name in ("CARD_SELECT", "CARD_PASS"):
            next_obs = self._advance_card_pick(action_index)
            if next_obs is not None:
                self._current_options = next_obs.options
                return GraphScenarioStepResult(
                    obs=next_obs, rewards={}, done=False, info={"node": next_obs.node_id}
                )
            # else: the pick just completed and was submitted to the
            # engine — fall through to advance past it below.
        else:
            response = self._finalize_single_shot(request, action_index)
            self._submit(response)

        obs = self._advance_to_decision_or_terminal()
        if obs is not None:
            self._current_options = obs.options
            return GraphScenarioStepResult(obs=obs, rewards={}, done=False, info={"node": obs.node_id})

        rewards = self._compute_rewards()
        outcome = "showdown" if self._engine.last_showdown is not None else "fold"
        return GraphScenarioStepResult(obs=None, rewards=rewards, done=True, info={"outcome": outcome})

    def run_hand(
        self, policy: GraphTrainingPolicy
    ) -> Tuple[Dict[int, List[GraphSeatDecision]], Dict[int, float]]:
        """
        Play one full hand from reset() to completion under ONE shared
        `policy`, called at EVERY decision point regardless of seat —
        the self-play entry point graph_training_loop.py's rollout
        collection uses instead of driving reset()/step() by hand.

        Parameters
        ----------
        policy : GraphTrainingPolicy
            Callable(obs) -> GraphPolicyDecision. Called once per
            decision point (including once per CARD_SELECT/CARD_PASS
            sub-pick — each sub-pick is its own decision point with its
            own GraphDecisionObs).

        Returns
        -------
        (decisions, rewards)
        decisions : Dict[int, List[GraphSeatDecision]]
            One list per seat index in [0, config.num_seats), in
            chronological order. A seat that never acted this hand
            (e.g. folded to before its turn) gets an empty list.
        rewards : Dict[int, float]
            Chip change per seat this hand (final stack - starting
            stack), one entry per seat regardless of whether that seat
            ever acted.
        """
        n = self.config.num_seats
        decisions: Dict[int, List[GraphSeatDecision]] = {i: [] for i in range(n)}

        obs = self.reset()
        while True:
            decision = policy(obs)
            decisions[obs.seat].append(
                GraphSeatDecision(
                    obs=obs,
                    action_index=decision.action_index,
                    log_prob=decision.log_prob,
                    value=decision.value,
                    rep_input=decision.rep_input,
                )
            )
            result = self.step(decision.action_index)
            if result.done:
                return decisions, result.rewards
            obs = result.obs

    # ------------------------------------------------------------------
    # Fast-forward start (config.start) -- see StartConfig's docstring
    # ------------------------------------------------------------------

    @staticmethod
    def _passive_betting_response(request: DecisionRequest) -> DecisionResponse:
        """Check if legal, else call. Used ONLY to auto-play decisions
        BEFORE config.start.node_id (never involves the policy)."""
        by_name = {opt.action_type.name: opt for opt in request.options}
        opt = by_name.get("CHECK") or by_name.get("CALL")
        if opt is None:
            raise RuntimeError(
                f"cannot fast-forward past BETTING node {request.node_id!r}: "
                f"neither CHECK nor CALL is legal (options: {sorted(by_name)})"
            )
        return DecisionResponse(
            node_id=request.node_id,
            domain=request.domain,
            player_index=request.player_index,
            value=Action(type=opt.action_type, amount=opt.min_amount),
        )

    def _fast_forward_to_start(self) -> None:
        target = self.config.start.node_id
        for _ in range(256):   # generous guard against a non-terminating flow
            if self._engine.is_complete:
                raise RuntimeError(
                    f"scenario {self.config.name!r}: hand completed before "
                    f"reaching start node {target!r}"
                )
            request = self._engine.pending_request
            if request.node_id == target:
                self._apply_start_overrides()
                return
            if request.domain.name != "BETTING":
                raise RuntimeError(
                    f"scenario {self.config.name!r}: cannot fast-forward past "
                    f"{request.domain.name} decision at {request.node_id!r} "
                    f"(only BETTING decisions are auto-played)"
                )
            # Straight to the engine: deliberately NOT self._submit(), so
            # these passive pre-start actions are not recorded in history.
            self._engine.submit_decision(self._passive_betting_response(request))
        raise RuntimeError(
            f"scenario {self.config.name!r}: did not reach start node "
            f"{target!r} within 256 decisions"
        )

    def _apply_start_overrides(self) -> None:
        start = self.config.start
        bb = self.config.stack.big_blind_size
        state = self._engine.state
        pot = int(round(start.pot_bb * bb))
        behind = int(round(start.stack_behind_bb * bb))
        share = pot // len(state.players)

        for p in state.players:
            p.stack = behind
            p.current_bet = 0
            p.total_contribution = share
            p.is_all_in = False
        state.pot = pot

        # Rewards are measured against each seat's chips at scenario start
        # (stack behind + its share of the pot). Zero-sum by construction:
        # sum(starting) == chips in play.
        self._starting_stacks = tuple(p.stack + p.total_contribution for p in state.players)

        # The pending request was built before the override: rebuild it so
        # min/max/call amounts reflect the new pot and stacks.
        self._engine.refresh_pending_request()

    # ------------------------------------------------------------------
    # Internal: drive the engine forward
    # ------------------------------------------------------------------

    def _advance_to_decision_or_terminal(self) -> Optional[GraphDecisionObs]:
        if self._engine.is_complete:
            self._done = True
            return None

        request = self._engine.pending_request
        if request is None:
            raise RuntimeError(
                "GraphEngine has neither a pending request nor is_complete "
                "— violates GraphEngine's own start()/submit_decision() contract"
            )

        domain_name = request.domain.name
        if not self.config.scope.covers_node(request.node_id, domain_name):
            raise RuntimeError(
                f"seat {request.player_index} reached DECISION node "
                f"{request.node_id!r} (domain {domain_name}), which is "
                f"OUTSIDE scenario {self.config.name!r}'s declared scope "
                f"— the scenario's coverage claim doesn't match the "
                f"compiled graph"
            )
        return self._build_decision_obs(request)

    def _build_decision_obs(self, request: DecisionRequest) -> GraphDecisionObs:
        domain_name = request.domain.name
        self._pending_pick = None

        if domain_name == "BETTING":
            options = self._betting_options(request)
        elif domain_name == "BOOLEAN":
            options = self._boolean_options(request)
        elif domain_name == "CHOICE":
            options = self._choice_options(request)
        elif domain_name in ("CARD_SELECT", "CARD_PASS"):
            options = self._start_card_pick(request)
        else:
            raise RuntimeError(f"unsupported DecisionDomain {domain_name!r}")

        self._current_options = options
        return GraphDecisionObs(
            node_id=request.node_id,
            domain=domain_name,
            seat=request.player_index,
            options=options,
            legal_mask=self._padded_legal_mask(options),
        )

    def _padded_legal_mask(self, options: Tuple[GraphActionOption, ...]) -> Tuple[bool, ...]:
        """
        Pad `options` (variable length) out to
        self.config.max_action_options (fixed) — True for real options,
        False past len(options). This is what lets ONE policy network
        action head (sized to max_action_options) serve every domain and
        every individual decision a scenario trains, regardless of how
        many options any particular decision actually offers — see
        GraphScenarioConfig.max_action_options's own docstring.

        Raises RuntimeError (not a silent truncation) if `options`
        somehow exceeds the configured ceiling — every per-domain
        abstraction already validates this at the point options are
        built (_betting_options / _choice_options / _card_pick_options),
        so reaching this branch means max_action_options itself was
        computed from a stale/inconsistent config, not that any single
        decision was to blame.
        """
        width = self.config.max_action_options
        n = len(options)
        if n > width:
            raise RuntimeError(
                f"node offered {n} options, exceeding "
                f"config.max_action_options ({width}) — this should have "
                f"been caught earlier by the domain-specific abstraction "
                f"check; max_action_options may be stale relative to "
                f"action_abstraction"
            )
        return tuple(i < n for i in range(width))

    # ------------------------------------------------------------------
    # BETTING
    # ------------------------------------------------------------------

    def _betting_options(self, request: DecisionRequest) -> Tuple[GraphActionOption, ...]:
        # Per-node override (config.node_action_abstraction) wins over the
        # domain-wide action_abstraction["BETTING"] fallback.
        abstraction: Optional[BettingAbstraction] = self.config.abstraction_for(
            "BETTING", request.node_id
        )
        if abstraction is None:
            raise RuntimeError(
                f"scenario {self.config.name!r} has no BETTING action_abstraction "
                f"configured, but reached BETTING node {request.node_id!r}"
            )

        pot = self._engine.state.pot
        bb = self.config.stack.big_blind_size
        seen = set()
        resolved: List[GraphActionOption] = []

        # Bets/raises already made on THIS street (blinds are posted by an
        # AUTO node and never appear in history, so they don't count).
        street = self.graph.node(request.node_id).metadata.get("street_index", 0)
        street = street if isinstance(street, int) and street >= 0 else 0
        aggressions = sum(
            1 for e in self._history if e.is_aggressor and e.street_index == street
        )

        for opt in request.options:
            # allow_raise=False: facing a bet, the only continuing options
            # are FOLD / CALL — never re-raise.
            if not abstraction.allow_raise and opt.action_type == EngineActionType.RAISE:
                continue
            # max_aggressions: once the cap is reached, no more BET/RAISE.
            if (
                abstraction.max_aggressions is not None
                and aggressions >= abstraction.max_aggressions
                and opt.action_type in (EngineActionType.BET, EngineActionType.RAISE)
            ):
                continue
            # no_limp: with no bet/raise yet this street, CALL (i.e. limp
            # preflop) is not offered.
            if (
                abstraction.no_limp
                and aggressions == 0
                and opt.action_type == EngineActionType.CALL
            ):
                continue

            if opt.min_amount == opt.max_amount:
                key = (opt.action_type, opt.min_amount)
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(
                    GraphActionOption(
                        label=opt.action_type.name,
                        amount=opt.min_amount,
                        engine_action_type=opt.action_type,
                    )
                )
                continue

            for frac in abstraction.fractions_at(aggressions):
                amount = max(opt.min_amount, min(opt.max_amount, round(frac * pot)))
                key = (opt.action_type, amount)
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(
                    GraphActionOption(
                        label=f"{opt.action_type.name}_{frac:g}xPOT",
                        amount=amount,
                        engine_action_type=opt.action_type,
                    )
                )

            for size_bb in abstraction.sizes_bb_at(aggressions):
                amount = max(
                    opt.min_amount, min(opt.max_amount, int(round(size_bb * bb)))
                )
                key = (opt.action_type, amount)
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(
                    GraphActionOption(
                        label=f"{opt.action_type.name}_{size_bb:g}BB",
                        amount=amount,
                        engine_action_type=opt.action_type,
                    )
                )

            if abstraction.all_in_at(aggressions):
                key = (opt.action_type, opt.max_amount)
                if key not in seen:
                    seen.add(key)
                    resolved.append(
                        GraphActionOption(
                            label=f"{opt.action_type.name}_ALL_IN",
                            amount=opt.max_amount,
                            engine_action_type=opt.action_type,
                        )
                    )

        if not resolved:
            raise RuntimeError(
                f"BETTING abstraction produced zero resolved actions at "
                f"node {request.node_id!r} — check bet_fractions/include_all_in/"
                f"allow_raise"
            )
        return tuple(resolved)

    # ------------------------------------------------------------------
    # BOOLEAN
    # ------------------------------------------------------------------

    @staticmethod
    def _boolean_options(request: DecisionRequest) -> Tuple[GraphActionOption, ...]:
        return (
            GraphActionOption(label="FALSE"),
            GraphActionOption(label="TRUE"),
        )

    # ------------------------------------------------------------------
    # CHOICE
    # ------------------------------------------------------------------

    def _choice_options(self, request: DecisionRequest) -> Tuple[GraphActionOption, ...]:
        if len(request.options) != 1:
            raise NotImplementedError(
                f"CHOICE node {request.node_id!r} has {len(request.options)} "
                f"ChoiceOption entries — GraphScenarioEnv only supports exactly one"
            )
        opt = request.options[0]
        abstraction = self.config.abstraction_for("CHOICE")
        if abstraction is not None and len(opt.labels) > abstraction.max_options:
            raise RuntimeError(
                f"CHOICE node {request.node_id!r} offers {len(opt.labels)} "
                f"labels, exceeding action_abstraction.CHOICE.max_options "
                f"({abstraction.max_options})"
            )
        return tuple(GraphActionOption(label=l, choice_label=l) for l in opt.labels)

    # ------------------------------------------------------------------
    # CARD_SELECT / CARD_PASS — multi-pick decomposition
    # ------------------------------------------------------------------

    def _start_card_pick(self, request: DecisionRequest) -> Tuple[GraphActionOption, ...]:
        domain_name = request.domain.name
        if len(request.options) != 1:
            raise NotImplementedError(
                f"{domain_name} node {request.node_id!r} has "
                f"{len(request.options)} option entries — GraphScenarioEnv "
                f"only supports exactly one"
            )
        opt = request.options[0]
        candidates = _cards_in_zone(opt.source, self._engine.state)

        if domain_name == "CARD_SELECT":
            min_count, max_count = opt.min_count, opt.max_count
        else:  # CARD_PASS — exact count, no STOP action ever offered
            min_count = max_count = opt.count

        self._pending_pick = _PendingCardPick(
            request=request,
            remaining_candidates=list(candidates),
            chosen=[],
            min_count=min_count,
            max_count=max_count,
        )
        return self._card_pick_options()

    def _card_pick_options(self) -> Tuple[GraphActionOption, ...]:
        pending = self._pending_pick
        domain_name = pending.request.domain.name
        abstraction = self.config.abstraction_for(domain_name)

        options = [
            GraphActionOption(label=str(c), card_id=c) for c in pending.remaining_candidates
        ]
        can_stop = (
            len(pending.chosen) >= pending.min_count
            and len(pending.chosen) < pending.max_count
        )
        if can_stop:
            options.append(GraphActionOption(label="STOP", is_stop=True))

        if abstraction is not None and len(options) > abstraction.max_options:
            raise RuntimeError(
                f"{pending.request.node_id!r} sub-pick offers {len(options)} "
                f"options, exceeding action_abstraction[{domain_name!r}]"
                f".max_options ({abstraction.max_options})"
            )
        return tuple(options)

    def _advance_card_pick(self, action_index: int) -> Optional[GraphDecisionObs]:
        pending = self._pending_pick
        options = self._current_options
        if not (0 <= action_index < len(options)):
            raise ValueError(f"action_index {action_index} out of range for {len(options)} options")
        chosen = options[action_index]

        if chosen.is_stop:
            self._submit_card_pick()
            return None

        pending.chosen.append(chosen.card_id)
        pending.remaining_candidates.remove(chosen.card_id)

        if len(pending.chosen) >= pending.max_count:
            self._submit_card_pick()
            return None

        new_options = self._card_pick_options()
        return GraphDecisionObs(
            node_id=pending.request.node_id,
            domain=pending.request.domain.name,
            seat=pending.request.player_index,
            options=new_options,
            legal_mask=self._padded_legal_mask(new_options),
        )

    def _submit_card_pick(self) -> None:
        pending = self._pending_pick
        response = DecisionResponse(
            node_id=pending.request.node_id,
            domain=pending.request.domain,
            player_index=pending.request.player_index,
            value=tuple(pending.chosen),
        )
        self._pending_pick = None
        self._submit(response)

    def _submit(self, response: DecisionResponse) -> None:
        """
        Every DecisionResponse — any seat, any domain — passes through
        here on its way to GraphEngine.submit_decision(), so BETTING
        history can be recorded in exactly one place regardless of
        which call site (single-shot finalize, or card-pick
        completion) produced it. BETTING actions and CARD_SELECT draws
        are recorded into self._history (see GraphHistoryEvent);
        CARD_PASS/BOOLEAN/CHOICE are not yet.
        """
        if response.domain == DecisionDomain.BETTING:
            action: Action = response.value
            pot_before = self._engine.state.pot
            is_aggressor = action.type in (EngineActionType.BET, EngineActionType.RAISE)
            street_index = self.graph.node(response.node_id).metadata.get("street_index", 0)
            self._history.append(
                GraphHistoryEvent(
                    seat=response.player_index,
                    engine_action_type=action.type,
                    street_index=street_index if isinstance(street_index, int) and street_index >= 0 else 0,
                    pot_fraction=(action.amount / pot_before if pot_before > 0 else 0.0),
                    is_aggressor=is_aggressor,
                )
            )
        elif response.domain == DecisionDomain.CARD_SELECT:
            # A completed draw/discard: record HOW MANY cards were taken
            # (not which ones -- the opponent can see the count, not the
            # cards). Appended at submit time, in chronological order, so
            # visibility follows draw order automatically.
            last_street = self._history[-1].street_index if self._history else 0
            self._history.append(
                GraphHistoryEvent(
                    seat=response.player_index,
                    engine_action_type=None,
                    street_index=last_street,
                    pot_fraction=0.0,
                    is_aggressor=False,
                    draw_count=len(response.value),
                )
            )
        self._engine.submit_decision(response)

    # ------------------------------------------------------------------
    # Single-shot finalization (BETTING / BOOLEAN / CHOICE)
    # ------------------------------------------------------------------

    def _finalize_single_shot(self, request: DecisionRequest, action_index: int) -> DecisionResponse:
        options = self._current_options
        if not (0 <= action_index < len(options)):
            raise ValueError(f"action_index {action_index} out of range for {len(options)} options")
        chosen = options[action_index]
        domain_name = request.domain.name

        if domain_name == "BETTING":
            value: Any = Action(type=chosen.engine_action_type, amount=chosen.amount)
        elif domain_name == "BOOLEAN":
            value = chosen.label == "TRUE"
        elif domain_name == "CHOICE":
            value = chosen.choice_label
        else:
            raise RuntimeError(f"_finalize_single_shot called for unsupported domain {domain_name!r}")

        return DecisionResponse(
            node_id=request.node_id,
            domain=request.domain,
            player_index=request.player_index,
            value=value,
        )

    # ------------------------------------------------------------------
    # Setup / rewards
    # ------------------------------------------------------------------

    def _size_stack(self) -> int:
        sc = self.config.stack
        if sc.mode == "fixed_bb":
            return max(1, int(round(sc.stack_bb * sc.big_blind_size)))
        min_chips = max(1, int(sc.min_stack_bb * sc.big_blind_size))
        max_chips = max(min_chips, int(sc.max_stack_bb * sc.big_blind_size))
        return self._rng.randint(min_chips, max_chips)

    def _build_players(self) -> List[PlayerState]:
        stack = self._size_stack()
        return [
            PlayerState(
                stack=stack,
                hand_mask=0,
                current_bet=0,
                total_contribution=0,
                has_folded=False,
                is_all_in=False,
                sitting_out=False,
            )
            for _ in range(self.config.num_seats)
        ]

    def _compute_rewards(self) -> Dict[int, float]:
        final = [p.stack for p in self._engine.state.players]
        return {
            i: float(final[i] - self._starting_stacks[i])
            for i in range(self.config.num_seats)
        }
"""
scenarios/graph_scenario_config.py — Declarative schema + loader for RL Lab
training scenarios built directly on the Engine's compiled GameGraph.

Supersedes scenarios/duo_game_config.py's DuoGameConfig
--------------------------------------------------------
DuoGameConfig authored its OWN fixed action-tree graph in YAML (nodes:/
terminals:, hand-mapped ActionType -> target id) — a parallel, hand-
maintained duplicate of whatever the Engine's variant YAML already
expresses via `flow:`. Per CAP_Technical_Quick_Reference.md:

    "New integration work should target: poker_engine.games.loader.
    load_game() -> GameGraph / GraphGameState / GraphEngine /
    DecisionRequest / DecisionResponse ... Do not build new
    functionality around PokerState, Phase, GameState.legal_actions(),
    or the old EngineCallbacks."

and (§4.2) PokerState/Phase-based RL Lab work is EXPECTED to move onto
this same graph contract now (this project's own migration decision,
not just a Backend one) — see the coordination note this module
replaces.

A GraphScenarioConfig therefore does NOT author a graph. It names:
  - which real Engine variant YAML to compile (`engine_variant`,
    resolved via the graph-native loader — see graph_scenario_env.py),
  - how many seats the scenario deals (`num_seats`) — EVERY seat is
    trained via self-play under ONE shared policy network (see
    graph_scenario_env.py / graph_training_loop.py): there is no
    hero/villain split. A single set of weights plays whichever seat
    the graph currently asks to decide, conditioned on
    PositionFeatures (button/blinds/seat_fraction/is_next_to_act —
    representation/position_encoder.py) so it can tell which seat it's
    currently playing. This mirrors standard shared-weight self-play
    (e.g. a symmetric two-player self-play agent), just generalized to
    any seat count the graph is dealt with,
  - which DECISION nodes within the compiled graph are actually
    in-scope for training (`scope`) — ANY DECISION node reached (by any
    seat) OUTSIDE scope is a config error at validation/runtime (see
    graph_scenario_env.py's `validate_against_graph`), not a silent
    skip, since it means the scenario's coverage claim ("what this
    scenario trains") was wrong,
  - how to size starting stacks (`stack`) — the compiled graph has no
    notion of chip stacks; blinds/antes are already posted by the
    graph's own AUTO nodes, but WHO starts with HOW MANY chips is
    outside `flow:` and must be supplied by the scenario/generator,
    exactly like BaseScenario.players[*].stack always has been,
  - how to turn each in-scope decision's raw DecisionRequest.options
    into the model's fixed, small discrete action space
    (`action_abstraction`), domain by domain — and, for BETTING, node
    by node (`node_action_abstraction`), so e.g. the turn and river of
    one scenario can offer different bet-size menus.

This module is why the RL Lab doesn't need a new hand-authored *_duo.yaml
per game shape anymore: any Engine variant YAML that already exists
(poker_engine/games/*.yaml) is trainable by writing ONE short scenario
YAML that just says which decisions/stacks to use — the graph itself is
compiled fresh from the real Engine loader, never duplicated here.

Deliberately Engine-import-free
--------------------------------
Same rationale as duo_game_config.py: this module must be loadable and
unit-testable without poker_engine / the compiled C++ extension on the
path. It knows the DECISION_DOMAINS vocabulary as plain strings (see
CAP_Technical_Quick_Reference.md §3.3 — BETTING / CARD_SELECT /
CARD_PASS / BOOLEAN / CHOICE) rather than importing an Engine enum.
Anything that actually needs to compile/walk a real GameGraph (loading
the variant YAML, checking that scope.node_ids/domains actually exist
in the compiled graph, resolving DecisionRequest.options at runtime)
lives in graph_scenario_env.py instead, and is validated lazily via
`validate_against_graph()` — called BY that module, not at
load_graph_scenario() time.

Domain-general from the start
--------------------------------
Per instruction, this does not special-case BETTING as the only
supported domain. `scope.domains` may list any subset of
DECISION_DOMAINS, and `action_abstraction` may carry an abstraction
entry for each domain a scenario actually uses. BETTING gets a real,
opinionated abstraction (BettingAbstraction — fixed pot-fraction
buckets, mirroring actions.abstract_action.ActionType's existing
vocabulary so the rest of the RL Lab's bet-sizing intuition carries
over unchanged). CARD_SELECT / CARD_PASS / CHOICE get a much simpler
PassthroughAbstraction: the model's action space at such a node is just
"pick one of DecisionRequest.options's own entries, in order" — there's
no RL-Lab-invented bucketing to layer on top of "which card(s)" or
"which named choice", unlike a continuous bet size. BOOLEAN needs no
abstraction object at all (the space is always exactly {True, False}).

Dependencies
------------
    PyYAML, dataclasses, pathlib. No poker_engine, no poker_eval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import yaml

BASE_PATH = Path(__file__).parent.parent / "graph_scenarios"

# Matches poker_engine's DecisionRequest.domain vocabulary exactly (see
# CAP_Technical_Quick_Reference.md §3.3/§3.4). Kept as plain strings, not
# an imported enum, so this module stays Engine-import-free — see module
# docstring.
DECISION_DOMAINS = ("BETTING", "CARD_SELECT", "CARD_PASS", "BOOLEAN", "CHOICE")


class GraphScenarioConfigError(ValueError):
    """Raised for any schema/reference validation failure in a
    GraphScenarioConfig — distinguishes config-authoring bugs from
    ordinary ValueErrors raised elsewhere in the RL Lab."""


# ---------------------------------------------------------------------------
# Stack sizing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackConfig:
    """
    Starting stack sizing for every seat this scenario deals. The
    compiled GameGraph has no notion of chip stacks at all (see module
    docstring) — this is purely an RL Lab concern, same shape as
    scenarios.duo_game_config.StackConfig (kept identical on purpose,
    so existing generator code / intuition about the two modes carries
    over unchanged).

    mode : "fixed_bb" | "randomized_bb"
        fixed_bb       every seat gets `stack_bb` big blinds, every
                       episode.
        randomized_bb  every seat gets the SAME randomly drawn stack in
                       [min_stack_bb, max_stack_bb] big blinds that
                       episode (never asymmetric — matches every other
                       StackConfig in this codebase).

    big_blind_size, small_blind_size : int
        Chip size of one big/small blind, used to convert stack_bb into
        an actual chip count AND (small_blind_size) to feed
        representation.stack_encoder.StackFeatures.sb_bb downstream in
        graph_observation_adapter.py. The compiled graph itself doesn't
        expose either (see module docstring), so unlike the old
        DuoGameConfig (which took small_blind/big_blind at the top
        level and fed them to its own hand-authored blind-posting
        node), this scenario config must be told both directly — they
        must agree with whatever blind amounts the variant YAML's own
        flow: posts (post_blinds' config: {small_blind, big_blind}),
        since chip stacks and the graph's blind-posting AUTO node are
        otherwise unable to cross-check each other. Defaults 2 / 1
        (matches every existing scenario's small_blind=1/big_blind=2
        convention).
    fixed_dealer_position : Optional[int]
        If given, GraphScenarioEnv.reset() uses this EXACT dealer
        position every hand instead of drawing one from
        self._rng.randrange(num_seats). Default None (random dealer,
        rotated every hand) — the right choice for a genuinely N-seat
        scenario, where dealer rotation is what teaches the shared
        policy to generalize across positions rather than memorizing
        one literal seat index.

        For a 2-seat scenario like push_fold_duo, though, rotation adds
        no generalization value at all (there are only two positions
        total, and the network already receives is_small_blind/
        is_big_blind as explicit PositionFeatures regardless of which
        physical seat is dealer) — it only adds noise the network must
        learn to ignore before it can start differentiating by hand
        strength at all, which is exactly what the old, pre-graph
        pipeline (scenarios/push_fold_duo_generator.py /
        envs/push_fold_duo_env.py, and equally
        scenarios/duo_game_config.py's dealer_seat / envs/
        generic_tree_env.py's `dealer_position = self.config.
        seat_index(dealer_seat_id)`) never had to deal with: SB was a
        LITERAL, permanently-fixed seat in every one of those, never
        re-rolled per hand. Setting fixed_dealer_position restores that
        same fast, low-variance convergence for scenarios where
        positional generalization isn't the point.
    """
    mode: str
    stack_bb: float = 0.0
    min_stack_bb: float = 0.0
    max_stack_bb: float = 0.0
    big_blind_size: int = 2
    small_blind_size: int = 1
    fixed_dealer_position: Optional[int] = None

    def __post_init__(self) -> None:
        if self.mode not in ("fixed_bb", "randomized_bb"):
            raise GraphScenarioConfigError(
                f"stack.mode must be 'fixed_bb' or 'randomized_bb', got {self.mode!r}"
            )
        if self.big_blind_size <= 0:
            raise GraphScenarioConfigError(
                f"stack.big_blind_size must be > 0, got {self.big_blind_size}"
            )
        if self.small_blind_size <= 0:
            raise GraphScenarioConfigError(
                f"stack.small_blind_size must be > 0, got {self.small_blind_size}"
            )
        if self.mode == "fixed_bb":
            if self.stack_bb <= 0:
                raise GraphScenarioConfigError(
                    f"stack.stack_bb must be > 0 for mode 'fixed_bb', got {self.stack_bb}"
                )
        else:
            if self.min_stack_bb <= 0:
                raise GraphScenarioConfigError(
                    f"stack.min_stack_bb must be > 0, got {self.min_stack_bb}"
                )
            if self.max_stack_bb < self.min_stack_bb:
                raise GraphScenarioConfigError(
                    f"stack.max_stack_bb ({self.max_stack_bb}) must be >= "
                    f"min_stack_bb ({self.min_stack_bb})"
                )
        if self.fixed_dealer_position is not None and self.fixed_dealer_position < 0:
            raise GraphScenarioConfigError(
                f"stack.fixed_dealer_position must be >= 0, got "
                f"{self.fixed_dealer_position}"
            )


# ---------------------------------------------------------------------------
# Decision scope
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecisionScope:
    """
    Which DECISION nodes in the compiled graph this scenario claims to
    train — for EVERY seat (see module docstring: there's no
    hero/villain split, one shared policy plays every seat).

    domains : Sequence[str]
        Decision domains this scenario trains at all. A DECISION node
        whose domain is not listed here is out of scope regardless of
        which seat it belongs to. Must be non-empty; every entry must
        be one of DECISION_DOMAINS.
    node_ids : Optional[Sequence[str]]
        If given, further restricts scope to EXACTLY these graph node
        ids (still filtered by `domains` too — a listed node_id whose
        actual domain isn't in `domains` is a config error, checked by
        validate_against_graph). None (default) means every DECISION
        node of an in-scope domain is in scope, however many the
        compiled graph turns out to have — the common case for "train
        every betting decision this hand has" scenarios (e.g. a full
        multi-street hand). An explicit list is for narrower scenarios
        (e.g. "only the preflop shove/fold decision", mirroring the old
        push_fold_duo.yaml's single-node coverage).
    """
    domains: Sequence[str] = field(default_factory=lambda: ("BETTING",))
    node_ids: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if not self.domains:
            raise GraphScenarioConfigError("scope.domains must be non-empty")
        for d in self.domains:
            if d not in DECISION_DOMAINS:
                raise GraphScenarioConfigError(
                    f"scope.domains: unknown domain {d!r} (expected one of "
                    f"{DECISION_DOMAINS})"
                )
        if self.node_ids is not None and not self.node_ids:
            raise GraphScenarioConfigError(
                "scope.node_ids, if given, must be non-empty (omit the key "
                "entirely to mean 'every node of an in-scope domain')"
            )

    def covers_domain(self, domain: str) -> bool:
        return domain in self.domains

    def covers_node(self, node_id: str, domain: str) -> bool:
        """True iff `node_id` (of the given `domain`) is in scope."""
        if not self.covers_domain(domain):
            return False
        if self.node_ids is None:
            return True
        return node_id in self.node_ids


# ---------------------------------------------------------------------------
# Fast-forward start (e.g. "river only" scenarios on a FULL engine variant)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StartConfig:
    """
    Begin the scenario at a LATER decision node of the real compiled graph
    instead of at its entry point, so a "river-only" (or turn-only, ...)
    training scenario can run on the ordinary engine variant (e.g. `omaha`)
    without a purpose-built engine YAML for it.

    On reset(), GraphScenarioEnv plays the real graph forward from its
    entry node, auto-resolving every BETTING decision before `node_id`
    PASSIVELY (check, else call) -- no policy involvement, nothing recorded
    into history -- until the engine asks for a decision at `node_id`. It
    then overrides the pot and every seat's stack to the values below
    (blinds/antes/pre-river betting rarely produce the pot/SPR a scenario
    wants to study) and refreshes the pending request, so bet-size options
    reflect the overridden stacks. Hole cards and board are dealt by the
    graph's own AUTO nodes as usual.

    Attributes
    ----------
    node_id : str
        Graph DECISION node to hand control to the policy at (e.g.
        "bet_river"). Must be in scope.
    pot_bb : float
        Pot in big blinds on arrival. Split evenly across seats as each
        seat's total_contribution (so side-pot / payout logic stays
        consistent); pot in chips must be divisible by num_seats.
    stack_behind_bb : float
        Every seat's remaining stack, in big blinds, on arrival.

    Rewards are measured relative to (stack_behind + share of pot) per
    seat, i.e. each seat's chips at the START of the scenario including its
    contribution to the pot, matching how the dedicated river engine
    variants measured them.
    """
    node_id: str
    pot_bb: float
    stack_behind_bb: float

    def __post_init__(self) -> None:
        if not self.node_id:
            raise GraphScenarioConfigError("start.node_id must be non-empty")
        if self.pot_bb <= 0:
            raise GraphScenarioConfigError(f"start.pot_bb must be > 0, got {self.pot_bb}")
        if self.stack_behind_bb <= 0:
            raise GraphScenarioConfigError(
                f"start.stack_behind_bb must be > 0, got {self.stack_behind_bb}"
            )


# ---------------------------------------------------------------------------
# Per-domain action abstraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BettingAbstraction:
    """
    Maps a small, fixed discrete action space onto a BETTING
    DecisionRequest's actual (min, max) sizing range at runtime.

    Per CAP_Technical_Quick_Reference.md §3.4, a BETTING
    DecisionRequest's options already carry sized boundary amounts
    (min raise / max raise) — the Engine deliberately does not enumerate
    every possible bet size ("The resolver provides sized betting
    options... The caller/UI is responsible for choosing/interpolating a
    size within the allowed range"). This is exactly
    actions.abstract_action's existing job (compute_bet_amount,
    ActionType.BET_* pot fractions); this class carries the same
    fraction vocabulary as scenario-level CONFIGURATION instead of a
    hardcoded enum, so a scenario can narrow/widen its own bet-sizing
    menu (e.g. a shove-only preflop scenario vs. a full multi-size
    river scenario) without touching Python.

    fold_check_call are never listed here — they aren't sizes, they're
    presence/absence of a DecisionRequest option, always surfaced
    whenever the Engine's own options include them (see
    graph_scenario_env.py's per-domain resolver).

    bet_fractions : Sequence[float]
        Pot-fraction bet/raise sizes this scenario's action space
        offers, e.g. [0.5, 1.0] for "half pot, pot". Translated to a
        concrete total bet amount at runtime via
        actions.abstract_action.compute_bet_amount-equivalent logic
        against the DecisionRequest's own min/max — see
        graph_scenario_env.py.
    include_all_in : bool
        Whether "all-in" is always offered as its own action (in
        addition to whatever bet_fractions produces) whenever the
        DecisionRequest's max sizing option equals the acting player's
        full stack. Default True.
    allow_raise : bool
        If False, engine RAISE options are never offered at nodes using
        this abstraction: a facing-a-bet decision is just FOLD/CALL.
        Gives a "bet or check, then call or fold" tree with no
        re-raises. Default True (unchanged behaviour).
    sizes_bb : Sequence[float]
        Absolute "bet/raise TO" sizes in big blinds (e.g. [3.0] = raise
        to 3bb), as an alternative to pot fractions. Each is converted to
        chips with stack.big_blind_size and clamped into the engine's
        (min, max) range at runtime. Default empty.
    max_aggressions : Optional[int]
        If set, once this many bets/raises have been made on the current
        street (blinds do NOT count), no further BET/RAISE option is
        offered — only fold/check/call remain. max_aggressions=1 gives
        "one raise, then call or fold". Default None (unlimited).
    no_limp : bool
        If True, CALL is not offered while no bet/raise has been made on
        the street yet. Preflop this removes the limp: the first player
        can only fold or raise. Default False.
    bet_fractions_by_aggression, sizes_bb_by_aggression : Sequence[Sequence[float]]
        Per-aggression-level sizing menus. Entry i is the menu offered
        when i bets/raises have ALREADY been made on this street (blinds
        do not count): entry 0 is the opening bet/raise, entry 1 the
        re-raise (3-bet preflop), and so on. Indexes past the end offer
        no sized options of that kind (only all-in, if allowed). When a
        by_aggression field is non-empty it REPLACES the matching flat
        field (bet_fractions / sizes_bb) at every level. Example for a
        draw game's preflop: sizes_bb_by_aggression: [[3.0], [9.0]] =
        "open to 3bb; 3-bet to 9bb", so the opener is never offered 9bb.
    all_in_min_aggressions : int
        All-in is only offered once at least this many bets/raises have
        been made on the street (default 0 = always, unchanged). Use 2
        to allow a shove only as a 4-bet.
    """
    bet_fractions: Sequence[float] = field(
        default_factory=lambda: (0.25, 0.5, 0.75, 1.0, 1.5)
    )
    include_all_in: bool = True
    allow_raise: bool = True
    sizes_bb: Sequence[float] = field(default_factory=tuple)
    max_aggressions: Optional[int] = None
    no_limp: bool = False
    bet_fractions_by_aggression: Sequence[Sequence[float]] = field(default_factory=tuple)
    sizes_bb_by_aggression: Sequence[Sequence[float]] = field(default_factory=tuple)
    all_in_min_aggressions: int = 0

    def __post_init__(self) -> None:
        any_sized = (
            bool(self.bet_fractions)
            or bool(self.sizes_bb)
            or any(self.bet_fractions_by_aggression)
            or any(self.sizes_bb_by_aggression)
        )
        if not any_sized and not self.include_all_in:
            raise GraphScenarioConfigError(
                "betting abstraction must offer at least one sized bet "
                "(bet_fractions / sizes_bb, flat or by_aggression, non-empty) "
                "or include_all_in=True — otherwise a BETTING decision with "
                "no legal check/call would have zero legal actions"
            )
        if self.all_in_min_aggressions < 0:
            raise GraphScenarioConfigError(
                f"all_in_min_aggressions must be >= 0, got {self.all_in_min_aggressions}"
            )
        for lvl in self.bet_fractions_by_aggression:
            for f in lvl:
                if f <= 0:
                    raise GraphScenarioConfigError(
                        f"bet_fractions_by_aggression values must be > 0, got {f}"
                    )
        for lvl in self.sizes_bb_by_aggression:
            for v in lvl:
                if v <= 0:
                    raise GraphScenarioConfigError(
                        f"sizes_bb_by_aggression values must be > 0, got {v}"
                    )
        for f in self.bet_fractions:
            if f <= 0:
                raise GraphScenarioConfigError(
                    f"betting abstraction bet_fractions must all be > 0, got {f}"
                )
        for s in self.sizes_bb:
            if s <= 0:
                raise GraphScenarioConfigError(
                    f"betting abstraction sizes_bb must all be > 0, got {s}"
                )
        if self.max_aggressions is not None and self.max_aggressions < 0:
            raise GraphScenarioConfigError(
                f"max_aggressions must be >= 0, got {self.max_aggressions}"
            )

    @property
    def max_options(self) -> int:
        """
        Upper bound on resolved actions a BETTING decision under this
        abstraction can ever produce, for sizing the policy's fixed
        categorical action head (see GraphScenarioConfig.max_action_options).
        3 fixed slots (fold/check/call — never all three legal at once
        in practice, but this is a ceiling, not an exact count) + one
        slot per configured bet_fractions entry + one for all-in.
        Deduplication at resolve-time (see graph_scenario_env.py's
        _betting_options) only ever REMOVES entries relative to this
        bound, never adds — so sizing the head to this value is always
        safe, if occasionally a little wider than strictly needed.
        """
        n_levels = max(
            len(self.bet_fractions_by_aggression),
            len(self.sizes_bb_by_aggression),
            1,
        )
        widest = max(
            len(self.fractions_at(i)) + len(self.sizes_bb_at(i))
            for i in range(n_levels + 1)
        )
        return 3 + widest + (1 if self.include_all_in else 0)

    # ------------------------------------------------------------------
    # Per-aggression-level menus (used by GraphScenarioEnv._betting_options)
    # ------------------------------------------------------------------

    def fractions_at(self, aggressions: int) -> Tuple[float, ...]:
        """Pot-fraction menu when `aggressions` bets/raises have been made
        this street. See bet_fractions_by_aggression."""
        if self.bet_fractions_by_aggression:
            if aggressions < len(self.bet_fractions_by_aggression):
                return tuple(self.bet_fractions_by_aggression[aggressions])
            return ()
        return tuple(self.bet_fractions)

    def sizes_bb_at(self, aggressions: int) -> Tuple[float, ...]:
        """Absolute-bb menu when `aggressions` bets/raises have been made
        this street. See sizes_bb_by_aggression."""
        if self.sizes_bb_by_aggression:
            if aggressions < len(self.sizes_bb_by_aggression):
                return tuple(self.sizes_bb_by_aggression[aggressions])
            return ()
        return tuple(self.sizes_bb)

    def all_in_at(self, aggressions: int) -> bool:
        """Whether all-in is offered at this aggression level."""
        return self.include_all_in and aggressions >= self.all_in_min_aggressions


@dataclass(frozen=True)
class PassthroughAbstraction:
    """
    Abstraction for domains where the Engine's own DecisionRequest
    options ARE already the right discrete action space, with no
    RL-Lab-invented bucketing layered on top — CARD_SELECT, CARD_PASS,
    CHOICE. The model's action index simply selects
    DecisionRequest.options[index] directly (order as returned by the
    Engine at that node).

    max_options : int
        Upper bound on how many options this scenario's model output
        head needs to support for this domain, across every in-scope
        node of this domain. Validated against the compiled graph at
        validate_against_graph() time — a node offering MORE options
        than this is a config error (the model has nowhere to put the
        extra logit), not silently truncated.
    """
    max_options: int

    def __post_init__(self) -> None:
        if self.max_options < 1:
            raise GraphScenarioConfigError(
                f"max_options must be >= 1, got {self.max_options}"
            )


AnyAbstraction = Union[BettingAbstraction, PassthroughAbstraction]


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphScenarioConfig:
    """
    A fully validated RL Lab training scenario over a real, compiled
    Engine GameGraph. See module docstring for the full rationale.

    Attributes
    ----------
    name : str
        Scenario identifier — matches the YAML filename stem, same
        loud-not-silent convention duo_game_config.load_duo_game()
        enforced for its `key:`/filename pairing.
    engine_variant : str
        Name passed to the graph-native Engine loader (resolves a
        poker_engine/games/<engine_variant>.yaml). This is NOT
        necessarily the same string as `name` — several scenarios can
        train different scope slices of the SAME engine_variant (e.g.
        "holdem_preflop_only" and "holdem_full_hand" both naming
        engine_variant: holdem).
    num_seats : int
        Number of seats in this scenario. The compiled graph itself
        doesn't declare a seat count (see module docstring) — this is
        purely how many PlayerState objects GraphScenarioEnv deals in
        each hand. Every seat is trained (self-play, one shared
        policy) — there is no hero_seats/villain_seats distinction.
    start_node : Optional[str]
        (Unused / reserved.) See `start` for fast-forwarding to a later node.
    start : Optional[StartConfig]
        Fast-forward to a later DECISION node with an overridden pot and
        stacks (e.g. river-only training on a full engine variant). None
        (default) starts at the graph's own entry point. See StartConfig.
    stack : StackConfig
    scope : DecisionScope
    action_abstraction : Dict[str, AnyAbstraction]
        One entry per domain actually reachable in this scenario's
        scope. Keys must be a subset of scope.domains — see
        __post_init__. Missing an in-scope domain here is allowed only
        for BOOLEAN (always {True, False}, needs no configuration);
        every other domain in scope.domains MUST have an entry.
    node_action_abstraction : Dict[str, BettingAbstraction]
        Per-node BETTING abstraction overrides, keyed by graph node id
        (e.g. {"bet_turn": ..., "bet_river": ...}). A BETTING node
        listed here uses its own abstraction instead of
        action_abstraction["BETTING"], which remains the fallback for
        every other BETTING node. Lets one scenario offer different
        bet-size menus (and allow_raise settings) on different streets.
    """
    name: str
    engine_variant: str
    description: str = ""

    num_seats: int = 2

    start_node: Optional[str] = None

    stack: StackConfig = field(
        default_factory=lambda: StackConfig(mode="fixed_bb", stack_bb=100.0)
    )
    scope: DecisionScope = field(default_factory=DecisionScope)
    action_abstraction: Dict[str, AnyAbstraction] = field(default_factory=dict)
    start: Optional[StartConfig] = None
    node_action_abstraction: Dict[str, BettingAbstraction] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_seats < 2:
            raise GraphScenarioConfigError(
                f"num_seats must be >= 2, got {self.num_seats}"
            )
        if (
            self.stack.fixed_dealer_position is not None
            and self.stack.fixed_dealer_position >= self.num_seats
        ):
            raise GraphScenarioConfigError(
                f"stack.fixed_dealer_position ({self.stack.fixed_dealer_position}) "
                f"must be < num_seats ({self.num_seats})"
            )

        if self.start is not None:
            if (
                self.scope.node_ids is not None
                and self.start.node_id not in self.scope.node_ids
            ):
                raise GraphScenarioConfigError(
                    f"start.node_id {self.start.node_id!r} must be listed in "
                    f"scope.node_ids ({list(self.scope.node_ids)})"
                )
            pot_chips = int(round(self.start.pot_bb * self.stack.big_blind_size))
            if pot_chips % self.num_seats != 0:
                raise GraphScenarioConfigError(
                    f"start.pot_bb ({self.start.pot_bb}) * big_blind_size "
                    f"({self.stack.big_blind_size}) = {pot_chips} chips is not "
                    f"divisible by num_seats ({self.num_seats})"
                )

        for domain in self.action_abstraction:
            if domain not in self.scope.domains:
                raise GraphScenarioConfigError(
                    f"action_abstraction declares domain {domain!r}, which "
                    f"is not in scope.domains ({list(self.scope.domains)})"
                )
        for domain in self.scope.domains:
            if domain == "BOOLEAN":
                continue  # needs no configuration — space is always {True, False}
            if domain not in self.action_abstraction:
                raise GraphScenarioConfigError(
                    f"scope.domains includes {domain!r} but "
                    f"action_abstraction has no entry for it — every "
                    f"in-scope domain except BOOLEAN needs an abstraction"
                )

        if self.node_action_abstraction:
            if "BETTING" not in self.scope.domains:
                raise GraphScenarioConfigError(
                    "node_action_abstraction requires BETTING in scope.domains"
                )
            if self.scope.node_ids is not None:
                for nid in self.node_action_abstraction:
                    if nid not in self.scope.node_ids:
                        raise GraphScenarioConfigError(
                            f"node_action_abstraction key {nid!r} is not in "
                            f"scope.node_ids ({list(self.scope.node_ids)})"
                        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def abstraction_for(
        self, domain: str, node_id: Optional[str] = None
    ) -> Optional[AnyAbstraction]:
        """The configured abstraction for `domain` (and, for BETTING,
        optionally a specific `node_id` whose per-node override wins),
        or None for BOOLEAN (which needs none) / a domain not in scope
        at all."""
        if domain == "BETTING" and node_id is not None:
            override = self.node_action_abstraction.get(node_id)
            if override is not None:
                return override
        return self.action_abstraction.get(domain)

    @property
    def max_action_options(self) -> int:
        """
        The fixed width every hero decision's action space is padded to
        — i.e. the categorical action head size the policy network must
        be built with (see models.actor_critic_factory.ActorCriticConfig
        .action_dim). Computed as the max, across every domain actually
        in scope (and every per-node BETTING override), of that
        domain's own upper bound:

            BETTING              -> BettingAbstraction.max_options
            CARD_SELECT/CARD_PASS/CHOICE -> PassthroughAbstraction.max_options
            BOOLEAN              -> always exactly 2 (True/False)

        GraphScenarioEnv pads every GraphDecisionObs.legal_mask to
        exactly this length (True for real options, False past
        len(options)) regardless of which domain produced that specific
        decision, so one policy network head serves every domain a
        scenario trains, with illegal padding slots simply masked off —
        see apply_legal_mask (models/policy_value_network.py), unchanged.
        """
        widths = []
        for domain in self.scope.domains:
            if domain == "BOOLEAN":
                widths.append(2)
                continue
            abstraction = self.action_abstraction[domain]
            widths.append(abstraction.max_options)
        widths.extend(a.max_options for a in self.node_action_abstraction.values())
        return max(widths)


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

def _parse_stack(raw: dict) -> StackConfig:
    return StackConfig(
        mode=raw["mode"],
        stack_bb=float(raw.get("stack_bb", 0.0)),
        min_stack_bb=float(raw.get("min_stack_bb", 0.0)),
        max_stack_bb=float(raw.get("max_stack_bb", 0.0)),
        big_blind_size=int(raw.get("big_blind_size", 2)),
        small_blind_size=int(raw.get("small_blind_size", 1)),
        fixed_dealer_position=(
            int(raw["fixed_dealer_position"])
            if raw.get("fixed_dealer_position") is not None
            else None
        ),
    )


def _parse_start(raw: Optional[dict]) -> Optional[StartConfig]:
    if raw is None:
        return None
    for key in ("node_id", "pot_bb", "stack_behind_bb"):
        if key not in raw:
            raise GraphScenarioConfigError(f"start: missing required key {key!r}")
    return StartConfig(
        node_id=str(raw["node_id"]),
        pot_bb=float(raw["pot_bb"]),
        stack_behind_bb=float(raw["stack_behind_bb"]),
    )


def _parse_scope(raw: dict) -> DecisionScope:
    node_ids = raw.get("node_ids")
    return DecisionScope(
        domains=tuple(raw.get("domains", ("BETTING",))),
        node_ids=tuple(node_ids) if node_ids is not None else None,
    )


def _parse_abstraction(domain: str, raw: dict) -> AnyAbstraction:
    if domain == "BETTING":
        return BettingAbstraction(
            bet_fractions=tuple(raw.get("bet_fractions", (0.25, 0.5, 0.75, 1.0, 1.5))),
            include_all_in=bool(raw.get("include_all_in", True)),
            allow_raise=bool(raw.get("allow_raise", True)),
            sizes_bb=tuple(raw.get("sizes_bb", ())),
            max_aggressions=(
                int(raw["max_aggressions"])
                if raw.get("max_aggressions") is not None
                else None
            ),
            no_limp=bool(raw.get("no_limp", False)),
            bet_fractions_by_aggression=tuple(
                tuple(float(x) for x in lvl)
                for lvl in raw.get("bet_fractions_by_aggression", ())
            ),
            sizes_bb_by_aggression=tuple(
                tuple(float(x) for x in lvl)
                for lvl in raw.get("sizes_bb_by_aggression", ())
            ),
            all_in_min_aggressions=int(raw.get("all_in_min_aggressions", 0)),
        )
    # CARD_SELECT / CARD_PASS / CHOICE all share the same passthrough shape.
    if "max_options" not in raw:
        raise GraphScenarioConfigError(
            f"action_abstraction[{domain!r}] must declare max_options "
            f"(passthrough domains have no RL-Lab-invented bucketing — "
            f"see PassthroughAbstraction's docstring)"
        )
    return PassthroughAbstraction(max_options=int(raw["max_options"]))


def parse_graph_scenario_config(data: dict) -> GraphScenarioConfig:
    """
    Parse an already-loaded YAML dict into a validated
    GraphScenarioConfig. Exposed separately from load_graph_scenario()
    so tests can construct a config from an in-memory dict without
    touching the filesystem — mirrors
    duo_game_config.parse_duo_game_config's own split.
    """
    stack = _parse_stack(data["stack"])
    scope = _parse_scope(data.get("scope", {}))

    abstraction_raw = data.get("action_abstraction", {})
    action_abstraction = {
        domain: _parse_abstraction(domain, raw)
        for domain, raw in abstraction_raw.items()
    }

    node_abstraction = {
        nid: _parse_abstraction("BETTING", raw)
        for nid, raw in (data.get("node_action_abstraction") or {}).items()
    }

    return GraphScenarioConfig(
        name=data["name"],
        engine_variant=data["engine_variant"],
        description=data.get("description", ""),
        num_seats=int(data.get("num_seats", 2)),
        start_node=data.get("start_node"),
        stack=stack,
        scope=scope,
        action_abstraction=action_abstraction,
        start=_parse_start(data.get("start")),
        node_action_abstraction=node_abstraction,
    )


def load_graph_scenario(name: str, path: str | Path | None = None) -> GraphScenarioConfig:
    """
    Load and validate a graph scenario configuration.

    Parameters
    ----------
    name : str
        File stem, e.g. "holdem_preflop", "holdem_full_hand". Ignored
        when `path` is provided. Must equal the loaded config's own
        `name:` field, or this raises loudly (same convention
        duo_game_config.load_duo_game() established for its filename/
        `key:` pairing) — a mismatch usually means a copy-pasted file
        whose internal name wasn't updated.
    path : str | Path | None, optional
        Explicit path to a YAML file. If omitted, loaded from
        `graph_scenarios/{name}.yaml`.

    Returns
    -------
    GraphScenarioConfig
    """
    config_path = Path(path) if path is not None else BASE_PATH / f"{name}.yaml"

    if not config_path.exists():
        raise GraphScenarioConfigError(f"no graph scenario config found at {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    config = parse_graph_scenario_config(data)
    if path is None and config.name != name:
        raise GraphScenarioConfigError(
            f"graph_scenarios/{name}.yaml declares name: {config.name!r}, "
            f"which does not match its filename stem {name!r}"
        )
    return config
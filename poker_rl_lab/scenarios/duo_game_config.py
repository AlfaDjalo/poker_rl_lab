"""
scenarios/duo_game_config.py — Declarative schema + loader for fixed
action-tree N-seat mini-games (scenarios/duo_games/*.yaml), per
RL_Lab_Scenario_Framework - Addendum, Grounding and Migration.md §3.1:

    "A first-generation Scenario Package / Game State Machine ...
    should be the seed the new framework grows from."

This is that seed. envs/generic_tree_env.py (the executor) and
scenarios/generic_tree_scenario_generator.py (the initial-state
generator) both already depend on the types defined here
(DuoGameConfig, NodeConfig, DealNodeConfig, TerminalConfig,
StackConfig, PotConfig, DealConfig, SeatConfig) and on load_duo_game().

Design
------
Mirrors poker_engine/games/loader.py's own shape deliberately: one YAML
file per game, one loader function, plain dataclasses as the compiled
in-memory representation. Unlike the Engine loader (which only ever
reads a trusted, hand-authored small set of files), everything here is
validated aggressively at load time — bad config should fail loudly
before training starts, never surface as a confusing runtime KeyError
three layers deep in GenericTreeEnv.

Node types
----------
    NodeConfig      — a Player Decision Node. `seat` (seat id string)
                       + `actions` (ActionType -> target node/terminal
                       id). One seat acts; the environment presents
                       exactly these actions as the legal mask.
    DealNodeConfig   — a Chance/Engine-Transition Node (§2's terms).
                       Reveals `node_indices` of the board and falls
                       through to `next` automatically — no seat acts,
                       no AbstractAction consumed. See
                       generic_tree_env.py's
                       `_advance_through_deal_nodes`.

Both are found in `DuoGameConfig.nodes`, discriminated at load time by
whether the raw YAML block has a `seat` key (-> NodeConfig) or
`type: deal` (-> DealNodeConfig).

Terminals
---------
    TerminalConfig — `type` is "fold" (some seat(s) folded; the
    survivors split the pot — see `folders`) or "showdown" (every seat
    still live; resolved by GenericTreeEnv's scoring_engine). Terminals
    live in `DuoGameConfig.terminals`, a namespace disjoint from
    `nodes` — `is_terminal(id)` is exactly `id in self.terminals`.

Stack / Pot / Deal
------------------
    StackConfig — every seat gets the SAME stack (see
    generic_tree_scenario_generator.py's `_size_stack` docstring for
    why this schema never sizes seats asymmetrically): either a fixed
    `stack_bb`, or `[min_stack_bb, max_stack_bb]` randomised per
    episode.
    PotConfig — "blinds": seat 0 posts small_blind, seat 1 posts
    big_blind (both clamped to stack), everyone else posts 0, no dead
    pot. "randomized_bb": a `[min_pot_bb, max_pot_bb]` dead pot with no
    per-seat attribution (river-style scenarios that start mid-street).
    DealConfig — `street_index` for HistoryEventObs/BettingObs bookkeeping,
    `hole_cards_per_player`, `node_count` (total board slots) and
    whether a board is dealt at all (`deal_board`). `reveal_at_start`
    controls whether the FULL board is visible from the first
    observation (single-street games like river_duo — no DealNodeConfig
    needed) or only progressively revealed through DealNodeConfig nodes
    reached during play (turn_river_duo).

Validation performed at load time
-------------------------------------
  - schema shape (required keys present, correct types)
  - every seat id referenced (node.seat, dealer_seat, terminal.folders)
    exists in `seats`
  - exactly one seat has acts_first=True
  - every action name is a real actions.abstract_action.ActionType
  - every action target and every DealNodeConfig.next resolves to a
    real node OR terminal id
  - every DealNodeConfig.node_indices is within [0, deal.node_count)
  - no node graph reachable from start_node passes through an
    unbounded chain of DealNodeConfig-only nodes (a cycle with no
    decision/terminal ever reached) — this is the invariant
    generic_tree_env.py's `_advance_through_deal_nodes` relies on to
    guarantee termination without its own cycle counter
  - start_node exists

Dependencies
------------
    PyYAML, dataclasses, pathlib. actions.abstract_action.ActionType
    (pure Python — no poker_eval dependency, so a config can be loaded
    and validated without the Engine's C++ extension present).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from poker_rl_lab.actions.abstract_action import ActionType

BASE_PATH = Path(__file__).parent.parent / "duo_games"


class DuoGameConfigError(ValueError):
    """Raised for any schema/reference/graph validation failure in a
    DuoGameConfig — distinguishes config-authoring bugs from ordinary
    ValueErrors raised elsewhere in the RL Lab."""


# ---------------------------------------------------------------------------
# Seats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeatConfig:
    """One seat at the table.

    Attributes
    ----------
    id : str
        Stable, human-readable identifier used throughout the YAML
        (node.seat, terminal.folder, dealer_seat) — e.g. "sb", "bb",
        "oop", "ip".
    acts_first : bool
        Exactly one seat across the whole config must have this True —
        it is the seat GenericTreeEnv.reset() hands the first
        observation to (see DuoGameConfig.first_seat_id).
    index : int
        0-based position, assigned by DuoGameConfig.__post_init__ from
        list order in the YAML's `seats:` block. This is the value
        every seat-keyed dict/tuple elsewhere in the RL Lab (rewards,
        RolloutBuffers, hole_masks, ...) is indexed by.
    """
    id: str
    acts_first: bool = False
    index: int = -1  # filled in by DuoGameConfig.__post_init__


# ---------------------------------------------------------------------------
# Stack / Pot / Deal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StackConfig:
    """
    mode : "fixed_bb" | "randomized_bb"
        fixed_bb   uses `stack_bb` for every seat, every episode.
        randomized_bb draws one value in [min_stack_bb, max_stack_bb]
                   per episode, used for EVERY seat that episode (see
                   module docstring — stacks are never asymmetric here).
    """
    mode: str
    stack_bb: float = 0.0
    min_stack_bb: float = 0.0
    max_stack_bb: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("fixed_bb", "randomized_bb"):
            raise DuoGameConfigError(
                f"stack.mode must be 'fixed_bb' or 'randomized_bb', got {self.mode!r}"
            )
        if self.mode == "fixed_bb":
            if self.stack_bb <= 0:
                raise DuoGameConfigError(
                    f"stack.stack_bb must be > 0 for mode 'fixed_bb', got {self.stack_bb}"
                )
        else:
            if self.min_stack_bb <= 0:
                raise DuoGameConfigError(
                    f"stack.min_stack_bb must be > 0, got {self.min_stack_bb}"
                )
            if self.max_stack_bb < self.min_stack_bb:
                raise DuoGameConfigError(
                    f"stack.max_stack_bb ({self.max_stack_bb}) must be >= "
                    f"min_stack_bb ({self.min_stack_bb})"
                )


@dataclass(frozen=True)
class PotConfig:
    """
    mode : "blinds" | "randomized_bb"
        blinds         seat index 0 posts small_blind, seat index 1
                       posts big_blind (both clamped to that seat's
                       stack), every other seat posts 0. No dead pot.
        randomized_bb  the entire starting pot is dead money in
                       [min_pot_bb, max_pot_bb] * big_blind, attributed
                       to no seat (a mid-street scenario like river-duo
                       where earlier streets' action already happened).
    """
    mode: str
    min_pot_bb: float = 0.0
    max_pot_bb: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in ("blinds", "randomized_bb"):
            raise DuoGameConfigError(
                f"pot.mode must be 'blinds' or 'randomized_bb', got {self.mode!r}"
            )
        if self.mode == "randomized_bb":
            if self.min_pot_bb <= 0:
                raise DuoGameConfigError(
                    f"pot.min_pot_bb must be > 0, got {self.min_pot_bb}"
                )
            if self.max_pot_bb < self.min_pot_bb:
                raise DuoGameConfigError(
                    f"pot.max_pot_bb ({self.max_pot_bb}) must be >= "
                    f"min_pot_bb ({self.min_pot_bb})"
                )


@dataclass(frozen=True)
class DealConfig:
    """
    street_index : int
        Engine street index this tree begins on (0=preflop, ..., see
        BettingObs.street_index / HistoryEventObs.street_index).
    hole_cards_per_player : int
        Defaults to 2 (hold'em).
    node_count : int
        Total board node slots (0 for a boardless game like
        push-fold-duo). Every DealNodeConfig.node_indices must be a
        subset of range(node_count).
    deal_board : bool
        Whether a board is dealt at all this episode.
    reveal_at_start : bool
        If True (default), every dealt board card is visible from the
        very first observation — correct for single-street games
        (river_duo: the flop+turn+river are all "already happened").
        If False, only cards revealed by a DealNodeConfig actually
        passed through are visible (turn_river_duo: flop+turn revealed
        immediately via a start-node deal, river revealed later only if
        both players check the turn).
    """
    street_index: int
    hole_cards_per_player: int = 2
    node_count: int = 0
    deal_board: bool = False
    reveal_at_start: bool = True

    def __post_init__(self) -> None:
        if self.street_index < 0:
            raise DuoGameConfigError(f"deal.street_index must be >= 0, got {self.street_index}")
        if self.hole_cards_per_player < 1:
            raise DuoGameConfigError(
                f"deal.hole_cards_per_player must be >= 1, got {self.hole_cards_per_player}"
            )
        if self.node_count < 0:
            raise DuoGameConfigError(f"deal.node_count must be >= 0, got {self.node_count}")
        if self.deal_board and self.node_count == 0:
            raise DuoGameConfigError(
                "deal.deal_board is True but deal.node_count is 0 — nothing to deal"
            )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeConfig:
    """A Player Decision Node: one seat, a fixed menu of legal actions."""
    id: str
    seat: str
    actions: Dict[ActionType, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DealNodeConfig:
    """
    An automatic Chance/Engine-Transition Node: reveals `node_indices`
    of the board, then falls through to `next` with no decision made.
    """
    id: str
    node_indices: List[int] = field(default_factory=list)
    next: str = ""


AnyNode = Union[NodeConfig, DealNodeConfig]


# ---------------------------------------------------------------------------
# Terminals
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TerminalConfig:
    """
    type : "fold" | "showdown"
        fold      one or more seats in `folders` have folded; every
                  OTHER seat splits the pot evenly (see
                  DuoGameConfig.fold_winners).
        showdown  every seat is still live; GenericTreeEnv resolves the
                  pot via its scoring_engine. `folders` must be empty.
    """
    id: str
    type: str
    folders: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.type not in ("fold", "showdown"):
            raise DuoGameConfigError(
                f"terminal {self.id!r}: type must be 'fold' or 'showdown', got {self.type!r}"
            )
        if self.type == "fold" and not self.folders:
            raise DuoGameConfigError(
                f"terminal {self.id!r}: type 'fold' requires at least one folder"
            )
        if self.type == "showdown" and self.folders:
            raise DuoGameConfigError(
                f"terminal {self.id!r}: type 'showdown' must not declare folders "
                f"(got {self.folders})"
            )


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DuoGameConfig:
    """
    A fully validated, compiled fixed-action-tree N-seat mini-game.

    See module docstring for the full field-by-field semantics; this
    class's __post_init__ is where every cross-reference (seat ids,
    node ids, terminal ids, action names) gets checked.
    """
    name: str
    game_name: str
    small_blind: int
    big_blind: int
    seats: List[SeatConfig]
    stack: StackConfig
    pot: PotConfig
    deal: DealConfig
    start_node: str
    nodes: Dict[str, AnyNode]
    terminals: Dict[str, TerminalConfig]
    dealer_seat: Optional[str] = None
    description: str = ""

    _seat_index_map: Dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.small_blind <= 0:
            raise DuoGameConfigError(f"small_blind must be > 0, got {self.small_blind}")
        if self.big_blind <= self.small_blind:
            raise DuoGameConfigError(
                f"big_blind ({self.big_blind}) must be > small_blind ({self.small_blind})"
            )
        if not self.seats:
            raise DuoGameConfigError("config must declare at least one seat")

        # -- assign seat indices, check uniqueness ------------------------
        seen_ids = set()
        indexed_seats = []
        for i, s in enumerate(self.seats):
            if s.id in seen_ids:
                raise DuoGameConfigError(f"duplicate seat id {s.id!r}")
            seen_ids.add(s.id)
            indexed_seats.append(SeatConfig(id=s.id, acts_first=s.acts_first, index=i))
        object.__setattr__(self, "seats", indexed_seats)
        object.__setattr__(
            self, "_seat_index_map", {s.id: s.index for s in indexed_seats}
        )

        # -- exactly one acts_first seat -----------------------------------
        first_seats = [s.id for s in self.seats if s.acts_first]
        if len(first_seats) != 1:
            raise DuoGameConfigError(
                f"exactly one seat must have acts_first: True, got {first_seats}"
            )

        # -- dealer_seat references a real seat ------------------------------
        if self.dealer_seat is not None and self.dealer_seat not in seen_ids:
            raise DuoGameConfigError(
                f"dealer_seat {self.dealer_seat!r} is not a declared seat id"
            )

        # -- start_node exists ------------------------------------------------
        if self.start_node not in self.nodes:
            raise DuoGameConfigError(
                f"start_node {self.start_node!r} is not a declared node"
            )

        all_target_ids = set(self.nodes) | set(self.terminals)

        # -- validate every node ------------------------------------------------
        for node_id, node in self.nodes.items():
            if node_id != node.id:
                raise DuoGameConfigError(
                    f"node key {node_id!r} does not match node.id {node.id!r}"
                )
            if isinstance(node, NodeConfig):
                if node.seat not in seen_ids:
                    raise DuoGameConfigError(
                        f"node {node_id!r}: seat {node.seat!r} is not a declared seat id"
                    )
                if not node.actions:
                    raise DuoGameConfigError(
                        f"node {node_id!r}: must declare at least one action"
                    )
                for atype, target in node.actions.items():
                    if target not in all_target_ids:
                        raise DuoGameConfigError(
                            f"node {node_id!r}: action {atype.name} targets "
                            f"unknown node/terminal {target!r}"
                        )
            elif isinstance(node, DealNodeConfig):
                if node.next not in all_target_ids:
                    raise DuoGameConfigError(
                        f"deal node {node_id!r}: next {node.next!r} is not a "
                        f"known node/terminal"
                    )
                for idx in node.node_indices:
                    if not (0 <= idx < self.deal.node_count):
                        raise DuoGameConfigError(
                            f"deal node {node_id!r}: node_indices contains {idx}, "
                            f"out of range [0, {self.deal.node_count})"
                        )
            else:
                raise DuoGameConfigError(
                    f"node {node_id!r}: unrecognised node type {type(node)!r}"
                )

        # -- validate every terminal ---------------------------------------------
        for term_id, term in self.terminals.items():
            if term_id != term.id:
                raise DuoGameConfigError(
                    f"terminal key {term_id!r} does not match terminal.id {term.id!r}"
                )
            for folder in term.folders:
                if folder not in seen_ids:
                    raise DuoGameConfigError(
                        f"terminal {term_id!r}: folder {folder!r} is not a "
                        f"declared seat id"
                    )

        # -- no unbounded deal-only cycle ---------------------------------------
        # Walk every node's outgoing chain of DealNodeConfig-only hops; if
        # we revisit a node before reaching a decision node or terminal,
        # that's a cycle with no way out (GenericTreeEnv's own
        # _advance_through_deal_nodes would infinite-loop).
        for node_id in self.nodes:
            visited = set()
            cur = node_id
            while cur in self.nodes and isinstance(self.nodes[cur], DealNodeConfig):
                if cur in visited:
                    raise DuoGameConfigError(
                        f"cycle detected among deal-only nodes starting at "
                        f"{node_id!r} (revisits {cur!r}) — no decision or "
                        f"terminal is ever reached"
                    )
                visited.add(cur)
                cur = self.nodes[cur].next

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def first_seat_id(self) -> str:
        """The seat id with acts_first=True — exactly one is guaranteed
        by __post_init__."""
        return next(s.id for s in self.seats if s.acts_first)

    def seat_index(self, seat_id: str) -> int:
        """0-based index for a seat id, matching SeatConfig.index."""
        try:
            return self._seat_index_map[seat_id]
        except KeyError:
            raise DuoGameConfigError(f"unknown seat id {seat_id!r}") from None

    def seat_id_for_index(self, index: int) -> str:
        """
        Reverse of seat_index(): the seat id string for a 0-based seat
        index (SeatConfig.index). Needed anywhere code has a seat index
        (e.g. from GenericTreeEnv.current_seat, or a rewards dict keyed
        by seat index) and wants the human-readable id back for
        logging — the inverse lookup seat_index() alone can't provide.
        """
        for s in self.seats:
            if s.index == index:
                return s.id
        raise DuoGameConfigError(f"unknown seat index {index!r}") from None

    def is_terminal(self, node_id: str) -> bool:
        """True iff node_id names a terminal (not a node)."""
        return node_id in self.terminals

    def fold_winners(self, terminal_id: str) -> List[str]:
        """
        Seat ids that split the pot at a 'fold' terminal — every seat
        NOT listed in that terminal's `folders`.
        """
        term = self.terminals[terminal_id]
        if term.type != "fold":
            raise DuoGameConfigError(
                f"fold_winners() called on non-fold terminal {terminal_id!r} "
                f"(type={term.type!r})"
            )
        folded = set(term.folders)
        return [s.id for s in self.seats if s.id not in folded]


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------

def _parse_actions(raw_actions: dict, node_id: str) -> Dict[ActionType, str]:
    parsed: Dict[ActionType, str] = {}
    for name, target in raw_actions.items():
        try:
            atype = ActionType[name]
        except KeyError:
            raise DuoGameConfigError(
                f"node {node_id!r}: unknown action name {name!r} "
                f"(not a member of actions.abstract_action.ActionType)"
            ) from None
        parsed[atype] = target
    return parsed


def _parse_node(node_id: str, raw: dict) -> AnyNode:
    if raw.get("type") == "deal":
        return DealNodeConfig(
            id=node_id,
            node_indices=list(raw.get("node_indices", [])),
            next=raw["next"],
        )
    if "seat" not in raw:
        raise DuoGameConfigError(
            f"node {node_id!r}: must declare either 'seat' (decision node) "
            f"or 'type: deal' (deal node)"
        )
    return NodeConfig(
        id=node_id,
        seat=raw["seat"],
        actions=_parse_actions(raw.get("actions", {}), node_id),
    )


def _parse_terminal(term_id: str, raw: dict) -> TerminalConfig:
    folders = raw.get("folders")
    if folders is None:
        folder = raw.get("folder")
        folders = [folder] if folder else []
    return TerminalConfig(id=term_id, type=raw["type"], folders=list(folders))


def parse_duo_game_config(data: dict) -> DuoGameConfig:
    """
    Parse an already-loaded YAML dict (e.g. from yaml.safe_load) into a
    validated DuoGameConfig. Exposed separately from load_duo_game() so
    tests can construct a config from an in-memory dict without touching
    the filesystem.
    """
    seats = [
        SeatConfig(id=s["id"], acts_first=bool(s.get("acts_first", False)))
        for s in data["seats"]
    ]

    stack_raw = data["stack"]
    stack = StackConfig(
        mode=stack_raw["mode"],
        stack_bb=float(stack_raw.get("stack_bb", 0.0)),
        min_stack_bb=float(stack_raw.get("min_stack_bb", 0.0)),
        max_stack_bb=float(stack_raw.get("max_stack_bb", 0.0)),
    )

    pot_raw = data["pot"]
    pot = PotConfig(
        mode=pot_raw["mode"],
        min_pot_bb=float(pot_raw.get("min_pot_bb", 0.0)),
        max_pot_bb=float(pot_raw.get("max_pot_bb", 0.0)),
    )

    deal_raw = data["deal"]
    deal = DealConfig(
        street_index=int(deal_raw["street_index"]),
        hole_cards_per_player=int(deal_raw.get("hole_cards_per_player", 2)),
        node_count=int(deal_raw.get("node_count", 0)),
        deal_board=bool(deal_raw.get("deal_board", False)),
        reveal_at_start=bool(deal_raw.get("reveal_at_start", True)),
    )

    nodes: Dict[str, AnyNode] = {
        node_id: _parse_node(node_id, raw)
        for node_id, raw in data.get("nodes", {}).items()
    }
    terminals: Dict[str, TerminalConfig] = {
        term_id: _parse_terminal(term_id, raw)
        for term_id, raw in data.get("terminals", {}).items()
    }

    return DuoGameConfig(
        name=data["name"],
        description=data.get("description", ""),
        game_name=data["game_name"],
        small_blind=int(data["small_blind"]),
        big_blind=int(data["big_blind"]),
        seats=seats,
        dealer_seat=data.get("dealer_seat"),
        stack=stack,
        pot=pot,
        deal=deal,
        start_node=data["start_node"],
        nodes=nodes,
        terminals=terminals,
    )



def load_duo_game(name: str, path: str | Path | None = None) -> DuoGameConfig:
    """
    Load and validate a duo-game configuration.

    Parameters
    ----------
    name : str
        File stem, e.g. "push_fold_duo", "river_duo", "turn_river_duo".
        Ignored when ``path`` is provided.
    path : str | Path | None, optional
        Explicit path to a YAML configuration file. If omitted, the
        configuration is loaded from
        ``scenarios/duo_games/{name}.yaml``.

    Returns
    -------
    DuoGameConfig
    """
    config_path = Path(path) if path is not None else BASE_PATH / f"{name}.yaml"

    if not config_path.exists():
        raise DuoGameConfigError(f"no duo-game config found at {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return parse_duo_game_config(data)

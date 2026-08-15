"""
envs/generic_tree_env.py — N-seat, config-driven executor for
DuoGameConfig (scenarios/duo_game_config.py) fixed-action-tree
mini-games.

This is the "GenericTreeEnv" scenarios/duo_game_config.py's own
docstring already refers to as scenarios/generic_tree_env.py's future
home. It interprets ANY validated DuoGameConfig — push-fold-duo,
river-duo, or a new N-seat YAML under scenarios/duo_games/ — without
per-game Python code. push_fold_duo_env.py / river_duo_env.py become
special cases of this one class once their trees are expressed as
DuoGameConfig YAML.

Step 1 of the sequencing in RL_Lab_Scenario_Framework - Addendum,
Grounding and Migration.md: "Generalize the node graph from 2 seats to
N seats" — this file is the executor half of that step (duo_game_config.py
was the schema half).

Showdown resolution — why an N-seat-native evaluator, not hero_wins()
--------------------------------------------------------------------------
push_fold_duo_env.py / river_duo_env.py both used a 2-seat-only
``hero_wins(hero_mask, villain_mask, board_mask) -> Optional[bool]``
evaluator contract — meaningless for N > 2 seats (there's no single
"villain"). This env instead takes a `scoring_engine` duck-typing
poker_engine.scoring's existing shared surface:
``evaluate(player_masks, board_mask, score_type, showdown_type) ->
List[ScoreResult]`` (poker_engine/scoring/scoring_engine.py,
mock_scoring_engine.py — the SAME interface ShowdownResolver and
envs/engine_scoring_evaluator.py already use). ScoreResult comparison
(`>`, `==`) is already N-player-native, so side-pot resolution below is
a straightforward layering over `evaluate()`'s output — no hero/villain
framing needed anywhere.

Why side-pot layering even for a "no re-raise" tree
--------------------------------------------------------
Every DuoGameConfig built so far uses equal starting stacks (see
duo_game_config.py's StackConfig docstring), so in practice every
showdown has equal commitments and side pots degenerate to one pot.
But nothing in the schema *forces* equal stacks, and getting side-pot
math right once, generically, costs little — so _resolve_showdown()
below builds real side-pot layers (mirroring
poker_engine/state/game_state.py's own build_side_pots()) rather than
assuming heads-up-style single-pot-plus-excess.

Monte Carlo showdown averaging (num_showdown_samples)
-----------------------------------------------------
**THIS WAS THE BUG.** A game like push-fold-duo is dealt with an EMPTY
board (0 known community cards) — every showdown therefore needs a full
5-card random runout before it can be scored at all (see
_complete_board_for_showdown). Previously this env drew exactly ONE
runout per hand and scored it — i.e. every hand's terminal reward was a
single noisy Bernoulli-ish sample of equity, not an estimate of it. That
is enormously higher-variance than what envs.monte_carlo_evaluator.
MonteCarloEquityEvaluator already exists to fix elsewhere in this
codebase (envs/push_fold_env.py's benchmark path) — but nothing in this
file ever called it. High-variance advantage estimates are exactly what
slows down PPO's convergence toward the Nash-shaped hand-range grids
(pairs/aces/broadways pushing/calling more) — the policy has to average
over many more hands before the useful gradient signal rises above the
runout noise floor.

_resolve_showdown() below now draws `num_showdown_samples` independent
runouts (only when the board is actually incomplete — river_duo already
deals all 5 board cards, so `needed <= 0` and this is a no-op, exactly
one score, same as before) and AVERAGES the resulting side-pot payouts.
This is unbiased (mean reward is unchanged — see benchmark_rollout_
collection.py's benchmark_mc_variance for the same before/after
argument applied to push_fold_env.py's older MockEvaluator) and reduces
reward variance by ~num_showdown_samples, at num_showdown_samples times
the per-hand scoring cost (cheap relative to the neural network forward
pass this reward eventually trains).

Default is 30, matching the K used elsewhere in this codebase's own
Monte Carlo evaluator benchmarks as a reasonable variance/cost
trade-off. Pass num_showdown_samples=1 to recover the old single-sample
behaviour exactly (e.g. for tests wanting the cheapest possible
resolution — MockScoringEngine-backed tests are unaffected either way,
since MockScoringEngine ignores the board entirely and always returns
the same ranking regardless of runout).

Why a showdown terminal never has a folded seat to exclude
------------------------------------------------------------
In this schema every branch is authored explicitly: a path that
includes a FOLD action leads to a "fold" terminal, never a "showdown"
terminal (see duo_game_config.py's TerminalConfig — a config where a
fold branch targets a showdown terminal is a config bug, not something
this env needs to defend against generically). So _resolve_showdown()
can safely assume every seat is still live.

What this does NOT do yet
-----------------------------
No scenario generator is provided here — GenericTreeScenario below is
the input shape a future GenericTreeScenarioGenerator must produce (the
next step in the sequencing). push_fold_duo_env.py / river_duo_env.py
and their training loops are NOT deleted or rewired to this class yet;
that migration happens once the scenario generator exists and
duo_training_loop.py / river_duo_training_loop.py are generalized to
call config.seat_index()-keyed seats instead of SB_SEAT/BB_SEAT /
OOP_SEAT/IP_SEAT constants.

Dependencies
------------
    scenarios.duo_game_config, envs.observations, actions.abstract_action
    (pure Python — no poker_eval needed to import this file). poker_eval
    / mock_poker_eval only touched at construction time, for the default
    ScoreType.HIGH / ShowdownType.HOLDEM values — same shim pattern as
    every other entry point in this codebase.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

try:
    import poker_rl_lab.poker_eval as poker_eval # noqa: F401  (prefer the real extension if present)
except ImportError:
    import poker_rl_lab.mock_poker_eval as mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)
    import poker_eval  # noqa: F401  (now resolves to the shim)

from poker_rl_lab.scenarios.duo_game_config import DuoGameConfig
from poker_rl_lab.envs.observations import (
    BettingObs,
    BoardObs,
    HistoryEventObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)
from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS
from poker_rl_lab.scenarios.duo_game_config import DuoGameConfig, DealNodeConfig

def _mask_to_cards(mask: int) -> Tuple[int, ...]:
    return tuple(i for i in range(52) if (mask >> i) & 1)


def _default_scoring_engine():
    """Prefer the real Engine scoring path; fall back to the documented
    mock — identical pattern to engine_scoring_evaluator.py's own
    _default_scoring_engine()."""
    try:
        from poker_engine.scoring.scoring_engine import CppScoringEngine

        return CppScoringEngine()
    except ImportError:
        from poker_engine.scoring.mock_scoring_engine import MockScoringEngine

        return MockScoringEngine()


# ---------------------------------------------------------------------------
# Scenario input shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenericTreeScenario:
    """
    Initial conditions for one GenericTreeEnv episode.

    Unlike scenarios.base_scenario.BaseScenario (which describes an
    arbitrary mid-hand Engine state for the standard single-hero envs),
    this only carries what a fixed-tree config's own scenario generator
    needs to produce: per-seat hole cards / starting stacks / chips
    already posted before the tree starts, and an optional dealt board.

    Attributes
    ----------
    config_name : str
        Must equal the DuoGameConfig.name this scenario targets — a
        cheap sanity check against feeding a push-fold-duo scenario
        into a river-duo env or vice versa.
    hole_masks : Tuple[int, ...]
        One 52-bit hand bitmask per seat, indexed by seat index.
    board_mask : int
        0 if config.deal.deal_board is False (no board this game).
    node_cards : Tuple[Optional[int], ...]
        Board cards by node index, for StructuredObservation.board —
        empty tuple if config.deal.deal_board is False.
    starting_stacks : Tuple[int, ...]
        Each seat's stack BEFORE any of `posted` is subtracted, one per
        seat, indexed by seat index.
    posted : Tuple[int, ...]
        Chips each seat has already committed before the first decision
        (blinds for a "blinds"-pot config; 0 for every seat in a
        "randomized_bb"-pot config, since that pot mode represents
        already-in-the-middle chips with no per-seat attribution — see
        duo_game_config.py's PotConfig docstring).
    dealer_position : int
        Kept for parity with BaseScenario; GenericTreeEnv actually
        derives dealer_position from config.dealer_seat at every
        observation instead (see _build_obs), so this field is
        currently unused by the env but reserved for a future generator
        that wants per-episode button rotation.
    """
    config_name: str
    hole_masks: Tuple[int, ...]
    board_mask: int = 0
    node_cards: Tuple[Optional[int], ...] = ()
    starting_stacks: Tuple[int, ...] = ()
    posted: Tuple[int, ...] = ()
    pot: int = 0
    dealer_position: int = 0

    def __post_init__(self) -> None:
        n = len(self.hole_masks)
        if len(self.starting_stacks) != n:
            raise ValueError(
                f"starting_stacks length ({len(self.starting_stacks)}) must "
                f"match hole_masks length ({n})"
            )
        if len(self.posted) != n:
            raise ValueError(
                f"posted length ({len(self.posted)}) must match "
                f"hole_masks length ({n})"
            )
        for i in range(n):
            if self.posted[i] > self.starting_stacks[i]:
                raise ValueError(
                    f"seat {i}: posted ({self.posted[i]}) exceeds "
                    f"starting_stacks ({self.starting_stacks[i]})"
                )


@dataclass(frozen=True)
class TreeStepResult:
    """
    Result of one GenericTreeEnv.step() call.

    Mirrors push_fold_duo_env.DuoStepResult / river_duo_env's own step
    result shape, generalized from SB_SEAT/BB_SEAT-keyed (or
    OOP_SEAT/IP_SEAT-keyed) rewards to arbitrary seat-INDEX keys.

    Attributes
    ----------
    obs : StructuredObservation, optional
        Observation for the NEXT acting seat. None iff done is True.
    rewards : Dict[int, float]
        Chip change per seat index. Empty ({}) on any intermediate
        (non-terminal) step; has one entry per seat on the terminal
        step.
    done : bool
    info : dict
        Diagnostic keys. Intermediate steps: {"node": <next node id>}.
        Terminal steps: {"outcome": "fold"|"showdown", "terminal": id,
        ...}.
    """
    obs: Optional[StructuredObservation]
    rewards: Dict[int, float]
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GenericTreeEnv:
    """
    Executes any validated DuoGameConfig's fixed action tree for N seats.

    Parameters
    ----------
    config : DuoGameConfig
    scoring_engine : optional
        See module docstring's "Showdown resolution" section. Defaults
        to CppScoringEngine, falling back to MockScoringEngine.
    score_type, showdown_type : optional
        Forwarded to scoring_engine.evaluate(). Default
        poker_eval.ScoreType.HIGH / poker_eval.ShowdownType.HOLDEM —
        every config built on this schema so far is a high-hand
        hold'em showdown.
    num_showdown_samples : int
        Number of independent board runouts averaged per showdown when
        the board is incomplete at terminal time (e.g. push-fold-duo,
        which deals no board at all). See module docstring's "Monte
        Carlo showdown averaging" section — THIS is the fix for slow /
        noisy convergence toward Nash-shaped hand-range grids. Default
        30. Has zero effect (falls back to exactly 1 sample) whenever
        the board is already fully dealt (river_duo and friends), and
        zero effect on outcome (only on wasted repeated calls) when
        scoring_engine is MockScoringEngine, since that mock ignores
        the board.
    """

    #: Bet-type actions this schema's fixed (non-continuous-sizing)
    #: trees actually use. ALL_IN = shove entire stack. BET_100 =
    #: pot-sized bet (river_duo.yaml's only bet action). Any other
    #: bet-type ActionType referenced by a node's YAML is a config
    #: error caught in _apply_bet() below, not silently mis-sized.
    _SUPPORTED_BET_TYPES = (ActionType.ALL_IN, ActionType.BET_100)

    def __init__(
        self,
        config: DuoGameConfig,
        scoring_engine=None,
        score_type=None,
        showdown_type=None,
        runout_seed=None,
        num_showdown_samples: int = 30,
    ) -> None:
        self.config = config
        self._scoring_engine = scoring_engine or _default_scoring_engine()
        self._score_type = (
            score_type if score_type is not None else poker_eval.ScoreType.HIGH
        )
        self._showdown_type = (
            showdown_type if showdown_type is not None else poker_eval.ShowdownType.HOLDEM
        )
        if num_showdown_samples < 1:
            raise ValueError(
                f"num_showdown_samples must be >= 1, got {num_showdown_samples}"
            )
        self._num_showdown_samples = int(num_showdown_samples)

        n = len(config.seats)
        self._scenario: Optional[GenericTreeScenario] = None
        self._done: bool = True
        self.current_seat: int = -1
        self._current_node: str = ""

        self._hole_masks: List[int] = [0] * n
        self._board_mask: int = 0
        self._node_cards: Tuple[Optional[int], ...] = ()
        self._starting_stack: List[int] = [0] * n
        self._committed: List[int] = [0] * n   # total chips committed so far
        self._pot_carry: int = 0
        self._stack: List[int] = [0] * n        # remaining (uncommitted) stack
        self._is_all_in: List[bool] = [False] * n
        self._history: List[HistoryEventObs] = []
        self._node_cards_full: Tuple[Optional[int], ...] = ()
        self._revealed_nodes: set = set()

        self._runout_rng = random.Random(runout_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, scenario: GenericTreeScenario) -> StructuredObservation:
        n = len(self.config.seats)
        if scenario.config_name != self.config.name:
            raise ValueError(
                f"scenario.config_name {scenario.config_name!r} does not "
                f"match this env's config {self.config.name!r}"
            )
        if len(scenario.hole_masks) != n:
            raise ValueError(
                f"scenario has {len(scenario.hole_masks)} seats' hole "
                f"cards, config {self.config.name!r} has {n} seats"
            )

        self._scenario = scenario
        self._done = False
        self._current_node = self.config.start_node
        self.current_seat = self.config.seat_index(self.config.first_seat_id)
        self._history = []

        self._hole_masks = list(scenario.hole_masks)
        self._board_mask = scenario.board_mask
        self._node_cards = scenario.node_cards

        self._node_cards_full = scenario.node_cards
        self._revealed_nodes = (
            set(range(len(scenario.node_cards)))
            if self.config.deal.reveal_at_start
            else set()
        )
        self._starting_stack = list(scenario.starting_stacks)
        self._committed = list(scenario.posted)
        self._pot_carry = scenario.pot
        self._stack = [
            scenario.starting_stacks[i] - scenario.posted[i] for i in range(n)
        ]
        self._is_all_in = [self._stack[i] == 0 for i in range(n)]

        resolved = self._advance_through_deal_nodes(self.config.start_node)
        if self.config.is_terminal(resolved):
            raise RuntimeError(
                f"config {self.config.name!r}: start_node "
                f"{self.config.start_node!r} leads directly to terminal "
                f"{resolved!r} via deal nodes with no decision in between "
                f"— degenerate tree."
            )
        self._current_node = resolved
        decision_node = self.config.nodes[resolved]
        self.current_seat = self.config.seat_index(decision_node.seat)
        return self._build_obs()

    def step(self, action: AbstractAction) -> TreeStepResult:
        if self._done or self._scenario is None:
            raise RuntimeError(
                "Call reset() before step(), or the episode has already ended."
            )

        node = self.config.nodes[self._current_node]
        atype = action.action_type

        if atype not in node.actions:
            raise ValueError(
                f"Illegal action {atype.name} at node {node.id!r} "
                f"(legal: {[a.name for a in node.actions]})"
            )

        target = node.actions[atype]
        acting_seat = self.current_seat
        pot_before = sum(self._committed)

        if atype == ActionType.FOLD:
            self._record_history(acting_seat, atype, pot_before, amount=0, is_aggressor=False)
        elif atype == ActionType.CALL:
            amount = self._apply_call(acting_seat)
            self._record_history(acting_seat, atype, pot_before, amount=amount, is_aggressor=False)
        elif atype == ActionType.CHECK:
            self._record_history(acting_seat, atype, pot_before, amount=0, is_aggressor=False)
        elif atype.is_bet:
            amount = self._apply_bet(acting_seat, atype)
            self._record_history(acting_seat, atype, pot_before, amount=amount, is_aggressor=True)
        else:
            raise ValueError(f"Unsupported action type in generic tree: {atype}")

        resolved = self._advance_through_deal_nodes(target)

        if self.config.is_terminal(resolved):
            return self._resolve_terminal(resolved)

        next_node = self.config.nodes[resolved]
        self._current_node = resolved
        self.current_seat = self.config.seat_index(next_node.seat)
        return TreeStepResult(
            obs=self._build_obs(), rewards={}, done=False,
            info={"node": resolved},
        )

    # ------------------------------------------------------------------
    # Chip application
    # ------------------------------------------------------------------

    def _apply_call(self, seat: int) -> int:
        """
        Match the largest total commitment among all OTHER seats. This
        schema has no re-raises, so at any CALL decision there is
        exactly one live bet size on the table to match — the max
        already committed by anyone else on this path.
        """
        to_match = max(
            (self._committed[i] for i in range(len(self._committed)) if i != seat),
            default=0,
        )
        to_call = max(to_match - self._committed[seat], 0)
        amount = min(to_call, self._stack[seat])
        self._committed[seat] += amount
        self._stack[seat] -= amount
        if self._stack[seat] == 0:
            self._is_all_in[seat] = True
        return amount

    def _apply_bet(self, seat: int, atype: ActionType) -> int:
        if atype not in self._SUPPORTED_BET_TYPES:
            raise ValueError(
                f"GenericTreeEnv only supports {[a.name for a in self._SUPPORTED_BET_TYPES]} "
                f"as bet actions (fixed-tree games have no continuous "
                f"sizing); got {atype.name}"
            )
        if atype == ActionType.ALL_IN:
            amount = self._stack[seat]
        else:  # BET_100 — pot-sized bet, relative to the FULL pot
            # (dead-money carry + everything committed this street), not
            # just this street's action — otherwise a river-duo bet
            # sized off pre-existing pot money is silently undersized.
            pot = self._pot_carry + sum(self._committed)
            amount = min(pot, self._stack[seat])

        self._committed[seat] += amount
        self._stack[seat] -= amount
        if self._stack[seat] == 0:
            self._is_all_in[seat] = True
        return amount

    def _advance_through_deal_nodes(self, node_id: str) -> str:
        """
        Walk forward from node_id through any chain of automatic
        DealNodeConfig nodes, revealing their board cards as each is
        passed through, until landing on a decision node or terminal.
        No seat acts and no AbstractAction is consumed for any node
        passed through here — this is the Environment executing the
        state machine's own automatic transitions.
        __post_init__'s deal-only-cycle check guarantees termination
        for any validated DuoGameConfig.
        """
        while not self.config.is_terminal(node_id):
            node = self.config.nodes[node_id]
            if not isinstance(node, DealNodeConfig):
                return node_id
            self._revealed_nodes.update(node.node_indices)
            node_id = node.next
        return node_id

    def _visible_node_cards(self) -> Tuple[Optional[int], ...]:
        """
        Board as currently OBSERVABLE: self._node_cards_full with every
        index not yet passed through by a deal node masked back to
        None. For deal.reveal_at_start=True configs (the default —
        every config before deal nodes existed) self._revealed_nodes is
        the full index set from reset() onward, so this is a no-op and
        behaviour is unchanged.
        """
        return tuple(
            c if i in self._revealed_nodes else None
            for i, c in enumerate(self._node_cards_full)
        )

    def _record_history(
        self, seat: int, atype: ActionType, pot_before: int, amount: int, is_aggressor: bool
    ) -> None:
        self._history.append(
            HistoryEventObs(
                seat=seat,
                action_type=int(atype),
                street_index=self.config.deal.street_index,
                pot_fraction=(amount / pot_before if pot_before > 0 else 0.0),
                is_aggressor=is_aggressor,
            )
        )

    # ------------------------------------------------------------------
    # Terminal resolution
    # ------------------------------------------------------------------

    def _resolve_terminal(self, terminal_id: str) -> TreeStepResult:
        term = self.config.terminals[terminal_id]
        self._done = True

        if term.type == "fold":
            winner_seats = [
                self.config.seat_index(sid) for sid in self.config.fold_winners(terminal_id)
            ]
            rewards = self._split_flat(winner_seats)
            info = {"outcome": "fold", "terminal": terminal_id, "folders": term.folders}
        else:
            rewards = self._resolve_showdown()
            info = {"outcome": "showdown", "terminal": terminal_id}

        return TreeStepResult(obs=None, rewards=rewards, done=True, info=info)

    def _split_flat(self, winner_seats: List[int]) -> Dict[int, float]:
        """
        Fold-terminal payout: the entire dead pot (everything every seat
        has committed) splits evenly across winner_seats. No side-pot
        layering needed here — folded seats simply forfeit what they
        committed, they are never owed anything back. Reduces exactly
        to push_fold_duo_env.py's own sb_fold/bb_fold reward formula
        for the 2-seat case.
        """
        n = len(self.config.seats)
        final_stack = list(self._stack)
        pot = self._pot_carry + sum(self._committed)

        share = pot // len(winner_seats)
        remainder = pot % len(winner_seats)
        for i, seat in enumerate(winner_seats):
            final_stack[seat] += share + (1 if i < remainder else 0)

        return {
            seat: float(final_stack[seat] - self._starting_stack[seat])
            for seat in range(n)
        }

    def _complete_board_for_showdown(self, player_masks: List[int]) -> int:
        """
        Real HOLDEM/HIGH scoring needs >= 5 known cards per player.
        Games with a partial or absent board (push-fold: 0 cards; a
        future flop/turn config: 3-4 cards) need the remainder filled
        with a random runout before scoring — mirrors
        envs.poker_eval_evaluator.PokerEvalEvaluator._complete_board /
        envs.engine_scoring_evaluator.EngineScoringEvaluator._complete_board.
        No-op for river_duo (already 5 board cards).

        Draws a FRESH runout on every call — callers that want a
        Monte-Carlo-averaged showdown (see _resolve_showdown) call this
        multiple times per hand and average the resulting payouts.
        """
        needed = 5 - bin(self._board_mask).count("1")
        if needed <= 0:
            return self._board_mask

        used_mask = self._board_mask
        for m in player_masks:
            used_mask |= m
        available = [c for c in range(52) if not (used_mask >> c) & 1]

        if len(available) < needed:
            raise RuntimeError(
                f"Not enough cards left to complete the board for showdown: "
                f"need {needed}, have {len(available)}."
            )
        runout = self._runout_rng.sample(available, needed)
        mask = self._board_mask
        for c in runout:
            mask |= 1 << c
        return mask

    def _resolve_showdown(self) -> Dict[int, float]:
        """
        Side-pot-aware, Monte-Carlo-averaged showdown payout.

        Side-pot STRUCTURE (which seats are eligible for which pot
        layer, and each layer's chip amount) depends only on
        self._committed — it is identical across every runout sample,
        so it's computed once. Only the WINNER of each pot layer
        depends on the sampled board, so that part is what gets
        resampled/averaged num_showdown_samples times (see module
        docstring's "Monte Carlo showdown averaging" section — this is
        the actual fix for slow-converging hand-range grids).

        When the board is already fully dealt (needed <= 0 inside
        _complete_board_for_showdown, e.g. river_duo), every "sample"
        returns the exact same board, so this degenerates to exactly
        one evaluate() call regardless of num_showdown_samples — no
        wasted work.
        """
        n = len(self.config.seats)
        final_stack_start = list(self._stack)

        contributions = sorted(
            ((i, self._committed[i]) for i in range(n) if self._committed[i] > 0),
            key=lambda x: x[1],
        )
        side_pots: List[Tuple[int, List[int]]] = []
        prev = 0
        remaining = [i for i, _ in contributions]
        for i, amount in contributions:
            if amount > prev:
                layer = amount - prev
                side_pots.append((layer * len(remaining), list(remaining)))
                prev = amount
            remaining.remove(i)

        if self._pot_carry and side_pots:
            amount, eligible = side_pots[-1]
            side_pots[-1] = (amount + self._pot_carry, eligible)
        elif self._pot_carry:
            side_pots = [(self._pot_carry, list(range(n)))]

        player_masks = [self._hole_masks[i] for i in range(n)]

        # Board is complete (river-style configs) -> exactly 1 sample,
        # regardless of self._num_showdown_samples, since every runout
        # would be identical anyway.
        needed = 5 - bin(self._board_mask).count("1")
        num_samples = self._num_showdown_samples if needed > 0 else 1

        accumulated_payout = [0.0] * n

        for _ in range(num_samples):
            showdown_board_mask = self._complete_board_for_showdown(player_masks)
            scores = self._scoring_engine.evaluate(
                player_masks, showdown_board_mask, self._score_type, self._showdown_type
            )

            for pot_amount, eligible in side_pots:
                best = max(scores[i] for i in eligible)
                winners = [i for i in eligible if scores[i] == best]
                share = pot_amount // len(winners)
                remainder = pot_amount % len(winners)
                for j, seat in enumerate(winners):
                    accumulated_payout[seat] += share + (1 if j < remainder else 0)

        final_stack = [
            final_stack_start[i] + accumulated_payout[i] / num_samples
            for i in range(n)
        ]

        return {
            seat: float(final_stack[seat] - self._starting_stack[seat])
            for seat in range(n)
        }

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    @staticmethod
    def _legal_mask_for_node(node) -> Tuple[bool, ...]:
        mask = [False] * NUM_ACTIONS
        for atype in node.actions:
            mask[atype] = True
        return tuple(mask)

    def _build_obs(self) -> StructuredObservation:
        n = len(self.config.seats)

        players = tuple(
            PlayerObs(
                seat=i,
                stack=self._stack[i],
                current_bet=self._committed[i],
                has_folded=False,
                is_all_in=self._is_all_in[i],
                is_hero=(i == self.current_seat),
            )
            for i in range(n)
        )

        pot = self._pot_carry + sum(self._committed)
        to_match = max(
            (self._committed[i] for i in range(n) if i != self.current_seat),
            default=0,
        )
        bet_to_call = max(to_match - self._committed[self.current_seat], 0)

        node = self.config.nodes[self._current_node]
        dealer_seat_id = self.config.dealer_seat or self.config.first_seat_id
        dealer_position = self.config.seat_index(dealer_seat_id)

        return StructuredObservation(
            hero_seat=self.current_seat,
            hole_cards=HoleCardsObs(cards=_mask_to_cards(self._hole_masks[self.current_seat])),
            board=BoardObs(node_cards=self._visible_node_cards()),
            players=players,
            betting=BettingObs(
                pot=pot,
                bet_to_call=bet_to_call,
                min_raise=self.config.big_blind,
                street_index=self.config.deal.street_index,
                raises_this_street=0,
            ),
            legal_mask=self._legal_mask_for_node(node),
            game_name=self.config.game_name,
            history=HistoryObs(events=tuple(self._history)),
            dealer_position=dealer_position,
            is_terminal=False,
        )
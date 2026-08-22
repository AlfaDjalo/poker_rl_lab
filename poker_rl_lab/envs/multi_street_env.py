"""
envs/multi_street_env.py — Engine-backed, multi-street, re-raise-capable
RL environment. The Phase D piece: unlike envs.generic_tree_env.GenericTreeEnv
(a fixed action TREE authored in YAML — no re-raises, one bet size per
node by construction), this env drives the REAL poker_engine.state.game_state.GameState
through poker_engine.state.poker_state.PokerState, so arbitrary raise
sequences across arbitrary streets are handled by the Engine itself,
not re-implemented here. This is the pattern envs/engine_scoring_evaluator.py's
docstring alludes to ("surfaced by envs/full_holdem_env.py under
multi-street, multi-raise play") — this module is that pattern, built
directly against the primitives already in this codebase
(scenarios.base_scenario.BaseScenario, scenarios.hand_snapshot_adapter,
actions.abstract_action).

Why not extend DuoGameConfig instead
--------------------------------------
A fixed tree needs one YAML node per distinct betting SEQUENCE. Even a
single street with 2 bet sizes and unlimited re-raises has unboundedly
many sequences (bet, raise, re-raise, ...) — DuoGameConfig's schema
(and GenericTreeEnv's "no re-raise, one live bet size to match"
_apply_call) is deliberately scoped away from this. Multi-street play
compounds that per street. So Phase D is a different env SHAPE, not an
extension of GenericTreeEnv's schema.

Contract
--------
Not a BasePokerEnv subclass — envs.base_env.BasePokerEnv.step() returns
a single (obs, reward, done, info) per hero action, which is what this
class provides too, but reset() additionally takes `hero_seat` (which
seat this episode's returned observations/rewards belong to — a
BaseScenario has no fixed hero the way push_fold_env.py/river_env.py's
scenarios do) and a `villain_policies` mapping so every OTHER seat's
decisions are resolved internally between hero decisions, exactly like
push_fold_env.py/river_env.py must already do internally (not shown in
the files available here, but implied by BasePokerEnv's own
single-hero contract).

Action shape assumption (see module docstring above / this file's
header comment) — poker_engine.state.game_state.GameState.apply_action()
only ever reads `action.type` / `action.amount` (no isinstance check),
so `_EngineAction` below is a minimal duck-typed stand-in. Replace with
the real poker_engine.actions.action.Action class if one exists and
something else in the Engine (callbacks, logging) requires the exact
type — this file has no hard dependency on which.

History tracking
------------------
Same gap GenericTreeEnv already had to fill (see history_encoder.py's
own TODO note: neither push_fold_env.py nor river_env.py maintains a
live HistoryObs). GameState itself keeps no action log, so this class
maintains its own list of HistoryEventObs, appended on every applied
action, exactly like GenericTreeEnv._record_history — see
_apply_abstract_action below.

What this does NOT do yet
-----------------------------
No scenario generator for earlier streets is provided here (e.g. a
"start on the flop with random ranges" generator) — this class only
consumes whatever BaseScenario it's given, the same relationship
GenericTreeEnv has to GenericTreeScenarioGenerator. A
FlopStartScenarioGenerator (or similar) producing street_index=1
BaseScenarios is the natural next piece.

Dependencies
------------
    poker_engine.state.poker_state.PokerState, Phase.
    poker_engine.actions.action_type.ActionType (aliased EngineActionType
    — name collision with actions.abstract_action.ActionType, which is
    a DIFFERENT enum: RL Lab's abstract, discretized action space vs.
    the Engine's own FOLD/CHECK/CALL/BET/RAISE/ANTE/BLIND).
    poker_engine.cards.mask.mask_to_card_ids.
    scenarios.base_scenario.BaseScenario, scenarios.hand_snapshot_adapter.scenario_to_snapshot.
    actions.abstract_action (AbstractAction, ActionType, NUM_ACTIONS,
    compute_bet_amount, legal_action_mask).
    envs.observations (StructuredObservation and friends).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

from poker_engine.state.poker_state import PokerState, Phase
from poker_engine.actions.action_type import ActionType as EngineActionType
from poker_engine.cards.mask import mask_to_card_ids

from poker_rl_lab.scenarios.base_scenario import BaseScenario
from poker_rl_lab.scenarios.hand_snapshot_adapter import scenario_to_snapshot
from poker_rl_lab.actions.abstract_action import (
    AbstractAction,
    ActionType,
    compute_bet_amount,
    legal_action_mask,
)
from poker_rl_lab.envs.observations import (
    BettingObs,
    BoardObs,
    HistoryEventObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)


# ---------------------------------------------------------------------------
# Duck-typed Engine action stand-in — see module docstring
# ---------------------------------------------------------------------------

@dataclass
class _EngineAction:
    type: EngineActionType
    amount: int = 0


def _default_scoring_engine():
    """Same fallback chain as every other entry point in this codebase
    (e.g. envs.generic_tree_env._default_scoring_engine)."""
    try:
        from poker_engine.scoring.scoring_engine import CppScoringEngine

        return CppScoringEngine()
    except ImportError:
        from poker_engine.scoring.mock_scoring_engine import MockScoringEngine

        return MockScoringEngine()


# ---------------------------------------------------------------------------
# Module-level core logic (shared by MultiStreetEnv's hero/villain contract
# AND trainers.multi_street_training_loop's self-play collection — see
# that module's own docstring for why self-play needs every seat's
# decisions, not just one hero seat's, which is why this logic lives at
# module level instead of as MultiStreetEnv-only private methods).
# ---------------------------------------------------------------------------

def apply_abstract_action(
    poker_state: PokerState,
    seat: int,
    action: AbstractAction,
    history: list,
) -> None:
    """
    Translate `action` into a real Engine Action, apply it via
    `poker_state.step()`, and append the resulting HistoryEventObs to
    `history` (mutated in place). See module docstring's "Action shape
    assumption" section for _EngineAction.
    """
    g = poker_state.game
    player = g.players[seat]
    atype = action.action_type

    pot_before = g.pot
    street_before = g.street_index

    if atype == ActionType.FOLD:
        engine_action = _EngineAction(type=EngineActionType.FOLD)
        chip_amount = 0
        is_aggressor = False
    elif atype == ActionType.CHECK:
        engine_action = _EngineAction(type=EngineActionType.CHECK)
        chip_amount = 0
        is_aggressor = False
    elif atype == ActionType.CALL:
        to_call = max(g.bet_to_call - player.current_bet, 0)
        chip_amount = min(to_call, player.stack)
        engine_action = _EngineAction(type=EngineActionType.CALL)
        is_aggressor = False
    elif atype.is_bet:
        to_call = max(g.bet_to_call - player.current_bet, 0)
        total = compute_bet_amount(
            atype,
            pot=g.pot,
            stack=player.stack,
            to_call=to_call,
            min_raise=g.min_raise,
            current_bet=player.current_bet,
        )
        chip_amount = total - player.current_bet
        engine_type = EngineActionType.BET if g.bet_to_call == 0 else EngineActionType.RAISE
        engine_action = _EngineAction(type=engine_type, amount=total)
        is_aggressor = True
    else:
        raise ValueError(f"Unsupported abstract action type: {atype}")

    poker_state.step(engine_action)

    history.append(
        HistoryEventObs(
            seat=seat,
            action_type=int(atype),
            street_index=street_before,
            pot_fraction=(chip_amount / pot_before if pot_before > 0 else 0.0),
            is_aggressor=is_aggressor,
        )
    )


def build_obs(poker_state: PokerState, game_name: str, history: list, seat: int) -> StructuredObservation:
    """Build a StructuredObservation for `seat` from poker_state's
    current GameState. `game_name` is the label stored on the
    observation (StructuredObservation.game_name) — pass
    self.game_def.game_name from a MultiStreetEnv, or the equivalent
    from a caller driving PokerState directly (e.g. the self-play
    training loop)."""
    g = poker_state.game
    player = g.players[seat]
    to_call = max(g.bet_to_call - player.current_bet, 0)

    available = [a.name.lower() for a in g.legal_actions()]
    mask = legal_action_mask(
        available=available,
        pot=g.pot,
        stack=player.stack,
        to_call=to_call,
        min_raise=g.min_raise,
        max_raise=player.current_bet + player.stack,
    )

    players = tuple(
        PlayerObs(
            seat=i,
            stack=p.stack,
            current_bet=p.current_bet,
            has_folded=p.has_folded,
            is_all_in=p.is_all_in,
            is_hero=(i == seat),
        )
        for i, p in enumerate(g.players)
    )

    return StructuredObservation(
        hero_seat=seat,
        hole_cards=HoleCardsObs(cards=tuple(mask_to_card_ids(player.hand_mask))),
        board=BoardObs(node_cards=tuple(g.node_cards)),
        players=players,
        betting=BettingObs(
            pot=g.pot,
            bet_to_call=to_call,
            min_raise=g.min_raise,
            street_index=g.street_index,
            raises_this_street=g.raises_this_street,
        ),
        legal_mask=mask,
        game_name=game_name,
        history=HistoryObs(events=tuple(history)),
        dealer_position=g.dealer_position,
        is_terminal=False,
    )


# ---------------------------------------------------------------------------
# Villain policy type
# ---------------------------------------------------------------------------

VillainPolicy = Callable[[StructuredObservation], AbstractAction]


@dataclass(frozen=True)
class PolicyDecision:
    """
    What a TRAINING policy (as opposed to a plain VillainPolicy) returns
    from run_hand() — same three fields RolloutBuffer.add() needs
    besides the observation itself, mirroring
    models.policy_value_network.ActionResult's shape without pulling in
    a torch dependency here.
    """
    action: AbstractAction
    log_prob: float
    value: float


TrainingPolicy = Callable[[StructuredObservation], PolicyDecision]


@dataclass(frozen=True)
class SeatDecision:
    """One recorded decision from run_hand() — obs is the
    StructuredObservation the policy was actually shown; action/log_prob/
    value are copied straight from that policy's PolicyDecision."""
    obs: StructuredObservation
    action: AbstractAction
    log_prob: float
    value: float


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiStreetStepResult:
    obs: Optional[StructuredObservation]
    reward: float
    done: bool
    info: dict


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class MultiStreetEnv:
    """
    Engine-backed, multi-street, re-raise-capable environment for one
    hero seat against N-1 villain seats driven by `villain_policies`.

    Parameters
    ----------
    game_name : str
        Passed to poker_engine.games.loader.load_game() at construction
        — same (game_def, rules) pair used for the entire env's
        lifetime (reset() may be called repeatedly for new hands of the
        SAME game; construct a new MultiStreetEnv for a different game).
    scoring_engine : optional
        Defaults to CppScoringEngine, falling back to MockScoringEngine
        — same chain as every other entry point in this codebase.
    """

    def __init__(self, game_name: str, scoring_engine=None) -> None:
        from poker_engine.games.loader import load_game

        self.game_def, self.rules = load_game(game_name)
        self._scoring_engine = scoring_engine or _default_scoring_engine()

        self._poker_state: Optional[PokerState] = None
        self._hero_seat: int = -1
        self._villain_policies: Dict[int, VillainPolicy] = {}
        self._history: list = []
        self._starting_stacks: Tuple[int, ...] = ()
        self._done: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        scenario: BaseScenario,
        hero_seat: int,
        villain_policies: Dict[int, VillainPolicy],
    ) -> StructuredObservation:
        """
        Start a new episode from `scenario`, playing forward through
        any villain decisions until the hero's first decision point.

        Parameters
        ----------
        scenario : BaseScenario
            May start on any street — street_index=0 (preflop) for
            "earlier street betting," or any later street for a
            river-shaped scenario, exactly like GenericTreeEnv's own
            scenarios but without the fixed-tree restriction.
        hero_seat : int
            Which seat this episode's returned observations/rewards
            belong to.
        villain_policies : Dict[int, VillainPolicy]
            One entry per NON-hero seat in `scenario`. Missing an entry
            for a seat that actually needs to act raises RuntimeError
            from _advance() rather than silently stalling.

        Returns
        -------
        StructuredObservation
            Non-terminal, at the hero's first decision point.
        """
        if not (0 <= hero_seat < scenario.player_count):
            raise ValueError(
                f"hero_seat {hero_seat} out of range for "
                f"{scenario.player_count} players"
            )

        snapshot = scenario_to_snapshot(scenario)
        self._poker_state = PokerState.load_state(
            snapshot, self.game_def, self.rules, self._scoring_engine
        )
        self._hero_seat = hero_seat
        self._villain_policies = villain_policies
        self._history = []
        self._starting_stacks = tuple(p.stack for p in self._poker_state.game.players)
        self._done = False

        obs = self._advance_to_hero_or_terminal()
        if obs is None:
            raise RuntimeError(
                "reset() reached a terminal state before the hero's first "
                "decision — scenario is already resolved (e.g. only one "
                "seat not folded/all-in). Check the scenario generator."
            )
        return obs

    def step(self, action: AbstractAction) -> MultiStreetStepResult:
        """
        Apply the hero's `action`, then play forward through any
        villain decisions until the hero's next decision point or a
        terminal state.
        """
        if self._done or self._poker_state is None:
            raise RuntimeError(
                "Call reset() before step(), or the episode has already ended."
            )

        g = self._poker_state.game
        if g.current_player != self._hero_seat:
            raise RuntimeError(
                f"step() called but current_player ({g.current_player}) is "
                f"not hero_seat ({self._hero_seat}) — internal bookkeeping bug."
            )

        apply_abstract_action(self._poker_state, self._hero_seat, action, self._history)

        obs = self._advance_to_hero_or_terminal()
        if obs is not None:
            return MultiStreetStepResult(obs=obs, reward=0.0, done=False, info={})

        reward = self._hero_reward()
        return MultiStreetStepResult(
            obs=None, reward=reward, done=True, info={"outcome": self._outcome}
        )

    def run_hand(
        self,
        scenario: BaseScenario,
        policies: Dict[int, TrainingPolicy],
    ) -> Tuple[Dict[int, list], Dict[int, float]]:
        """
        Play one full hand from `scenario` to completion, calling
        `policies[seat](obs) -> PolicyDecision` whenever `seat` must
        act — for EVERY seat, no hero/villain distinction. This is the
        entry point self-play training uses (see
        trainers.multi_street_training_loop.collect_multi_street_rollouts)
        instead of reset()/step(), which only ever trains one seat per
        call and discards every other seat's log_prob/value.

        Does not touch self._hero_seat / self._villain_policies / the
        reset()-step() state machine at all — safe to call on a fresh
        MultiStreetEnv instance, or interleaved with reset()/step()
        calls on the SAME instance (each call fully re-initialises
        self._poker_state from `scenario`).

        Parameters
        ----------
        scenario : BaseScenario
        policies : Dict[int, TrainingPolicy]
            One entry per seat in `scenario`. Every seat that must act
            needs an entry, or this raises RuntimeError the same way
            _advance_to_hero_or_terminal() does for a missing
            VillainPolicy.

        Returns
        -------
        (decisions, rewards)
        decisions : Dict[int, List[SeatDecision]]
            One list per seat, chronological, possibly empty for a
            seat that never got to act (e.g. folded to before their
            turn — cannot happen preflop, but a future non-preflop
            start scenario could produce this).
        rewards : Dict[int, float]
            Chip change per seat this hand (final stack - starting
            stack), one entry per seat regardless of whether that seat
            ever acted.
        """
        n = scenario.player_count
        snapshot = scenario_to_snapshot(scenario)
        self._poker_state = PokerState.load_state(
            snapshot, self.game_def, self.rules, self._scoring_engine
        )
        self._history = []
        starting_stacks = tuple(p.stack for p in self._poker_state.game.players)

        decisions: Dict[int, list] = {i: [] for i in range(n)}
        ps = self._poker_state

        while True:
            if ps.phase == Phase.DEAL_BOARD:
                ps.step(None)
                continue
            if ps.phase in (Phase.HAND_COMPLETE, Phase.SHOWDOWN):
                break

            seat = ps.game.current_player
            policy = policies.get(seat)
            if policy is None:
                raise RuntimeError(
                    f"No policy registered for seat {seat}, which must "
                    f"act this hand. `policies` must cover every seat "
                    f"in `scenario`."
                )

            obs = build_obs(ps, self.game_def.game_name, self._history, seat)
            decision = policy(obs)
            decisions[seat].append(
                SeatDecision(
                    obs=obs,
                    action=decision.action,
                    log_prob=decision.log_prob,
                    value=decision.value,
                )
            )
            apply_abstract_action(ps, seat, decision.action, self._history)

        final_stacks = tuple(p.stack for p in ps.game.players)
        rewards = {
            i: float(final_stacks[i] - starting_stacks[i]) for i in range(n)
        }
        return decisions, rewards

    # ------------------------------------------------------------------
    # Internal: drive the engine forward
    # ------------------------------------------------------------------

    def _advance_to_hero_or_terminal(self) -> Optional[StructuredObservation]:
        """
        Repeatedly: auto-advance DEAL_BOARD phases, resolve villain
        decisions via their registered policy, until either the hero
        must act (returns that observation) or the hand is over
        (returns None, with self._done/self._outcome set).
        """
        ps = self._poker_state

        while True:
            if ps.phase == Phase.DEAL_BOARD:
                ps.step(None)
                continue

            if ps.phase in (Phase.HAND_COMPLETE, Phase.SHOWDOWN):
                self._done = True
                self._outcome = "showdown" if ps.phase == Phase.SHOWDOWN else "fold"
                return None

            # Phase.BETTING
            seat = ps.game.current_player
            if seat == self._hero_seat:
                return build_obs(ps, self.game_def.game_name, self._history, seat)

            policy = self._villain_policies.get(seat)
            if policy is None:
                raise RuntimeError(
                    f"No villain policy registered for seat {seat}, which "
                    f"must act this hand. villain_policies passed to "
                    f"reset() must cover every non-hero seat."
                )
            villain_obs = build_obs(ps, self.game_def.game_name, self._history, seat)
            villain_action = policy(villain_obs)
            apply_abstract_action(ps, seat, villain_action, self._history)

    # ------------------------------------------------------------------
    # Internal: reward
    # ------------------------------------------------------------------

    def _hero_reward(self) -> float:
        final_stack = self._poker_state.game.players[self._hero_seat].stack
        return float(final_stack - self._starting_stacks[self._hero_seat])
"""
envs/full_holdem_env.py — Full-table hold'em environment backed by the
REAL Engine (Phase 4 · Step 3).

Per RL_lab_implementation_guide.md's curriculum (§1):

    1. Push/Fold
    2. River Hold'em
    3. Full Hold'em      <-- this file

Why this env is structurally different from PushFoldEnv / RiverEnv
--------------------------------------------------------------------
PushFoldEnv and RiverEnv are deliberately self-contained: they hand-roll
their own tiny slice of betting logic (single decision, or one street of
heads-up betting) so they can run with zero dependency on the Engine or
the poker_eval C++ extension — see each of those files' own docstrings.
That was always a scoped simplification, not the guide's actual
target architecture. §3 ("Environment") is explicit:

    The Environment must never implement: betting rules, hand evaluation,
    showdown logic, legal action generation, variant-specific poker rules.
    These responsibilities belong entirely to the CAP Engine.

FullHoldemEnv is the first environment that honours this literally: every
hand is played out by a real poker_engine.state.poker_state.PokerState
instance. This class's only jobs are (1) translating a BaseScenario into
a HandSnapshot the Engine can load, (2) translating the hero's abstract
action into a concrete Engine Action, (3) driving the Engine forward
(dealing streets, running villain turns) until it's the hero's turn again
or the hand is over, and (4) packaging the Engine's state back into a
StructuredObservation. No betting/turn-order/showdown rule is
reimplemented anywhere in this file.

Hero seat convention
----------------------
Unlike PushFoldEnv/RiverEnv (fixed HERO_SEAT), a full table has no single
canonical hero seat. The hero is whichever seat the scenario's
current_player names at reset() time (for FullHoldemScenarioGenerator,
this is UTG — see that module's docstring) and remains the hero for the
rest of the episode. Every other seat is driven automatically, between
hero decisions, by an injectable per-seat villain policy (defaulting to
river_env.CheckFoldVillainPolicy for every non-hero seat — those policy
classes only consume a legal_mask, so they are reused here unchanged;
see envs/river_env.py).

Engine phase handling
------------------------
PokerState.step() has three phase-dependent behaviours this class must
drive correctly (poker_engine/state/poker_state.py):

  Phase.DEAL_BOARD  — step(action) IGNORES `action` entirely and just
                       deals the next street (or resolves showdown if no
                       streets remain). Must be advanced with a dummy
                       step(None) call — see _advance()'s DEAL_BOARD arm.
  Phase.BETTING     — step(action) applies exactly one seat's action via
                       GameState.apply_action(). Whose turn it is is read
                       off game.current_player; this class must not
                       reimplement or precompute this, only read it.
  Phase.SHOWDOWN /
  Phase.HAND_COMPLETE — terminal. Notably, PokerState.step()'s own
                       SHOWDOWN -> HAND_COMPLETE auto-transition is
                       commented out in the Engine source ("Commenting
                       this out to let GameService control progression"),
                       so step() must NEVER be called again once phase
                       reaches SHOWDOWN — both phases are treated as
                       terminal here without further Engine interaction.
                       Player stacks are already final by the time either
                       phase is reached (payouts happen inside the same
                       step() call that transitions into them).

Legal-action masking
------------------------
_legal_mask_for(seat) builds the abstract action mask via
actions.abstract_action.legal_action_mask(), but critically the
`available` argument comes from calling the real
GameState.legal_actions() and lower-casing the returned Engine
ActionType names — never from a hand-written legality check. This is
what keeps "never duplicate poker rules" true even for masking: a short
stack that can only call-for-less (Engine offers CALL, not RAISE, when
`stack <= to_call`) is reflected automatically because avail comes
straight from the Engine.

poker_eval shim
------------------
Same as push_fold_experiment.py / river_experiment.py: PokerState
transitively imports poker_eval (via showdown_resolver.py /
rules.game_rules / rules.point_definition). Unlike those two experiment
files, this module imports PokerState at MODULE level (not lazily),
since the whole point of this env is being Engine-backed — so the shim
lives here, at the top of this file, executed once at import time.

Dependencies
------------
    scenarios.base_scenario, scenarios.hand_snapshot_adapter,
    envs.base_env, envs.observations, envs.river_env (villain policy
    classes only — no river-specific coupling), actions.abstract_action.
    poker_engine.state.poker_state, poker_engine.actions.action_type,
    poker_engine.actions.action, poker_engine.loader,
    poker_engine.scoring.mock_scoring_engine (default scoring engine).
"""

from __future__ import annotations

import sys

try:
    import poker_eval  # noqa: F401  (prefer the real extension if present)
except ImportError:
    import mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

from typing import Dict, List, Optional, Tuple

from scenarios.base_scenario import BaseScenario
from scenarios.hand_snapshot_adapter import scenario_to_snapshot

from envs.base_env import BasePokerEnv
from envs.observations import (
    BettingObs,
    BoardObs,
    HistoryEventObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)
from envs.river_env import CheckFoldVillainPolicy

from actions.abstract_action import (
    AbstractAction,
    ActionType,
    NUM_ACTIONS,
    compute_bet_amount,
    legal_action_mask,
)

from poker_engine.state.poker_state import PokerState, Phase
from poker_engine.actions.action_type import ActionType as EngineActionType
from poker_engine.actions.action import Action as EngineAction
from poker_engine.games.loader import load_game
from poker_engine.scoring.mock_scoring_engine import MockScoringEngine


# ── Card helpers (no dependency on poker_engine.cards.mask; mirrors the
#    local copy already used in push_fold_env.py / river_env.py) ──────────

def _mask_to_cards(mask: int) -> Tuple[int, ...]:
    return tuple(i for i in range(52) if (mask >> i) & 1)


# ── Villain policy protocol ────────────────────────────────────────────────

class VillainPolicyLike:
    """Documentation-only Protocol: any object with
    ``act(legal_mask: Tuple[bool, ...]) -> AbstractAction``. See
    envs/river_env.py's CheckFoldVillainPolicy / CallAnyVillainPolicy /
    FoldAnyVillainPolicy / RandomVillainPolicy for concrete examples —
    those classes are reused directly by this environment."""


# ── Environment ───────────────────────────────────────────────────────────────

class FullHoldemEnv(BasePokerEnv):
    """
    Full-table hold'em environment, driven end-to-end by the real Engine.

    Parameters
    ----------
    scoring_engine : optional
        Passed straight through to PokerState. Defaults to
        MockScoringEngine (deterministic, no poker knowledge — see that
        module's own docstring). Pass a real evaluator wrapping the
        poker_eval extension for training runs that need genuine hand
        strength.
    default_villain_policy : optional
        Policy used for any non-hero seat not given an explicit override
        in `villain_policies`. Defaults to CheckFoldVillainPolicy()
        (envs/river_env.py), matching RiverEnv's own default.
    villain_policies : Dict[int, VillainPolicyLike], optional
        Per-seat policy overrides, keyed by seat index. Seats not present
        here use `default_villain_policy`.

    Usage
    -----
        gen = FullHoldemScenarioGenerator(player_count=6)
        gen.seed(42)

        env = FullHoldemEnv()
        obs = env.reset(gen.generate())
        obs, reward, done, info = env.step(AbstractAction(ActionType.CALL))
    """

    def __init__(
        self,
        scoring_engine=None,
        default_villain_policy: Optional[VillainPolicyLike] = None,
        villain_policies: Optional[Dict[int, VillainPolicyLike]] = None,
    ) -> None:
        self._scoring_engine = scoring_engine or MockScoringEngine()
        self._default_villain_policy = default_villain_policy or CheckFoldVillainPolicy()
        self._villain_policies: Dict[int, VillainPolicyLike] = dict(villain_policies or {})

        self._game_cache: Dict[str, tuple] = {}

        self._state: Optional[PokerState] = None
        self._hero_seat: int = 0
        self._game_name: str = "holdem"
        self._done: bool = True
        self._initial_hero_stack: int = 0
        self._history: List[HistoryEventObs] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, scenario: BaseScenario) -> StructuredObservation:
        """
        Start a new episode from *scenario*, played through the real Engine.

        Parameters
        ----------
        scenario : BaseScenario
            Typically produced by FullHoldemScenarioGenerator, but any
            BaseScenario valid for `scenario.game_name`'s GameDefinition
            works — this class has no push-fold/river/full-table-specific
            assumptions baked in beyond "hero = scenario.current_player".

        Returns
        -------
        StructuredObservation
            Non-terminal observation for the hero, at the hero's first
            decision point (after any auto-played villain turns / board
            deals the Engine performs before the hero is first on
            action — normally none, since generators hand the hero the
            very first decision, but this env doesn't assume that).
        """
        game_def, rules = self._get_game(scenario.game_name)

        snapshot = scenario_to_snapshot(scenario)
        self._state = PokerState.load_state(
            snapshot, game_def, rules, self._scoring_engine
        )

        self._hero_seat = scenario.current_player
        self._game_name = scenario.game_name
        self._done = False
        self._history = []
        
        # Track the baseline stack before the current round's bets were posted
        hero_player = self._state.game.players[self._hero_seat]
        self._initial_hero_stack = hero_player.stack + hero_player.current_bet

        self._advance()

        return self._build_obs(is_terminal=self._done)

    def step(
        self,
        action: AbstractAction,
    ) -> Tuple[StructuredObservation, float, bool, Dict]:
        """
        Apply the hero's *action*, then let the Engine (and villain
        policies) run forward until the hero's next decision or the hand
        ends.

        Parameters
        ----------
        action : AbstractAction
            Must be legal for the hero (True in the current legal_mask).

        Returns
        -------
        obs : StructuredObservation
        reward : float
            0.0 at intermediate steps; hero's total chip change (final
            stack - stack recorded at reset) once the hand ends.
        done : bool
        info : dict
            Keys: ``pot``, ``street_index``, ``phase``.

        Raises
        ------
        RuntimeError
            If called before reset(), after the episode has ended, or
            when it is not currently the hero's turn.
        """
        if self._done or self._state is None:
            raise RuntimeError("Call reset() before step().")
        if self._state.game.current_player != self._hero_seat:
            raise RuntimeError("It is not the hero's turn.")

        self._apply_and_step(self._hero_seat, action)
        self._advance()

        obs = self._build_obs(is_terminal=self._done)
        reward = self._compute_reward() if self._done else 0.0
        return obs, reward, self._done, self._make_info()

    # ── Engine-driving internals ────────────────────────────────────────────

    def _advance(self) -> None:
        """
        Drive the Engine forward (dealing streets, running villain turns)
        until it is the hero's turn in Phase.BETTING, or the hand has
        reached a terminal phase. Never reads or writes betting/turn-order
        state directly — every decision comes from PokerState/GameState.
        """
        while True:
            phase = self._state.phase

            if phase in (Phase.SHOWDOWN, Phase.HAND_COMPLETE):
                self._done = True
                return

            if phase == Phase.DEAL_BOARD:
                # step()'s DEAL_BOARD branch ignores its argument entirely
                # (see poker_state.py) — deals the next street or resolves
                # showdown if none remain.
                self._state.step(None)
                continue

            if phase == Phase.BETTING:
                g = self._state.game
                if g.current_player == self._hero_seat:
                    return
                self._apply_villain_action(g.current_player)
                continue

            raise RuntimeError(
                f"FullHoldemEnv cannot advance from unexpected engine "
                f"phase {phase!r}; expected DEAL_BOARD, BETTING, "
                f"SHOWDOWN, or HAND_COMPLETE post-reset."
            )

    def _apply_villain_action(self, seat: int) -> None:
        legal_mask = self._legal_mask_for(seat)
        
        if not any(legal_mask):
            # Fallback to prevent crash if the abstract mask is empty
            g = self._state.game
            avail = [a.name.lower() for a in g.legal_actions()]
            if "check" in avail:
                action = AbstractAction(ActionType.CHECK)
            elif "fold" in avail:
                action = AbstractAction(ActionType.FOLD)
            elif "call" in avail:
                action = AbstractAction(ActionType.CALL)
            else:
                action = AbstractAction(ActionType.FOLD)
        else:
            policy = self._villain_policies.get(seat, self._default_villain_policy)
            action = policy.act(legal_mask)
            
        self._apply_and_step(seat, action)

    def _apply_and_step(self, seat: int, action: AbstractAction) -> None:
        """Translate one abstract action for *seat* into a concrete Engine
        Action, record it in history, then hand it to PokerState.step()."""
        g = self._state.game
        player = g.players[seat]
        atype = action.action_type

        pot_before = g.pot
        amount_added = 0
        is_aggressor = False

        if atype == ActionType.FOLD:
            engine_action = EngineAction(type=EngineActionType.FOLD)

        elif atype == ActionType.CHECK:
            engine_action = EngineAction(type=EngineActionType.CHECK)

        elif atype == ActionType.CALL:
            # Always a plain CALL, even when it puts the player all-in for
            # less — that is GameState.apply_action's own CALL semantics
            # (amount = min(to_call, stack)); there is no separate engine
            # action for a short-stack call.
            to_call = min(g.bet_to_call - player.current_bet, player.stack)
            amount_added = to_call
            engine_action = EngineAction(type=EngineActionType.CALL)

        else:  # BET_* / ALL_IN
            to_call = g.bet_to_call - player.current_bet
            amount = compute_bet_amount(
                atype,
                pot=g.pot,
                stack=player.stack,
                to_call=to_call,
                min_raise=g.min_raise,
                current_bet=player.current_bet,
            )
            amount_added = amount - player.current_bet
            is_aggressor = True
            engine_type = (
                EngineActionType.BET if g.bet_to_call == 0 else EngineActionType.RAISE
            )
            engine_action = EngineAction(type=engine_type, amount=amount)

        street_index = g.street_index
        self._state.step(engine_action)

        self._history.append(
            HistoryEventObs(
                seat=seat,
                action_type=int(atype),
                street_index=street_index,
                pot_fraction=(amount_added / pot_before if pot_before > 0 else 0.0),
                is_aggressor=is_aggressor,
            )
        )

    # ── Legal action mask ─────────────────────────────────────────────────────

    def _legal_mask_for(self, seat: int) -> Tuple[bool, ...]:
        """
        Build the abstract action mask for *seat*, grounded entirely in
        the Engine's own GameState.legal_actions() — never reimplemented.
        """
        g = self._state.game
        player = g.players[seat]

        avail = [a.name.lower() for a in g.legal_actions()]
        to_call = g.bet_to_call - player.current_bet
        max_raise = player.stack + to_call

        return tuple(
            legal_action_mask(
                avail, g.pot, player.stack, to_call, g.min_raise, max_raise
            )
        )

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Hero chip change this episode (positive = gain, negative = loss)."""
        final_stack = self._state.game.players[self._hero_seat].stack
        return float(final_stack - self._initial_hero_stack)

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_obs(self, is_terminal: bool) -> StructuredObservation:
        g = self._state.game
        hero = g.players[self._hero_seat]

        legal_mask = (
            tuple(False for _ in range(NUM_ACTIONS))
            if is_terminal
            else self._legal_mask_for(self._hero_seat)
        )

        return StructuredObservation(
            hero_seat=self._hero_seat,
            hole_cards=HoleCardsObs(cards=_mask_to_cards(hero.hand_mask)),
            board=BoardObs(node_cards=tuple(g.node_cards)),
            players=tuple(
                PlayerObs(
                    seat=i,
                    stack=p.stack,
                    current_bet=p.current_bet,
                    has_folded=p.has_folded,
                    is_all_in=p.is_all_in,
                    is_hero=(i == self._hero_seat),
                )
                for i, p in enumerate(g.players)
            ),
            betting=BettingObs(
                pot=g.pot,
                bet_to_call=g.bet_to_call - hero.current_bet,
                min_raise=g.min_raise,
                street_index=g.street_index,
                raises_this_street=g.raises_this_street,
            ),
            legal_mask=legal_mask,
            game_name=self._game_name,
            history=HistoryObs(events=tuple(self._history)),
            dealer_position=g.dealer_position,
            is_terminal=is_terminal,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_game(self, game_name: str) -> tuple:
        """Cache (game_def, rules) per game_name — mirrors
        representation.observation_adapter.ObservationAdapter's own
        caching rationale (avoid re-parsing YAML every episode)."""
        if game_name not in self._game_cache:
            self._game_cache[game_name] = load_game(game_name)
        return self._game_cache[game_name]

    def _make_info(self) -> Dict:
        g = self._state.game
        return {
            "pot": g.pot,
            "street_index": g.street_index,
            "phase": self._state.phase.name,
        }
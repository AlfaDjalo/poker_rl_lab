"""
envs/river_env.py  —  Heads-up river environment  (Phase 4 · Step 2)

Adds multi-step betting on top of the push-fold foundation.

  · Seat 0: in-position (acts second; villain; controlled by VillainPolicy).
  · Seat 1: out-of-position (acts first; hero; the RL agent).
  · Five board cards are already dealt in the scenario.
  · Full action space: FOLD / CHECK / CALL / BET_* / ALL_IN.
  · Episode ends at showdown or when a player folds.

Design
------
  * Still self-contained — no PokerState, no C++ extension.
  * Villain acts automatically inside step() so the hero always receives
    observations only at its own decision points.
  * Injectable evaluator and villain policy keep the class testable and
    replaceable without touching any other code.

Reward convention
-----------------
  Reward = hero final stack − hero starting stack for this episode.

  check/check → lose  :  0     (no river chips at risk)
  check/check → win   :  +pot
  bet → call  → lose  : −bet
  bet → call  → win   :  +pot  (pot includes the bet)
  fold                : −chips committed before folding
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from poker_rl_lab.scenarios.base_scenario import BaseScenario
from poker_rl_lab.envs.observations import (
    BoardObs,
    BettingObs,
    HistoryEventObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)
from poker_rl_lab.actions.abstract_action import (
    AbstractAction,
    ActionType,
    NUM_ACTIONS,
    compute_bet_amount,
)

from poker_rl_lab.envs.base_env import BasePokerEnv

# ── Seat constants ────────────────────────────────────────────────────────────

HERO_SEAT:    int = 1   # OOP; acts first; RL agent
VILLAIN_SEAT: int = 0   # IP; acts second; controlled by policy


# ── Card helpers ──────────────────────────────────────────────────────────────

def _mask_to_cards(mask: int) -> Tuple[int, ...]:
    return tuple(i for i in range(52) if (mask >> i) & 1)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class MockEvaluator:
    """
    Deterministic stub: the player with the larger hand_mask integer wins.
    Replace with a real poker_eval wrapper before training.
    """
    def hero_wins(
        self,
        hero_mask: int,
        villain_mask: int,
        board_mask: int,
    ) -> Optional[bool]:
        """True = hero wins, False = villain wins, None = chop."""
        if hero_mask > villain_mask:
            return True
        if hero_mask < villain_mask:
            return False
        return None


# ── Villain policies ──────────────────────────────────────────────────────────

class CheckFoldVillainPolicy:
    """
    Passive villain: checks when possible, calls if facing a bet, folds
    only if it has no other option. Good for deterministic test scenarios.
    """
    def act(self, legal_mask: Tuple[bool, ...]) -> AbstractAction:
        if legal_mask[ActionType.CHECK]:
            return AbstractAction(ActionType.CHECK)
        if legal_mask[ActionType.CALL]:
            return AbstractAction(ActionType.CALL)
        return AbstractAction(ActionType.FOLD)


class CallAnyVillainPolicy:
    """
    Station villain: always calls, never raises.
    Useful for testing value-bet reward signals.
    """
    def act(self, legal_mask: Tuple[bool, ...]) -> AbstractAction:
        if legal_mask[ActionType.CALL]:
            return AbstractAction(ActionType.CALL)
        if legal_mask[ActionType.CHECK]:
            return AbstractAction(ActionType.CHECK)
        return AbstractAction(ActionType.FOLD)


class FoldAnyVillainPolicy:
    """
    Villain that folds the moment it faces a bet. Never raises.
    Use to test that hero earns a positive reward by bluffing.
    """
    def act(self, legal_mask: Tuple[bool, ...]) -> AbstractAction:
        if legal_mask[ActionType.CHECK]:
            return AbstractAction(ActionType.CHECK)
        return AbstractAction(ActionType.FOLD)


class RandomVillainPolicy:
    """
    Uniform-random villain over legal actions.
    Provides varied opponent behaviour for training.
    """
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def act(self, legal_mask: Tuple[bool, ...]) -> AbstractAction:
        legal = [i for i, m in enumerate(legal_mask) if m]
        return AbstractAction(ActionType(self._rng.choice(legal)))


# ── Environment ───────────────────────────────────────────────────────────────

class RiverEnv(BasePokerEnv):
    """
    Heads-up single-street river environment.

    Parameters
    ----------
    evaluator : optional
        Object with ``hero_wins(hero_mask, villain_mask, board_mask) ->
        Optional[bool]``. Defaults to MockEvaluator.
    villain_policy : optional
        Object with ``act(legal_mask: Tuple[bool, ...]) -> AbstractAction``.
        Defaults to CheckFoldVillainPolicy.
    max_raises : int
        Cap on raises per street. Default 4 (standard limit rules).
        Use a large value (e.g. 100) for effectively uncapped no-limit.

    Usage
    -----
        gen = RiverScenarioGenerator()
        gen.seed(42)

        env = RiverEnv()
        obs = env.reset(gen.generate())
        obs, reward, done, info = env.step(AbstractAction(ActionType.BET_50))
    """

    def __init__(
        self,
        evaluator=None,
        villain_policy=None,
        max_raises: int = 4,
    ) -> None:
        self._evaluator      = evaluator      or MockEvaluator()
        self._villain_policy = villain_policy or CheckFoldVillainPolicy()
        self._max_raises     = max_raises

        self._scenario: Optional[BaseScenario] = None
        self._done: bool = True

        # Mutable per-episode state — initialised in reset()
        self._pot:                 int            = 0
        self._stacks:              List[int]      = [0, 0]
        self._player_bets:         List[int]      = [0, 0]   # this-street bets
        self._hole_masks:          List[int]      = [0, 0]
        self._has_folded:          List[bool]     = [False, False]
        self._is_all_in:           List[bool]     = [False, False]
        self._current_player:      int            = HERO_SEAT
        self._bet_to_call:         int            = 0
        self._min_raise:           int            = 0
        self._raises_this_street:  int            = 0
        self._last_aggressor:      Optional[int]  = None
        self._players_to_act:      set            = set()
        self._node_cards:          tuple          = ()
        self._game_name:           str            = "holdem"
        self._big_blind:           int            = 2
        self._initial_hero_stack:  int            = 0
        self._outcome:             str            = ""
        self._history:             List[HistoryEventObs] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, scenario: BaseScenario) -> StructuredObservation:
        """
        Start a new episode.

        Parameters
        ----------
        scenario : BaseScenario
            Produced by RiverScenarioGenerator. Expects street_index=3,
            bet_to_call=0, and five node_cards.

        Returns
        -------
        StructuredObservation
            Non-terminal observation for the hero (seat 1).
        """
        sc             = scenario
        self._scenario = sc
        self._done     = False
        self._outcome  = ""
        self._history  = []

        self._pot                = sc.pot
        self._stacks             = [p.stack       for p in sc.players]
        self._player_bets        = [p.current_bet  for p in sc.players]
        self._hole_masks         = [p.hand_mask    for p in sc.players]
        self._has_folded         = [p.has_folded   for p in sc.players]
        self._is_all_in          = [p.is_all_in    for p in sc.players]
        self._bet_to_call        = sc.bet_to_call
        self._min_raise          = sc.min_raise
        self._raises_this_street = sc.raises_this_street
        self._last_aggressor     = sc.last_aggressor
        self._node_cards         = sc.node_cards
        self._game_name          = sc.game_name
        self._big_blind          = sc.big_blind
        self._current_player     = sc.current_player
        self._initial_hero_stack = sc.players[HERO_SEAT].stack

        # Seats that still need to make a decision this street
        self._players_to_act = {
            s for s, p in enumerate(sc.players)
            if not p.has_folded and not p.is_all_in
        }

        return self._build_obs(is_terminal=False)

    def step(
        self,
        action: AbstractAction,
    ) -> Tuple[StructuredObservation, float, bool, Dict]:
        """
        Apply hero's *action* and advance the episode.

        The villain acts automatically between hero turns, so the caller
        always receives an observation at a hero decision point or at a
        terminal state.

        Parameters
        ----------
        action : AbstractAction
            Must be legal for the hero (i.e. True in the legal_mask).

        Returns
        -------
        obs : StructuredObservation
        reward : float
            0.0 at intermediate steps; final chip-change at terminal.
        done : bool
        info : dict
            Keys: ``outcome``, ``pot``, ``hero_bets``, ``raises``.

        Raises
        ------
        RuntimeError
            If called before reset() or after the episode has ended.
        """
        if self._done or self._scenario is None:
            raise RuntimeError("Call reset() before step().")
        if self._current_player != HERO_SEAT:
            raise RuntimeError("It is not the hero's turn.")

        # ── Hero acts ─────────────────────────────────────────────────────
        self._apply_action(HERO_SEAT, action)

        if self._has_folded[HERO_SEAT]:
            return self._end_episode("fold_hero")

        if self._is_betting_done():
            return self._resolve_and_end()

        # ── Villain acts until it's hero's turn again, or round ends ──────
        while (
            self._current_player == VILLAIN_SEAT
            and not self._is_betting_done()
        ):
            v_mask   = tuple(self._compute_legal_mask(VILLAIN_SEAT))
            v_action = self._villain_policy.act(v_mask)
            self._apply_action(VILLAIN_SEAT, v_action)

            if self._has_folded[VILLAIN_SEAT]:
                return self._end_episode("fold_villain")

        if self._is_betting_done():
            return self._resolve_and_end()

        # ── Hero must act again (villain raised) ──────────────────────────
        return self._build_obs(is_terminal=False), 0.0, False, self._make_info()

    # ── Action application ────────────────────────────────────────────────────

    def _apply_action(self, seat: int, action: AbstractAction) -> None:
        """Mutate internal state to reflect *action* taken by *seat*."""
        atype = action.action_type
        other = 1 - seat

        pot_before   = self._pot
        amount_added = 0
        is_aggressor = False

        if atype == ActionType.FOLD:
            self._has_folded[seat] = True
            # Nobody else needs to act
            self._players_to_act.discard(seat)
            self._players_to_act.discard(other)

        elif atype == ActionType.CHECK:
            self._players_to_act.discard(seat)
            self._advance_player()

        elif atype == ActionType.CALL:
            to_call = min(
                self._bet_to_call - self._player_bets[seat],
                self._stacks[seat],
            )
            self._stacks[seat]      -= to_call
            self._player_bets[seat] += to_call
            self._pot               += to_call
            if self._stacks[seat] == 0:
                self._is_all_in[seat] = True
            self._players_to_act.discard(seat)
            self._advance_player()
            amount_added = to_call

        else:  # BET_* or ALL_IN
            to_call = self._bet_to_call - self._player_bets[seat]
            amount  = compute_bet_amount(
                atype,
                pot       = self._pot,
                stack     = self._stacks[seat],
                to_call   = to_call,
                min_raise = self._min_raise,
                current_bet = self._player_bets[seat],
            )
            prev_bet = self._bet_to_call
            additional = amount - self._player_bets[seat]

            self._stacks[seat]       -= additional
            self._player_bets[seat]   = amount
            self._pot                += additional
            if self._stacks[seat] == 0:
                self._is_all_in[seat] = True

            self._bet_to_call         = amount
            self._min_raise           = amount - prev_bet
            self._raises_this_street += 1
            self._last_aggressor      = seat

            # Only the other player needs to respond
            if not self._has_folded[other] and not self._is_all_in[other]:
                self._players_to_act = {other}
                self._current_player = other
            else:
                self._players_to_act = set()

            amount_added = additional
            is_aggressor = True

        self._history.append(
            HistoryEventObs(
                seat=seat,
                action_type=int(atype),
                street_index=self._scenario.street_index,
                pot_fraction=(amount_added / pot_before if pot_before > 0 else 0.0),
                is_aggressor=is_aggressor,
            )
        )

    def _advance_player(self) -> None:
        """Rotate current_player to the next eligible (not folded, not all-in) seat."""
        n = len(self._scenario.players)
        for i in range(1, n + 1):
            nxt = (self._current_player + i) % n
            if not self._has_folded[nxt] and not self._is_all_in[nxt]:
                self._current_player = nxt
                return

    # ── Terminal conditions ───────────────────────────────────────────────────

    def _is_betting_done(self) -> bool:
        if any(self._has_folded):
            return True
        return len(self._players_to_act) == 0

    # ── Episode completion ────────────────────────────────────────────────────

    def _end_episode(
        self, outcome: str
    ) -> Tuple[StructuredObservation, float, bool, Dict]:
        self._done    = True
        self._outcome = outcome

        # Award uncontested pot on fold
        if outcome == "fold_villain":
            self._stacks[HERO_SEAT] += self._pot
            self._pot = 0
        elif outcome == "fold_hero":
            self._stacks[VILLAIN_SEAT] += self._pot
            self._pot = 0

        return (
            self._build_obs(is_terminal=True),
            self._compute_reward(),
            True,
            self._make_info(),
        )

    def _resolve_and_end(self) -> Tuple[StructuredObservation, float, bool, Dict]:
        self._do_showdown()
        return self._end_episode("showdown")

    # ── Showdown ──────────────────────────────────────────────────────────────

    def _do_showdown(self) -> None:
        """Award the pot; handle uncalled portion for 2-player all-in side pots."""
        # Return any uncalled chips (one player went all-in for less)
        hero_bet = self._player_bets[HERO_SEAT]
        vill_bet = self._player_bets[VILLAIN_SEAT]
        excess   = abs(hero_bet - vill_bet)
        if excess:
            bigger = HERO_SEAT if hero_bet > vill_bet else VILLAIN_SEAT
            self._stacks[bigger] += excess
            self._pot            -= excess

        board_mask = self._compute_board_mask()
        result = self._evaluator.hero_wins(
            self._hole_masks[HERO_SEAT],
            self._hole_masks[VILLAIN_SEAT],
            board_mask,
        )

        if result is True:
            self._stacks[HERO_SEAT]    += self._pot
        elif result is False:
            self._stacks[VILLAIN_SEAT] += self._pot
        else:  # chop
            half = self._pot // 2
            self._stacks[HERO_SEAT]    += half
            self._stacks[VILLAIN_SEAT] += self._pot - half

        self._pot = 0

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self) -> float:
        """Hero chip change this episode (positive = gain, negative = loss)."""
        return float(self._stacks[HERO_SEAT] - self._initial_hero_stack)

    # ── Legal action mask ─────────────────────────────────────────────────────

    def _compute_legal_mask(self, seat: int) -> List[bool]:
        mask = [False] * NUM_ACTIONS

        if self._has_folded[seat] or self._is_all_in[seat]:
            return mask

        to_call  = self._bet_to_call - self._player_bets[seat]
        stack    = self._stacks[seat]
        max_bet  = stack + self._player_bets[seat]

        mask[ActionType.FOLD] = True

        if to_call == 0:
            mask[ActionType.CHECK] = True
        else:
            # Must at least call (or go all-in for less)
            if stack >= to_call:
                mask[ActionType.CALL] = True
            else:
                # Short stack: only option is all-in partial call
                mask[ActionType.ALL_IN] = True
                return mask

        # Can raise / bet if stack allows and raise cap not reached
        can_aggress = (
            stack > to_call
            and self._raises_this_street < self._max_raises
        )
        if can_aggress:
            for at in (
                ActionType.BET_25,
                ActionType.BET_50,
                ActionType.BET_75,
                ActionType.BET_100,
                ActionType.BET_150,
            ):
                amount = compute_bet_amount(
                    at,
                    pot       = self._pot,
                    stack     = stack,
                    to_call   = to_call,
                    min_raise = self._min_raise,
                    current_bet=self._player_bets[seat],
                )
                if self._min_raise <= amount <= max_bet:
                    mask[at] = True
            mask[ActionType.ALL_IN] = True

        return mask

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_obs(self, is_terminal: bool) -> StructuredObservation:
        sc = self._scenario

        legal_mask = (
            tuple(False for _ in range(NUM_ACTIONS))
            if is_terminal
            else tuple(self._compute_legal_mask(HERO_SEAT))
        )

        return StructuredObservation(
            hero_seat  = HERO_SEAT,
            hole_cards = HoleCardsObs(
                cards=_mask_to_cards(self._hole_masks[HERO_SEAT])
            ),
            board      = BoardObs(node_cards=self._node_cards),
            players    = tuple(
                PlayerObs(
                    seat        = s,
                    stack       = self._stacks[s],
                    current_bet = self._player_bets[s],
                    has_folded  = self._has_folded[s],
                    is_all_in   = self._is_all_in[s],
                    is_hero     = (s == HERO_SEAT),
                )
                for s in range(len(sc.players))
            ),
            betting = BettingObs(
                pot                = self._pot,
                bet_to_call        = self._bet_to_call - self._player_bets[HERO_SEAT],
                min_raise          = self._min_raise,
                street_index       = sc.street_index,
                raises_this_street = self._raises_this_street,
            ),
            legal_mask  = legal_mask,
            game_name   = self._game_name,
            history     = HistoryObs(events=tuple(self._history)),
            dealer_position = sc.dealer_position,
            is_terminal = is_terminal,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_board_mask(self) -> int:
        mask = 0
        for card in self._node_cards:
            if card is not None:
                mask |= 1 << card
        return mask

    def _make_info(self) -> Dict:
        return {
            "outcome"   : self._outcome,
            "pot"       : self._pot,
            "hero_bets" : self._player_bets[HERO_SEAT],
            "raises"    : self._raises_this_street,
        }
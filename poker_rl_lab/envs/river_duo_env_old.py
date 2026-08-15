"""
envs/river_duo_env.py — Two-agent heads-up river betting environment.

Sibling of envs/push_fold_duo_env.py — mirrors that module's design
exactly (two-agent, turn-based, DuoStepResult with per-seat rewards),
adapted to river_duo's richer (but still fixed, non-reraising) action
tree. See scenarios/river_duo_generator.py's module docstring for the
full decision tree:

    OOP (seat 1) acts first — CHECK or BET_100 (pot-sized bet only).
        OOP CHECK  -> IP (seat 0) acts — CHECK or BET_100.
            IP CHECK -> showdown (check-check).
            IP BET   -> OOP acts again — CALL or FOLD.
        OOP BET    -> IP (seat 0) acts — CALL or FOLD.

Why this does NOT implement BasePokerEnv
------------------------------------------
Same rationale as push_fold_duo_env.py: both seats are being trained
simultaneously, and which seat is "the agent" flips within a single
episode (OOP acts, then possibly IP, then possibly OOP again). See
that module's own docstring for the full argument against forcing this
into the single-hero BasePokerEnv contract; not repeated here.

Routing OOP's two possible decision points
--------------------------------------------
OOP can be self.current_seat at two structurally different points: the
INITIAL check/bet decision, and (only if OOP checked and IP then bet) a
later call/fold decision facing IP's bet. `self._oop_facing_bet` is the
flag that disambiguates which of those step() should route to — it is
only ever set True inside _step_ip()'s BET_100 branch and is never
relevant once IP has acted first (in the OOP-bets-first branch, IP's
own call/fold decision is unambiguous since IP never has a first,
different-shaped decision the way OOP does).

Bet sizing
-----------
Only one bet size exists in this game: pot-sized (BET_100,
actions.abstract_action.ActionType.BET_100's pot_fraction == 1.0). The
actual chip amount committed is min(pot_before_bet, remaining_stack) —
capped by the bettor's stack (effectively all-in if stack < pot). A
subsequent call mirrors that exactly: min(amount_to_call, caller's
stack) — if the caller has fewer chips than the bet, they call all-in
for less and any uncalled excess is returned at showdown, exactly
mirroring push_fold_duo_env.py's _resolve_showdown() excess-handling.

Evaluator
----------
Same duck-typed contract as push_fold_duo_env.py:
``hero_wins(hero_mask, villain_mask, board_mask) -> Optional[bool]``.
Called with OOP as "hero" — True means OOP wins, False means IP wins,
None means chop. Defaults to a MockEvaluator (duplicated here rather
than imported, matching every other env's self-contained pattern —
see push_fold_env.py's own module docstring); pass
envs.engine_scoring_evaluator.EngineScoringEvaluator or
envs.poker_eval_evaluator.PokerEvalEvaluator for real showdown
resolution (the board is already fully dealt at reset() — see
scenarios/river_duo_generator.py — so no runout completion is needed
here, unlike push-fold's empty board).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from scenarios.base_scenario import BaseScenario
from scenarios.river_duo_generator import OOP_SEAT, IP_SEAT
from envs.observations import (
    BettingObs,
    BoardObs,
    HistoryEventObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)
from actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS


def _mask_to_cards(mask: int) -> Tuple[int, ...]:
    return tuple(i for i in range(52) if (mask >> i) & 1)


class MockEvaluator:
    """Deterministic stub: larger hand_mask integer wins. See module docstring."""

    def hero_wins(self, hero_mask: int, villain_mask: int, board_mask: int) -> Optional[bool]:
        if hero_mask > villain_mask:
            return True
        if hero_mask < villain_mask:
            return False
        return None


@dataclass(frozen=True)
class DuoStepResult:
    """
    Result of one RiverDuoEnv.step() call.

    Attributes
    ----------
    obs : StructuredObservation, optional
        Observation for the NEXT acting seat. None iff done is True.
    rewards : Dict[int, float]
        Chip change for each seat, keyed by seat index. Empty ({}) for
        any intermediate (non-terminal) step; has exactly
        {0: ..., 1: ...} on the terminal step.
    done : bool
    info : dict
        Diagnostic ``outcome`` key: one of "check_check",
        "oop_bet_ip_call", "oop_bet_ip_fold", "ip_bet_oop_call",
        "ip_bet_oop_fold".
    """
    obs: Optional[StructuredObservation]
    rewards: Dict[int, float]
    done: bool
    info: dict


class RiverDuoEnv:
    """
    Two-agent heads-up river betting environment. See module docstring.

    Parameters
    ----------
    evaluator : optional
        ``hero_wins(oop_mask, ip_mask, board_mask) -> Optional[bool]``,
        called with OOP as hero. Defaults to MockEvaluator.
    """

    def __init__(self, evaluator=None) -> None:
        self._evaluator = evaluator or MockEvaluator()

        self._scenario: Optional[BaseScenario] = None
        self._done: bool = True
        self.current_seat: int = OOP_SEAT

        self._starting_stack: List[int] = [0, 0]
        self._total_committed: List[int] = [0, 0]
        self._stack: List[int] = [0, 0]
        self._is_all_in: List[bool] = [False, False]
        self._pot_before_street: int = 0
        self._history: List[HistoryEventObs] = []

        self._oop_facing_bet: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, scenario: BaseScenario) -> StructuredObservation:
        if scenario.current_player != OOP_SEAT:
            raise ValueError(
                f"RiverDuoEnv expects scenario.current_player == "
                f"OOP_SEAT ({OOP_SEAT}), got {scenario.current_player}"
            )

        self._scenario = scenario
        self._done = False
        self.current_seat = OOP_SEAT
        self._oop_facing_bet = False
        self._history = []

        for seat in (OOP_SEAT, IP_SEAT):
            p = scenario.players[seat]
            self._stack[seat] = p.stack
            self._starting_stack[seat] = p.stack
            self._total_committed[seat] = 0
            self._is_all_in[seat] = p.is_all_in

        self._pot_before_street = scenario.pot

        return self._build_obs(OOP_SEAT, legal_mask=self._check_bet_legal_mask())

    def step(self, action: AbstractAction) -> DuoStepResult:
        """
        Apply *action* for self.current_seat and advance the episode.

        Raises
        ------
        RuntimeError
            If called before reset() or after the episode has ended.
        ValueError
            If the action is not legal for the currently-acting seat.
        """
        if self._done or self._scenario is None:
            raise RuntimeError(
                "Call reset() before step(), or the episode has already ended."
            )

        atype = action.action_type

        if self.current_seat == OOP_SEAT and not self._oop_facing_bet:
            return self._step_oop_first(atype)
        if self.current_seat == IP_SEAT:
            return self._step_ip(atype)
        return self._step_oop_facing_bet(atype)

    # ── OOP's first decision: CHECK or BET_100 ──────────────────────────────

    def _step_oop_first(self, atype: ActionType) -> DuoStepResult:
        pot_before = self._pot_before_street

        if atype == ActionType.CHECK:
            self._history.append(
                HistoryEventObs(seat=OOP_SEAT, action_type=int(atype),
                                 street_index=self._scenario.street_index,
                                 pot_fraction=0.0, is_aggressor=False)
            )
            self.current_seat = IP_SEAT
            return DuoStepResult(
                obs=self._build_obs(IP_SEAT, legal_mask=self._check_bet_legal_mask()),
                rewards={}, done=False, info={"outcome": "oop_check"},
            )

        if atype == ActionType.BET_100:
            amount = self._commit(OOP_SEAT, pot_before)
            self._history.append(
                HistoryEventObs(
                    seat=OOP_SEAT, action_type=int(atype),
                    street_index=self._scenario.street_index,
                    pot_fraction=(amount / pot_before if pot_before > 0 else 0.0),
                    is_aggressor=True,
                )
            )
            self.current_seat = IP_SEAT
            return DuoStepResult(
                obs=self._build_obs(IP_SEAT, legal_mask=self._call_fold_legal_mask()),
                rewards={}, done=False, info={"outcome": "oop_bet"},
            )

        raise ValueError(f"Illegal OOP action: {atype}. Only CHECK and BET_100 are legal.")

    # ── IP's decision: either {CHECK, BET_100} (after OOP check) or
    #    {CALL, FOLD} (facing OOP's bet) ─────────────────────────────────────

    def _step_ip(self, atype: ActionType) -> DuoStepResult:
        facing_bet = self._total_committed[OOP_SEAT] > self._total_committed[IP_SEAT]

        if facing_bet:
            return self._resolve_call_or_fold(
                acting_seat=IP_SEAT, other_seat=OOP_SEAT, atype=atype,
                fold_outcome="oop_bet_ip_fold", call_outcome="oop_bet_ip_call",
            )

        # IP facing a check — CHECK or BET_100
        pot_before = self._pot_before_street

        if atype == ActionType.CHECK:
            self._history.append(
                HistoryEventObs(seat=IP_SEAT, action_type=int(atype),
                                 street_index=self._scenario.street_index,
                                 pot_fraction=0.0, is_aggressor=False)
            )
            self._done = True
            rewards = self._resolve_showdown()
            return DuoStepResult(obs=None, rewards=rewards, done=True,
                                  info={"outcome": "check_check"})

        if atype == ActionType.BET_100:
            amount = self._commit(IP_SEAT, pot_before)
            self._history.append(
                HistoryEventObs(
                    seat=IP_SEAT, action_type=int(atype),
                    street_index=self._scenario.street_index,
                    pot_fraction=(amount / pot_before if pot_before > 0 else 0.0),
                    is_aggressor=True,
                )
            )
            self.current_seat = OOP_SEAT
            self._oop_facing_bet = True
            return DuoStepResult(
                obs=self._build_obs(OOP_SEAT, legal_mask=self._call_fold_legal_mask()),
                rewards={}, done=False, info={"outcome": "ip_bet"},
            )

        raise ValueError(f"Illegal IP action: {atype}. Only CHECK and BET_100 are legal here.")

    # ── OOP facing IP's bet (after OOP's own initial check): CALL or FOLD ──

    def _step_oop_facing_bet(self, atype: ActionType) -> DuoStepResult:
        return self._resolve_call_or_fold(
            acting_seat=OOP_SEAT, other_seat=IP_SEAT, atype=atype,
            fold_outcome="ip_bet_oop_fold", call_outcome="ip_bet_oop_call",
        )

    # ── Shared CALL/FOLD resolution ─────────────────────────────────────────

    def _resolve_call_or_fold(
        self, acting_seat: int, other_seat: int, atype: ActionType,
        fold_outcome: str, call_outcome: str,
    ) -> DuoStepResult:
        if atype == ActionType.FOLD:
            self._history.append(
                HistoryEventObs(seat=acting_seat, action_type=int(atype),
                                 street_index=self._scenario.street_index,
                                 pot_fraction=0.0, is_aggressor=False)
            )
            self._done = True
            rewards = self._resolve_fold(folder=acting_seat, other=other_seat)
            return DuoStepResult(obs=None, rewards=rewards, done=True,
                                  info={"outcome": fold_outcome})

        if atype == ActionType.CALL:
            to_call = self._total_committed[other_seat] - self._total_committed[acting_seat]
            call_amount = min(to_call, self._stack[acting_seat])
            pot_before = self._total_committed[OOP_SEAT] + self._total_committed[IP_SEAT]

            self._total_committed[acting_seat] += call_amount
            self._stack[acting_seat] -= call_amount
            if self._stack[acting_seat] == 0:
                self._is_all_in[acting_seat] = True

            self._history.append(
                HistoryEventObs(
                    seat=acting_seat, action_type=int(atype),
                    street_index=self._scenario.street_index,
                    pot_fraction=(call_amount / pot_before if pot_before > 0 else 0.0),
                    is_aggressor=False,
                )
            )
            self._done = True
            rewards = self._resolve_showdown()
            return DuoStepResult(obs=None, rewards=rewards, done=True,
                                  info={"outcome": call_outcome})

        raise ValueError(f"Illegal action: {atype}. Only CALL and FOLD are legal here.")

    # ── Chip commitment helper (pot-sized bet, capped at stack) ─────────────

    def _commit(self, seat: int, pot_before: int) -> int:
        """Commit a pot-sized bet for `seat`, capped at their remaining
        stack. Returns the amount actually committed."""
        amount = min(pot_before, self._stack[seat])
        self._total_committed[seat] += amount
        self._stack[seat] -= amount
        if self._stack[seat] == 0:
            self._is_all_in[seat] = True
        return amount

    # ── Showdown / fold resolution ──────────────────────────────────────────

    def _resolve_fold(self, folder: int, other: int) -> Dict[int, float]:
        """
        Uncontested pot (pre-street pot + both seats' street commitments)
        goes entirely to `other`.
        """
        folder_final = self._starting_stack[folder] - self._total_committed[folder]
        other_final = (
            self._starting_stack[other]
            - self._total_committed[other]
            + self._total_committed[OOP_SEAT]
            + self._total_committed[IP_SEAT]
        )
        return self._finalize_rewards({folder: folder_final, other: other_final})

    def _resolve_showdown(self) -> Dict[int, float]:
        """
        Resolve a contested pot (check-check or a call), returning any
        uncalled excess to whichever seat committed more this street
        (mirrors push_fold_duo_env.py's _resolve_showdown() excess-
        handling exactly), then awards the pot via self._evaluator.
        """
        sc = self._scenario
        oop_total = self._total_committed[OOP_SEAT]
        ip_total = self._total_committed[IP_SEAT]

        oop_final = self._starting_stack[OOP_SEAT] - oop_total
        ip_final = self._starting_stack[IP_SEAT] - ip_total

        pot = self._pot_before_street + oop_total + ip_total
        excess = abs(oop_total - ip_total)
        if excess:
            if oop_total > ip_total:
                oop_final += excess
            else:
                ip_final += excess
            pot -= excess

        result = self._evaluator.hero_wins(
            sc.players[OOP_SEAT].hand_mask, sc.players[IP_SEAT].hand_mask, self._board_mask()
        )

        if result is True:
            oop_final += pot
        elif result is False:
            ip_final += pot
        else:
            half = pot // 2
            oop_final += half
            ip_final += pot - half

        return self._finalize_rewards({OOP_SEAT: oop_final, IP_SEAT: ip_final})

    def _finalize_rewards(self, final_stacks: Dict[int, float]) -> Dict[int, float]:
        return {
            seat: float(final_stacks[seat] - self._starting_stack[seat])
            for seat in (OOP_SEAT, IP_SEAT)
        }

    def _board_mask(self) -> int:
        mask = 0
        for card in self._scenario.node_cards:
            if card is not None:
                mask |= 1 << card
        return mask

    # ── Legal masks ───────────────────────────────────────────────────────────

    @staticmethod
    def _check_bet_legal_mask() -> Tuple[bool, ...]:
        mask = [False] * NUM_ACTIONS
        mask[ActionType.CHECK] = True
        mask[ActionType.BET_100] = True
        return tuple(mask)

    @staticmethod
    def _call_fold_legal_mask() -> Tuple[bool, ...]:
        mask = [False] * NUM_ACTIONS
        mask[ActionType.FOLD] = True
        mask[ActionType.CALL] = True
        return tuple(mask)

    # ── Observation ──────────────────────────────────────────────────────────

    def _build_obs(self, seat: int, legal_mask: Tuple[bool, ...]) -> StructuredObservation:
        sc = self._scenario

        players = tuple(
            PlayerObs(
                seat=s,
                stack=self._stack[s],
                current_bet=self._total_committed[s],
                has_folded=False,
                is_all_in=self._is_all_in[s],
                is_hero=(s == seat),
            )
            for s in (IP_SEAT, OOP_SEAT)
        )

        other = IP_SEAT if seat == OOP_SEAT else OOP_SEAT
        # pot = pre-street pot (already includes both blinds/earlier
        # streets — see base_scenario.py's inclusive pot convention)
        # plus whatever either seat has committed on THIS street so far.
        pot = self._pot_before_street + self._total_committed[OOP_SEAT] + self._total_committed[IP_SEAT]
        bet_to_call = self._total_committed[other] - self._total_committed[seat]

        return StructuredObservation(
            hero_seat=seat,
            hole_cards=HoleCardsObs(cards=_mask_to_cards(sc.players[seat].hand_mask)),
            board=BoardObs(node_cards=tuple(sc.node_cards)),
            players=players,
            betting=BettingObs(
                pot=pot,
                bet_to_call=bet_to_call,
                min_raise=sc.big_blind,
                street_index=sc.street_index,
                raises_this_street=0,
            ),
            legal_mask=legal_mask,
            game_name=sc.game_name,
            history=HistoryObs(events=tuple(self._history)),
            dealer_position=sc.dealer_position,
            is_terminal=False,
        )
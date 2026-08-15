"""
envs/push_fold_duo_env.py — Two-agent heads-up push-fold environment.

Design note — why this does NOT implement BasePokerEnv
-----------------------------------------------------------
envs/base_env.py's BasePokerEnv contract is single-hero: one seat is
"the agent", every other seat is driven internally by a fixed villain
policy, and step() returns one scalar reward for that one hero. That
contract is wrong for this game: SB (seat 0) and BB (seat 1) are BOTH
being trained, and which one is "the agent" flips every episode
depending on who's on move. Forcing this into BasePokerEnv would mean
picking a fixed hero per env instance and hard-coding the other seat's
policy inside the env — exactly the coupling BasePokerEnv exists to
avoid (see full_holdem_env.py's villain_policy pattern, which is
correct there because only ONE seat is ever the learner).

Instead this env is deliberately a plain two-agent, turn-based env:
reset() returns SB's observation; step(action) always applies to
whichever seat self.current_seat currently names, and returns a
DuoStepResult carrying a per-seat rewards dict (populated only once the
episode actually terminates) rather than a single float. A duo training
loop (see duo_training_loop.py) is expected to read self.current_seat
before every step() to know which policy to query — mirroring how
controllers/base_controller.py's Controller.act(obs) already works,
just without forcing a Controller object through this file.

Game shape
-----------
    SB (seat 0): FOLD or ALL_IN (push). No bet sizing — push-fold has
                 exactly these two options by definition.
    BB (seat 1): only acts if SB pushed. FOLD or CALL. Since
                 scenarios.push_fold_duo_generator.PushFoldDuoScenarioGenerator
                 always starts both seats with an IDENTICAL stack,
                 calling always puts BB all-in for exactly SB's shove
                 — but the showdown accounting below does not assume
                 that (see _resolve_showdown), so a future generator
                 with unequal stacks works unchanged (uncalled excess
                 is returned to whichever seat committed more, exactly
                 like envs/river_env.py's own _do_showdown()).

Evaluator
----------
Same duck-typed contract used throughout the RL Lab:
``hero_wins(hero_mask, villain_mask, board_mask) -> Optional[bool]``.
Called with SB as "hero" — True means SB wins, False means BB wins,
None means chop. Pass envs.poker_eval_evaluator.PokerEvalEvaluator or
envs.engine_scoring_evaluator.EngineScoringEvaluator for a real
(runout-completed) preflop equity evaluation; defaults to a MockEvaluator
for pipeline testing only (see push_fold_env.py's own MockEvaluator —
duplicated here rather than imported, matching that module's own
"self-contained, no cross-env coupling" pattern).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from poker_rl_lab.scenarios.base_scenario import BaseScenario
from poker_rl_lab.scenarios.push_fold_duo_generator import SB_SEAT, BB_SEAT
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
    Result of one PushFoldDuoEnv.step() call.

    Attributes
    ----------
    obs : StructuredObservation, optional
        Observation for the NEXT acting seat. None iff done is True.
    rewards : Dict[int, float]
        Chip change for each seat, keyed by seat index. Empty ({}) for
        the intermediate SB-pushes step (nobody's outcome is decided
        yet); has exactly {0: ..., 1: ...} on the terminal step.
    done : bool
    info : dict
        Diagnostic keys: ``outcome`` (one of "sb_fold", "sb_push",
        "bb_fold", "showdown").
    """
    obs: Optional[StructuredObservation]
    rewards: Dict[int, float]
    done: bool
    info: dict


class PushFoldDuoEnv:
    """
    Two-agent heads-up push-fold environment. See module docstring.

    Parameters
    ----------
    evaluator : optional
        ``hero_wins(sb_mask, bb_mask, board_mask) -> Optional[bool]``,
        called with SB as hero. Defaults to MockEvaluator.
    """

    def __init__(self, evaluator=None) -> None:
        self._evaluator = evaluator or MockEvaluator()

        self._scenario: Optional[BaseScenario] = None
        self._done: bool = True
        self.current_seat: int = SB_SEAT

        self._starting_stack: List[int] = [0, 0]
        self._posted: List[int] = [0, 0]          # blind posted at reset()
        self._total_committed: List[int] = [0, 0]  # chips committed so far
        self._stack: List[int] = [0, 0]            # remaining stack
        self._is_all_in: List[bool] = [False, False]
        self._history: List[HistoryEventObs] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, scenario: BaseScenario) -> StructuredObservation:
        if scenario.current_player != SB_SEAT:
            raise ValueError(
                f"PushFoldDuoEnv expects scenario.current_player == "
                f"SB_SEAT ({SB_SEAT}), got {scenario.current_player}"
            )

        self._scenario = scenario
        self._done = False
        self.current_seat = SB_SEAT
        self._history = []

        for seat in (SB_SEAT, BB_SEAT):
            p = scenario.players[seat]
            self._posted[seat] = p.current_bet
            self._stack[seat] = p.stack
            self._starting_stack[seat] = p.stack + p.current_bet
            self._total_committed[seat] = p.current_bet
            self._is_all_in[seat] = p.is_all_in

        return self._build_obs(SB_SEAT, legal_mask=self._sb_legal_mask())

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

        if self.current_seat == SB_SEAT:
            return self._step_sb(atype)
        return self._step_bb(atype)

    # ── SB's decision ─────────────────────────────────────────────────────────

    def _step_sb(self, atype: ActionType) -> DuoStepResult:
        pot_before = self._posted[SB_SEAT] + self._posted[BB_SEAT]

        if atype == ActionType.FOLD:
            self._history.append(
                HistoryEventObs(seat=SB_SEAT, action_type=int(atype),
                                 street_index=0, pot_fraction=0.0, is_aggressor=False)
            )
            self._done = True
            sb_r = -float(self._posted[SB_SEAT])
            bb_r = float(self._posted[SB_SEAT])
            return DuoStepResult(
                obs=None, rewards={SB_SEAT: sb_r, BB_SEAT: bb_r},
                done=True, info={"outcome": "sb_fold"},
            )

        if atype == ActionType.ALL_IN:
            amount_added = self._stack[SB_SEAT]
            self._total_committed[SB_SEAT] += amount_added
            self._stack[SB_SEAT] = 0
            self._is_all_in[SB_SEAT] = True

            self._history.append(
                HistoryEventObs(
                    seat=SB_SEAT, action_type=int(atype), street_index=0,
                    pot_fraction=(amount_added / pot_before if pot_before > 0 else 0.0),
                    is_aggressor=True,
                )
            )
            self.current_seat = BB_SEAT
            return DuoStepResult(
                obs=self._build_obs(BB_SEAT, legal_mask=self._bb_legal_mask()),
                rewards={}, done=False, info={"outcome": "sb_push"},
            )

        raise ValueError(f"Illegal SB action: {atype}. Only FOLD and ALL_IN are legal.")

    # ── BB's decision (only reachable after SB pushed) ─────────────────────────

    def _step_bb(self, atype: ActionType) -> DuoStepResult:
        if atype == ActionType.FOLD:
            self._history.append(
                HistoryEventObs(seat=BB_SEAT, action_type=int(atype),
                                 street_index=0, pot_fraction=0.0, is_aggressor=False)
            )
            self._done = True
            sb_r = float(self._posted[BB_SEAT])
            bb_r = -float(self._posted[BB_SEAT])
            return DuoStepResult(
                obs=None, rewards={SB_SEAT: sb_r, BB_SEAT: bb_r},
                done=True, info={"outcome": "bb_fold"},
            )

        if atype == ActionType.CALL:
            to_call = self._total_committed[SB_SEAT] - self._posted[BB_SEAT]
            call_amount = min(to_call, self._stack[BB_SEAT])
            pot_before = self._total_committed[SB_SEAT] + self._posted[BB_SEAT]

            self._total_committed[BB_SEAT] = self._posted[BB_SEAT] + call_amount
            self._stack[BB_SEAT] -= call_amount
            if self._stack[BB_SEAT] == 0:
                self._is_all_in[BB_SEAT] = True

            self._history.append(
                HistoryEventObs(
                    seat=BB_SEAT, action_type=int(atype), street_index=0,
                    pot_fraction=(call_amount / pot_before if pot_before > 0 else 0.0),
                    is_aggressor=False,
                )
            )
            self._done = True
            rewards = self._resolve_showdown()
            return DuoStepResult(obs=None, rewards=rewards, done=True,
                                  info={"outcome": "showdown"})

        raise ValueError(f"Illegal BB action: {atype}. Only FOLD and CALL are legal.")

    # ── Showdown ──────────────────────────────────────────────────────────────

    def _resolve_showdown(self) -> Dict[int, float]:
        """
        Resolve the all-in showdown, returning any uncalled excess to
        whichever seat committed more (only relevant for future
        unequal-stack generators — see module docstring), then awards
        the contested pot via self._evaluator.

        Mirrors envs/river_env.py's RiverEnv._do_showdown() heads-up
        excess-handling exactly, adapted to seat-indexed bookkeeping.
        """
        sc = self._scenario
        sb_total = self._total_committed[SB_SEAT]
        bb_total = self._total_committed[BB_SEAT]

        sb_final = self._starting_stack[SB_SEAT] - sb_total
        bb_final = self._starting_stack[BB_SEAT] - bb_total

        pot = sb_total + bb_total
        excess = abs(sb_total - bb_total)
        if excess:
            if sb_total > bb_total:
                sb_final += excess
            else:
                bb_final += excess
            pot -= excess

        result = self._evaluator.hero_wins(
            sc.players[SB_SEAT].hand_mask, sc.players[BB_SEAT].hand_mask, 0
        )

        if result is True:
            sb_final += pot
        elif result is False:
            bb_final += pot
        else:
            half = pot // 2
            sb_final += half
            bb_final += pot - half

        return {
            SB_SEAT: float(sb_final - self._starting_stack[SB_SEAT]),
            BB_SEAT: float(bb_final - self._starting_stack[BB_SEAT]),
        }

    # ── Legal masks ───────────────────────────────────────────────────────────

    @staticmethod
    def _sb_legal_mask() -> Tuple[bool, ...]:
        mask = [False] * NUM_ACTIONS
        mask[ActionType.FOLD] = True
        mask[ActionType.ALL_IN] = True
        return tuple(mask)

    @staticmethod
    def _bb_legal_mask() -> Tuple[bool, ...]:
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
            for s in (SB_SEAT, BB_SEAT)
        )

        other = BB_SEAT if seat == SB_SEAT else SB_SEAT
        # pot = everything either seat has committed so far (both
        # blinds, plus SB's push once it's happened) — see
        # base_scenario.py's inclusive pot convention. bet_to_call stays
        # a delta (BettingObs's own contract: "additional chips the
        # hero must add to call").
        pot = self._total_committed[SB_SEAT] + self._total_committed[BB_SEAT]
        bet_to_call = self._total_committed[other] - self._total_committed[seat]

        return StructuredObservation(
            hero_seat=seat,
            hole_cards=HoleCardsObs(cards=_mask_to_cards(sc.players[seat].hand_mask)),
            board=BoardObs(node_cards=tuple()),
            players=players,
            betting=BettingObs(
                pot=pot,
                bet_to_call=bet_to_call,
                min_raise=sc.big_blind,
                street_index=0,
                raises_this_street=0,
            ),
            legal_mask=legal_mask,
            game_name=sc.game_name,
            history=HistoryObs(events=tuple(self._history)),
            dealer_position=SB_SEAT,
            is_terminal=False,
        )
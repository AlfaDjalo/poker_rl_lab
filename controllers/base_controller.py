"""
controllers/base_controller.py — Controller abstraction, per
RL_lab_implementation_guide.md (§9, "Self Play"):

    Each seat in the Environment owns a Controller.

    Controllers may include:
    - RandomController
    - RuleBasedController
    - FrozenPolicyController
    - LearningPolicyController
    - HumanController

    The Environment communicates only through the Controller interface.

    Initially only one learning seat is required.

Interface level: full StructuredObservation, not legal_mask
-------------------------------------------------------------
The guide's diagram (§2) is explicit that Structured Observations sit
between the Environment and everything downstream — Controllers are
"downstream" of the Environment exactly like the Policy Network is, so
a Controller must be able to see whatever a hero can see (hole cards,
board, betting state, history) to make a real decision. A mask-only
interface would make FrozenPolicyController/LearningPolicyController
strictly worse than the hero they're paired against for no principled
reason (they'd be blind to their own cards), and RuleBasedController
strategies that key off board texture or stack depth would be
impossible to express. So Controller.act() below takes a full
StructuredObservation, matching BasePokerEnv.step()'s own contract for
the hero (envs/base_env.py) exactly — a Controller is "a hero that
happens not to be the one being trained," nothing less.

KNOWN GAP — today's envs cannot yet feed this interface for non-hero
seats
-------------------------------------------------------------------
envs/river_env.py's `villain_policy.act(legal_mask)` and
envs/full_holdem_env.py's `VillainPolicyLike.act(legal_mask)` (see that
module's own Protocol docstring) only ever pass the villain seat's
legal_mask — never its hole cards, board, or history. This was a
reasonable scope cut when those envs were built (CheckFoldVillainPolicy
/ CallAnyVillainPolicy / etc. — see river_env.py — only ever needed the
mask), but it means a Controller with real card knowledge (FrozenPolicy/
LearningPolicy/RuleBased-on-hand-strength) CANNOT currently be wired in
as a live opponent in RiverEnv or FullHoldemEnv without an env change:
there is no seam that hands a non-hero seat's StructuredObservation to
anything.

This file does not paper over that gap. `LegalMaskControllerAdapter`
below (the bridge between this module's Controller interface and the
existing `act(legal_mask) -> AbstractAction` envs actually call) is
explicit that it can only forward the mask — the wrapped Controller
receives a synthesized, deliberately-degenerate StructuredObservation
with empty hole cards / empty board / empty history, and only the mask
itself is real. That is sufficient for RandomController and
RuleBasedController strategies that only read legal_mask (both are
therefore genuinely drop-in compatible today), but FrozenPolicyController
/ LearningPolicyController used through the adapter would be acting
blind — the adapter's own docstring says so loudly rather than silently
producing a plausible-looking but unfounded decision. Wiring real
per-seat observations through RiverEnv/FullHoldemEnv is a follow-up env
change (give each non-hero seat its own _build_obs()-equivalent), not
something this file can fix from the Controller side.

Dependencies
------------
    actions.abstract_action, envs.observations. No torch, no engine
    rules, no poker_eval — mirrors evaluation.baseline_policies' own
    "no heavy dependency" scope (that module's HeroPolicy callables are
    reused directly by RuleBasedController below).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

from actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS
from envs.observations import (
    BettingObs,
    BoardObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class Controller(ABC):
    """
    Owns the decision-making for one seat.

    The Environment (or, today, LegalMaskControllerAdapter standing in
    for the Environment — see module docstring) communicates with a
    seat only through this interface. A Controller must never be
    special-cased by seat number inside the Environment: swapping which
    Controller a seat owns must never require an Environment change.

    Contract
    --------
    act(obs) -> AbstractAction
        obs is never terminal (obs.is_terminal is False) and
        obs.legal_mask has at least one True entry — identical
        precondition to envs.base_env.BasePokerEnv's hero_policy contract.
        The returned action must be legal (True in obs.legal_mask).
    """

    @abstractmethod
    def act(self, obs: StructuredObservation) -> AbstractAction:
        ...


# ---------------------------------------------------------------------------
# RandomController
# ---------------------------------------------------------------------------

class RandomController(Controller):
    """
    Uniform-random over legal actions.

    Deliberately the Controller-interface twin of
    envs.base_env.RandomHeroPolicy — same behaviour, same seeding
    contract — just typed against Controller.act(obs) instead of a bare
    HeroPolicy callable, so it can be assigned to a non-hero seat.

    Parameters
    ----------
    seed : int, optional
        Seed for reproducible action selection.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: StructuredObservation) -> AbstractAction:
        legal = [i for i, m in enumerate(obs.legal_mask) if m]
        if not legal:
            raise ValueError("No legal actions available — is the observation terminal?")
        return AbstractAction(ActionType(self._rng.choice(legal)))


# ---------------------------------------------------------------------------
# RuleBasedController
# ---------------------------------------------------------------------------

#: Same shape as evaluation.baseline_policies.HeroPolicy — a Controller
#: is a strict superset of that contract (it additionally promises not
#: to be terminal-called), so any existing HeroPolicy callable can be
#: wrapped without modification.
RuleBasedStrategy = Callable[[StructuredObservation], AbstractAction]


class RuleBasedController(Controller):
    """
    Wraps a fixed, non-learning strategy function as a Controller.

    This is intentionally a thin adapter rather than a new hierarchy of
    rule-based strategy classes: evaluation/baseline_policies.py already
    defines HeroPolicy-shaped strategies (always_fold_policy,
    calling_station_policy, preference_policy(...)) that read only
    obs.legal_mask and work unchanged across every environment (see that
    module's own docstring). RuleBasedController lets the exact same
    callables be assigned to a table seat instead of only being used as
    the hero in evaluation.policy_evaluator.

    A strategy is free to read any field of obs (hole_cards, board,
    betting, history, ...), not just legal_mask — e.g. a future
    "bet big with a pair or better" strategy. Whether that richer
    information is actually available at the call site today depends on
    which path handed it its StructuredObservation — see the module
    docstring's KNOWN GAP section.

    Parameters
    ----------
    strategy : RuleBasedStrategy
        e.g. evaluation.baseline_policies.always_fold_policy,
        .calling_station_policy, or any
        preference_policy([...]) built from that module.
    """

    def __init__(self, strategy: RuleBasedStrategy) -> None:
        self._strategy = strategy

    def act(self, obs: StructuredObservation) -> AbstractAction:
        return self._strategy(obs)


# ---------------------------------------------------------------------------
# Legal-mask adapter — bridges this module's Controller interface to the
# `act(legal_mask) -> AbstractAction` surface RiverEnv / FullHoldemEnv
# actually call for non-hero seats today.
# ---------------------------------------------------------------------------

class LegalMaskControllerAdapter:
    """
    Wraps a Controller so it satisfies the existing mask-only villain
    policy Protocol (`act(legal_mask: Tuple[bool, ...]) -> AbstractAction`
    — see envs/river_env.py's CheckFoldVillainPolicy and
    envs/full_holdem_env.py's VillainPolicyLike), so a Controller can be
    passed directly as RiverEnv(villain_policy=...) or
    FullHoldemEnv(default_villain_policy=...) / villain_policies={...}
    without an env change.

    THIS IS A LOSSY BRIDGE — see module docstring's KNOWN GAP section.
    Only legal_mask is real; every other field of the StructuredObservation
    handed to the wrapped Controller is a synthesized placeholder (empty
    hole cards, empty board, zeroed betting/position info, empty
    history). Concretely:

        - Safe to wrap: RandomController, RuleBasedController strategies
          that only branch on obs.legal_mask (always_fold_policy,
          calling_station_policy, any preference_policy([...])) — these
          produce IDENTICAL decisions whether given a real or synthesized
          observation, since they never look past legal_mask.
        - NOT meaningfully usable through this adapter: FrozenPolicyController
          / LearningPolicyController, or any RuleBasedController strategy
          that reads hole_cards/board/betting/history. Wrapping one of
          these will not raise — obs is a real (if degenerate)
          StructuredObservation, so nothing crashes — but the Controller
          is deciding with fabricated inputs (e.g. hole_cards=() always),
          which is worse than useless for anything hand-strength-aware.
          Do not wrap those two Controllers with this adapter for a real
          opponent; wire an env-level fix instead (see module docstring).

    Parameters
    ----------
    controller : Controller
    hero_seat : int
        Seat this adapter is standing in for. Only used to populate the
        synthesized observation's hero_seat/is_hero fields plausibly;
        does not affect which actions are legal (that's entirely
        determined by the legal_mask argument passed to __call__/act()
        at call time by the env).
    game_name : str
        Forwarded into the synthesized StructuredObservation's
        game_name field. Purely cosmetic here (no strategy this adapter
        is safe for reads it) but kept so the object is a well-formed
        StructuredObservation regardless.
    """

    def __init__(
        self,
        controller: Controller,
        hero_seat: int = 0,
        game_name: str = "holdem",
    ) -> None:
        self._controller = controller
        self._seat = hero_seat
        self._game_name = game_name

    def act(self, legal_mask: Sequence[bool]) -> AbstractAction:
        obs = self._synthesize_obs(tuple(legal_mask))
        return self._controller.act(obs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _synthesize_obs(self, legal_mask: tuple) -> StructuredObservation:
        """
        Build the minimum StructuredObservation that satisfies its own
        __post_init__ validation (envs/observations.py) and lets a
        mask-only strategy act correctly. Every non-mask field is a
        placeholder — see class docstring.
        """
        if len(legal_mask) != NUM_ACTIONS:
            raise ValueError(
                f"legal_mask must have length {NUM_ACTIONS}, got {len(legal_mask)}"
            )

        placeholder_player = PlayerObs(
            seat=self._seat,
            stack=0,
            current_bet=0,
            has_folded=False,
            is_all_in=False,
            is_hero=True,
        )

        return StructuredObservation(
            hero_seat=self._seat,
            hole_cards=HoleCardsObs(cards=()),
            board=BoardObs(node_cards=()),
            players=(placeholder_player,),
            betting=BettingObs(
                pot=0, bet_to_call=0, min_raise=0,
                street_index=0, raises_this_street=0,
            ),
            legal_mask=legal_mask,
            game_name=self._game_name,
            history=HistoryObs(events=()),
            dealer_position=0,
            is_terminal=False,
        )
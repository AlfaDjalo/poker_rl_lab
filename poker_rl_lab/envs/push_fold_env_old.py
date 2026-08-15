"""
envs/push_fold_env.py  —  Push-or-fold RL environment  (Phase 4 · Step 1)

The simplest complete poker environment.

  · Hero is always seat 1 (the big-blind caller).
  · Villain (seat 0) is already all-in when the episode begins.
  · Exactly two legal actions: CALL or FOLD.
  · Episode terminates immediately after the hero acts.

Design
------
  * Self-contained: no dependency on PokerState, the CAP Engine, or any
    C++ extension.  Safe to run on QuantConnect free tier.
  * Injectable evaluator: swap MockEvaluator for a real hand ranker without
    changing any other code.
  * Rewards are in chips so callers can normalise by BB however they like.

Reward convention
-----------------
  FOLD          -hero.current_bet          (lost the posted blind)
  CALL + win    sc.pot + effective         (net gain; villain's chips + pot)
  CALL + loss   -hero_total               (lost everything committed)
  CALL + chop    0                         (pots cancel)

  where effective = min(hero_total, villain_total)
"""

from __future__ import annotations

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
from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS

from poker_rl_lab.envs.base_env import BasePokerEnv

# ── Seat convention ──────────────────────────────────────────────────────────

HERO_SEAT: int = 1      # the caller is always seat 1 in push-fold scenarios


# ── Card helpers (no dependency on cards.mask) ───────────────────────────────

def _mask_to_cards(mask: int) -> Tuple[int, ...]:
    """Expand a 52-bit hand bitmask into a sorted tuple of card ids."""
    return tuple(i for i in range(52) if (mask >> i) & 1)


# ── Default (mock) evaluator ─────────────────────────────────────────────────

class MockEvaluator:
    """
    Deterministic stub for unit tests.

    The player whose hand_mask integer is numerically larger wins.
    Because card ids are unique, masks are always distinct — no ties.

    Replace with a real evaluator (wrapping poker_eval or an equivalent
    pure-Python ranker) before training.
    """

    def hero_wins(
        self,
        hero_mask: int,
        villain_mask: int,
        board_mask: int,        # unused by mock; real evaluator needs it
    ) -> Optional[bool]:
        """
        Returns
        -------
        True   hero wins
        False  villain wins
        None   chop (included for protocol completeness)
        """
        if hero_mask > villain_mask:
            return True
        if hero_mask < villain_mask:
            return False
        return None


# ── Environment ───────────────────────────────────────────────────────────────

class PushFoldEnv(BasePokerEnv):
    """
    Push-or-fold RL environment.

    Parameters
    ----------
    evaluator : optional
        Any object with the method::

            hero_wins(hero_mask, villain_mask, board_mask) -> Optional[bool]

        ``True`` = hero wins, ``False`` = villain wins, ``None`` = chop.
        Defaults to :class:`MockEvaluator`.

    Example
    -------
        gen = PushFoldScenarioGenerator()
        gen.seed(42)

        env = PushFoldEnv()
        obs = env.reset(gen.generate())
        obs, reward, done, info = env.step(AbstractAction(ActionType.CALL))
    """

    def __init__(self, evaluator=None) -> None:
        self._evaluator          = evaluator or MockEvaluator()
        self._scenario: Optional[BaseScenario] = None
        self._done: bool         = True
        self._history: List[HistoryEventObs] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, scenario: BaseScenario) -> StructuredObservation:
        """
        Begin a new episode.

        Parameters
        ----------
        scenario : BaseScenario
            Produced by :class:`PushFoldScenarioGenerator`.
            Must have ``street_index == 0`` and empty ``node_cards``.

        Returns
        -------
        StructuredObservation
            Non-terminal observation for the hero (seat 1).

        Notes
        -----
        Push-fold scenarios begin *after* the villain has already shoved
        (villain is already all-in — see module docstring). There is no
        literal event log to draw from at this point, so reset() seeds
        ``history`` with a single reconstructed HistoryEventObs
        representing that shove, approximating the pot immediately
        before it as ``small_blind + big_blind`` (the forced bets
        already posted). This is a best-effort reconstruction, not a
        replay of real sequential events — push-fold scenarios are
        generated already-shoved.
        """
        self._scenario = scenario
        self._done     = False
        self._history  = [self._reconstruct_shove_event(scenario)]
        return self._build_obs(is_terminal=False)

    def step(
        self,
        action: AbstractAction,
    ) -> Tuple[StructuredObservation, float, bool, Dict]:
        """
        Apply *action* and end the episode.

        Parameters
        ----------
        action : AbstractAction
            ``FOLD`` or ``CALL`` are the only legal actions.

        Returns
        -------
        obs : StructuredObservation
            Terminal observation; ``legal_mask`` is all-False.
        reward : float
            Chip change for the hero.
        done : bool
            Always ``True`` (push-fold terminates in one step).
        info : dict
            Diagnostic keys: ``action``, ``outcome``, ``pot``.

        Raises
        ------
        RuntimeError
            If called before :meth:`reset` or after the episode has ended.
        ValueError
            If ``action_type`` is not ``FOLD`` or ``CALL``.
        """
        if self._done or self._scenario is None:
            raise RuntimeError(
                "Cannot call step() before reset(), or after the episode has ended."
            )

        sc   = self._scenario
        hero = sc.players[HERO_SEAT]
        vill = sc.players[0]
        atype = action.action_type

        # sc.pot already includes every player's current_bet (shover's
        # full push + caller's posted blind) — see base_scenario.py's
        # pot docstring. Nothing needs to be added back on top of it.
        pot_before_hero = sc.pot  # everything committed so far

        # ── FOLD ──────────────────────────────────────────────────────────
        if atype == ActionType.FOLD:
            reward  = -float(hero.current_bet)   # lose the posted blind
            outcome = "fold"
            pot     = sc.pot  # villain takes uncontested
            hero_action_amount = 0

        # ── CALL ──────────────────────────────────────────────────────────
        elif atype == ActionType.CALL:
            # Chips hero adds on top of already-posted blind
            to_call    = min(sc.bet_to_call - hero.current_bet, hero.stack)
            hero_total = hero.current_bet + to_call
            hero_action_amount = to_call

            # Effective amount: neither player wins more than they put in
            effective  = min(hero_total, vill.current_bet)

            board_mask = self._board_mask(sc)
            result     = self._evaluator.hero_wins(
                hero.hand_mask, vill.hand_mask, board_mask
            )

            # sc.pot already includes vill.current_bet and hero's posted
            # blind (hero.current_bet); only the additional chips hero
            # adds to call (to_call) aren't in it yet.
            pot = sc.pot + to_call

            if result is True:
                reward  = float(sc.pot + effective)   # net gain
                outcome = "win"
            elif result is False:
                reward  = -float(hero_total)           # net loss
                outcome = "loss"
            else:
                reward  = 0.0                          # chop
                outcome = "chop"

        # ── ILLEGAL ───────────────────────────────────────────────────────
        else:
            raise ValueError(
                f"Illegal action in push-fold: {atype}. "
                "Only FOLD and CALL are legal."
            )

        self._done = True
        self._history.append(
            HistoryEventObs(
                seat=HERO_SEAT,
                action_type=int(atype),
                street_index=sc.street_index,
                pot_fraction=(
                    hero_action_amount / pot_before_hero if pot_before_hero > 0 else 0.0
                ),
                is_aggressor=False,   # hero can only FOLD or CALL in push-fold
            )
        )
        obs  = self._build_obs(is_terminal=True)
        info = {"action": atype.name, "outcome": outcome, "pot": pot}
        return obs, reward, True, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _reconstruct_shove_event(sc: BaseScenario) -> HistoryEventObs:
        """
        Best-effort reconstruction of villain's (seat 0) initial shove as
        a HistoryEventObs. See reset()'s docstring for why this is an
        approximation rather than a literal replay.
        """
        pot_before    = sc.small_blind + sc.big_blind
        amount_shoved = max(sc.bet_to_call - sc.small_blind, 0)
        pot_fraction  = amount_shoved / pot_before if pot_before > 0 else 0.0

        return HistoryEventObs(
            seat=0,
            action_type=int(ActionType.ALL_IN),
            street_index=sc.street_index,
            pot_fraction=pot_fraction,
            is_aggressor=True,
        )

    @staticmethod
    def _board_mask(sc: BaseScenario) -> int:
        """Bitmask of all dealt board cards (empty for preflop push-fold)."""
        mask = 0
        for card in sc.node_cards:
            if card is not None:
                mask |= 1 << card
        return mask

    def _legal_mask(self) -> Tuple[bool, ...]:
        mask                   = [False] * NUM_ACTIONS
        mask[ActionType.FOLD]  = True
        mask[ActionType.CALL]  = True
        return tuple(mask)

    def _build_obs(self, is_terminal: bool) -> StructuredObservation:
        sc   = self._scenario
        hero = sc.players[HERO_SEAT]

        return StructuredObservation(
            hero_seat  = HERO_SEAT,
            hole_cards = HoleCardsObs(cards=_mask_to_cards(hero.hand_mask)),
            board      = BoardObs(node_cards=tuple(sc.node_cards)),
            players    = tuple(
                PlayerObs(
                    seat        = i,
                    stack       = p.stack,
                    current_bet = p.current_bet,
                    has_folded  = p.has_folded,
                    is_all_in   = p.is_all_in,
                    is_hero     = (i == HERO_SEAT),
                )
                for i, p in enumerate(sc.players)
            ),
            betting = BettingObs(
                pot                = sc.pot,
                # bet_to_call in BettingObs is *additional* chips hero needs
                bet_to_call        = sc.bet_to_call - hero.current_bet,
                min_raise          = sc.min_raise,
                street_index       = sc.street_index,
                raises_this_street = sc.raises_this_street,
            ),
            legal_mask  = (
                tuple(False for _ in range(NUM_ACTIONS))
                if is_terminal
                else self._legal_mask()
            ),
            game_name   = sc.game_name,
            history     = HistoryObs(events=tuple(self._history)),
            dealer_position = sc.dealer_position,
            is_terminal = is_terminal,
        )
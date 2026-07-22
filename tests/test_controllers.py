"""
test_controllers.py — Unit tests for controllers/.

Scope
-----
    - RandomController / RuleBasedController act correctly against a
      hand-built StructuredObservation (no env, no engine, no torch).
    - LegalMaskControllerAdapter forwards masks correctly and is
      behaviourally IDENTICAL to the wrapped Controller for mask-only
      strategies (the documented "safe to wrap" case).
    - LegalMaskControllerAdapter's synthesized observation is provably
      card-blind, i.e. the documented gap is real and observable, not
      just asserted in a docstring.
    - FrozenPolicyController / LearningPolicyController against a
      lightweight stub actor_critic + adapter (same pattern as
      test_ppo_trainer.py's own stubs — no TF hand encoder, no
      poker_eval extension needed).
    - HumanController's re-prompt-on-invalid-input loop and rendering,
      driven via injected prompt_fn/print_fn (no real terminal).

Dependencies
------------
    Standard library unittest, torch (only for the tiny stub
    ActionResult tensors FrozenPolicyController/LearningPolicyController
    consume). No poker_eval, no TensorFlow.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS
from envs.observations import (
    BettingObs,
    BoardObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)
from evaluation.baseline_policies import (
    always_fold_policy,
    calling_station_policy,
    preference_policy,
)

from controllers.base_controller import (
    Controller,
    RandomController,
    RuleBasedController,
    LegalMaskControllerAdapter,
)
from controllers.policy_controller import (
    FrozenPolicyController,
    LearningPolicyController,
)
from controllers.human_controller import HumanController


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _make_obs(
    legal_mask: Sequence[bool],
    hero_seat: int = 0,
    hole_cards: Tuple[int, ...] = (0, 1),
    board_cards: Tuple[Optional[int], ...] = (None, None, None, None, None),
    pot: int = 10,
    bet_to_call: int = 2,
) -> StructuredObservation:
    """Hand-build a well-formed, non-terminal StructuredObservation."""
    players = (
        PlayerObs(seat=0, stack=100, current_bet=0, has_folded=False,
                   is_all_in=False, is_hero=(hero_seat == 0)),
        PlayerObs(seat=1, stack=100, current_bet=2, has_folded=False,
                   is_all_in=False, is_hero=(hero_seat == 1)),
    )
    return StructuredObservation(
        hero_seat=hero_seat,
        hole_cards=HoleCardsObs(cards=hole_cards),
        board=BoardObs(node_cards=board_cards),
        players=players,
        betting=BettingObs(pot=pot, bet_to_call=bet_to_call, min_raise=4,
                            street_index=0, raises_this_street=0),
        legal_mask=tuple(legal_mask),
        game_name="holdem",
        history=HistoryObs(events=()),
        dealer_position=0,
        is_terminal=False,
    )


def _mask(*legal_types: ActionType) -> Tuple[bool, ...]:
    m = [False] * NUM_ACTIONS
    for t in legal_types:
        m[t] = True
    return tuple(m)


# ---------------------------------------------------------------------------
# RandomController
# ---------------------------------------------------------------------------

class TestRandomController(unittest.TestCase):
    def test_returns_legal_action(self):
        controller = RandomController(seed=0)
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL))
        for _ in range(50):
            action = controller.act(obs)
            self.assertTrue(obs.legal_mask[action.action_type])

    def test_deterministic_given_seed(self):
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL, ActionType.CHECK))
        a = RandomController(seed=42)
        b = RandomController(seed=42)
        seq_a = [a.act(obs).action_type for _ in range(20)]
        seq_b = [b.act(obs).action_type for _ in range(20)]
        self.assertEqual(seq_a, seq_b)

    def test_raises_on_no_legal_actions(self):
        controller = RandomController(seed=0)
        obs = _make_obs(_mask())  # nothing legal
        with self.assertRaises(ValueError):
            controller.act(obs)


# ---------------------------------------------------------------------------
# RuleBasedController
# ---------------------------------------------------------------------------

class TestRuleBasedController(unittest.TestCase):
    def test_always_fold(self):
        controller = RuleBasedController(always_fold_policy)
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL))
        self.assertEqual(controller.act(obs).action_type, ActionType.FOLD)

    def test_calling_station_prefers_call(self):
        controller = RuleBasedController(calling_station_policy)
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL, ActionType.CHECK))
        self.assertEqual(controller.act(obs).action_type, ActionType.CALL)

    def test_calling_station_falls_back_to_check(self):
        controller = RuleBasedController(calling_station_policy)
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CHECK))
        self.assertEqual(controller.act(obs).action_type, ActionType.CHECK)

    def test_custom_preference_order(self):
        strategy = preference_policy([ActionType.BET_50, ActionType.CALL, ActionType.FOLD])
        controller = RuleBasedController(strategy)
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL))
        self.assertEqual(controller.act(obs).action_type, ActionType.CALL)


# ---------------------------------------------------------------------------
# LegalMaskControllerAdapter
# ---------------------------------------------------------------------------

class TestLegalMaskControllerAdapter(unittest.TestCase):
    def test_forwards_mask_and_matches_wrapped_controller(self):
        """Safe-to-wrap case: a mask-only strategy behaves identically
        whether called directly or through the adapter."""
        mask = _mask(ActionType.FOLD, ActionType.CALL, ActionType.CHECK)
        direct = RuleBasedController(calling_station_policy)
        adapted = LegalMaskControllerAdapter(
            RuleBasedController(calling_station_policy), hero_seat=1
        )

        direct_action = direct.act(_make_obs(mask))
        adapted_action = adapted.act(mask)

        self.assertEqual(direct_action.action_type, adapted_action.action_type)

    def test_rejects_wrong_length_mask(self):
        adapted = LegalMaskControllerAdapter(RandomController(seed=0))
        with self.assertRaises(ValueError):
            adapted.act([True, False])  # wrong length

    def test_synthesized_observation_is_card_blind(self):
        """Documents (and proves) the KNOWN GAP: the observation handed
        to the wrapped Controller carries no real hole cards or board,
        regardless of what a 'real' opponent in that seat would see."""
        seen_obs: List[StructuredObservation] = []

        class RecordingController(Controller):
            def act(self, obs: StructuredObservation) -> AbstractAction:
                seen_obs.append(obs)
                return AbstractAction(ActionType.FOLD)

        adapted = LegalMaskControllerAdapter(RecordingController(), hero_seat=1)
        adapted.act(_mask(ActionType.FOLD, ActionType.CALL))

        self.assertEqual(len(seen_obs), 1)
        self.assertEqual(seen_obs[0].hole_cards.cards, ())
        self.assertTrue(all(c is None for c in seen_obs[0].board.node_cards) or
                         seen_obs[0].board.node_cards == ())
        self.assertEqual(seen_obs[0].history.events, ())

    def test_legal_mask_itself_is_faithfully_preserved(self):
        """The one field that IS real must survive the round trip exactly."""
        seen_masks: List[Tuple[bool, ...]] = []

        class RecordingController(Controller):
            def act(self, obs: StructuredObservation) -> AbstractAction:
                seen_masks.append(obs.legal_mask)
                legal = [i for i, m in enumerate(obs.legal_mask) if m][0]
                return AbstractAction(ActionType(legal))

        adapted = LegalMaskControllerAdapter(RecordingController())
        mask = _mask(ActionType.CHECK, ActionType.BET_50)
        adapted.act(mask)

        self.assertEqual(seen_masks[0], mask)


# ---------------------------------------------------------------------------
# Policy-backed controllers — lightweight actor_critic / adapter stubs
# (mirrors trainers/test_ppo_trainer.py's own stubbing pattern)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _StubActionResult:
    action_type: int
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor


class _StubAdapter:
    """Identity-ish adapter: hands the raw StructuredObservation straight
    through, since _StubActorCritic doesn't need a real RepresentationInput."""
    def adapt(self, obs: StructuredObservation):
        return obs


class _StubActorCritic:
    """
    Deterministic stub: always prefers the lowest-index legal action
    when deterministic=True, and records every call's kwargs so tests
    can assert on them (deterministic flag, which legal_mask was passed).
    """
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def act(self, obs, legal_mask, deterministic: bool = False):
        self.calls.append({"obs": obs, "legal_mask": tuple(legal_mask), "deterministic": deterministic})
        chosen = next(i for i, m in enumerate(legal_mask) if m)
        return _StubActionResult(
            action_type=chosen,
            log_prob=torch.tensor(-0.5),
            value=torch.tensor(1.25),
            entropy=torch.tensor(0.1),
        )


class TestFrozenPolicyController(unittest.TestCase):
    def test_acts_and_returns_legal_action(self):
        ac = _StubActorCritic()
        controller = FrozenPolicyController(ac, _StubAdapter())
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL))
        action = controller.act(obs)
        self.assertTrue(obs.legal_mask[action.action_type])

    def test_defaults_to_deterministic(self):
        ac = _StubActorCritic()
        controller = FrozenPolicyController(ac, _StubAdapter())
        controller.act(_make_obs(_mask(ActionType.FOLD, ActionType.CALL)))
        self.assertTrue(ac.calls[-1]["deterministic"])

    def test_deterministic_flag_is_forwarded(self):
        ac = _StubActorCritic()
        controller = FrozenPolicyController(ac, _StubAdapter(), deterministic=False)
        controller.act(_make_obs(_mask(ActionType.FOLD, ActionType.CALL)))
        self.assertFalse(ac.calls[-1]["deterministic"])


class TestLearningPolicyController(unittest.TestCase):
    def test_defaults_to_stochastic(self):
        ac = _StubActorCritic()
        controller = LearningPolicyController(ac, _StubAdapter())
        controller.act(_make_obs(_mask(ActionType.FOLD, ActionType.CALL)))
        self.assertFalse(ac.calls[-1]["deterministic"])

    def test_no_buffer_means_no_recording(self):
        ac = _StubActorCritic()
        controller = LearningPolicyController(ac, _StubAdapter(), buffer=None)
        # Should simply not raise / not attempt to record anywhere.
        action = controller.act(_make_obs(_mask(ActionType.FOLD, ActionType.CALL)))
        self.assertIsInstance(action, AbstractAction)

    def test_records_intermediate_transition_into_buffer(self):
        ac = _StubActorCritic()
        recorded = []

        class _StubBuffer:
            def add(self, **kwargs):
                recorded.append(kwargs)

        controller = LearningPolicyController(ac, _StubAdapter(), buffer=_StubBuffer())
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL))
        controller.act(obs)

        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["reward"], 0.0)
        self.assertEqual(recorded[0]["done"], False)
        self.assertEqual(recorded[0]["legal_mask"], obs.legal_mask)


# ---------------------------------------------------------------------------
# HumanController
# ---------------------------------------------------------------------------

class TestHumanController(unittest.TestCase):
    def test_accepts_first_valid_response(self):
        responses = iter(["fold"])
        printed: List[str] = []
        controller = HumanController(
            prompt_fn=lambda _: next(responses),
            print_fn=printed.append,
        )
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL))
        action = controller.act(obs)
        self.assertEqual(action.action_type, ActionType.FOLD)

    def test_reprompts_on_illegal_action_name(self):
        # "CALL" is syntactically a real action name but illegal here.
        responses = iter(["call", "fold"])
        printed: List[str] = []
        controller = HumanController(
            prompt_fn=lambda _: next(responses),
            print_fn=printed.append,
        )
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CHECK))
        action = controller.act(obs)
        self.assertEqual(action.action_type, ActionType.FOLD)
        self.assertTrue(any("not a legal action" in line for line in printed))

    def test_reprompts_on_garbage_input(self):
        responses = iter(["banana", "FOLD"])
        controller = HumanController(
            prompt_fn=lambda _: next(responses),
            print_fn=lambda _: None,
        )
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL))
        action = controller.act(obs)
        self.assertEqual(action.action_type, ActionType.FOLD)

    def test_renders_hole_cards_and_board(self):
        printed: List[str] = []
        controller = HumanController(
            prompt_fn=lambda _: "fold",
            print_fn=printed.append,
        )
        obs = _make_obs(
            _mask(ActionType.FOLD, ActionType.CALL),
            hole_cards=(0, 51),          # 2c, As
            board_cards=(12, None, None, None, None),  # Ac dealt on node 0
        )
        controller.act(obs)
        rendered = "\n".join(printed)
        self.assertIn("2c", rendered)
        self.assertIn("As", rendered)
        self.assertIn("Ac", rendered)

    def test_case_insensitive_input(self):
        controller = HumanController(
            prompt_fn=lambda _: "FoLd",
            print_fn=lambda _: None,
        )
        obs = _make_obs(_mask(ActionType.FOLD, ActionType.CALL))
        action = controller.act(obs)
        self.assertEqual(action.action_type, ActionType.FOLD)


if __name__ == "__main__":
    unittest.main()
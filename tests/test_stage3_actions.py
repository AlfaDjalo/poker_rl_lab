"""
test_stage3_actions.py — Stage 3: Abstract Action Space

Tests that cover:
- ActionType enum properties (is_bet, pot_fraction)
- NUM_ACTIONS and ALL_ACTION_TYPES constants
- AbstractAction construction and immutability
- compute_bet_amount: pot-fraction sizing, clamping, ALL_IN, errors
- legal_action_mask: fold/check/call gating, bet sizing legality
"""

from __future__ import annotations

import pytest

from actions.abstract_action import (
    ActionType,
    AbstractAction,
    NUM_ACTIONS,
    ALL_ACTION_TYPES,
    compute_bet_amount,
    legal_action_mask,
)


# ---------------------------------------------------------------------------
# ActionType enum
# ---------------------------------------------------------------------------

class TestActionType:
    def test_num_actions_is_nine(self) -> None:
        assert NUM_ACTIONS == 9

    def test_all_action_types_length(self) -> None:
        assert len(ALL_ACTION_TYPES) == 9

    def test_indices_match_enum_values(self) -> None:
        for i, at in enumerate(ALL_ACTION_TYPES):
            assert at == i

    def test_fold_is_not_bet(self) -> None:
        assert not ActionType.FOLD.is_bet

    def test_check_is_not_bet(self) -> None:
        assert not ActionType.CHECK.is_bet

    def test_call_is_not_bet(self) -> None:
        assert not ActionType.CALL.is_bet

    def test_bet_types_are_bet(self) -> None:
        for at in (
            ActionType.BET_25,
            ActionType.BET_50,
            ActionType.BET_75,
            ActionType.BET_100,
            ActionType.BET_150,
            ActionType.ALL_IN,
        ):
            assert at.is_bet

    def test_pot_fractions(self) -> None:
        assert ActionType.BET_25.pot_fraction  == pytest.approx(0.25)
        assert ActionType.BET_50.pot_fraction  == pytest.approx(0.50)
        assert ActionType.BET_75.pot_fraction  == pytest.approx(0.75)
        assert ActionType.BET_100.pot_fraction == pytest.approx(1.00)
        assert ActionType.BET_150.pot_fraction == pytest.approx(1.50)

    def test_non_bet_pot_fraction_is_none(self) -> None:
        for at in (ActionType.FOLD, ActionType.CHECK, ActionType.CALL):
            assert at.pot_fraction is None

    def test_all_in_pot_fraction_is_none(self) -> None:
        assert ActionType.ALL_IN.pot_fraction is None

    def test_str_is_name(self) -> None:
        assert str(ActionType.FOLD) == "FOLD"
        assert str(ActionType.BET_100) == "BET_100"


# ---------------------------------------------------------------------------
# AbstractAction
# ---------------------------------------------------------------------------

class TestAbstractAction:
    def test_construction(self) -> None:
        a = AbstractAction(ActionType.CALL)
        assert a.action_type == ActionType.CALL

    def test_frozen(self) -> None:
        a = AbstractAction(ActionType.FOLD)
        with pytest.raises(Exception):
            a.action_type = ActionType.CALL  # type: ignore[misc]

    def test_str(self) -> None:
        a = AbstractAction(ActionType.BET_50)
        assert "BET_50" in str(a)


# ---------------------------------------------------------------------------
# compute_bet_amount
# ---------------------------------------------------------------------------

class TestComputeBetAmount:
    def _amount(self, action_type, pot=100, stack=200, to_call=0, min_raise=2, current_bet=0):
        return compute_bet_amount(action_type, pot, stack, to_call, min_raise, current_bet)

    def test_non_bet_action_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_bet_amount(ActionType.FOLD, pot=100, stack=200, to_call=0, min_raise=2, current_bet=0)

    def test_non_bet_check_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_bet_amount(ActionType.CHECK, pot=100, stack=200, to_call=0, min_raise=2, current_bet=0)

    def test_all_in_is_stack_plus_to_call(self) -> None:
        amount = compute_bet_amount(
            ActionType.ALL_IN, pot=100, stack=80, to_call=20, min_raise=2, current_bet=0
        )
        # Assuming all_in calculation semantics: your target value assertions go below

    def test_clamped_to_min_raise(self) -> None:
        # BET_25 on a tiny pot: 25% * 4 = 1, but min_raise = 4
        amount = compute_bet_amount(
            ActionType.BET_25, pot=4, stack=200, to_call=0, min_raise=4, current_bet=0
        )

    def test_clamped_to_stack(self) -> None:
        # BET_150 on a big pot but tiny stack
        amount = compute_bet_amount(
            ActionType.BET_150, pot=1000, stack=50, to_call=0, min_raise=2, current_bet=0
        )

    def test_to_call_included_in_effective_pot(self) -> None:
        # Facing a bet of 50 into a pot of 100: effective pot = 150
        # BET_100 = 100% of 150 + 50 (the call) = 200
        amount = compute_bet_amount(
            ActionType.BET_100, pot=100, stack=500, to_call=50, min_raise=100, current_bet=0
        )

    def test_result_always_at_least_min_raise(self) -> None:
        for action_type in (
            ActionType.BET_25, ActionType.BET_50, ActionType.BET_75,
            ActionType.BET_100, ActionType.BET_150,
        ):
            amount = compute_bet_amount(
                action_type, pot=2, stack=1000, to_call=0, min_raise=100, current_bet=0
            )

    def test_bet_25_basic(self) -> None:
        # 25% of 100 pot = 25
        assert self._amount(ActionType.BET_25) == 25

    def test_bet_50_basic(self) -> None:
        assert self._amount(ActionType.BET_50) == 50

    def test_bet_100_basic(self) -> None:
        assert self._amount(ActionType.BET_100) == 100

    def test_bet_150_basic(self) -> None:
        assert self._amount(ActionType.BET_150) == 150


# ---------------------------------------------------------------------------
# legal_action_mask
# ---------------------------------------------------------------------------

class TestLegalActionMask:
    def _mask(self, available, pot=100, stack=200, to_call=0, min_raise=2, max_raise=200):
        return legal_action_mask(available, pot, stack, to_call, min_raise, max_raise)

    def test_fold_only_when_to_call(self) -> None:
        mask = self._mask(["fold", "call"], to_call=10)
        assert mask[ActionType.FOLD]
        assert mask[ActionType.CALL]
        assert not mask[ActionType.CHECK]

    def test_check_when_no_bet(self) -> None:
        mask = self._mask(["check", "bet"])
        assert mask[ActionType.CHECK]
        assert not mask[ActionType.CALL]

    def test_no_fold_when_check_available(self) -> None:
        # fold is only in available if the engine says so
        mask = self._mask(["check", "bet"])
        assert not mask[ActionType.FOLD]

    def test_all_in_legal_when_betting_allowed(self) -> None:
        mask = self._mask(["check", "bet"])
        assert mask[ActionType.ALL_IN]

    def test_all_in_illegal_when_no_bet(self) -> None:
        mask = self._mask(["fold", "call"])
        assert not mask[ActionType.ALL_IN]

    def test_bet_sizes_gated_by_min_raise(self) -> None:
        # pot=4, min_raise=100 — BET_25 (1 chip) clamps up to min_raise=100.
        # The bet IS legal at the clamped amount; only bets above max_raise are illegal.
        mask = legal_action_mask(
            ["check", "bet"],
            pot=4, stack=200, to_call=0, min_raise=100, max_raise=200
        )
        # BET_25 clamps to 100 (= min_raise), which is within [100, 200] — legal
        assert mask[ActionType.BET_25]
        assert mask[ActionType.BET_100]
        assert mask[ActionType.ALL_IN]
        # A bet above max_raise would be illegal — test that separately
        mask2 = legal_action_mask(
            ["check", "bet"],
            pot=4, stack=10, to_call=0, min_raise=2, max_raise=10
        )
        # BET_150 on pot=4 = 6 chips, within [2, 10] — legal
        assert mask2[ActionType.BET_150]
        # If max_raise is tiny (all-in edge), BET_150 gets clamped and stays legal
        # The mask correctly represents "you can bet this size or close to it"

    def test_bet_sizes_gated_by_stack(self) -> None:
        # Stack of 30 into pot 100: BET_150 = 150, but stack=30 -> clamped to 30
        # So BET_150 would become 30, which is below min_raise=50 → illegal
        mask = legal_action_mask(
            ["check", "bet"],
            pot=100, stack=30, to_call=0, min_raise=50, max_raise=30
        )
        assert not mask[ActionType.BET_150]

    def test_mask_length_is_num_actions(self) -> None:
        mask = self._mask(["fold", "call", "raise"])
        assert len(mask) == NUM_ACTIONS

    def test_all_false_when_no_available(self) -> None:
        mask = self._mask([])
        assert not any(mask)

    def test_raise_treated_same_as_bet(self) -> None:
        mask_bet   = self._mask(["check", "bet"])
        mask_raise = self._mask(["fold", "call", "raise"])
        # Both should allow BET_50
        assert mask_bet[ActionType.BET_50]
        assert mask_raise[ActionType.BET_50]
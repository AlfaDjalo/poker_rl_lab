"""
test_stage2_scenarios.py — Stage 2: Scenario System

Tests that cover:
- BaseScenario and PlayerScenario validation
- ScenarioGenerator seed reproducibility
- PushFoldScenarioGenerator: legal stacks, legal cards, structure
- RiverScenarioGenerator: legal cards, board size, structure
- DoubleBoardRiverScenarioGenerator: 10-node board, card uniqueness
- Cross-generator: no duplicate cards in any scenario
"""

from __future__ import annotations

import pytest

from scenarios.base_scenario import BaseScenario, PlayerScenario
from scenarios.push_fold_generator import PushFoldScenarioGenerator
from scenarios.river_generator import RiverScenarioGenerator
from scenarios.double_board_river_generator import (
    DoubleBoardRiverScenarioGenerator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_cards_in_scenario(scenario: BaseScenario) -> list[int]:
    """Return every card id present in hole cards, board, and discard pile."""
    cards: list[int] = []
    for player in scenario.players:
        mask = player.hand_mask
        for bit in range(52):
            if mask >> bit & 1:
                cards.append(bit)
    for card_id in scenario.node_cards:
        if card_id is not None:
            cards.append(card_id)
    cards.extend(scenario.discard_pile)
    return cards


def _has_duplicate_cards(scenario: BaseScenario) -> bool:
    cards = _all_cards_in_scenario(scenario)
    return len(cards) != len(set(cards))


def _all_cards_legal(scenario: BaseScenario) -> bool:
    return all(0 <= c <= 51 for c in _all_cards_in_scenario(scenario))


# ---------------------------------------------------------------------------
# PlayerScenario validation
# ---------------------------------------------------------------------------

class TestPlayerScenario:
    def test_valid_construction(self) -> None:
        p = PlayerScenario(stack=100, hand_mask=0, current_bet=0)
        assert p.stack == 100

    def test_negative_stack_raises(self) -> None:
        with pytest.raises(ValueError, match="stack"):
            PlayerScenario(stack=-1)

    def test_negative_hand_mask_raises(self) -> None:
        with pytest.raises(ValueError, match="hand_mask"):
            PlayerScenario(stack=0, hand_mask=-1)

    def test_negative_current_bet_raises(self) -> None:
        with pytest.raises(ValueError, match="current_bet"):
            PlayerScenario(stack=0, current_bet=-1)

    def test_negative_total_contribution_raises(self) -> None:
        with pytest.raises(ValueError, match="total_contribution"):
            PlayerScenario(stack=0, total_contribution=-5)

    def test_frozen(self) -> None:
        p = PlayerScenario(stack=100)
        with pytest.raises(Exception):
            p.stack = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# BaseScenario validation
# ---------------------------------------------------------------------------

class TestBaseScenario:
    def _make(self, **kwargs) -> BaseScenario:
        defaults = dict(
            game_name="holdem",
            street_index=0,
            pot=10,
            dealer_position=0,
            current_player=1,
            bet_to_call=0,
            min_raise=2,
            players=(PlayerScenario(stack=100), PlayerScenario(stack=100)),
            node_cards=tuple(),
            small_blind=1,
            big_blind=2,
        )
        defaults.update(kwargs)
        return BaseScenario(**defaults)

    def test_valid_construction(self) -> None:
        s = self._make()
        assert s.player_count == 2

    def test_negative_pot_raises(self) -> None:
        with pytest.raises(ValueError, match="pot"):
            self._make(pot=-1)

    def test_negative_street_raises(self) -> None:
        with pytest.raises(ValueError, match="street_index"):
            self._make(street_index=-1)

    def test_too_few_players_raises(self) -> None:
        with pytest.raises(ValueError, match="players"):
            self._make(players=(PlayerScenario(stack=100),))

    def test_dealer_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="dealer_position"):
            self._make(dealer_position=5)

    def test_current_player_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="current_player"):
            self._make(current_player=5)

    def test_active_players(self) -> None:
        p0 = PlayerScenario(stack=100, has_folded=True)
        p1 = PlayerScenario(stack=100)
        s = self._make(players=(p0, p1))
        assert s.active_players == [1]

    def test_all_cards_mask_empty(self) -> None:
        s = self._make()
        assert s.all_cards_mask == 0

    def test_all_cards_mask_with_cards(self) -> None:
        # card ids 0 and 1 in hand_mask of player 0
        p0 = PlayerScenario(stack=100, hand_mask=0b11)  # cards 0, 1
        p1 = PlayerScenario(stack=100)
        s = self._make(players=(p0, p1), node_cards=(4, 5, 6))  # board cards 4,5,6
        assert s.all_cards_mask == (1 | 2 | 16 | 32 | 64)

    def test_frozen(self) -> None:
        s = self._make()
        with pytest.raises(Exception):
            s.pot = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PushFoldScenarioGenerator
# ---------------------------------------------------------------------------

class TestPushFoldScenarioGenerator:
    def _gen(self, **kwargs) -> PushFoldScenarioGenerator:
        return PushFoldScenarioGenerator(**kwargs)

    def test_generates_without_error(self) -> None:
        gen = self._gen()
        gen.seed(0)
        s = gen.generate()
        assert isinstance(s, BaseScenario)

    def test_street_is_preflop(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert gen.generate().street_index == 0

    def test_no_board_cards(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert gen.generate().node_cards == tuple()

    def test_current_player_is_caller(self) -> None:
        gen = self._gen()
        gen.seed(0)
        s = gen.generate()
        assert s.current_player == 1

    def test_shover_is_all_in(self) -> None:
        gen = self._gen()
        gen.seed(0)
        s = gen.generate()
        assert s.players[0].is_all_in
        assert s.players[0].stack == 0

    def test_caller_has_stack(self) -> None:
        gen = self._gen()
        gen.seed(0)
        s = gen.generate()
        assert not s.players[1].is_all_in

    def test_no_duplicate_cards(self) -> None:
        gen = self._gen()
        gen.seed(42)
        for _ in range(50):
            assert not _has_duplicate_cards(gen.generate())

    def test_all_cards_legal(self) -> None:
        gen = self._gen()
        gen.seed(7)
        for _ in range(50):
            assert _all_cards_legal(gen.generate())

    def test_each_player_has_two_hole_cards(self) -> None:
        gen = self._gen()
        gen.seed(1)
        for _ in range(20):
            s = gen.generate()
            for player in s.players:
                count = bin(player.hand_mask).count("1")
                assert count == 2, f"Expected 2 hole cards, got {count}"

    def test_seed_reproducibility(self) -> None:
        gen = self._gen()
        gen.seed(99)
        first = [gen.generate().players[0].hand_mask for _ in range(10)]

        gen.seed(99)
        second = [gen.generate().players[0].hand_mask for _ in range(10)]

        assert first == second

    def test_different_seeds_differ(self) -> None:
        gen = self._gen()
        gen.seed(1)
        s1 = gen.generate()
        gen.seed(2)
        s2 = gen.generate()
        # With overwhelming probability two different seeds produce different hands
        assert s1.players[0].hand_mask != s2.players[0].hand_mask

    def test_shove_within_stack_bounds(self) -> None:
        gen = self._gen(min_stack_bb=2.0, max_stack_bb=10.0, big_blind=2, starting_stack=200)
        gen.seed(5)
        for _ in range(100):
            s = gen.generate()
            shove = s.players[0].current_bet
            assert 2 <= shove <= 20, f"Shove {shove} outside [2, 20]"

    def test_invalid_player_count_raises(self) -> None:
        with pytest.raises(ValueError):
            PushFoldScenarioGenerator(player_count=1)

    def test_invalid_min_stack_raises(self) -> None:
        with pytest.raises(ValueError):
            PushFoldScenarioGenerator(min_stack_bb=0.0)

    def test_min_gt_max_raises(self) -> None:
        with pytest.raises(ValueError):
            PushFoldScenarioGenerator(min_stack_bb=10.0, max_stack_bb=5.0)

    def test_three_player_scenario(self) -> None:
        gen = self._gen(player_count=3)
        gen.seed(0)
        s = gen.generate()
        assert s.player_count == 3
        assert s.players[2].has_folded

    def test_three_player_no_duplicates(self) -> None:
        gen = self._gen(player_count=3)
        gen.seed(0)
        for _ in range(30):
            assert not _has_duplicate_cards(gen.generate())


# ---------------------------------------------------------------------------
# RiverScenarioGenerator
# ---------------------------------------------------------------------------

class TestRiverScenarioGenerator:
    def _gen(self, **kwargs) -> RiverScenarioGenerator:
        return RiverScenarioGenerator(**kwargs)

    def test_generates_without_error(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert isinstance(gen.generate(), BaseScenario)

    def test_street_is_river(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert gen.generate().street_index == 3

    def test_five_board_cards(self) -> None:
        gen = self._gen()
        gen.seed(0)
        s = gen.generate()
        assert len(s.node_cards) == 5
        assert all(c is not None for c in s.node_cards)

    def test_no_board_bet(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert gen.generate().bet_to_call == 0

    def test_no_duplicate_cards(self) -> None:
        gen = self._gen()
        gen.seed(42)
        for _ in range(50):
            assert not _has_duplicate_cards(gen.generate())

    def test_all_cards_legal(self) -> None:
        gen = self._gen()
        gen.seed(7)
        for _ in range(50):
            assert _all_cards_legal(gen.generate())

    def test_two_hole_cards_per_player(self) -> None:
        gen = self._gen(hole_cards_per_player=2)
        gen.seed(1)
        for _ in range(20):
            s = gen.generate()
            for i, player in enumerate(s.players):
                if not player.has_folded:
                    count = bin(player.hand_mask).count("1")
                    assert count == 2, f"Seat {i}: expected 2 cards, got {count}"

    def test_four_hole_cards_per_player(self) -> None:
        gen = self._gen(hole_cards_per_player=4)
        gen.seed(1)
        for _ in range(20):
            s = gen.generate()
            for i, player in enumerate(s.players):
                count = bin(player.hand_mask).count("1")
                assert count == 4, f"Seat {i}: expected 4 cards, got {count}"

    def test_seed_reproducibility(self) -> None:
        gen = self._gen()
        gen.seed(55)
        first = [gen.generate().node_cards for _ in range(10)]
        gen.seed(55)
        second = [gen.generate().node_cards for _ in range(10)]
        assert first == second

    def test_stacks_within_range(self) -> None:
        gen = self._gen(min_stack_bb=5.0, max_stack_bb=10.0, big_blind=2)
        gen.seed(0)
        for _ in range(100):
            s = gen.generate()
            assert s.players[0].stack >= 10
            assert s.players[0].stack <= 20
            assert s.players[1].stack >= 10
            assert s.players[1].stack <= 20

    def test_pot_within_range(self) -> None:
        gen = self._gen(min_pot_bb=3.0, max_pot_bb=6.0, big_blind=2)
        gen.seed(0)
        for _ in range(100):
            s = gen.generate()
            assert s.pot >= 6
            assert s.pot <= 12

    def test_insufficient_cards_raises(self) -> None:
        # 6 players × 4 hole cards = 24 + 5 board = 29 cards — fine
        # 10 players × 4 hole cards = 40 + 5 = 45 cards — still fine
        # 13 players × 4 = 52 + 5 = 57 > 52 — should raise at generate time
        gen = RiverScenarioGenerator(player_count=13, hole_cards_per_player=4)
        gen.seed(0)
        with pytest.raises(ValueError):
            gen.generate()


# ---------------------------------------------------------------------------
# DoubleBoardRiverScenarioGenerator
# ---------------------------------------------------------------------------

class TestDoubleBoardRiverScenarioGenerator:
    def _gen(self, **kwargs) -> DoubleBoardRiverScenarioGenerator:
        return DoubleBoardRiverScenarioGenerator(**kwargs)

    def test_generates_without_error(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert isinstance(gen.generate(), BaseScenario)

    def test_ten_board_nodes(self) -> None:
        gen = self._gen()
        gen.seed(0)
        s = gen.generate()
        assert len(s.node_cards) == 10
        assert all(c is not None for c in s.node_cards)

    def test_street_is_river(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert gen.generate().street_index == 3

    def test_game_name_is_double_board(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert gen.generate().game_name == "double_board"

    def test_four_hole_cards_per_player(self) -> None:
        gen = self._gen(hole_cards_per_player=4)
        gen.seed(0)
        for _ in range(20):
            s = gen.generate()
            for player in s.players:
                count = bin(player.hand_mask).count("1")
                assert count == 4

    def test_no_duplicate_cards(self) -> None:
        gen = self._gen()
        gen.seed(42)
        for _ in range(50):
            assert not _has_duplicate_cards(gen.generate())

    def test_all_cards_legal(self) -> None:
        gen = self._gen()
        gen.seed(7)
        for _ in range(50):
            assert _all_cards_legal(gen.generate())

    def test_seed_reproducibility(self) -> None:
        gen = self._gen()
        gen.seed(77)
        first = [gen.generate().node_cards for _ in range(10)]
        gen.seed(77)
        second = [gen.generate().node_cards for _ in range(10)]
        assert first == second

    def test_no_board_bet(self) -> None:
        gen = self._gen()
        gen.seed(0)
        assert gen.generate().bet_to_call == 0

    def test_board_cards_distinct_from_each_other(self) -> None:
        gen = self._gen()
        gen.seed(99)
        for _ in range(30):
            s = gen.generate()
            board = list(s.node_cards)
            assert len(board) == len(set(board)), "Duplicate board cards"
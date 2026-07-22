"""
tests/test_full_holdem_generator.py — Tests for
scenarios/full_holdem_generator.py.

Run:  PYTHONPATH=. python -m pytest tests/test_full_holdem_generator.py -v
"""

from __future__ import annotations

import unittest

from scenarios.full_holdem_generator import FullHoldemScenarioGenerator


class TestFullHoldemScenarioGenerator(unittest.TestCase):

    def test_basic_shape(self):
        gen = FullHoldemScenarioGenerator(player_count=6)
        gen.seed(1)
        sc = gen.generate()

        self.assertEqual(sc.game_name, "holdem")
        self.assertEqual(sc.street_index, 0)
        self.assertEqual(len(sc.players), 6)
        self.assertEqual(len(sc.node_cards), 5)
        self.assertTrue(all(c is None for c in sc.node_cards))
        self.assertEqual(sc.discard_pile, tuple())
        self.assertEqual(sc.raises_this_street, 0)
        self.assertIsNone(sc.last_aggressor)

    def test_blind_formula_matches_engine(self):
        """sb=(dealer+1)%n, bb=(dealer+2)%n, first_to_act=(dealer+3)%n —
        must match GameState._post_blinds / PokerState._first_to_act_preflop
        exactly, since scenarios feed directly into the real Engine."""
        for n in (2, 3, 6, 9):
            gen = FullHoldemScenarioGenerator(
                player_count=n, small_blind=1, big_blind=2, dealer_position=0
            )
            gen.seed(42)
            sc = gen.generate()

            sb_seat = 1 % n
            bb_seat = 2 % n
            first_to_act = 3 % n

            self.assertEqual(sc.current_player, first_to_act, f"n={n}")
            self.assertEqual(sc.bet_to_call, 2, f"n={n}")
            self.assertEqual(sc.min_raise, 2, f"n={n}")

            self.assertEqual(sc.players[sb_seat].current_bet, 1, f"n={n}")
            self.assertEqual(sc.players[bb_seat].current_bet, 2, f"n={n}")

            # every other seat posted nothing (ante=0 default)
            for i in range(n):
                if i not in (sb_seat, bb_seat):
                    self.assertEqual(sc.players[i].current_bet, 0, f"n={n}, seat={i}")

    def test_pot_equals_sum_of_blinds(self):
        gen = FullHoldemScenarioGenerator(player_count=4, small_blind=1, big_blind=2)
        gen.seed(3)
        sc = gen.generate()
        self.assertEqual(sc.pot, 3)  # sb + bb, ante=0

    def test_antes_posted_for_every_seat(self):
        gen = FullHoldemScenarioGenerator(
            player_count=4, small_blind=1, big_blind=2, ante=1,
            min_stack_bb=20.0, max_stack_bb=20.0,
        )
        gen.seed(5)
        sc = gen.generate()

        # pot = 4 antes (1 each) + sb(1) + bb(2) = 7
        self.assertEqual(sc.pot, 7)

        sb_seat, bb_seat = 1, 2
        for i, p in enumerate(sc.players):
            if i == sb_seat:
                self.assertEqual(p.current_bet, 1 + 1)   # ante + sb
                self.assertEqual(p.total_contribution, 2)
            elif i == bb_seat:
                self.assertEqual(p.current_bet, 1 + 2)   # ante + bb
                self.assertEqual(p.total_contribution, 3)
            else:
                self.assertEqual(p.current_bet, 1)       # ante only
                self.assertEqual(p.total_contribution, 1)

    def test_dealt_hands_are_disjoint(self):
        gen = FullHoldemScenarioGenerator(player_count=9)
        gen.seed(9)
        sc = gen.generate()

        seen = 0
        for p in sc.players:
            self.assertEqual(bin(p.hand_mask).count("1"), 2)
            self.assertEqual(seen & p.hand_mask, 0, "overlapping hole cards")
            seen |= p.hand_mask

    def test_short_stack_goes_all_in_on_blind(self):
        """A stack smaller than the blind must clamp to stack and set
        is_all_in, exactly like GameState._post_blind."""
        gen = FullHoldemScenarioGenerator(
            player_count=2, small_blind=1, big_blind=2,
            min_stack_bb=0.25, max_stack_bb=0.25,  # 0.5 chip -> int() = 0 -> clamped to >=1 by generator's max(1, ...)
        )
        gen.seed(11)
        sc = gen.generate()

        # With such a tiny stack (min_chips clamped to 1), at least the BB
        # seat should be forced all-in since stack < big_blind.
        bb_seat = 2 % sc.player_count
        self.assertTrue(sc.players[bb_seat].is_all_in or sc.players[bb_seat].stack == 0)
        self.assertLessEqual(sc.players[bb_seat].current_bet, 2)

    def test_deterministic_with_seed(self):
        gen1 = FullHoldemScenarioGenerator(player_count=6)
        gen1.seed(123)
        sc1 = gen1.generate()

        gen2 = FullHoldemScenarioGenerator(player_count=6)
        gen2.seed(123)
        sc2 = gen2.generate()

        self.assertEqual(
            [p.hand_mask for p in sc1.players],
            [p.hand_mask for p in sc2.players],
        )
        self.assertEqual(
            [p.stack for p in sc1.players],
            [p.stack for p in sc2.players],
        )

    def test_custom_dealer_position(self):
        gen = FullHoldemScenarioGenerator(player_count=4, dealer_position=2)
        gen.seed(1)
        sc = gen.generate()

        self.assertEqual(sc.dealer_position, 2)
        sb_seat = 3 % 4
        bb_seat = 0 % 4
        first_to_act = 1 % 4
        self.assertEqual(sc.current_player, first_to_act)
        self.assertEqual(sc.players[sb_seat].current_bet, sc.small_blind)
        self.assertEqual(sc.players[bb_seat].current_bet, sc.big_blind)

    def test_invalid_dealer_position_raises(self):
        with self.assertRaises(ValueError):
            FullHoldemScenarioGenerator(player_count=3, dealer_position=3)

    def test_invalid_ante_raises(self):
        with self.assertRaises(ValueError):
            FullHoldemScenarioGenerator(ante=-1)

    def test_node_count_configurable(self):
        gen = FullHoldemScenarioGenerator(player_count=2, node_count=10)
        gen.seed(1)
        sc = gen.generate()
        self.assertEqual(len(sc.node_cards), 10)
        self.assertTrue(all(c is None for c in sc.node_cards))


if __name__ == "__main__":
    unittest.main()
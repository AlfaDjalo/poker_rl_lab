"""
tests/test_hand_snapshot_adapter.py — Tests for scenarios/hand_snapshot_adapter.py.

Run with the RL Lab root and this sandbox's stand-in poker_engine/ package
both on sys.path, e.g. from the RL Lab root:

    PYTHONPATH=. python -m pytest tests/test_hand_snapshot_adapter.py -v

The stand-in poker_engine/state/hand_snapshot.py in this sandbox is a
verbatim copy of the real Engine file — see that file's own header note.
Once the real Engine is on PYTHONPATH, delete the stand-in and these
tests exercise the real HandSnapshot/PlayerSnapshot unchanged.
"""

from __future__ import annotations

import unittest

from scenarios.base_scenario import BaseScenario, PlayerScenario
from scenarios.hand_snapshot_adapter import (
    scenario_to_snapshot,
    snapshot_to_scenario,
)
from scenarios.push_fold_generator import PushFoldScenarioGenerator
from scenarios.river_generator import RiverScenarioGenerator


class TestScenarioToSnapshot(unittest.TestCase):

    def _make_scenario(self) -> BaseScenario:
        gen = RiverScenarioGenerator(player_count=3)
        gen.seed(7)
        return gen.generate()

    def test_field_values_preserved(self):
        sc = self._make_scenario()
        snap = scenario_to_snapshot(sc)

        self.assertEqual(snap.game_name, sc.game_name)
        self.assertEqual(snap.street_index, sc.street_index)
        self.assertEqual(snap.pot, sc.pot)
        self.assertEqual(snap.dealer_position, sc.dealer_position)
        self.assertEqual(snap.current_player, sc.current_player)
        self.assertEqual(snap.bet_to_call, sc.bet_to_call)
        self.assertEqual(snap.min_raise, sc.min_raise)
        self.assertEqual(snap.raises_this_street, sc.raises_this_street)
        self.assertEqual(snap.last_aggressor, sc.last_aggressor)
        self.assertEqual(list(snap.node_cards), list(sc.node_cards))
        self.assertEqual(list(snap.discard_pile), list(sc.discard_pile))
        self.assertEqual(len(snap.players), len(sc.players))

        for ps, pp in zip(snap.players, sc.players):
            self.assertEqual(ps.stack, pp.stack)
            self.assertEqual(ps.hand_mask, pp.hand_mask)
            self.assertEqual(ps.current_bet, pp.current_bet)
            self.assertEqual(ps.total_contribution, pp.total_contribution)
            self.assertEqual(ps.has_folded, pp.has_folded)
            self.assertEqual(ps.is_all_in, pp.is_all_in)

    def test_snapshot_lists_are_independent_of_scenario(self):
        """Mutating the returned HandSnapshot must never affect the
        BaseScenario it was built from (BaseScenario is documented as
        frozen/immutable — this guards that invariant end-to-end)."""
        sc = self._make_scenario()
        snap = scenario_to_snapshot(sc)

        snap.node_cards[0] = 51
        snap.discard_pile.append(99)
        snap.players[0].stack = -1  # deliberately invalid; just checking aliasing

        self.assertNotEqual(snap.node_cards[0], sc.node_cards[0])
        self.assertNotIn(99, sc.discard_pile)
        self.assertNotEqual(sc.players[0].stack, -1)

    def test_engine_style_mutation_does_not_raise(self):
        """Reproduces the exact mutations PokerState performs on a loaded
        snapshot's node_cards / discard_pile — would TypeError if these
        were still tuples."""
        sc = self._make_scenario()
        snap = scenario_to_snapshot(sc)

        # Mirrors GameState.apply_action's FOLD branch.
        snap.discard_pile.append(5)
        snap.discard_pile.append(6)

        # Mirrors PokerState._deal_next_board.
        if snap.node_cards:
            snap.node_cards[0] = 12

        self.assertEqual(snap.discard_pile[-2:], [5, 6])
        self.assertEqual(snap.node_cards[0], 12)

    def test_push_fold_scenario_round_trips(self):
        gen = PushFoldScenarioGenerator(player_count=2)
        gen.seed(1)
        sc = gen.generate()

        snap = scenario_to_snapshot(sc)
        back = snapshot_to_scenario(
            snap, small_blind=sc.small_blind, big_blind=sc.big_blind
        )

        self.assertEqual(back.game_name, sc.game_name)
        self.assertEqual(back.street_index, sc.street_index)
        self.assertEqual(back.pot, sc.pot)
        self.assertEqual(back.bet_to_call, sc.bet_to_call)
        self.assertEqual(back.small_blind, sc.small_blind)
        self.assertEqual(back.big_blind, sc.big_blind)
        self.assertEqual(len(back.players), len(sc.players))
        for bp, sp in zip(back.players, sc.players):
            self.assertEqual(bp.hand_mask, sp.hand_mask)
            self.assertEqual(bp.stack, sp.stack)

    def test_round_trip_produces_frozen_tuple_backed_scenario(self):
        sc = self._make_scenario()
        snap = scenario_to_snapshot(sc)
        back = snapshot_to_scenario(
            snap, small_blind=sc.small_blind, big_blind=sc.big_blind
        )

        self.assertIsInstance(back.node_cards, tuple)
        self.assertIsInstance(back.discard_pile, tuple)
        self.assertIsInstance(back.players, tuple)
        with self.assertRaises(Exception):
            back.pot = 999  # frozen dataclass — must reject mutation


if __name__ == "__main__":
    unittest.main()
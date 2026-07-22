"""
scenarios/hand_snapshot_adapter.py — Converts between BaseScenario
(scenarios/base_scenario.py, RL-Lab-native) and HandSnapshot
(poker_engine/state/hand_snapshot.py, Engine-native).

Why this exists
----------------
Every RL Lab ScenarioGenerator produces a BaseScenario. The real Engine's
PokerState.load_state(snapshot, ...) — the hand editor's own resume-mid-hand
entry point — expects a HandSnapshot instead. The two types are almost
field-for-field identical (by design: BaseScenario's docstring says it
contains "exactly what is needed to construct a PokerState"), but differ in
two structural ways this module exists to bridge:

  1. Mutability. BaseScenario is frozen and uses tuples throughout
     (players, node_cards, discard_pile). HandSnapshot uses plain lists.
     This is not cosmetic: PokerState.load_state() assigns
     ``g.node_cards = snapshot.node_cards`` and ``g.discard_pile =
     snapshot.discard_pile`` *by reference*, and later engine code does
     in-place mutation of both (``g.node_cards[node] = card`` in
     _deal_next_board, ``g.discard_pile.append(card_id)`` in
     GameState.apply_action's FOLD branch). Handing the engine a tuple
     would raise TypeError the first time it deals a card or a fold is
     processed. scenario_to_snapshot() always returns fresh, independent
     lists — never the scenario's own tuples reinterpreted — so mutating
     a HandSnapshot returned from this function can never alias, and
     therefore never corrupt, the BaseScenario it was built from.
  2. Blind sizes. BaseScenario carries small_blind/big_blind directly
     (a scenario is fully self-describing). HandSnapshot does not — the
     Engine treats blind sizes as a property of the *game* (GameDefinition,
     via loader.load_game(name)), not of any particular hand snapshot.
     So snapshot_to_scenario() requires them as explicit arguments rather
     than inventing a default that could silently disagree with the
     variant's actual YAML config.

Both directions are provided because both are needed:
  scenario_to_snapshot   — feeding a generated scenario into the real
                            Engine (e.g. envs/full_holdem_env.py's reset()).
  snapshot_to_scenario   — capturing a live/edited Engine hand back into
                            RL-Lab scenario form (e.g. for a future
                            "train from real played hands" data source).

Dependencies
------------
    scenarios.base_scenario (BaseScenario, PlayerScenario) — always
    available (pure RL Lab). poker_engine.state.hand_snapshot
    (HandSnapshot, PlayerSnapshot) is imported lazily inside each
    function, not at module level, so this file stays importable without
    the Engine on sys.path — only actually calling one of these two
    functions requires it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .base_scenario import BaseScenario, PlayerScenario


# ---------------------------------------------------------------------------
# BaseScenario -> HandSnapshot
# ---------------------------------------------------------------------------

def scenario_to_snapshot(scenario: BaseScenario):
    """
    Convert a BaseScenario into an Engine-native HandSnapshot.

    Parameters
    ----------
    scenario : BaseScenario
        Produced by any ScenarioGenerator.

    Returns
    -------
    poker_engine.state.hand_snapshot.HandSnapshot
        Fresh, independent lists throughout (see module docstring §1) —
        safe to hand to PokerState.load_state() and let the Engine
        mutate freely without touching `scenario`.

    Notes
    -----
    Blind sizes (scenario.small_blind / scenario.big_blind) are dropped
    here, not lost: HandSnapshot has no field for them because the
    Engine sources blinds from the game's own GameDefinition
    (loader.load_game(scenario.game_name)), not from the hand snapshot.
    A caller wiring this into PokerState should already be constructing
    that GameDefinition separately (see envs/full_holdem_env.py).
    """
    from poker_engine.state.hand_snapshot import HandSnapshot, PlayerSnapshot

    players: List[PlayerSnapshot] = [
        PlayerSnapshot(
            stack=p.stack,
            hand_mask=p.hand_mask,
            current_bet=p.current_bet,
            total_contribution=p.total_contribution,
            has_folded=p.has_folded,
            is_all_in=p.is_all_in,
        )
        for p in scenario.players
    ]

    node_cards: List[Optional[int]] = list(scenario.node_cards)
    discard_pile: List[int] = list(scenario.discard_pile)

    return HandSnapshot(
        game_name=scenario.game_name,
        street_index=scenario.street_index,
        pot=scenario.pot,
        dealer_position=scenario.dealer_position,
        current_player=scenario.current_player,
        bet_to_call=scenario.bet_to_call,
        min_raise=scenario.min_raise,
        players=players,
        node_cards=node_cards,
        discard_pile=discard_pile,
        raises_this_street=scenario.raises_this_street,
        last_aggressor=scenario.last_aggressor,
    )


# ---------------------------------------------------------------------------
# HandSnapshot -> BaseScenario
# ---------------------------------------------------------------------------

def snapshot_to_scenario(
    snapshot,  # poker_engine.state.hand_snapshot.HandSnapshot
    small_blind: int,
    big_blind: int,
) -> BaseScenario:
    """
    Convert an Engine-native HandSnapshot back into a BaseScenario.

    Parameters
    ----------
    snapshot : poker_engine.state.hand_snapshot.HandSnapshot
        A live or hand-edited Engine snapshot (e.g. PokerState's own
        current game state, captured via HandSnapshot(...) construction
        from a GameState — see hand_snapshot.py).
    small_blind : int
    big_blind : int
        Not present on HandSnapshot (see module docstring §2) — pass the
        values from the same game's GameDefinition
        (loader.load_game(snapshot.game_name)[0].small_blind / .big_blind).

    Returns
    -------
    BaseScenario
        Immutable, tuple-backed — safe to hand to any ScenarioGenerator
        consumer / Environment.reset() without risking aliasing back into
        live Engine state.
    """
    players = tuple(
        PlayerScenario(
            stack=p.stack,
            hand_mask=p.hand_mask,
            current_bet=p.current_bet,
            total_contribution=p.total_contribution,
            has_folded=p.has_folded,
            is_all_in=p.is_all_in,
        )
        for p in snapshot.players
    )

    return BaseScenario(
        game_name=snapshot.game_name,
        street_index=snapshot.street_index,
        pot=snapshot.pot,
        dealer_position=snapshot.dealer_position,
        current_player=snapshot.current_player,
        bet_to_call=snapshot.bet_to_call,
        min_raise=snapshot.min_raise,
        players=players,
        node_cards=tuple(snapshot.node_cards),
        small_blind=small_blind,
        big_blind=big_blind,
        discard_pile=tuple(snapshot.discard_pile),
        raises_this_street=snapshot.raises_this_street,
        last_aggressor=snapshot.last_aggressor,
    )
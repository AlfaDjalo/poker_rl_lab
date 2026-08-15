"""
controllers/human_controller.py — HumanController, per
RL_lab_implementation_guide.md (§9, "Self Play"):

    Controllers may include:
    ...
    - HumanController

Purpose
-------
Lets a person occupy a seat directly — for manual sanity-checking of a
trained policy ("does this agent actually play sensibly against me?"),
demoing a curriculum stage, or debugging an Environment's legal-mask /
reward wiring interactively. Not part of any automated training or
evaluation path (evaluation.policy_evaluator's evaluate_policy /
compare_policies always run unattended) — this Controller exists purely
for a person sitting at a REPL.

Why this needs the full StructuredObservation (unlike a rule-based mask
reader)
--------------------------------------------------------------------------
A human deciding what to do needs to actually see their hole cards, the
board, and the pot/bet-to-call — legal_mask alone ("here are 3 numbers
you may pick") is useless to a person. So HumanController's rendering
depends on the same full-observation Controller interface
base_controller.py establishes (see that module's docstring), and is
subject to the identical KNOWN GAP: if ever wrapped via
LegalMaskControllerAdapter to sit in RiverEnv's/FullHoldemEnv's existing
villain slot, the human would be shown a card-blind synthesized
observation and asked to play in the dark. That adapter's own docstring
already calls this combination unsupported; repeated here because it's
the single most likely way someone reaches for HumanController today
(pitting themselves against the current villain_policy slot) and hits
the gap in practice, not just in the abstract.

Multi-board rendering
----------------------
StructuredObservation.board.node_cards is a single flat tuple indexed by
engine node position (see envs/observations.py's BoardObs docstring) —
it carries no "which board" grouping. For single-board games (holdem)
that's fine: every node belongs to the one board. For multi-board games
(double_board's two independent 5-card boards; hopscotch_bomb_pot's two
overlapping boards that share the turn/river nodes) flattening every
node into one comma-joined string is actively misleading — it reads as
one 10-card board instead of two 5-card ones, and for hopscotch it would
also print shared nodes twice with no indication they're shared.

This module does NOT import rules.game_rules / PointDefinition to derive
node_sets generically (that would pull the poker_eval dependency into a
pure-stdlib, terminal-only Controller — see the module dependencies
note). Instead, _BOARD_LAYOUTS below is a small, explicit table of the
node groupings for every multi-board game_name that exists today,
mirroring the node_sets already declared in each game's YAML
(poker_engine/games/double_board.yaml, .../hopscotch_bomb_pot.yaml).
Unlisted game_names fall back to a single board containing every dealt
node, in node order — correct for holdem.yaml and any other
single-board variant.

Dependencies
------------
    Standard library only. actions.abstract_action, envs.observations,
    controllers.base_controller.Controller.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from poker_rl_lab.actions.abstract_action import AbstractAction, ActionType, NUM_ACTIONS
from poker_rl_lab.controllers.base_controller import Controller
from poker_rl_lab.envs.observations import StructuredObservation
from poker_rl_lab.representation.card_utils import card_id_to_str


PromptFn = Callable[[str], str]
PrintFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# Known multi-board node layouts (see module docstring — kept in sync with
# each game's own YAML node_sets, not derived from it).
# ---------------------------------------------------------------------------

_BOARD_LAYOUTS: Dict[str, List[Tuple[int, ...]]] = {
    "double_board": [
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
    ],
    "hopscotch_bomb_pot": [
        (0, 1, 2, 3, 4),
        (5, 6, 7, 3, 4),   # shares turn/river nodes (3, 4) with board 1
    ],
}


def _boards_for_display(
    game_name: str, node_cards: Sequence[Optional[int]]
) -> List[Tuple[str, List[int]]]:
    """
    Split node_cards into one or more (label, dealt_card_ids) groups for
    rendering.

    Falls back to a single "Board" group containing every dealt node (in
    node order) for any game_name not in _BOARD_LAYOUTS.
    """
    layout = _BOARD_LAYOUTS.get(game_name)

    if layout is None:
        dealt = [c for c in node_cards if c is not None]
        return [("Board", dealt)]

    groups = []
    for i, node_set in enumerate(layout, start=1):
        dealt = [
            node_cards[n]
            for n in node_set
            if n < len(node_cards) and node_cards[n] is not None
        ]
        groups.append((f"Board {i}", dealt))
    return groups


class HumanController(Controller):
    """
    Prompts a person for an action at every decision point.

    Parameters
    ----------
    prompt_fn : callable, optional
        ``prompt_fn(prompt_text: str) -> str``. Defaults to the builtin
        ``input``. Override in tests to script a sequence of canned
        responses without a real terminal.
    print_fn : callable, optional
        ``print_fn(text: str) -> None``. Defaults to the builtin
        ``print``. Override to redirect rendering (e.g. into a GUI log)
        instead of stdout.
    name : str
        Label used in the rendered prompt (e.g. "Player 2"). Purely
        cosmetic.

    Notes
    -----
    Invalid input (unknown action name, or a syntactically valid action
    that's illegal in the current obs.legal_mask) is never allowed to
    silently fall back to a random/default action — this Controller
    exists so a person's actual intent reaches the Environment, so it
    just re-prompts until a legal choice is given. An unrecognised
    response is treated as invalid input, not e.g. an implicit fold.
    """

    def __init__(
        self,
        prompt_fn: Optional[PromptFn] = None,
        print_fn: Optional[PrintFn] = None,
        name: str = "Human",
    ) -> None:
        self._prompt_fn = prompt_fn or input
        self._print_fn = print_fn or print
        self._name = name

    def act(self, obs: StructuredObservation) -> AbstractAction:
        self._render(obs)

        legal_types = [ActionType(i) for i, m in enumerate(obs.legal_mask) if m]
        legal_by_name = {a.name: a for a in legal_types}

        while True:
            raw = self._prompt_fn(
                f"{self._name}, choose an action "
                f"[{', '.join(a.name for a in legal_types)}]: "
            ).strip().upper()

            chosen = legal_by_name.get(raw)
            if chosen is not None:
                return AbstractAction(chosen)

            self._print_fn(
                f"'{raw}' is not a legal action right now. "
                f"Legal actions: {', '.join(a.name for a in legal_types)}"
            )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, obs: StructuredObservation) -> None:
        hole = ", ".join(card_id_to_str(c) for c in obs.hole_cards.cards)

        self._print_fn("")
        self._print_fn(f"--- {self._name}'s turn (seat {obs.hero_seat}) ---")
        self._print_fn(f"Your hole cards : {hole}")

        for label, dealt in _boards_for_display(obs.game_name, obs.board.node_cards):
            board_str = ", ".join(card_id_to_str(c) for c in dealt) if dealt else "(none)"
            self._print_fn(f"{label:<16}: {board_str}")

        self._print_fn(
            f"Pot: {obs.betting.pot}   "
            f"To call: {obs.betting.bet_to_call}   "
            f"Min raise: {obs.betting.min_raise}"
        )
        for p in obs.players:
            tag = " (you)" if p.is_hero else ""
            status = []
            if p.has_folded:
                status.append("folded")
            if p.is_all_in:
                status.append("all-in")
            status_str = f" [{', '.join(status)}]" if status else ""
            self._print_fn(
                f"  seat {p.seat}{tag}: stack={p.stack} "
                f"current_bet={p.current_bet}{status_str}"
            )
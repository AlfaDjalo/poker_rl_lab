"""
representation/showdown_constraints.py — Legal hole/board card-count
splits for made-hand decomposition, per showdown_type.

Used by CompositeBoardEncoder (representation/composite_board_encoder.py)
when decomposing a hole∪board card set into candidate 5-card hands: which
(hole_count, board_count) splits are even legal depends on the game's
showdown_type, and that differs per Dealer's Choice variant:

    HOLDEM  — any number of hole cards (0-5, in practice 0-2 since only
              2 exist) may combine with the rest from the board, total 5.
    OMAHA   — EXACTLY 2 hole + 3 board, always.
    MAKE5   — board is fixed at n cards (1 <= n <= 4); the hand must use
              ALL n board cards plus exactly 5-n hole cards. No other
              split is legal.

This module holds ONLY these counting bounds — never actual poker
legality/scoring logic (that stays in the Engine, per this codebase's
core rule). Getting an entry wrong here means the made-hand encoder
pools some illegal or missing candidate windows; it can never make a
real showdown outcome incorrect, since scoring is untouched.

showdown_type values
---------------------
Integer keys match poker_eval.ShowdownType / mock_poker_eval.ShowdownType
by value — same NUM_SHOWDOWN_TYPES vocabulary already shared by
point_encoder.py and game_encoder.py (HOLDEM, OMAHA, MAKE5, DRAW, BADUGI).

Config-driven
-------------
Loaded from showdown_constraints.yaml (same directory) so a new Dealer's
Choice variant's showdown_type can get a constraint entry without a code
change — mirrors scenarios/duo_game_config.py's config-over-code pattern.
DRAW/BADUGI are intentionally absent from the shipped YAML (draw-style
games typically have no board at all, so made-hand decomposition doesn't
apply the same way) — get_constraint() falls back to a generic HOLDEM-
shaped constraint for any missing entry, WITH a loud warning, rather
than raising, so an unmodelled variant degrades gracefully instead of
crashing training.
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)

_YAML_PATH = Path(__file__).resolve().parent / "showdown_constraints.yaml"

# Matches poker_eval.ShowdownType / mock_poker_eval.ShowdownType by value
# — same vocabulary as point_encoder.NUM_SHOWDOWN_TYPES / game_encoder.py.
_SHOWDOWN_TYPE_NAMES: Tuple[str, ...] = ("HOLDEM", "OMAHA", "MAKE5", "DRAW", "BADUGI")

# Used whenever a showdown_type has no registry entry (missing YAML key,
# out-of-range int, or the YAML file itself is missing) — permissive
# "any split totaling 5" behaviour, clipped to whatever cards actually
# exist, so decomposition degrades rather than failing outright.
_FALLBACK_CONSTRAINT_NAME = "HOLDEM"


@dataclass(frozen=True)
class CardUsageConstraint:
    """
    Bounds on how many hole cards vs. board cards may combine into one
    candidate 5-card made hand.

    Attributes
    ----------
    min_hole, max_hole : int
        Inclusive bounds on hole-card count per candidate hand.
    min_board, max_board : int
        Inclusive bounds on board-card count per candidate hand, used
        only when exact_board is False.
    exact_board : bool
        If True (Make-5 semantics), the board count is pinned to
        whatever board cards are actually available for this point —
        min_board/max_board are ignored — and hole count is forced to
        exactly 5 - (that board count). If False (Hold'em/Omaha
        semantics), min_board/max_board bound the board count directly
        and every (hole, board) pair summing to 5 within both seats'
        bounds is enumerated.
    """
    min_hole: int = 0
    max_hole: int = 5
    min_board: int = 0
    max_board: int = 5
    exact_board: bool = False

    def __post_init__(self) -> None:
        if not (0 <= self.min_hole <= self.max_hole <= 5):
            raise ValueError(
                f"min_hole/max_hole must satisfy 0 <= min_hole <= max_hole <= 5, "
                f"got min_hole={self.min_hole}, max_hole={self.max_hole}"
            )
        if not (0 <= self.min_board <= self.max_board <= 5):
            raise ValueError(
                f"min_board/max_board must satisfy 0 <= min_board <= max_board <= 5, "
                f"got min_board={self.min_board}, max_board={self.max_board}"
            )

    def valid_splits(self, n_hole: int, n_board: int) -> List[Tuple[int, int]]:
        """
        Every legal (hole_count, board_count) pair for this constraint,
        given the actual number of hole/board cards available, summing
        to exactly 5.

        Parameters
        ----------
        n_hole : int
            Hole cards actually available (e.g. 2 for hold'em, 4 for
            Omaha).
        n_board : int
            Board cards actually dealt/visible for this point right now
            (may be < the game's eventual full board mid-hand).

        Returns
        -------
        List[Tuple[int, int]]
            Possibly empty — e.g. Omaha before the flop (n_board < 3)
            legitimately has zero valid 5-card made hands yet; callers
            (CompositeBoardEncoder) must handle the empty-list case
            (fall back to the empty/placeholder embedding) rather than
            assume at least one split always exists.
        """
        if self.exact_board:
            board_count = n_board
            hole_count = 5 - board_count
            if hole_count < 0 or hole_count > n_hole:
                return []
            return [(hole_count, board_count)]

        splits = []
        lo_board = max(self.min_board, 0)
        hi_board = min(self.max_board, n_board)
        for board_count in range(lo_board, hi_board + 1):
            hole_count = 5 - board_count
            if not (self.min_hole <= hole_count <= self.max_hole):
                continue
            if not (0 <= hole_count <= n_hole):
                continue
            splits.append((hole_count, board_count))
        return splits


def _parse_constraint(raw: dict, name: str) -> CardUsageConstraint:
    try:
        return CardUsageConstraint(
            min_hole=int(raw.get("min_hole", 0)),
            max_hole=int(raw.get("max_hole", 5)),
            min_board=int(raw.get("min_board", 0)),
            max_board=int(raw.get("max_board", 5)),
            exact_board=bool(raw.get("exact_board", False)),
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"showdown_constraints.yaml entry {name!r} is invalid: {exc}") from exc


@lru_cache(maxsize=1)
def _load_registry() -> Dict[int, CardUsageConstraint]:
    """
    Load + parse showdown_constraints.yaml once per process. Missing
    file is not fatal here — logged loudly, empty registry returned —
    so get_constraint()'s per-lookup fallback (also logged) is what
    actually kicks in for every showdown_type in that case.
    """
    if not _YAML_PATH.exists():
        msg = (
            f"[showdown_constraints] {_YAML_PATH} not found — no "
            f"showdown-type-specific card-usage constraints loaded. "
            f"Every made-hand decomposition will use the generic "
            f"{_FALLBACK_CONSTRAINT_NAME!r} fallback regardless of the "
            f"game's actual showdown_type."
        )
        warnings.warn(msg, RuntimeWarning)
        logger.warning(msg)
        print(f"WARNING: {msg}", file=sys.stderr)
        return {}

    with open(_YAML_PATH) as f:
        data = yaml.safe_load(f) or {}

    registry: Dict[int, CardUsageConstraint] = {}
    for name, raw in data.items():
        if name not in _SHOWDOWN_TYPE_NAMES:
            msg = (
                f"[showdown_constraints] {_YAML_PATH} declares unknown "
                f"showdown_type name {name!r} (expected one of "
                f"{_SHOWDOWN_TYPE_NAMES}) — ignoring this entry."
            )
            warnings.warn(msg, RuntimeWarning)
            logger.warning(msg)
            print(f"WARNING: {msg}", file=sys.stderr)
            continue
        value = _SHOWDOWN_TYPE_NAMES.index(name)
        registry[value] = _parse_constraint(raw, name)

    logger.info(
        f"[showdown_constraints] loaded {len(registry)} constraint(s) from "
        f"{_YAML_PATH}: {[_SHOWDOWN_TYPE_NAMES[v] for v in registry]}"
    )
    return registry


def get_constraint(showdown_type: int) -> CardUsageConstraint:
    """
    Look up the CardUsageConstraint for a showdown_type integer value.

    Falls back to a generic Hold'em-shaped constraint — logged loudly,
    once per distinct missing showdown_type per process — for any
    value with no registry entry (including DRAW/BADUGI, which the
    shipped YAML deliberately omits; see module docstring). This keeps
    an unmodelled Dealer's Choice variant training-capable rather than
    crashing, at the cost of a possibly-too-permissive decomposition
    for that variant until a real entry is added.
    """
    registry = _load_registry()

    if showdown_type in registry:
        return registry[showdown_type]

    name = (
        _SHOWDOWN_TYPE_NAMES[showdown_type]
        if 0 <= showdown_type < len(_SHOWDOWN_TYPE_NAMES)
        else f"<unknown:{showdown_type}>"
    )
    _warn_missing_constraint(showdown_type, name)
    return registry.get(
        _SHOWDOWN_TYPE_NAMES.index(_FALLBACK_CONSTRAINT_NAME),
        CardUsageConstraint(),  # ultimate fallback if even HOLDEM's own
                                 # entry is missing from the YAML
    )


@lru_cache(maxsize=None)
def _warn_missing_constraint(showdown_type: int, name: str) -> None:
    """lru_cache'd so this warning fires once per distinct missing
    showdown_type per process, not once per decomposition call."""
    msg = (
        f"[showdown_constraints] no CardUsageConstraint registered for "
        f"showdown_type={showdown_type} ({name}) — falling back to "
        f"{_FALLBACK_CONSTRAINT_NAME!r}'s (generic Hold'em) bounds. Add "
        f"an entry to {_YAML_PATH.name} for this showdown_type if it "
        f"needs different made-hand decomposition rules."
    )
    warnings.warn(msg, RuntimeWarning)
    logger.warning(msg)
    print(f"WARNING: {msg}", file=sys.stderr)
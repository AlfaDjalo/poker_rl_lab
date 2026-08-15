"""
push_fold_duo_hand_grids.py — SB push-probability and BB call-probability
hand-range grids for a trained/checkpointed push-fold-duo
DualSeatActorCritic, per stack depth.

Generic-tree edition
---------------------
This module now targets experiments.generic_tree_experiment /
scenarios.duo_game_config.DuoGameConfig instead of the retired
experiments.push_fold_duo_experiment (SB_SEAT/BB_SEAT-hardcoded). Seat
indices are resolved from the loaded DuoGameConfig via
game_config.seat_index("sb") / .seat_index("bb") rather than assumed to
be 0/1 — a config-driven game makes no such guarantee (see
generic_tree_env.py's own docstring). game_name / small_blind /
big_blind are likewise read off game_config rather than passed
separately on the CLI, so the synthesized observations below can never
drift from what the checkpoint was actually trained against.

Why this builds StructuredObservations directly (no env/generator)
-----------------------------------------------------------------------
A hand-range grid needs EVERY starting-hand combo at a FIXED stack
depth, evaluated at each seat's own decision point. GenericTreeEnv's
reset() only accepts a scenario at the config's start_node (SB's
decision point for push_fold_duo), and there is no seam to jump
straight to "BB facing a specific push with a specific hand" without
first driving SB through its own (irrelevant, policy-dependent)
decision. So this module builds the two decision points'
StructuredObservations by hand, mirroring GenericTreeEnv._build_obs()'s
field construction for push_fold_duo's specific two-node tree — see
build_sb_obs / build_bb_obs below. This does not duplicate any BETTING
RULE (push-fold-duo's legal actions are fixed by construction: {FOLD,
ALL_IN} for SB, {FOLD, CALL} for BB), only the observation-shape
bookkeeping the env itself does. This is inherently push_fold_duo-tree-
shaped (it hardcodes "SB pushes, then BB decides"), not generic across
arbitrary DuoGameConfig trees.

Reading action probabilities (not sampled/argmax actions)
--------------------------------------------------------------
models.dual_seat_policy.DualSeatActorCritic.act() only returns a
SAMPLED (or argmax) action, not the full distribution — insufficient
for a probability heatmap. _action_probs() below reaches into the
concrete variant (SharedSingleHeadActorCritic /
SharedTrunkTwoHeadsActorCritic — see models/shared_single_head_policy.py
and models/shared_trunk_two_heads_policy.py) to run
representation_model -> head -> apply_legal_mask -> softmax directly,
duck-typing on which concrete class was given rather than adding a new
required method to the DualSeatActorCritic interface just for this
analysis tool.

Dependencies
------------
    PyTorch, numpy. matplotlib for the optional heatmap plots (only
    imported inside plot_heatmap, so grid computation alone doesn't
    require it). actions.abstract_action, envs.observations,
    representation.card_utils, representation.observation_adapter,
    models.dual_seat_policy / shared_single_head_policy /
    shared_trunk_two_heads_policy, models.policy_value_network
    (apply_legal_mask), experiments.generic_tree_experiment.
"""

from __future__ import annotations

import sys

try:
    import poker_rl_lab.poker_eval as poker_eval  # noqa: F401
except ImportError:
    import poker_rl_lab.mock_poker_eval as mock_poker_eval

    sys.modules.setdefault("poker_eval", mock_poker_eval)

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from poker_rl_lab.actions.abstract_action import ActionType, NUM_ACTIONS
from poker_rl_lab.envs.observations import (
    BettingObs,
    BoardObs,
    HistoryEventObs,
    HistoryObs,
    HoleCardsObs,
    PlayerObs,
    StructuredObservation,
)
from poker_rl_lab.models.dual_seat_policy import DualSeatActorCritic
from poker_rl_lab.models.policy_value_network import apply_legal_mask
from poker_rl_lab.models.shared_single_head_policy import SharedSingleHeadActorCritic
from poker_rl_lab.models.shared_trunk_two_heads_policy import SharedTrunkTwoHeadsActorCritic
from poker_rl_lab.representation.card_utils import str_to_card_id
from poker_rl_lab.representation.observation_adapter import ObservationAdapter

RANKS: Tuple[str, ...] = ("A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2")
SUITS: Tuple[str, ...] = ("c", "d", "h", "s")


# ---------------------------------------------------------------------------
# Synthetic observation construction (mirrors GenericTreeEnv._build_obs
# for push_fold_duo's specific two-node tree)
# ---------------------------------------------------------------------------

def _mask_to_cards(mask: int) -> Tuple[int, ...]:
    return tuple(i for i in range(52) if (mask >> i) & 1)


def _cards_to_mask(card_ids: Sequence[int]) -> int:
    mask = 0
    for c in card_ids:
        mask |= 1 << c
    return mask


def _sb_legal_mask() -> Tuple[bool, ...]:
    mask = [False] * NUM_ACTIONS
    mask[ActionType.FOLD] = True
    mask[ActionType.ALL_IN] = True
    return tuple(mask)


def _bb_legal_mask() -> Tuple[bool, ...]:
    mask = [False] * NUM_ACTIONS
    mask[ActionType.FOLD] = True
    mask[ActionType.CALL] = True
    return tuple(mask)


def build_sb_obs(
    hero_mask: int,
    stack_bb: float,
    small_blind: int,
    big_blind: int,
    sb_seat: int,
    bb_seat: int,
    game_name: str = "holdem",
) -> StructuredObservation:
    """SB's decision-point observation: both blinds posted, no shove yet."""
    starting_stack = max(big_blind, int(round(stack_bb * big_blind)))
    sb_posted = min(small_blind, starting_stack)
    bb_posted = min(big_blind, starting_stack)
    sb_stack = starting_stack - sb_posted
    bb_stack = starting_stack - bb_posted

    seat_players = {
        sb_seat: PlayerObs(seat=sb_seat, stack=sb_stack, current_bet=sb_posted,
                            has_folded=False, is_all_in=(sb_stack == 0), is_hero=True),
        bb_seat: PlayerObs(seat=bb_seat, stack=bb_stack, current_bet=bb_posted,
                            has_folded=False, is_all_in=(bb_stack == 0), is_hero=False),
    }
    players = tuple(seat_players[s] for s in sorted(seat_players))

    return StructuredObservation(
        hero_seat=sb_seat,
        hole_cards=HoleCardsObs(cards=_mask_to_cards(hero_mask)),
        board=BoardObs(node_cards=tuple()),
        players=players,
        betting=BettingObs(
            # Both blinds are already posted before SB's decision — pot
            # must include them (base_scenario.py's inclusive pot
            # convention; mirrors GenericTreeEnv._build_obs).
            pot=sb_posted + bb_posted, bet_to_call=bb_posted - sb_posted,
            min_raise=big_blind, street_index=0, raises_this_street=0,
        ),
        legal_mask=_sb_legal_mask(),
        game_name=game_name,
        history=HistoryObs(events=()),
        dealer_position=sb_seat,
        is_terminal=False,
    )


def build_bb_obs(
    hero_mask: int,
    stack_bb: float,
    small_blind: int,
    big_blind: int,
    sb_seat: int,
    bb_seat: int,
    game_name: str = "holdem",
) -> StructuredObservation:
    """BB's decision-point observation: SB has just shoved all-in."""
    starting_stack = max(big_blind, int(round(stack_bb * big_blind)))
    sb_posted = min(small_blind, starting_stack)
    bb_posted = min(big_blind, starting_stack)
    bb_stack = starting_stack - bb_posted

    sb_amount_added = starting_stack - sb_posted  # SB's push, on top of its blind
    pot_before_push = sb_posted + bb_posted
    push_event = HistoryEventObs(
        seat=sb_seat,
        action_type=int(ActionType.ALL_IN),
        street_index=0,
        pot_fraction=(sb_amount_added / pot_before_push if pot_before_push > 0 else 0.0),
        is_aggressor=True,
    )

    seat_players = {
        sb_seat: PlayerObs(seat=sb_seat, stack=0, current_bet=starting_stack,
                            has_folded=False, is_all_in=True, is_hero=False),
        bb_seat: PlayerObs(seat=bb_seat, stack=bb_stack, current_bet=bb_posted,
                            has_folded=False, is_all_in=(bb_stack == 0), is_hero=True),
    }
    players = tuple(seat_players[s] for s in sorted(seat_players))

    return StructuredObservation(
        hero_seat=bb_seat,
        hole_cards=HoleCardsObs(cards=_mask_to_cards(hero_mask)),
        board=BoardObs(node_cards=tuple()),
        players=players,
        betting=BettingObs(
            # SB's full push AND BB's own already-posted blind are both
            # already committed by the time BB decides — pot must
            # include both (base_scenario.py's inclusive pot convention;
            # mirrors GenericTreeEnv._build_obs).
            pot=starting_stack + bb_posted, bet_to_call=starting_stack - bb_posted,
            min_raise=big_blind, street_index=0, raises_this_street=0,
        ),
        legal_mask=_bb_legal_mask(),
        game_name=game_name,
        history=HistoryObs(events=(push_event,)),
        dealer_position=sb_seat,
        is_terminal=False,
    )


# ---------------------------------------------------------------------------
# Action-probability extraction (duck-types on the concrete variant)
# ---------------------------------------------------------------------------

def _trunk_and_head(actor_critic: DualSeatActorCritic, seat: int, sb_seat: int):
    if isinstance(actor_critic, SharedSingleHeadActorCritic):
        pac = actor_critic.actor_critic
        return pac.representation_model, pac.policy_value_network
    if isinstance(actor_critic, SharedTrunkTwoHeadsActorCritic):
        head = actor_critic.head_sb if seat == sb_seat else actor_critic.head_bb
        return actor_critic.representation_model, head
    raise TypeError(
        f"Don't know how to extract trunk/head from {type(actor_critic).__name__}; "
        f"expected SharedSingleHeadActorCritic or SharedTrunkTwoHeadsActorCritic."
    )


def action_probs(
    actor_critic: DualSeatActorCritic,
    seat: int,
    sb_seat: int,
    adapter: ObservationAdapter,
    obs: StructuredObservation,
) -> np.ndarray:
    """
    Full (masked, softmaxed) action-probability vector for one seat at
    one observation. Shape (NUM_ACTIONS,).
    """
    trunk, head = _trunk_and_head(actor_critic, seat, sb_seat)
    rep_input = adapter.adapt(obs)

    with torch.no_grad():
        latent = trunk(rep_input)
        logits, _value = head(latent)
        mask_t = torch.as_tensor(obs.legal_mask, dtype=torch.bool)
        masked_logits = apply_legal_mask(logits, mask_t)
        probs = torch.softmax(masked_logits, dim=-1)

    return probs.numpy()


# ---------------------------------------------------------------------------
# Starting-hand combo enumeration
# ---------------------------------------------------------------------------

def _all_starting_hand_combos(rank1: str, rank2: str, suited: bool) -> List[Tuple[int, int]]:
    """All concrete 2-card combos for one 13x13 grid cell (engine card ids)."""
    combos = []
    if rank1 == rank2:
        for s1, s2 in combinations(SUITS, 2):
            combos.append((str_to_card_id(rank1 + s1), str_to_card_id(rank2 + s2)))
    elif suited:
        for s in SUITS:
            combos.append((str_to_card_id(rank1 + s), str_to_card_id(rank2 + s)))
    else:
        for s1 in SUITS:
            for s2 in SUITS:
                if s1 != s2:
                    combos.append((str_to_card_id(rank1 + s1), str_to_card_id(rank2 + s2)))
    return combos


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_sb_push_grid(
    actor_critic: DualSeatActorCritic,
    adapter: ObservationAdapter,
    stack_bb: float,
    small_blind: int,
    big_blind: int,
    sb_seat: int,
    bb_seat: int,
    game_name: str = "holdem",
) -> np.ndarray:
    """13x13 grid of P(ALL_IN) for SB, at a fixed stack depth."""
    grid = np.zeros((13, 13), dtype=np.float32)

    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            suited = i < j
            combos = _all_starting_hand_combos(r1, r2, suited=suited)
            probs = []
            for c1, c2 in combos:
                mask = _cards_to_mask([c1, c2])
                obs = build_sb_obs(mask, stack_bb, small_blind, big_blind, sb_seat, bb_seat, game_name)
                p = action_probs(actor_critic, sb_seat, sb_seat, adapter, obs)
                probs.append(p[ActionType.ALL_IN])
            grid[i, j] = float(np.mean(probs))

    return grid


def generate_bb_call_grid(
    actor_critic: DualSeatActorCritic,
    adapter: ObservationAdapter,
    stack_bb: float,
    small_blind: int,
    big_blind: int,
    sb_seat: int,
    bb_seat: int,
    game_name: str = "holdem",
) -> np.ndarray:
    """13x13 grid of P(CALL) for BB facing an SB shove, at a fixed stack depth."""
    grid = np.zeros((13, 13), dtype=np.float32)

    for i, r1 in enumerate(RANKS):
        for j, r2 in enumerate(RANKS):
            suited = i < j
            combos = _all_starting_hand_combos(r1, r2, suited=suited)
            probs = []
            for c1, c2 in combos:
                mask = _cards_to_mask([c1, c2])
                obs = build_bb_obs(mask, stack_bb, small_blind, big_blind, sb_seat, bb_seat, game_name)
                p = action_probs(actor_critic, bb_seat, sb_seat, adapter, obs)
                probs.append(p[ActionType.CALL])
            grid[i, j] = float(np.mean(probs))

    return grid


def generate_all_grids(
    actor_critic: DualSeatActorCritic,
    adapter: ObservationAdapter,
    stack_bb_values: Sequence[float],
    small_blind: int,
    big_blind: int,
    sb_seat: int,
    bb_seat: int,
    game_name: str = "holdem",
) -> Dict[float, Dict[str, np.ndarray]]:
    """
    Returns {stack_bb: {"sb_push": grid, "bb_call": grid}} for every
    stack depth in `stack_bb_values`.
    """
    out: Dict[float, Dict[str, np.ndarray]] = {}
    for stack_bb in stack_bb_values:
        out[stack_bb] = {
            "sb_push": generate_sb_push_grid(
                actor_critic, adapter, stack_bb, small_blind, big_blind,
                sb_seat, bb_seat, game_name,
            ),
            "bb_call": generate_bb_call_grid(
                actor_critic, adapter, stack_bb, small_blind, big_blind,
                sb_seat, bb_seat, game_name,
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_heatmap(grid: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Action Probability")
    plt.xticks(range(13), RANKS)
    plt.yticks(range(13), RANKS)
    plt.xlabel("Second Card")
    plt.ylabel("First Card")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    from poker_rl_lab.experiments.generic_tree_experiment import (
        GenericTreeExperimentConfig,
        PolicyVariant,
        build_generic_tree_experiment,
        load_checkpoint,
    )

    parser = argparse.ArgumentParser(
        description="Generate SB push / BB call hand-range heatmaps from a "
                    "generic-tree push-fold-duo checkpoint."
    )
    parser.add_argument("--config-name", type=str, default="push_fold_duo",
                         help="Name of scenarios/duo_games/<config_name>.yaml. "
                              "Must be a config with a push_fold_duo-shaped tree "
                              "(seats named 'sb'/'bb', SB pushes then BB calls/folds) — "
                              "this script's obs-building is specific to that shape.")
    parser.add_argument("--checkpoint-in", type=str, required=True)
    parser.add_argument(
        "--variant", type=str, default="shared_trunk_two_heads",
        choices=[v.value for v in PolicyVariant],
        help="Must match the variant the checkpoint was trained with.",
    )
    parser.add_argument(
        "--stack-bb-values", type=str, default="5,10,15,20,25",
        help="Comma-separated stack depths (in big blinds) to grid.",
    )
    parser.add_argument("--output-dir", type=str, default="hand_grids")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device for the actor_critic, e.g. 'cpu', 'cuda', 'cuda:0'.",
    )
    args = parser.parse_args()

    cfg = GenericTreeExperimentConfig(
        config_name=args.config_name,
        variant=PolicyVariant(args.variant),
        device=args.device,
    )
    exp = build_generic_tree_experiment(cfg)
    load_checkpoint(exp, args.checkpoint_in)
    exp.actor_critic.eval()

    game_config = exp.game_config
    sb_seat = game_config.seat_index("sb")
    bb_seat = game_config.seat_index("bb")
    small_blind = game_config.small_blind
    big_blind = game_config.big_blind
    game_name = game_config.game_name

    stack_bb_values = [float(s) for s in args.stack_bb_values.split(",") if s.strip()]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_grids = generate_all_grids(
        exp.actor_critic, exp.adapter, stack_bb_values,
        small_blind=small_blind, big_blind=big_blind,
        sb_seat=sb_seat, bb_seat=bb_seat, game_name=game_name,
    )

    for stack_bb, grids in all_grids.items():
        print(f"stack={stack_bb}bb  "
              f"SB mean push%={grids['sb_push'].mean():.3f}  "
              f"BB mean call%={grids['bb_call'].mean():.3f}")

        plot_heatmap(
            grids["sb_push"], f"SB Push Probability at {stack_bb}bb",
            out_dir / f"sb_push_{stack_bb}bb.png",
        )
        plot_heatmap(
            grids["bb_call"], f"BB Call Probability at {stack_bb}bb",
            out_dir / f"bb_call_{stack_bb}bb.png",
        )

    print(f"\nHeatmaps saved to {out_dir}/")


if __name__ == "__main__":
    _main()
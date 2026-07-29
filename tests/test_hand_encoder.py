"""
test_hand_encoder.py — Standalone sanity check for a pretrained hand
encoder .keras file, before wiring it into a full PPO training run.

Loads the file via representation.pretrained_hand_encoder.PretrainedHandEncoder,
runs a handful of probe hands through it, and prints diagnostics: shape,
embedding_dim, L2 norm, and cosine similarity between related/unrelated
hands (a sanity signal — e.g. two hands sharing a card should usually be
at least somewhat closer to each other than two disjoint random hands,
though this is a soft check, not a strict assertion, since the encoder's
actual similarity geometry depends on how it was trained).

This does NOT touch PyTorch, PPO, the Engine, or poker_eval — it only
exercises the TF/Keras loading path (PretrainedHandEncoder.from_file's
own architecture fingerprint check already does most of the heavy
lifting; see that method's docstring). Use this when you just want to
know "does my .keras file load and produce something sane" without
spinning up a training run to find out.

Usage
-----
    PYTHONPATH=. python3 test_hand_encoder.py
    PYTHONPATH=. python3 test_hand_encoder.py --encoder-path encoders/encoder_model.keras --embedding-dim 32
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from representation.card_utils import card_id_to_str, str_to_card_id
from representation.pretrained_hand_encoder import (
    HandEncoderArchitectureMismatch,
    PretrainedHandEncoder,
)


def _print_header(title: str) -> None:
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanity-check a pretrained hand encoder .keras file."
    )
    parser.add_argument("--encoder-path", type=str,
                         default="encoders/encoder_model.keras")
    parser.add_argument("--embedding-dim", type=int, default=32,
                         help="Expected embedding_dim. Pass 0 to skip the check.")
    args = parser.parse_args()

    _print_header("LOADING")
    print(f"path: {args.encoder_path}")
    try:
        encoder = PretrainedHandEncoder.from_file(
            args.encoder_path,
            expected_embedding_dim=args.embedding_dim or None,
        )
    except FileNotFoundError:
        print(f"\nFAILED: no file found at {args.encoder_path!r}.")
        sys.exit(1)
    except HandEncoderArchitectureMismatch as exc:
        print(f"\nFAILED architecture validation:\n  {exc}")
        sys.exit(1)

    print(f"Loaded OK. embedding_dim={encoder.embedding_dim}")

    # ------------------------------------------------------------------
    # Probe hands
    # ------------------------------------------------------------------
    _print_header("PROBE HANDS")

    hands = {
        "AA (pocket rockets)": ["As", "Ah"],
        "AA (different suits)": ["As", "Ad"],
        "72o (worst hand)": ["7c", "2d"],
        "KK": ["Ks", "Kh"],
        "AKs": ["As", "Ks"],
    }

    embeddings = {}
    for name, cards in hands.items():
        card_ids = [str_to_card_id(c) for c in cards]
        emb = encoder.encode_hole(card_ids)
        embeddings[name] = emb
        norm = float(np.linalg.norm(emb))
        print(f"{name:<24} cards={cards}  shape={emb.shape}  "
              f"L2 norm={norm:.4f}  finite={bool(np.all(np.isfinite(emb)))}")

    # ------------------------------------------------------------------
    # Similarity checks (soft sanity signal, not a hard assertion)
    # ------------------------------------------------------------------
    _print_header("PAIRWISE COSINE SIMILARITY")

    names = list(embeddings.keys())
    header = " " * 26 + "".join(f"{n[:10]:>12}" for n in names)
    print(header)
    for a in names:
        row = f"{a[:24]:<26}"
        for b in names:
            row += f"{_cosine_sim(embeddings[a], embeddings[b]):>12.3f}"
        print(row)

    print(
        "\nExpect: diagonal ~1.000 (self-similarity). Off-diagonal values "
        "are informative but encoder-specific — no fixed pass/fail "
        "threshold here, this is for a human sanity glance, not a unit test."
    )

    # ------------------------------------------------------------------
    # Board + combined encoding paths
    # ------------------------------------------------------------------
    _print_header("BOARD / COMBINED ENCODING")

    board_cards = [str_to_card_id(c) for c in ["Th", "9h", "2c"]]
    board_emb = encoder.encode_board(board_cards)
    print(f"encode_board(Th,9h,2c)     shape={board_emb.shape}  "
          f"L2 norm={float(np.linalg.norm(board_emb)):.4f}")

    hole_cards = [str_to_card_id(c) for c in ["As", "Ks"]]
    combined_emb = encoder.encode_combined(hole_cards, board_cards)
    print(f"encode_combined(AKs, board) shape={combined_emb.shape}  "
          f"L2 norm={float(np.linalg.norm(combined_emb)):.4f}")

    # ------------------------------------------------------------------
    # Batch path
    # ------------------------------------------------------------------
    _print_header("BATCH ENCODING")

    batch = [
        [str_to_card_id(c) for c in cards] for cards in hands.values()
    ]
    batch_emb = encoder.encode_batch(batch)
    print(f"encode_batch({len(batch)} hands)  shape={batch_emb.shape}")

    matches = all(
        np.allclose(batch_emb[i], embeddings[name], atol=1e-4)
        for i, name in enumerate(hands.keys())
    )
    print(f"Batch output matches per-hand encode_hole() calls: {matches}")
    if not matches:
        print("WARNING: batch and single-hand encodings disagree — check "
              "for train/inference-mode differences (e.g. BatchNorm) in "
              "the loaded model.")

    _print_header("DONE")
    print("Encoder file looks structurally sound. Wire it into "
          "run_push_fold_demo.py / push_fold_experiment.py via "
          "--encoder-path (or the hand_encoder= argument directly).")


if __name__ == "__main__":
    main()
"""
representation/pretrained_hand_encoder.py — Wrapper around the pretrained
CardSetEncoder / CardStateEncoder (Keras), vendored from another project.

Per the RL Lab implementation guide (Phase 5, "Existing Embeddings"), the
RL Lab must initialise using the existing pretrained embeddings for
two-card, three-card, and five-card hands rather than random weights.

Where these embeddings come from
----------------------------------
The pretrained weights (encoder_model.keras) were trained in a SEPARATE
project, using that project's own training pipeline (data generation,
losses, ML/training/trainer.py, DualOptimizerModel, value heads, etc.).
None of that pipeline exists in — or is needed by — the CAP project.
CAP only consumes the finished, trained weights file.

To load that file we still need the two Keras layer classes it was built
from (CardSetEncoder, CardStateEncoder), because Keras reconstructs custom
layers from their class + saved config, not from source code embedded in
the file. Those two classes have been vendored into
representation/encoders.py for exactly this purpose — see that file's
docstring for the full explanation. This module's get_custom_objects()
below only ever needs those two classes; it does not call into, or
depend on, the other project's trainer module in any way.

A single trained model is reused across "modes" by changing *which cards
go into the grid*, not by swapping models:

    hole-only grid   (2 cards)  -> "two-card" hand strength embedding
    board-only grid  (3 cards)  -> "three-card" (flop) embedding
    hole+board grid  (5 cards)  -> composite "five-card" embedding

This mirrors the other project's `cards_to_tensor(mode=...)` behaviour,
but built on engine card ids (0-51) via card_utils instead of card
strings ("As", "Kd", ...), so it slots into the rest of this codebase
without a string-conversion round trip.

IMPORTANT — framework boundary
-------------------------------
This wrapper is TensorFlow/Keras, NOT PyTorch. card_encoder.py (the
per-card rank/suit embedding used elsewhere in this Representation
Model) is PyTorch. These two will need to be reconciled before the
Representation Model is assembled end-to-end — see the note at the
bottom of this file. Downstream code should depend on the
BaseHandEncoder interface (base_hand_encoder.py), not on this class
directly, so a future PyTorch backend can be swapped in without any
call-site changes.

Loading the real pretrained weights
------------------------------------
    encoder = PretrainedHandEncoder.from_file("rl_lab/pretrained/encoder_model.keras")

Until that file is available, PretrainedHandEncoder can also build a
freshly-initialised (untrained) copy of the same architecture, purely
so the surrounding pipeline (grid construction, shape contracts, batching)
can be tested end-to-end today. Swapping in the real weights later is a
one-line change (see from_file).
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import tensorflow as tf

from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.encoders import CardSetEncoder, CardStateEncoder
from poker_rl_lab.representation.card_utils import card_ids_to_grid, GRID_SHAPE


class HandEncoderArchitectureMismatch(RuntimeError):
    """
    Raised when a loaded .keras file doesn't match what this vendored
    architecture (representation/encoders.py) expects.

    Because encoders.py is a manually-maintained copy of the other
    project's architecture (see that file's docstring), the two can
    silently drift — Keras will happily load a file whose layer configs
    differ from what CAP assumes, and CardStateEncoder.from_config()
    on a config dict it doesn't recognise. That would produce an encoder
    that runs without error but returns meaningless or wrongly-shaped
    embeddings. This exception exists so that mismatch fails loudly at
    load time instead of silently corrupting training later.
    """


def get_custom_objects() -> dict:
    """
    custom_objects dict needed to deserialize the pretrained .keras file.

    Only CardSetEncoder and CardStateEncoder (vendored in
    representation/encoders.py) are needed — the saved encoder-only
    checkpoint shouldn't reference anything from the other project's
    training pipeline (value heads, DualOptimizerModel, etc.), since
    those are training-time-only wrappers around the encoder, not part
    of the encoder itself. If loading ever fails with an "Unknown layer"
    error, that assumption was wrong for this particular checkpoint —
    the missing class needs to be vendored here too.
    """
    return {
        "CardSetEncoder": CardSetEncoder,
        "CardStateEncoder": CardStateEncoder,
    }


class PretrainedHandEncoder(BaseHandEncoder):
    """
    Produces embeddings for 2-card, 3-card, and 5-card composite card sets
    using one shared CardStateEncoder.

    Implements BaseHandEncoder. This is the TF/Keras backend; downstream
    code should type-hint against BaseHandEncoder, not this class, so a
    PyTorch backend can be swapped in later without any call-site changes.

    Parameters
    ----------
    model : CardStateEncoder
        The (ideally pretrained) shared encoder.
    """

    def __init__(self, model: CardStateEncoder) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str,
        custom_objects: Optional[dict] = None,
        expected_embedding_dim: Optional[int] = None,
        expected_input_shape: Optional[tuple] = None,
        skip_validation: bool = False,
    ) -> "PretrainedHandEncoder":
        """
        Load the pretrained encoder from a .keras file.

        Parameters
        ----------
        path : str
            Path to encoder_model.keras (or equivalent).
        custom_objects : dict, optional
            Extra custom objects to merge in, if the saved file references
            classes beyond CardSetEncoder / CardStateEncoder.
        expected_embedding_dim : int, optional
            If given, the loaded model's output dimensionality must match
            exactly or loading fails. Use this once you know the real
            embedding_dim the other project trained with, so an
            accidental swap-in of the wrong file (or a future re-export
            with different config) is caught immediately rather than
            producing silently-wrong-shaped downstream tensors.
        expected_input_shape : tuple, optional
            If given, the loaded model's expected grid input shape must
            match exactly (should normally be GRID_SHAPE = (14, 4, 1) —
            see card_utils.py — since that's the grid this wrapper feeds
            it via card_ids_to_grid).
        skip_validation : bool
            Skip the fingerprint check entirely. Only for debugging a
            mismatch itself; leave False in normal use.

        Raises
        ------
        HandEncoderArchitectureMismatch
            If the loaded model doesn't behave like a valid card-set
            encoder (wrong input rank, non-finite output, non-normalised
            output) or doesn't match an explicitly given expected shape.
        """
        objs = get_custom_objects()
        if custom_objects:
            objs.update(custom_objects)

        model = tf.keras.models.load_model(path, custom_objects=objs, compile=False)
        if not isinstance(model, CardStateEncoder):
            # Some checkpoints save the raw CardSetEncoder directly rather
            # than the CardStateEncoder wrapper; normalise either way.
            inner = model
            model = CardStateEncoder.__new__(CardStateEncoder)
            tf.keras.Model.__init__(model)
            model.card_set_encoder = inner
            model._encoder_config = getattr(inner, "encoder_input_shape", None)

        instance = cls(model)
        if not skip_validation:
            instance._validate_fingerprint(expected_embedding_dim, expected_input_shape)
        return instance

    @classmethod
    def untrained(cls, embedding_dim: int = 32, input_shape=GRID_SHAPE) -> "PretrainedHandEncoder":
        """
        Build a freshly-initialised copy of the same architecture.

        For pipeline/shape testing ONLY — these weights carry no poker
        knowledge. Use from_file() for the real pretrained embeddings.
        """
        config = {"encoder_input_shape": input_shape, "embedding_dim": embedding_dim}
        model = CardStateEncoder(config)
        return cls(model)

    # ------------------------------------------------------------------
    # Public encoding API
    # ------------------------------------------------------------------

    def encode_hole(self, card_ids: Sequence[int]) -> np.ndarray:
        """Embed a 2-card hole hand. Returns shape (embedding_dim,)."""
        return self._encode_single(card_ids)

    def encode_board(self, card_ids: Sequence[int]) -> np.ndarray:
        """Embed a board-only card set (e.g. 3-card flop). Returns shape (embedding_dim,)."""
        return self._encode_single(card_ids)

    def encode_combined(self, hole_ids: Sequence[int], board_ids: Sequence[int]) -> np.ndarray:
        """
        Embed hole + board together in one grid (the 5-card composite,
        matching the old `mode="combined"` behaviour). Returns shape
        (embedding_dim,).
        """
        return self._encode_single(list(hole_ids) + list(board_ids))

    def encode_batch(self, card_id_lists: Sequence[Sequence[int]]) -> np.ndarray:
        """
        Embed a batch of card sets in one forward pass.

        Parameters
        ----------
        card_id_lists : Sequence[Sequence[int]]
            One card-id list per example (mix of hole/board/combined is
            fine — grid construction doesn't care what the cards "mean").

        Returns
        -------
        np.ndarray
            Shape (batch, embedding_dim).
        """
        grids = np.stack([card_ids_to_grid(ids) for ids in card_id_lists], axis=0)
        out = self._model(tf.constant(grids), training=False)
        embedded = out[0] if isinstance(out, (list, tuple)) else out
        return embedded.numpy()

    @property
    def embedding_dim(self) -> int:
        return self._model.card_set_encoder.embedding_dim

    # ------------------------------------------------------------------
    # Architecture fingerprint check
    # ------------------------------------------------------------------

    def _validate_fingerprint(
        self,
        expected_embedding_dim: Optional[int],
        expected_input_shape: Optional[tuple],
    ) -> None:
        """
        Sanity-check a just-loaded model before trusting it.

        Runs a real forward pass (not just config inspection) because
        the failure mode we actually care about — encoders.py having
        silently drifted from the architecture that produced the
        .keras file — can load without a Keras-level error and still
        produce garbage. A forward pass catches shape mismatches;
        checking the L2 norm catches "loaded a different, structurally
        compatible model whose output isn't actually this encoder's
        normalized embedding" (e.g. a raw CardSetEncoder saved without
        its final l2_norm layer wired up correctly).
        """
        input_shape = getattr(self._model.card_set_encoder, "encoder_input_shape", None)
        if input_shape is None:
            raise HandEncoderArchitectureMismatch(
                "Loaded model has no encoder_input_shape — this is not a "
                "valid CardSetEncoder/CardStateEncoder checkpoint."
            )
        input_shape = tuple(input_shape)

        if expected_input_shape is not None and input_shape != tuple(expected_input_shape):
            raise HandEncoderArchitectureMismatch(
                f"Loaded model's input shape {input_shape} does not match "
                f"expected_input_shape {tuple(expected_input_shape)}. "
                f"encoders.py may be out of sync with the file that "
                f"produced this checkpoint."
            )

        try:
            # Use a real one-hot two-card grid, not an all-zero grid: an
            # all-zero input degenerates to an all-zero output through any
            # conv/batchnorm/relu stack regardless of whether the loaded
            # architecture is even correct (relu(0)=0, batchnorm centers
            # zero-mean input to 0 at init), which would make the
            # L2-normalization check below fail for a reason that has
            # nothing to do with an actual architecture mismatch.
            probe_cards = [0, 51]  # 2c, As — arbitrary, just non-degenerate
            probe_grid = card_ids_to_grid(probe_cards)
            if probe_grid.shape != input_shape:
                # Grid helper is fixed to GRID_SHAPE; if the loaded model
                # expects a different input_shape we can't build a probe
                # for it, so fall back to a random (still non-zero) tensor
                # of the model's own declared shape.
                rng = np.random.default_rng(0)
                probe_grid = rng.uniform(0.0, 1.0, size=input_shape).astype(np.float32)
            dummy = probe_grid[np.newaxis, ...]
            out = self._model(tf.constant(dummy), training=False)
            out = out[0] if isinstance(out, (list, tuple)) else out
            out = out.numpy()
        except Exception as exc:  # noqa: BLE001 — deliberately broad; any failure here means "not usable"
            raise HandEncoderArchitectureMismatch(
                f"Loaded model failed a test forward pass with input shape "
                f"{(1,) + input_shape}: {exc}"
            ) from exc

        if out.ndim != 2 or out.shape[0] != 1:
            raise HandEncoderArchitectureMismatch(
                f"Loaded model's output shape {out.shape} is not (1, embedding_dim)."
            )

        if not np.all(np.isfinite(out)):
            raise HandEncoderArchitectureMismatch(
                "Loaded model produced non-finite values (NaN/Inf) on a "
                "smoke-test forward pass."
            )

        norm = float(np.linalg.norm(out[0]))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise HandEncoderArchitectureMismatch(
                f"Loaded model's output is not L2-normalized (norm={norm:.4f}, "
                f"expected ~1.0). This encoder is documented to always "
                f"L2-normalize its embedding — a non-normalized output "
                f"means the loaded architecture doesn't actually match "
                f"CardSetEncoder as vendored in encoders.py."
            )

        embedding_dim = out.shape[1]
        if expected_embedding_dim is not None and embedding_dim != expected_embedding_dim:
            raise HandEncoderArchitectureMismatch(
                f"Loaded model's embedding_dim ({embedding_dim}) does not "
                f"match expected_embedding_dim ({expected_embedding_dim})."
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode_single(self, card_ids: Iterable[int]) -> np.ndarray:
        grid = card_ids_to_grid(card_ids)[np.newaxis, ...]  # add batch dim
        out = self._model(tf.constant(grid), training=False)
        embedded = out[0] if isinstance(out, (list, tuple)) else out
        return embedded.numpy()[0]


# ---------------------------------------------------------------------------
# NOTE — TensorFlow / PyTorch boundary (needs a decision before Phase 5
# is assembled end-to-end; not resolved in this file):
#
#   Option A: Convert the pretrained Keras weights into an equivalent
#             torch.nn.Module once, at load time, so the whole
#             Representation Model (this + CardEncoder + future Board/
#             Point/Game/Position/History embeddings) lives in one
#             framework and trains under one optimiser (PPO in PyTorch).
#   Option B: Keep this encoder frozen and TF-native, run it once per
#             observation to produce a fixed feature vector, and only
#             train the PyTorch layers on top. Simpler, but contradicts
#             the guide's "pretrained embeddings should remain trainable"
#             requirement (§5, Fine Tuning).
#   Option C: Do the whole Representation Model in TF/Keras instead of
#             PyTorch, and rewrite CardEncoder to match.
#
# Recommend Option A once the real .keras file is available and we can
# inspect its actual architecture/weights.
# ---------------------------------------------------------------------------
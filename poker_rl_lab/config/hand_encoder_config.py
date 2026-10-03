"""
config/hand_encoder_config.py — Where the pretrained hand encoder lives.

A single well-known path, not a CLI flag. Every training entry point
(graph_training_loop.py and, previously, whatever built the retired
DualSeatActorCritic-based experiments) should use the SAME pretrained
weights by default -- which encoder backs the policy is a property of
the codebase's ML setup, not something that should vary per invocation
or be something a caller can silently forget to pass. Passing a
DIFFERENT encoder is still possible by constructing a PretrainedHandEncoder
directly and threading it through build_actor_critic(hand_encoder=...)
(see models/actor_critic_factory.py) -- this module only supplies the
default every normal training/grid-generation run should get for free.

If PRETRAINED_HAND_ENCODER_PATH doesn't exist on disk, load_default_hand_encoder()
falls back to an untrained encoder -- loudly, via
PretrainedHandEncoder.untrained() itself (see that classmethod's own
docstring/warning) -- rather than raising, so a fresh checkout without
the weights file still runs end-to-end for pipeline/shape testing. This
mirrors representation/showdown_constraints.py's own "missing file is
not fatal, but is loud" convention.
"""

from __future__ import annotations

from pathlib import Path

from poker_rl_lab.representation.base_hand_encoder import BaseHandEncoder
from poker_rl_lab.representation.pretrained_hand_encoder import PretrainedHandEncoder

# Single source of truth for the pretrained hand-encoder weights.
# Relative to this file so it resolves correctly regardless of the
# caller's cwd (matches showdown_constraints.py's own
# Path(__file__).resolve().parent pattern).
PRETRAINED_HAND_ENCODER_PATH: Path = (
    Path(__file__).resolve().parent.parent / "encoders" / "encoder_model.keras"
)


def load_default_hand_encoder() -> BaseHandEncoder:
    """
    Load the pretrained hand encoder from PRETRAINED_HAND_ENCODER_PATH.

    Falls back to PretrainedHandEncoder.untrained() (loudly warned by
    that classmethod itself) if the file isn't present, so callers don't
    need their own missing-file branching -- every caller just gets
    "the best encoder currently available", real weights when present.
    """
    if not PRETRAINED_HAND_ENCODER_PATH.exists():
        return PretrainedHandEncoder.untrained()
    return PretrainedHandEncoder.from_file(str(PRETRAINED_HAND_ENCODER_PATH))
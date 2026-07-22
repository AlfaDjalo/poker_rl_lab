"""
representation/base_hand_encoder.py — Abstract interface for hand-level
(composite) card-set encoders.

Everything downstream (the Representation Model, tests, training code)
should depend only on this interface, never on PretrainedHandEncoder
(Keras) directly. That is what makes it possible to drop in a PyTorch
implementation later — e.g. after converting the pretrained Keras
weights, or after training a fresh PyTorch hand encoder — without
touching any calling code.

Concrete implementations
-------------------------
    PretrainedHandEncoder (representation/pretrained_hand_encoder.py)
        TensorFlow/Keras. Wraps the existing pretrained CardSetEncoder.
        Current default backend.

    (future) TorchHandEncoder
        PyTorch equivalent. Same interface, swap the construction site
        (wherever PretrainedHandEncoder.from_file(...) / .untrained() is
        called) and nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class BaseHandEncoder(ABC):
    """
    Encodes card sets (2-card hole, 3-card board, 5-card composite, ...)
    into fixed-size embeddings.

    All methods return plain numpy arrays regardless of the underlying
    framework (TensorFlow, PyTorch, ...), so callers never need to know
    or care which backend produced them.
    """

    @abstractmethod
    def encode_hole(self, card_ids: Sequence[int]) -> np.ndarray:
        """Embed a hole-card hand. Returns shape (embedding_dim,)."""
        ...

    @abstractmethod
    def encode_board(self, card_ids: Sequence[int]) -> np.ndarray:
        """Embed a board-only card set. Returns shape (embedding_dim,)."""
        ...

    @abstractmethod
    def encode_combined(
        self, hole_ids: Sequence[int], board_ids: Sequence[int]
    ) -> np.ndarray:
        """Embed hole + board together. Returns shape (embedding_dim,)."""
        ...

    @abstractmethod
    def encode_batch(self, card_id_lists: Sequence[Sequence[int]]) -> np.ndarray:
        """Embed a batch of card sets. Returns shape (batch, embedding_dim)."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        ...
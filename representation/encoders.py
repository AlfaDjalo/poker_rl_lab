"""
representation/encoders.py — Card-set encoder architecture (CardSetEncoder /
CardStateEncoder), vendored into the CAP RL Lab.

Provenance
----------
This architecture was designed and TRAINED in a separate project, not in
the CAP codebase. That project owns the training pipeline (data
generation, loss functions, ML/training/trainer.py, DualOptimizerModel,
value heads, etc.) — none of which exists here, and none of which this
file depends on.

What lives here is only the two layer classes needed to reconstruct the
architecture: CardSetEncoder and CardStateEncoder. This is a HARD
requirement for loading the pretrained weights, not a convenience copy:
a .keras file saves each custom layer's config (constructor arguments)
and weights, but not its Python source. Keras reconstructs the layer by
calling the class's __init__ with the saved config — so the exact same
class definitions must exist wherever the file is deserialized, or
tf.keras.models.load_model() raises "Unknown layer: CardSetEncoder".
That's the whole reason this file exists in the CAP project: it is the
minimal vendored dependency representation/pretrained_hand_encoder.py
needs to load encoder_model.keras.

Consequences of vendoring rather than importing from the other project
------------------------------------------------------------------------
- This file must be kept byte-for-byte architecture-compatible with
  whatever produced encoder_model.keras (same layer types, same layer
  order, same hyperparameter names in `config`). If the other project's
  encoder architecture ever changes, this file needs to be re-synced
  manually — there is no shared import path enforcing that.
- CardSetEncoder's `embedding_dim` and `encoder_input_shape` are NOT
  hardcoded here; they come from the `config` dict passed to __init__,
  which in turn comes from whatever is baked into the saved .keras file
  (see PretrainedHandEncoder.from_file). Nothing here needs to guess
  those values.

What this module does
----------------------
Maps structured card grids (hand-only, board-only, or combined states,
see card_utils.card_ids_to_grid) into fixed-dimensional L2-normalized
embedding vectors:

- CardSetEncoder: single shared conv/dense stack for one card-grid input.
- CardStateEncoder: thin wrapper applying one CardSetEncoder to one or
  more inputs, so the same weights are reused across hand/board/combined
  calls (see PretrainedHandEncoder).

Design principles (as inherited from the source architecture):
- Preserve rank ordering semantics via progressive 1D convolutions
- Fully mix suit information early while avoiding suit-specific bias
- Produce normalized embeddings to stabilize similarity-based objectives
- Keep input semantics (e.g. suit permutation, augmentation) outside
  the encoder itself
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from keras.saving import register_keras_serializable

@register_keras_serializable(package="Poker")
class CardSetEncoder(Model):
    """
    Neural encoder for a single card set represented as a rank × suit grid.

    The encoder maps an input tensor of shape `(ranks, suits, channels)`
    (typically `(14, 4, C)` with duplicated Ace rank) into a fixed-size,
    L2-normalized embedding vector.

    Architectural overview:
    - 2D convolution to jointly mix suit information and local rank context
    - Reshape into a rank-sequential representation
    - Progressive 1D convolutions to model rank adjacency and ordering
    - Dense projection into an embedding space
    - L2 normalization for metric stability

    This encoder is intentionally agnostic to:
    - the semantic meaning of channels
    - whether the card set represents a hand, board, or other subset
    - suit permutation or other symmetry enforcement (handled upstream)

    Args:
        config (dict):
            Configuration dictionary containing:
            - "encoder_input_shape": Expected input shape (e.g. (14, 4, C))
            - "embedding_dim": Dimensionality of the output embedding

    Outputs:
        Tensor of shape `(batch_size, embedding_dim)`, L2-normalized.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        # Expected input shape (rank × suit × channels).
        # Typically ranks=14 to support Ace duplication for wheel straights.        
        self.encoder_input_shape = config["encoder_input_shape"]  # ideally (14,4,C)
        self.embedding_dim = config["embedding_dim"]

        # First stage: jointly mix suits and local rank patterns.
        # Kernel spans all suits (width=4) and 2 adjacent ranks.
        self.conv2d = layers.Conv2D(
            filters=32,
            kernel_size=(2,4),
            padding="valid",
            activation="relu"
        )
        self.bn2d = layers.BatchNormalization()
        
        # After Conv2D, collapse suit dimension and treat ranks as a sequence.
        # This enables rank-progressive 1D convolutions.        
        self.reshape = layers.Reshape((13, 32))
        # self.reshape = layers.Reshape((-1, 32))
        
        self.conv1 = layers.Conv1D(48, kernel_size=2, padding="valid", activation="relu")
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv1D(64, kernel_size=2, padding="valid", activation="relu")
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv1D(96, kernel_size=2, padding="valid", activation="relu")
        self.bn3 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(256, activation="relu")
        self.d2 = layers.Dense(64, activation="relu")
        self.embedding = layers.Dense(self.embedding_dim, activation=None)

        # L2 normalization stabilizes similarity-based objectives
        # (e.g. cosine similarity, contrastive loss, pairwise heads).
        self.l2_norm = layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=-1))

    def call(self, x, training=False):
        """
        Forward pass for a single card set.

        Args:
            x: Tensor of shape `(batch, ranks, suits, channels)`.
            training (bool): Whether the call is in training mode
                (affects batch normalization).

        Returns:
            L2-normalized embedding tensor of shape
            `(batch, embedding_dim)`.
        """
        x = self.conv2d(x)
        x = self.bn2d(x, training=training)

        x = self.reshape(x)

        x = self.conv1(x)
        x = self.bn1(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        x = self.embedding(x)
        x = self.l2_norm(x)

        return x

    def get_config(self):
        """
        Returns the configuration needed to serialize this encoder.

        The returned config is sufficient to fully reconstruct the encoder
        architecture and embedding dimensionality.
        """
        base = super().get_config()
        base.update({
            "config": {
                "input_shape_encoder": self.encoder_input_shape,
                "embedding_dim": self.embedding_dim,
            }
        })
        return base

    @classmethod
    def from_config(cls, config):
        if "config" in config:
            cfg = config.pop("config")
        else:
            cfg = config
        return cls(cfg, **config)


@register_keras_serializable(package="Poker")
class CardStateEncoder(tf.keras.Model):
    """
    Wrapper encoder that applies a shared CardSetEncoder to one or more inputs.

    This class enables flexible handling of different input modes:
    - single card set (e.g. hand-only)
    - multiple card sets (e.g. [hand, board])
    - paired inputs (e.g. A vs B comparisons)

    Each input tensor is encoded independently using the same
    underlying CardSetEncoder instance, ensuring weight sharing
    and consistent embedding geometry.

    Args:
        config (dict):
            Global or encoder-specific configuration. Only the
            fields relevant to the CardSetEncoder are extracted.

    Inputs:
        Either:
        - a single tensor `(batch, ranks, suits, channels)`, or
        - a list/tuple of such tensors

    Outputs:
        A list of embedding tensors, one per input, each of shape
        `(batch, embedding_dim)`.
    """
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self._encoder_config = dict(get_encoder_config(config)) if isinstance(config, dict) else dict(config)
        self.card_set_encoder = CardSetEncoder(self._encoder_config)

    def call(self, inputs, training=False):
        """
        Encode one or more card-state tensors using a shared encoder.

        Args:
            inputs:
                A tensor or a list/tuple of tensors, each representing
                a card set grid.
            training (bool):
                Whether the call is in training mode.

        Returns:
            List of L2-normalized embedding tensors, preserving input order.
        """
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        return [self.card_set_encoder(x, training=training) for x in inputs]        
           
    def get_config(self):
        base = super().get_config()
        base.update({
            "encoder_config": self._encoder_config,
        })
        return base

    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop("encoder_config")
        return cls(encoder_config, **config)


def get_encoder_config(global_config):
    """
    Extract the subset of global configuration relevant to the card encoder.

    This function acts as a boundary between higher-level model configuration
    (e.g. training modes, heads, losses) and the encoder architecture itself.

    Args:
        global_config (dict):
            Global experiment or model configuration.

    Returns:
        dict:
            Configuration dictionary suitable for initializing
            a CardSetEncoder.
    """
    return {
        "encoder_input_shape": global_config["encoder_input_shape"],
        "embedding_dim": global_config["embedding_dim"],
    }
"""Model definitions for the two required versions.

* build_baseline_cnn  -> Model Version 1 (baseline, trained from scratch on pixels)
* build_transfer_head -> Model Version 2 (improved: classifier head on frozen
  VGGFace ResNet50 embeddings + regularization)
"""
from tensorflow.keras import layers, models, regularizers

from src import config


def build_baseline_cnn(num_classes, img_size=None):
    """Small convolutional network trained from raw pixels (V1 baseline)."""
    img_size = img_size or config.IMG_SIZE
    model = models.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu", padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="v1_baseline_cnn",
    )
    return model


def build_transfer_head(num_classes, embed_dim=None):
    """Dense classifier head on top of frozen VGGFace embeddings (V2 improved).

    Transfer learning: the heavy VGGFace ResNet50 backbone stays frozen (its
    2048-d embeddings are pre-computed in embedding.pkl); we only train this
    lightweight regularized head. This is why V2 trains in seconds on CPU.
    """
    embed_dim = embed_dim or config.EMBED_DIM
    model = models.Sequential(
        [
            layers.Input(shape=(embed_dim,)),
            layers.Dense(512, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="v2_transfer_head",
    )
    return model

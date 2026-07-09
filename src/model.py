"""
Model Builder
=============

Builds the TensorFlow classification model.

Pipeline
--------
1. Build EfficientNetV2 backbone
2. Freeze backbone
3. Add classification head
4. Return compiled model structure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import tensorflow as tf
from keras import layers
from keras.applications import EfficientNetV2B0

from src.config import (
    CLASSIFIER_DROPOUT_RATE,
    DEFAULT_MODEL,
    FINE_TUNE_AT,
    INPUT_SHAPE,
    NUM_CLASSES,
)
from src.logger import logger

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================


@dataclass(slots=True)
class ModelConfiguration:
    """
    Model configuration.
    """

    input_shape: tuple[int, int, int] = INPUT_SHAPE

    num_classes: int = NUM_CLASSES

    model_name: str = DEFAULT_MODEL


@dataclass(slots=True)
class ModelArtifacts:
    """
    Built model artifacts.
    """

    base_model: tf.keras.Model
    model: tf.keras.Model


class ModelInformation(TypedDict):
    """
    Model statistics.
    """

    total_parameters: int
    trainable_parameters: int
    non_trainable_parameters: int


# =============================================================================
# MODEL BUILDER
# =============================================================================


class ModelBuilder:
    """
    TensorFlow model builder.
    """

    def __init__(
        self,
        configuration: ModelConfiguration | None = None,
    ) -> None:

        self.configuration = (
            configuration
            if configuration is not None
            else ModelConfiguration()
        )

        self.base_model: tf.keras.Model | None = None
        self.model: tf.keras.Model | None = None
        self.artifacts: ModelArtifacts | None = None

        logger.info("=" * 70)
        logger.info("Model Builder Initialized")
        logger.info(
            "Model : %s",
            self.configuration.model_name,
        )
        logger.info("=" * 70)

    # =========================================================================
    # BASE MODEL
    # =========================================================================

    def build_base_model(
        self,
    ) -> tf.keras.Model:
        """
        Build EfficientNetV2 backbone.
        """

        self.base_model = EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_shape=self.configuration.input_shape,
        )

        self.base_model.trainable = False

        logger.info("Base model created.")

        return self.base_model

    # =========================================================================
    # CLASSIFIER HEAD
    # =========================================================================

    def build_classifier(
        self,
        inputs: tf.Tensor,
    ) -> tf.Tensor:
        """
        Build classification head.
        """

        x = layers.GlobalAveragePooling2D(
            name="global_average_pooling",
        )(inputs)

        x = layers.Dropout(
            rate=CLASSIFIER_DROPOUT_RATE,
            name="dropout",
        )(x)

        outputs = layers.Dense(
            units=self.configuration.num_classes,
            activation="softmax",
            name="predictions",
        )(x)

        logger.info("Classification head created.")

        return outputs

    # =========================================================================
    # COMPLETE MODEL
    # =========================================================================

    def build_model(
        self,
    ) -> tf.keras.Model:
        """
        Build complete classification model.
        """

        if self.base_model is None:
            self.build_base_model()

        inputs = layers.Input(
            shape=self.configuration.input_shape,
            name="input_image",
        )

        x = self.base_model(
            inputs,
            training=False,
        )

        outputs = self.build_classifier(x)

        self.model = tf.keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=self.configuration.model_name,
        )

        logger.info("Model created successfully.")

        self.artifacts = ModelArtifacts(
            base_model=self.base_model,
            model=self.model,
        )

        return self.model

    # =========================================================================
    # FREEZE / UNFREEZE
    # =========================================================================

    def freeze_base_model(
        self,
    ) -> None:
        """
        Freeze backbone layers.
        """

        if self.base_model is None:
            return

        self.base_model.trainable = False

        logger.info("Base model frozen.")

    def unfreeze_top_layers(
        self,
    ) -> None:
        """
        Unfreeze the top layers of the backbone.
        """

        if self.base_model is None:
            return

        self.base_model.trainable = True

        if FINE_TUNE_AT <= 0:
            raise ValueError("FINE_TUNE_AT must be greater than zero.")

        if FINE_TUNE_AT > len(self.base_model.layers):
            raise ValueError(
                "FINE_TUNE_AT exceeds " "the number of backbone layers."
            )

        for layer in self.base_model.layers[:-FINE_TUNE_AT]:
            layer.trainable = False

        logger.info(
            "Top %d layers unfrozen.",
            FINE_TUNE_AT,
        )

    # =========================================================================
    # INFORMATION
    # =========================================================================

    def summary(
        self,
    ) -> None:
        """
        Display model summary.
        """

        if not self.is_built():
            self.build_model()

        self.model.summary()

    def total_parameters(
        self,
    ) -> int:
        """
        Return total model parameters.
        """

        if not self.is_built():
            self.build_model()

        return self.model.count_params()

    def trainable_parameters(
        self,
    ) -> int:
        """
        Return trainable parameters.
        """

        if not self.is_built():
            self.build_model()

        return sum(
            tf.keras.backend.count_params(variable)
            for variable in self.model.trainable_weights
        )

    def non_trainable_parameters(
        self,
    ) -> int:
        """
        Return non-trainable parameters.
        """

        if not self.is_built():
            self.build_model()

        return sum(
            tf.keras.backend.count_params(variable)
            for variable in self.model.non_trainable_weights
        )

    def model_information(
        self,
    ) -> ModelInformation:
        """
        Return model information.
        """

        return {
            "total_parameters": self.total_parameters(),
            "trainable_parameters": self.trainable_parameters(),
            "non_trainable_parameters": self.non_trainable_parameters(),
        }

    def is_built(
        self,
    ) -> bool:
        """
        Return True if the model has been built.
        """

        return self.model is not None

    # =========================================================================
    # PIPELINE
    # =========================================================================

    def run(
        self,
    ) -> ModelArtifacts:
        """
        Build the complete model.
        """

        logger.info("=" * 70)
        logger.info("START MODEL BUILDING")
        logger.info("=" * 70)

        self.build_base_model()

        self.build_model()

        logger.info(
            "Total Parameters : %d",
            self.total_parameters(),
        )

        logger.info(
            "Trainable Parameters : %d",
            self.trainable_parameters(),
        )

        logger.info(
            "Non-trainable Parameters : %d",
            self.non_trainable_parameters(),
        )

        logger.info("=" * 70)
        logger.info("MODEL BUILDING COMPLETED")
        logger.info("=" * 70)

        if self.artifacts is None:
            raise RuntimeError("Model artifacts were not created.")

        return self.artifacts

    # =========================================================================
    # MAGIC METHODS
    # =========================================================================

    def __len__(
        self,
    ) -> int:
        """
        Return total number of parameters.
        """

        return self.total_parameters()

    def __repr__(
        self,
    ) -> str:

        information = self.model_information()

        return (
            "ModelBuilder("
            f"model='{self.configuration.model_name}', "
            f"classes={self.configuration.num_classes}, "
            f"parameters={information['total_parameters']}"
            ")"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Execute model building.
    """

    builder = ModelBuilder()

    artifacts = builder.run()

    builder.summary()

    logger.info(
        "Model Name : %s",
        artifacts.model.name,
    )


if __name__ == "__main__":
    main()

__all__ = [
    "ModelArtifacts",
    "ModelBuilder",
    "ModelConfiguration",
    "ModelInformation",
]

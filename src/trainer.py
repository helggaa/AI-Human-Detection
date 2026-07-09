"""
Model Trainer
=============

Train and fine-tune the TensorFlow model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tensorflow as tf
from keras.callbacks import (
    EarlyStopping,
    History,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from src.config import (
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    FINAL_MODEL_FILENAME,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    LEARNING_RATE,
    METRIC_DIR,
    MIN_LEARNING_RATE,
    MODEL_DIR,
    MODEL_FILENAME,
    REDUCE_LR_FACTOR,
    REDUCE_LR_PATIENCE,
    TRAINING_HISTORY_FILENAME,
)
from src.logger import logger
from src.model import (
    ModelArtifacts,
    ModelBuilder,
)
from src.preprocessing import (
    DatasetBundle,
    DatasetPreprocessor,
)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================


@dataclass(slots=True)
class TrainingConfiguration:

    learning_rate: float = LEARNING_RATE

    fine_tune_learning_rate: float = FINE_TUNE_LEARNING_RATE

    epochs: int = EPOCHS

    fine_tune_epochs: int = FINE_TUNE_EPOCHS


@dataclass(slots=True)
class TrainingResults:
    """
    Final training results.
    """

    initial_history: History

    fine_tune_history: History | None

    model: tf.keras.Model


# =============================================================================
# MODEL TRAINER
# =============================================================================


class ModelTrainer:

    def __init__(
        self,
        builder: ModelBuilder,
        artifacts: ModelArtifacts,
        dataset: DatasetBundle,
        configuration: TrainingConfiguration | None = None,
    ) -> None:
        self.builder = builder
        self.artifacts = artifacts

        self.dataset = dataset

        self.configuration = (
            configuration
            if configuration is not None
            else TrainingConfiguration()
        )

        self.history: History | None = None
        self.fine_tune_history: History | None = None

        logger.info("=" * 70)
        logger.info("Model Trainer Initialized")
        logger.info("=" * 70)

    # =========================================================================
    # MODEL COMPILATION
    # =========================================================================

    def compile_model(
        self,
        learning_rate: float,
    ) -> None:
        """
        Compile the TensorFlow model.
        """

        self.artifacts.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        )

        logger.info(
            "Model compiled (learning_rate=%f).",
            learning_rate,
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def build_callbacks(
        self,
    ) -> list[tf.keras.callbacks.Callback]:
        """
        Build training callbacks.
        """

        callbacks: list[tf.keras.callbacks.Callback] = [
            EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=REDUCE_LR_FACTOR,
                patience=REDUCE_LR_PATIENCE,
                min_lr=MIN_LEARNING_RATE,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=MODEL_DIR / MODEL_FILENAME,
                monitor="val_accuracy",
                save_weights_only=False,
                save_best_only=True,
                verbose=1,
            ),
        ]

        logger.info(
            "%d callbacks created.",
            len(callbacks),
        )

        return callbacks

    # =========================================================================
    # TRAINING
    # =========================================================================

    def train(
        self,
    ) -> History:
        """
        Train the classifier head.
        """

        self.compile_model(
            self.configuration.learning_rate,
        )

        history = self.artifacts.model.fit(
            self.dataset.train_dataset,
            validation_data=self.dataset.validation_dataset,
            epochs=self.configuration.epochs,
            callbacks=self.build_callbacks(),
            verbose=1,
        )

        self.history = history

        logger.info("Initial training completed.")

        return history

    # =========================================================================
    # FINE-TUNING
    # =========================================================================

    def fine_tune(
        self,
    ) -> History:
        """
        Fine-tune the pretrained backbone.
        """
        self.builder.unfreeze_top_layers()

        self.compile_model(
            self.configuration.fine_tune_learning_rate,
        )

        history = self.artifacts.model.fit(
            self.dataset.train_dataset,
            validation_data=self.dataset.validation_dataset,
            epochs=(
                self.configuration.epochs + self.configuration.fine_tune_epochs
            ),
            initial_epoch=self.configuration.epochs,
            callbacks=self.build_callbacks(),
            verbose=1,
        )

        self.fine_tune_history = history

        logger.info("Fine-tuning completed.")

        return history

    # =========================================================================
    # INFORMATION
    # =========================================================================

    def is_trained(
        self,
    ) -> bool:
        """
        Return True if the initial training has finished.
        """

        return self.history is not None

    def is_fine_tuned(
        self,
    ) -> bool:
        """
        Return True if fine-tuning has finished.
        """

        return self.fine_tune_history is not None

    def current_model(
        self,
    ) -> tf.keras.Model:
        """
        Return the current trained model.
        """

        return self.artifacts.model

    # =========================================================================
    # PIPELINE
    # =========================================================================

    def run(
        self,
        fine_tune: bool = True,
    ) -> TrainingResults:
        """
        Execute the complete training pipeline.
        """

        logger.info("=" * 70)
        logger.info("START TRAINING")
        logger.info("=" * 70)

        initial_history = self.train()

        fine_tune_history = None

        if fine_tune:

            fine_tune_history = self.fine_tune()

        logger.info("=" * 70)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 70)

        return TrainingResults(
            initial_history=initial_history,
            fine_tune_history=fine_tune_history,
            model=self.current_model(),
        )

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================

    @staticmethod
    def history_to_dataframe(
        history: History,
    ) -> pd.DataFrame:
        """
        Convert a TensorFlow History object to a DataFrame.
        """

        return pd.DataFrame(
            history.history,
        )

    def export_history(
        self,
        history: History,
        output_path: Path,
    ) -> None:
        """
        Export training history.
        """

        dataframe = self.history_to_dataframe(
            history,
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Training history saved: %s",
            output_path,
        )

    def export_results(
        self,
        results: TrainingResults,
    ) -> None:
        """
        Export training results.
        """

        history = (
            results.fine_tune_history
            if results.fine_tune_history is not None
            else results.initial_history
        )

        self.export_history(
            history,
            METRIC_DIR / TRAINING_HISTORY_FILENAME,
        )

    def save_final_model(
        self,
        output_path: Path | None = None,
    ) -> None:
        """
        Save the final trained model.
        """
        if output_path is None:
            output_path = MODEL_DIR / FINAL_MODEL_FILENAME

        self.current_model().save(
            output_path,
        )

        logger.info(
            "Final model saved: %s",
            output_path,
        )

    # =========================================================================
    # MAGIC METHODS
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        return (
            "ModelTrainer("
            f"epochs={self.configuration.epochs}, "
            f"fine_tune_epochs="
            f"{self.configuration.fine_tune_epochs}"
            ")"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Execute model training.
    """

    preprocessor = DatasetPreprocessor()

    dataset = preprocessor.run()

    builder = ModelBuilder()

    artifacts = builder.run()

    trainer = ModelTrainer(
        builder=builder,
        artifacts=artifacts,
        dataset=dataset,
    )

    results = trainer.run(
        fine_tune=True,
    )

    trainer.export_results(
        results,
    )

    trainer.save_final_model()

    logger.info("Training pipeline completed.")


if __name__ == "__main__":
    main()

__all__ = [
    "TrainingConfiguration",
    "TrainingResults",
    "ModelTrainer",
]

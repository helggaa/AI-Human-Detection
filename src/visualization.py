"""
Visualization Utilities
=======================

Visualization functions for training and evaluation results.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import History
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from src.config import (
    CONFUSION_MATRIX_FIGURE_SIZE,
    DEFAULT_FIGURE_SIZE,
    FIGURE_DIR,
    FIGURE_DPI,
    ROC_FIGURE_SIZE,
    SAVE_FIGURES,
)
from src.logger import logger

# =============================================================================
# VISUALIZATION
# =============================================================================


class Visualizer:
    """
    Visualization utilities.
    """

    # =========================================================================
    # HELPER
    # =========================================================================

    @staticmethod
    def save_figure(
        figure: plt.Figure,
        filename: str,
    ) -> None:
        """
        Save matplotlib figure.
        """

        if not SAVE_FIGURES:
            return

        output_path = FIGURE_DIR / filename

        figure.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
        )
        plt.close(figure)

        logger.info(
            "Figure saved: %s",
            output_path,
        )

    @staticmethod
    def show() -> None:
        """
        Display all figures.
        """

        plt.show()

    # =========================================================================
    # TRAINING HISTORY
    # =========================================================================

    @staticmethod
    def plot_training_history(
        history: History,
    ) -> plt.Figure:
        """
        Plot training accuracy.
        """

        figure, axis = plt.subplots(
            figsize=DEFAULT_FIGURE_SIZE,
        )

        axis.plot(
            history.history["accuracy"],
            label="Training",
        )

        axis.plot(
            history.history["val_accuracy"],
            label="Validation",
        )

        axis.set_title(
            "Training Accuracy",
        )

        axis.set_xlabel(
            "Epoch",
        )

        axis.set_ylabel(
            "Accuracy",
        )

        axis.legend()

        figure.tight_layout()

        return figure

    @staticmethod
    def plot_loss_history(
        history: History,
    ) -> plt.Figure:
        """
        Plot training loss.
        """

        figure, axis = plt.subplots(
            figsize=DEFAULT_FIGURE_SIZE,
        )

        axis.plot(
            history.history["loss"],
            label="Training",
        )

        axis.plot(
            history.history["val_loss"],
            label="Validation",
        )

        axis.set_title(
            "Training Loss",
        )

        axis.set_xlabel(
            "Epoch",
        )

        axis.set_ylabel(
            "Loss",
        )

        axis.legend()

        figure.tight_layout()

        return figure

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    @staticmethod
    def plot_confusion_matrix(
        matrix: pd.DataFrame,
        class_names: list[str],
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        """

        figure, axis = plt.subplots(
            figsize=CONFUSION_MATRIX_FIGURE_SIZE,
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix.values,
            display_labels=class_names,
        )

        display.plot(
            ax=axis,
            colorbar=False,
        )

        figure.tight_layout()

        return figure

    # =========================================================================
    # ROC CURVE
    # =========================================================================

    @staticmethod
    def plot_roc_curve(
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> plt.Figure:
        """
        Plot ROC curve.
        """

        figure, axis = plt.subplots(
            figsize=ROC_FIGURE_SIZE,
        )

        RocCurveDisplay.from_predictions(
            labels,
            probabilities,
            ax=axis,
        )

        figure.tight_layout()

        return figure

    # =========================================================================
    # EXPORT
    # =========================================================================

    @staticmethod
    def save_training_history(
        history: History,
    ) -> None:
        """
        Save training accuracy and loss figures.
        """

        accuracy = Visualizer.plot_training_history(
            history,
        )

        Visualizer.save_figure(
            accuracy,
            "training_accuracy.png",
        )

        loss = Visualizer.plot_loss_history(
            history,
        )

        Visualizer.save_figure(
            loss,
            "training_loss.png",
        )

    @staticmethod
    def save_confusion_matrix(
        matrix: pd.DataFrame,
        class_names: list[str],
    ) -> None:
        """
        Save confusion matrix figure.
        """

        figure = Visualizer.plot_confusion_matrix(
            matrix,
            class_names,
        )

        Visualizer.save_figure(
            figure,
            "confusion_matrix.png",
        )

    @staticmethod
    def save_roc_curve(
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        """
        Save ROC curve.
        """

        figure = Visualizer.plot_roc_curve(
            labels,
            probabilities,
        )

        Visualizer.save_figure(
            figure,
            "roc_curve.png",
        )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Visualization utilities.
    """

    logger.info("Visualization module loaded.")


if __name__ == "__main__":
    main()

__all__ = [
    "Visualizer",
]

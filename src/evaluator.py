"""
Model Evaluation
================

Evaluate the trained TensorFlow model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    CLASS_LABELS,
    CLASSIFICATION_REPORT_FILENAME,
    CONFUSION_MATRIX_FILENAME,
    METRIC_DIR,
    NUM_CLASSES,
)
from src.logger import logger
from src.model import ModelBuilder
from src.preprocessing import (
    DatasetBundle,
    DatasetPreprocessor,
)
from src.trainer import (
    ModelTrainer,
    TrainingResults,
)

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


@dataclass(slots=True)
class EvaluationMetrics:
    """
    Evaluation metrics.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float


@dataclass(slots=True)
class PredictionResults:
    """
    Prediction outputs.
    """

    predictions: np.ndarray
    probabilities: np.ndarray


@dataclass(slots=True)
class EvaluationArtifacts:
    """
    Evaluation outputs.
    """

    metrics: EvaluationMetrics

    classification_report: pd.DataFrame

    confusion_matrix: pd.DataFrame

    prediction_results: PredictionResults


# =============================================================================
# MODEL EVALUATOR
# =============================================================================


class ModelEvaluator:
    """
    Evaluate trained models.
    """

    def __init__(
        self,
        results: TrainingResults,
        dataset: DatasetBundle,
    ) -> None:
        self.results = results
        self.dataset = dataset

        logger.info("=" * 70)
        logger.info("Model Evaluator Initialized")
        logger.info("=" * 70)

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def labels(
        self,
    ) -> np.ndarray:
        """
        Return ground-truth labels.
        """

        return self.dataset.test_dataframe["label"].to_numpy()

    @property
    def class_names(
        self,
    ) -> list[str]:
        """
        Return class names.
        """

        return list(CLASS_LABELS.values())

    # =========================================================================
    # PREDICTION
    # =========================================================================

    def predict(
        self,
    ) -> PredictionResults:
        """
        Predict the test dataset.
        """
        probabilities = self.results.model.predict(
            self.dataset.test_dataset,
            verbose=0,
        )
        predictions = np.argmax(
            probabilities,
            axis=1,
        ).astype(np.int32)
        logger.info("Prediction completed.")
        return PredictionResults(
            predictions=predictions,
            probabilities=probabilities,
        )

    # =========================================================================
    # METRICS
    # =========================================================================

    def calculate_metrics(
        self,
        prediction_results: PredictionResults,
    ) -> EvaluationMetrics:
        """
        Calculate evaluation metrics.
        """
        average = self.average_method()

        try:
            if NUM_CLASSES == 2:
                roc_auc = roc_auc_score(
                    self.labels,
                    prediction_results.probabilities[:, 1],
                )
            else:
                roc_auc = roc_auc_score(
                    self.labels,
                    prediction_results.probabilities,
                    multi_class="ovr",
                )
        except ValueError:
            roc_auc = float("nan")

        return EvaluationMetrics(
            accuracy=accuracy_score(
                self.labels,
                prediction_results.predictions,
            ),
            precision=precision_score(
                self.labels,
                prediction_results.predictions,
                average=average,
                zero_division=0,
            ),
            recall=recall_score(
                self.labels,
                prediction_results.predictions,
                average=average,
                zero_division=0,
            ),
            f1_score=f1_score(
                self.labels,
                prediction_results.predictions,
                average=average,
                zero_division=0,
            ),
            roc_auc=roc_auc,
        )

    def classification_report_dataframe(
        self,
        prediction_results: PredictionResults,
    ) -> pd.DataFrame:
        """
        Generate classification report.
        """

        report = classification_report(
            self.labels,
            prediction_results.predictions,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        return pd.DataFrame(report).transpose()

    def confusion_matrix_dataframe(
        self,
        prediction_results: PredictionResults,
    ) -> pd.DataFrame:
        """
        Generate confusion matrix.
        """

        class_names = self.class_names

        matrix = confusion_matrix(
            self.labels,
            prediction_results.predictions,
        )

        return pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names,
        )

    @staticmethod
    def average_method() -> str:
        """
        Return averaging method based on the number of classes.
        """

        return "binary" if NUM_CLASSES == 2 else "macro"

    # =========================================================================
    # EXPORT
    # =========================================================================

    @staticmethod
    def export_dataframe(
        dataframe: pd.DataFrame,
        output_path: Path,
        index: bool = True,
    ) -> None:
        """
        Export a dataframe to CSV.
        """
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            output_path,
            index=index,
            encoding="utf-8",
        )

        logger.info(
            "Saved: %s",
            output_path,
        )

    def export_reports(
        self,
        artifacts: EvaluationArtifacts,
    ) -> None:
        """
        Export evaluation reports.
        """

        self.export_dataframe(
            artifacts.classification_report,
            METRIC_DIR / CLASSIFICATION_REPORT_FILENAME,
        )

        self.export_dataframe(
            artifacts.confusion_matrix,
            METRIC_DIR / CONFUSION_MATRIX_FILENAME,
        )

    # =========================================================================
    # PIPELINE
    # =========================================================================

    def run(
        self,
    ) -> EvaluationArtifacts:
        """
        Execute evaluation pipeline.
        """

        logger.info("=" * 70)
        logger.info("START EVALUATION")
        logger.info("=" * 70)

        prediction_results = self.predict()

        metrics = self.calculate_metrics(
            prediction_results,
        )

        report = self.classification_report_dataframe(
            prediction_results,
        )

        matrix = self.confusion_matrix_dataframe(
            prediction_results,
        )

        artifacts = EvaluationArtifacts(
            metrics=metrics,
            classification_report=report,
            confusion_matrix=matrix,
            prediction_results=prediction_results,
        )

        self.export_reports(
            artifacts,
        )

        logger.info("=" * 70)

        logger.info(
            "Accuracy  : %.4f",
            metrics.accuracy,
        )

        logger.info(
            "Precision : %.4f",
            metrics.precision,
        )

        logger.info(
            "Recall    : %.4f",
            metrics.recall,
        )

        logger.info(
            "F1-score  : %.4f",
            metrics.f1_score,
        )

        logger.info(
            "ROC-AUC   : %.4f",
            metrics.roc_auc,
        )

        logger.info("EVALUATION COMPLETED")
        logger.info("=" * 70)

        return artifacts


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Execute model evaluation.
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

    evaluator = ModelEvaluator(
        results=results,
        dataset=dataset,
    )

    evaluator.run()


if __name__ == "__main__":
    main()

__all__ = [
    "PredictionResults",
    "EvaluationMetrics",
    "EvaluationArtifacts",
    "ModelEvaluator",
]

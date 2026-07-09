"""
Model Inference
===============

Reusable inference engine for the AI Human Detection project.

This module provides a production-ready interface for loading the
trained TensorFlow model and performing inference on images.

The predictor is intentionally independent from any user interface.
It can be reused by:

- predict.py (CLI)
- app.py (Streamlit)
- unit tests
- future REST API
"""

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np
import tensorflow as tf
from keras.applications.efficientnet_v2 import preprocess_input
from keras.models import load_model
from PIL import Image, ImageOps

from src.config import (
    CLASS_LABELS,
    IMAGE_SIZE,
    MODEL_DIR,
    MODEL_FILENAME,
    MODEL_VERSION,
)
from src.logger import logger
from src.utils import convert_rgb

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(slots=True)
class PredictorConfiguration:
    """
    Predictor configuration.

    Parameters
    ----------
    model_path : Path
        Path to trained model.

    image_size : tuple[int, int]
        Input image size.

    class_labels : dict[int, str]
        Mapping between class index and class label.
    """

    model_path: Path = MODEL_DIR / MODEL_FILENAME

    image_size: tuple[int, int] = IMAGE_SIZE

    class_labels: Mapping[int, str] = field(
        default_factory=lambda: CLASS_LABELS.copy(),
    )


# =============================================================================
# RESULT
# =============================================================================


@dataclass(slots=True)
class PredictionResult:
    """
    Prediction result.

    Parameters
    ----------
    predicted_label : str
        Predicted class label.

    predicted_index : int
        Predicted class index.

    confidence : float
        Highest prediction confidence (%).

    probabilities : dict[str, float]
        Probability for each class (%).

    inference_time_ms : float
        Model inference time in milliseconds.
    """

    predicted_label: str

    predicted_index: int

    confidence: float

    probabilities: dict[str, float]

    inference_time_ms: float


# =============================================================================
# INFORMATION
# =============================================================================


@dataclass(slots=True)
class PredictorInformation:
    """
    Predictor metadata.

    Parameters
    ----------
    model_name : str
        Loaded model name.

    model_version : str
        Model version.

    model_path : Path
        Path to the trained model.

    image_size : tuple[int, int]
        Expected input image size.

    number_of_classes : int
        Number of supported classes.

    class_labels : tuple[str, ...]
        Ordered class labels.
    """

    model_name: str

    model_version: str

    model_path: Path

    image_size: tuple[int, int]

    number_of_classes: int

    class_labels: tuple[str, ...]


# =============================================================================
# PREDICTOR
# =============================================================================


class ImagePredictor:
    """
    Production-ready image predictor.

    Notes
    -----
    This class is responsible only for inference.

    It does NOT contain any CLI code or Streamlit code.
    """

    def __init__(
        self,
        configuration: PredictorConfiguration | None = None,
    ) -> None:
        """
        Initialize predictor.

        Parameters
        ----------
        configuration : PredictorConfiguration | None
            Predictor configuration.
        """

        self.configuration = (
            configuration
            if configuration is not None
            else PredictorConfiguration()
        )

        self._model: tf.keras.Model = self._load_model()

        logger.info("ImagePredictor initialized successfully.")

    # =========================================================================
    # PROPERTIES
    # =========================================================================

    @property
    def model(self) -> tf.keras.Model:
        """
        Loaded TensorFlow model.

        Returns
        -------
        tf.keras.Model
        """

        return self._model

    @property
    def model_name(
        self,
    ) -> str:
        """
        Return model name.

        Returns
        -------
        str
        """

        return self.model.name

    @property
    def image_size(
        self,
    ) -> tuple[int, int]:
        """
        Return inference image size.

        Returns
        -------
        tuple[int, int]
        """

        return self.configuration.image_size

    @property
    def number_of_classes(
        self,
    ) -> int:
        """
        Return number of supported classes.

        Returns
        -------
        int
        """

        return len(
            self.class_labels,
        )

    @property
    def class_labels(
        self,
    ) -> Mapping[int, str]:
        """
        Return class labels.
        """

        return self.configuration.class_labels

    @property
    def information(
        self,
    ) -> PredictorInformation:
        """
        Return predictor metadata.

        Returns
        -------
        PredictorInformation
        """

        return PredictorInformation(
            model_name=self.model_name,
            model_version=MODEL_VERSION,
            model_path=self.configuration.model_path,
            image_size=self.image_size,
            number_of_classes=self.number_of_classes,
            class_labels=tuple(
                self.class_labels[index] for index in sorted(self.class_labels)
            ),
        )

    # =========================================================================
    # MODEL
    # =========================================================================

    def _load_model(
        self,
    ) -> tf.keras.Model:
        """
        Load trained TensorFlow model.

        Returns
        -------
        tf.keras.Model

        Raises
        ------
        FileNotFoundError
            If model file does not exist.

        RuntimeError
            If model loading fails.
        """

        model_path = self.configuration.model_path

        logger.info(
            "Loading model: %s",
            model_path,
        )

        if not model_path.exists():

            logger.error(
                "Model file not found: %s",
                model_path,
            )

            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        try:

            model = load_model(
                model_path,
            )

            logger.info("Model loaded successfully.")

            return model

        except Exception as error:

            logger.exception("Unable to load trained model.")

            raise RuntimeError("Failed to load trained model.") from error

    # =========================================================================
    # MAGIC METHODS
    # =========================================================================

    def __repr__(
        self,
    ) -> str:

        return (
            "ImagePredictor("
            f"model_name='{self.model_name}', "
            f"image_size={self.image_size}, "
            f"classes={self.number_of_classes}"
            ")"
        )

    # =========================================================================
    # IMAGE LOADING
    # =========================================================================

    def _load_image(
        self,
        image_path: Path,
    ) -> Image.Image:
        """
        Load an image from disk.

        Parameters
        ----------
        image_path : Path
            Image path.

        Returns
        -------
        Image.Image
            RGB image.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.

        ValueError
            If the image cannot be opened.
        """

        logger.info(
            "Loading image: %s",
            image_path,
        )

        if not image_path.exists():

            logger.error(
                "Image not found: %s",
                image_path,
            )

            raise FileNotFoundError(f"Image not found: {image_path}")

        try:

            with Image.open(image_path) as image:

                image = ImageOps.exif_transpose(image)

                if image.width == 0 or image.height == 0:
                    raise ValueError("Image has invalid dimensions.")

                if image.mode != "RGB":
                    image = convert_rgb(image)
                else:
                    image = image.copy()

            return image

        except Exception as error:

            logger.exception("Unable to load image.")

            raise ValueError(f"Invalid image: {image_path}") from error

    # =========================================================================
    # PREPROCESSING
    # =========================================================================

    def _preprocess(
        self,
        image: Image.Image,
    ) -> np.ndarray:
        """
        Preprocess an image for inference.

        Parameters
        ----------
        image : Image.Image
            RGB image.

        Returns
        -------
        np.ndarray
            Preprocessed image tensor.
        """

        logger.info("Preprocessing image.")

        if image.mode != "RGB":
            image = convert_rgb(image)

        image = image.resize(
            size=self.image_size,
            resample=Image.Resampling.BILINEAR,
        )

        image_array = np.asarray(
            image,
            dtype=np.float32,
        )

        image_array = np.expand_dims(
            image_array,
            axis=0,
        )

        image_array = preprocess_input(
            image_array,
        )

        logger.info("Image preprocessing completed.")

        return image_array

    # =========================================================================
    # MODEL INFERENCE
    # =========================================================================

    def _predict(
        self,
        image_array: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Perform model inference.

        Parameters
        ----------
        image_array : np.ndarray
            Preprocessed image.

        Returns
        -------
        tuple[np.ndarray, float]
            Prediction probabilities and inference time
            in milliseconds.

        Raises
        ------
        RuntimeError
            If prediction output is invalid.
        """

        logger.info("Running inference.")

        start = perf_counter()

        predictions = np.asarray(
            self.model.predict(
                image_array,
                verbose=0,
            )[0],
            dtype=np.float32,
        )

        elapsed_ms = (perf_counter() - start) * 1000.0

        if predictions.ndim != 1:
            raise RuntimeError(
                "Expected a 1D prediction vector. "
                f"Received shape {predictions.shape}."
            )

        expected_classes = self.number_of_classes

        if predictions.shape[0] != expected_classes:
            raise RuntimeError(
                "Model output dimension does not "
                "match configured class labels. "
                f"Expected {expected_classes}, "
                f"received {predictions.shape[0]}."
            )

        if not np.isfinite(predictions).all():
            raise RuntimeError("Prediction contains invalid values.")

        probability_sum = float(np.sum(predictions))

        if not np.isclose(
            probability_sum,
            1.0,
            atol=1e-3,
        ):

            logger.warning(
                "Prediction probabilities sum " "to %.6f instead of 1.0.",
                probability_sum,
            )

        logger.info("Inference completed.")

        return (
            predictions,
            round(elapsed_ms, 2),
        )

    # =========================================================================
    # RESULT
    # =========================================================================

    def _build_result(
        self,
        predictions: np.ndarray,
        inference_time_ms: float,
    ) -> PredictionResult:
        """
        Build a PredictionResult from model predictions.

        Parameters
        ----------
        predictions : np.ndarray
            Prediction probabilities.

        inference_time_ms : float
            Model inference time in milliseconds.

        Returns
        -------
        PredictionResult
            Structured prediction result.
        """

        predicted_index = int(np.argmax(predictions))

        predicted_label = self.class_labels[predicted_index]

        confidence = round(
            np.max(predictions).item() * 100,
            2,
        )

        probabilities = {
            self.class_labels[index]: round(
                float(probability * 100),
                2,
            )
            for index, probability in enumerate(predictions)
        }

        logger.info(
            "Prediction completed: %s (%.2f%%)",
            predicted_label,
            confidence,
        )

        return PredictionResult(
            predicted_label=predicted_label,
            predicted_index=predicted_index,
            confidence=confidence,
            probabilities=probabilities,
            inference_time_ms=inference_time_ms,
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def predict_from_path(
        self,
        image_path: str | Path,
    ) -> PredictionResult:
        """
        Predict from an image file.

        Parameters
        ----------
        image_path : str | Path
            Image path.

        Returns
        -------
        PredictionResult
        """

        image = self._load_image(
            Path(image_path),
        )

        image_array = self._preprocess(
            image,
        )

        predictions, inference_time_ms = self._predict(
            image_array,
        )

        return self._build_result(
            predictions,
            inference_time_ms,
        )

    def predict_from_pil(
        self,
        image: Image.Image,
    ) -> PredictionResult:
        """
        Predict from a PIL image.

        Parameters
        ----------
        image : Image.Image
            Input image.

        Returns
        -------
        PredictionResult
        """

        image = ImageOps.exif_transpose(
            image,
        )

        if image.mode != "RGB":
            image = convert_rgb(image)

        image_array = self._preprocess(
            image,
        )

        predictions, inference_time_ms = self._predict(
            image_array,
        )

        return self._build_result(
            predictions,
            inference_time_ms,
        )

    def predict_from_numpy(
        self,
        image: np.ndarray,
    ) -> PredictionResult:
        """
        Predict from a NumPy array.

        Parameters
        ----------
        image : np.ndarray
            RGB image array.

        Returns
        -------
        PredictionResult
        """

        if image.ndim != 3:
            raise ValueError(
                "Expected image shape " "(height, width, channels)."
            )

        if image.shape[2] != 3:
            raise ValueError("Expected RGB image with 3 channels.")

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)

        pil_image = Image.fromarray(image)

        return self.predict_from_pil(
            pil_image,
        )

    def predict(
        self,
        image_path: str | Path,
    ) -> PredictionResult:
        """
        Predict an image from its file path.

        This is a convenience wrapper around
        ``predict_from_path()``.

        Parameters
        ----------
        image_path : str | Path
            Path to the input image.

        Returns
        -------
        PredictionResult
            Prediction result.
        """

        return self.predict_from_path(
            image_path,
        )

    def __len__(
        self,
    ) -> int:
        """
        Return number of classes.
        """

        return self.number_of_classes


__all__ = [
    "PredictorConfiguration",
    "PredictionResult",
    "PredictorInformation",
    "ImagePredictor",
]

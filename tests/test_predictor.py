"""
Tests for predictor.py
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.predictor import (
    ImagePredictor,
    PredictionResult,
)

# =============================================================================
# INITIALIZATION
# =============================================================================


def test_predictor_initialization(
    predictor: ImagePredictor,
) -> None:
    """
    Predictor should initialize correctly.
    """

    assert predictor is not None
    assert predictor.model is not None


def test_model_name(
    predictor: ImagePredictor,
) -> None:
    """
    Model name should not be empty.
    """

    assert predictor.model_name != ""


def test_number_of_classes(
    predictor: ImagePredictor,
) -> None:
    """
    Predictor should contain two classes.
    """

    assert predictor.number_of_classes == 2


# =============================================================================
# PIL
# =============================================================================


def test_predict_from_pil(
    predictor: ImagePredictor,
    rgb_image: Image.Image,
) -> None:
    """
    Predict from PIL image.
    """

    result = predictor.predict_from_pil(
        rgb_image,
    )

    assert isinstance(
        result,
        PredictionResult,
    )

    assert result.predicted_index in (
        0,
        1,
    )

    assert result.predicted_label in result.probabilities

    assert 0 <= result.confidence <= 100


def test_predict_grayscale_image(
    predictor: ImagePredictor,
    grayscale_image: Image.Image,
) -> None:
    """
    Predictor should accept grayscale image.
    """

    result = predictor.predict_from_pil(
        grayscale_image,
    )

    assert isinstance(
        result,
        PredictionResult,
    )


def test_predict_rgba_image(
    predictor: ImagePredictor,
    rgba_image: Image.Image,
) -> None:
    """
    Predictor should accept RGBA image.
    """

    result = predictor.predict_from_pil(
        rgba_image,
    )

    assert isinstance(
        result,
        PredictionResult,
    )


# =============================================================================
# NUMPY
# =============================================================================


def test_predict_numpy(
    predictor: ImagePredictor,
    numpy_image: np.ndarray,
) -> None:
    """
    Predict from NumPy image.
    """

    result = predictor.predict_from_numpy(
        numpy_image,
    )

    assert isinstance(
        result,
        PredictionResult,
    )


def test_invalid_numpy_dimension(
    predictor: ImagePredictor,
) -> None:
    """
    Invalid NumPy dimensions.
    """

    image = np.zeros(
        (
            224,
            224,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
    ):
        predictor.predict_from_numpy(
            image,
        )


def test_invalid_numpy_channels(
    predictor: ImagePredictor,
) -> None:
    """
    Invalid channel count.
    """

    image = np.zeros(
        (
            224,
            224,
            1,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
    ):
        predictor.predict_from_numpy(
            image,
        )


# =============================================================================
# FILE
# =============================================================================


def test_predict_from_path(
    predictor: ImagePredictor,
    image_file,
) -> None:
    """
    Predict from image path.
    """

    result = predictor.predict(
        image_file,
    )

    assert isinstance(
        result,
        PredictionResult,
    )


def test_missing_image(
    predictor: ImagePredictor,
) -> None:
    """
    Missing image should raise.
    """

    with pytest.raises(
        FileNotFoundError,
    ):
        predictor.predict(
            "missing.jpg",
        )


def test_invalid_image(
    predictor: ImagePredictor,
    invalid_image_file,
) -> None:
    """
    Invalid image should raise.
    """

    with pytest.raises(
        ValueError,
    ):
        predictor.predict(
            invalid_image_file,
        )


# =============================================================================
# RESULT
# =============================================================================


def test_probability_sum(
    predictor: ImagePredictor,
    rgb_image: Image.Image,
) -> None:
    """
    Probabilities should sum to ~100%.
    """

    result = predictor.predict_from_pil(
        rgb_image,
    )

    total = sum(result.probabilities.values())

    assert abs(total - 100) < 0.05


def test_inference_time(
    predictor: ImagePredictor,
    rgb_image: Image.Image,
) -> None:
    """
    Inference time should be positive.
    """

    result = predictor.predict_from_pil(
        rgb_image,
    )

    assert result.inference_time_ms > 0


# =============================================================================
# MAGIC METHODS
# =============================================================================


def test_len(
    predictor: ImagePredictor,
) -> None:
    """
    __len__ should return number of classes.
    """

    assert len(predictor) == predictor.number_of_classes


def test_repr(
    predictor: ImagePredictor,
) -> None:
    """
    __repr__ should contain class name.
    """

    representation = repr(
        predictor,
    )

    assert "ImagePredictor" in representation

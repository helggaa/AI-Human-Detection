"""
Shared pytest fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.predictor import ImagePredictor

# =============================================================================
# IMAGE FIXTURES
# =============================================================================


@pytest.fixture
def rgb_image() -> Image.Image:
    """
    Create a sample RGB image.
    """

    return Image.new(
        mode="RGB",
        size=(224, 224),
        color=(255, 255, 255),
    )


@pytest.fixture
def grayscale_image() -> Image.Image:
    """
    Create a sample grayscale image.
    """

    return Image.new(
        mode="L",
        size=(224, 224),
        color=128,
    )


@pytest.fixture
def rgba_image() -> Image.Image:
    """
    Create a sample RGBA image.
    """

    return Image.new(
        mode="RGBA",
        size=(224, 224),
        color=(255, 255, 255, 255),
    )


@pytest.fixture
def numpy_image() -> np.ndarray:
    """
    Create a sample NumPy RGB image.
    """

    return np.zeros(
        (
            224,
            224,
            3,
        ),
        dtype=np.uint8,
    )


# =============================================================================
# FILE FIXTURES
# =============================================================================


@pytest.fixture
def image_file(
    tmp_path: Path,
    rgb_image: Image.Image,
) -> Path:
    """
    Save a temporary RGB image.
    """

    image_path = tmp_path / "image.jpg"

    rgb_image.save(
        image_path,
        format="JPEG",
    )

    return image_path


@pytest.fixture
def invalid_image_file(
    tmp_path: Path,
) -> Path:
    """
    Create an invalid image file.
    """

    path = tmp_path / "invalid.jpg"

    path.write_text(
        "not an image",
        encoding="utf-8",
    )

    return path


# =============================================================================
# PREDICTOR
# =============================================================================


@pytest.fixture(scope="session")
def predictor() -> ImagePredictor:
    """
    Load the predictor once for the entire test session.
    """

    return ImagePredictor()

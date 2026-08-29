"""
Tests for gradcam.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.gradcam import (
    GradCAMResult,
    compute_gradcam_heatmap,
    generate_gradcam,
    overlay_heatmap,
)
from src.predictor import ImagePredictor


def test_compute_gradcam_heatmap(
    predictor: ImagePredictor,
    rgb_image: Image.Image,
):
    """compute_gradcam_heatmap should return a normalized 2D array."""
    preprocessed = predictor._preprocess(rgb_image)
    heatmap = compute_gradcam_heatmap(
        model=predictor.model,
        preprocessed_image=preprocessed,
        class_index=0,
    )

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.shape == (7, 7)
    assert np.all(heatmap >= 0.0)
    assert np.all(heatmap <= 1.0 + 1e-6)


def test_overlay_heatmap(rgb_image: Image.Image):
    """overlay_heatmap should produce a PIL Image of matching dimensions."""
    mock_heatmap = np.random.uniform(0, 1, size=(7, 7)).astype(np.float32)
    overlay = overlay_heatmap(rgb_image, mock_heatmap, alpha=0.5)

    assert isinstance(overlay, Image.Image)
    assert overlay.size == rgb_image.size
    assert overlay.mode == "RGB"


def test_generate_gradcam_complete(
    predictor: ImagePredictor,
    rgb_image: Image.Image,
):
    """generate_gradcam should produce a complete structured result."""
    preprocessed = predictor._preprocess(rgb_image)
    result = generate_gradcam(
        model=predictor.model,
        original_image=rgb_image,
        preprocessed_image=preprocessed,
        class_labels=predictor.class_labels,
    )

    assert isinstance(result, GradCAMResult)
    assert result.heatmap.shape == (7, 7)
    assert result.target_class_label in predictor.class_labels.values()

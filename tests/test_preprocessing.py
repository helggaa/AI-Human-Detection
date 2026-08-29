"""
Tests for preprocessing.py
"""

from __future__ import annotations

import pandas as pd
import tensorflow as tf

from src.preprocessing import DatasetPreprocessor, DatasetStatistics


def test_get_class_names():
    """DatasetPreprocessor should return configured class names."""
    classes = DatasetPreprocessor.get_class_names()
    assert len(classes) == 2
    assert "AI" in classes
    assert "Authentic" in classes


def test_dataset_statistics_defaults():
    """Default dataset statistics should start at zero."""
    stats = DatasetStatistics()
    assert stats.total_images == 0
    assert stats.train_images == 0
    assert stats.validation_images == 0
    assert stats.test_images == 0


def test_build_augmentation_layer():
    """
    Augmentation layer should construct a valid Sequential model.
    """
    augmentation = DatasetPreprocessor.build_augmentation_layer()
    assert isinstance(augmentation, tf.keras.Sequential)
    assert len(augmentation.layers) == 4

    # Test passing a mock image tensor through augmentation
    mock_input = tf.random.uniform((2, 224, 224, 3), dtype=tf.float32)
    output = augmentation(mock_input, training=True)
    assert output.shape == (2, 224, 224, 3)


def test_dataset_summary():
    """Dataset summary should compute per-class image counts."""
    preprocessor = DatasetPreprocessor()
    df = pd.DataFrame(
        {
            "filepath": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "label": [0, 1, 1],
            "class_name": ["Authentic", "AI", "AI"],
        }
    )
    summary = preprocessor.dataset_summary(df)
    assert len(summary) == 2
    ai_count = summary.loc[summary["class_name"] == "AI", "images"].values[0]
    auth_count = summary.loc[
        summary["class_name"] == "Authentic", "images"
    ].values[0]
    assert ai_count == 2
    assert auth_count == 1

"""
Tests for exporter.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import MODEL_DIR
from src.exporter import (
    TFLITE_FILENAME,
    export_to_tflite,
    verify_tflite_model,
)


def test_export_to_tflite_invalid_path(tmp_path: Path):
    """Missing source model should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        export_to_tflite(
            model_path=tmp_path / "non_existent.keras",
            output_path=tmp_path / "model.tflite",
        )


def test_verify_tflite_invalid_path(tmp_path: Path):
    """Missing tflite file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        verify_tflite_model(tmp_path / "missing.tflite")


def test_verify_tflite_model_inference():
    """
    Exported model should run inference and return predictions.
    """
    tflite_path = MODEL_DIR / TFLITE_FILENAME
    if not tflite_path.exists():
        pytest.skip("model.tflite not generated yet")

    test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    output = verify_tflite_model(tflite_path, test_input)

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2)
    assert np.isclose(np.sum(output), 1.0, atol=1e-2)

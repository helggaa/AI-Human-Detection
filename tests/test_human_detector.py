"""
Tests for human_detector.py
"""

from __future__ import annotations

import numpy as np
import pytest

from src.human_detector import (
    contains_human,
    load_cascade,
    load_detector,
)


def test_load_detector():
    """Primary detector should load successfully."""
    detector = load_detector()
    assert detector is not None
    assert not detector.empty()


def test_load_cascade_valid():
    """Valid cascade should load properly."""
    cascade = load_cascade("haarcascade_frontalface_default.xml")
    assert cascade is not None
    assert not cascade.empty()


def test_load_cascade_missing():
    """Missing cascade file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_cascade("non_existent_cascade.xml")


def test_contains_human_blank_image():
    """A plain blank image should not detect any humans."""
    blank = np.zeros((224, 224, 3), dtype=np.uint8)
    assert contains_human(blank) is False


def test_contains_human_grayscale_input():
    """Grayscale 2D array should be supported without errors."""
    blank_gray = np.zeros((224, 224), dtype=np.uint8)
    assert contains_human(blank_gray) is False


def test_contains_human_invalid_dimensions():
    """1D or 4D array should raise ValueError."""
    invalid_1d = np.zeros((224,), dtype=np.uint8)
    with pytest.raises(ValueError):
        contains_human(invalid_1d)

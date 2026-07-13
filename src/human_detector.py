from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=1)
def load_detector() -> cv2.CascadeClassifier:

    detector_path = (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )

    if not detector_path.exists():
        raise FileNotFoundError(f"Detector file not found: {detector_path}")

    detector = cv2.CascadeClassifier(str(detector_path))

    if detector.empty():
        raise RuntimeError("Failed to load Haar cascade detector.")

    return detector


def contains_human(
    image: np.ndarray,
) -> bool:

    detector = load_detector()

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    faces = detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    return len(faces) > 0

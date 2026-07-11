from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=1)
def load_detector() -> cv2.CascadeClassifier:

    detector_path = (
        Path(cv2.__file__).parent
        / "data"
        / "haarcascade_frontalface_default.xml"
    )

    detector = cv2.CascadeClassifier(
        str(detector_path)
    )

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
"""
Human Detector
==============

Detect whether an input image contains human subjects (faces or bodies)
using OpenCV Haar Cascade classifiers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=1)
def load_cascade(filename: str) -> cv2.CascadeClassifier:
    """
    Load a Haar cascade classifier from OpenCV data directory.

    Parameters
    ----------
    filename : str
        Cascade XML filename.

    Returns
    -------
    cv2.CascadeClassifier
    """
    cascade_path = Path(cv2.data.haarcascades) / filename

    if not cascade_path.exists():
        raise FileNotFoundError(f"Cascade file not found: {cascade_path}")

    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {filename}")

    return detector


@lru_cache(maxsize=1)
def load_detector() -> cv2.CascadeClassifier:
    """
    Load the primary frontal face cascade detector.

    Returns
    -------
    cv2.CascadeClassifier
    """
    return load_cascade("haarcascade_frontalface_default.xml")


def contains_human(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (40, 40),
) -> bool:
    """
    Check if an image contains human features
    (frontal face, profile face, or upper body).

    Parameters
    ----------
    image : np.ndarray
        Input image as RGB or grayscale NumPy array.
    scale_factor : float, default=1.1
        Scale factor for multiscale detection.
    min_neighbors : int, default=5
        Minimum neighbors for candidate bounding boxes.
    min_size : tuple[int, int], default=(40, 40)
        Minimum detection window size.

    Returns
    -------
    bool
        True if any human subject is detected, False otherwise.
    """
    if image.ndim == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        grayscale = image
    else:
        raise ValueError(
            f"Expected image with 2 or 3 dimensions, got {image.ndim}."
        )

    # 1. Check frontal face
    frontal_detector = load_detector()
    faces = frontal_detector.detectMultiScale(
        grayscale,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    if len(faces) > 0:
        return True

    # 2. Check profile face (side portraits)
    try:
        profile_detector = load_cascade("haarcascade_profileface.xml")
        profile_faces = profile_detector.detectMultiScale(
            grayscale,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        if len(profile_faces) > 0:
            return True
    except Exception:
        pass

    # 3. Check upper body fallback
    try:
        upperbody_detector = load_cascade("haarcascade_upperbody.xml")
        upper_bodies = upperbody_detector.detectMultiScale(
            grayscale,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(60, 60),
        )
        if len(upper_bodies) > 0:
            return True
    except Exception:
        pass

    return False


__all__ = [
    "load_cascade",
    "load_detector",
    "contains_human",
]

from __future__ import annotations

from functools import lru_cache

import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ultralytics import YOLO


@lru_cache(maxsize=1)
def load_detector() -> YOLO:
    """
    Load the YOLO model once.
    """

    from ultralytics import YOLO

    return YOLO("yolov8n.pt")


def contains_human(
    image: np.ndarray,
    confidence_threshold: float = 0.35,
) -> bool:
    """
    Detect whether the image contains at least one person.
    """

    model = load_detector()

    results = model.predict(
        source=image,
        verbose=False,
    )

    boxes = results[0].boxes

    if boxes is None:

        return False

    for box in boxes:

        class_id = int(box.cls.item())
        confidence = float(box.conf.item())

        # COCO class 0 = person

        if class_id == 0 and confidence >= confidence_threshold:

            return True

    return False

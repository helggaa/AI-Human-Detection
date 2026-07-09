"""
Command-Line Image Prediction
=============================

Command-line interface for the AI Human Detection project.

Example
-------
python predict.py image.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import SUPPORTED_IMAGE_EXTENSIONS
from src.logger import logger
from src.predictor import (
    ImagePredictor,
    PredictionResult,
)

APP_NAME = "AI Human Detection"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Predict whether an image is an authentic "
            "human or an AI-generated human."
        ),
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image.",
    )

    return parser.parse_args()


# =============================================================================
# VALIDATION
# =============================================================================


def validate_image_path(
    image_path: Path,
) -> None:
    """
    Validate the input image path.

    Parameters
    ----------
    image_path : Path
        Image path.

    Raises
    ------
    FileNotFoundError
        If the image does not exist.

    ValueError
        If the file extension is unsupported.
    """

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"{image_path} is not a file.")

    extension = image_path.suffix.lower()

    if extension not in SUPPORTED_IMAGE_EXTENSIONS:

        supported = ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))

        raise ValueError(
            "Unsupported image format.\n" f"Supported formats: {supported}"
        )


# =============================================================================
# DISPLAY
# =============================================================================


def display_prediction(
    image_path: Path,
    result: PredictionResult,
) -> None:
    """
    Display prediction results.

    Parameters
    ----------
    image_path : Path
        Input image path.

    result : PredictionResult
        Prediction result.
    """

    lines = [
        "=" * 60,
        APP_NAME,
        "=" * 60,
        "",
        "Image",
        "-----",
        image_path.name,
        "",
        "Prediction",
        "----------",
        result.predicted_label,
        "",
        "Confidence",
        "----------",
        f"{result.confidence:.2f}%",
        "",
        "Probabilities",
        "-------------",
    ]

    for label, probability in result.probabilities.items():
        lines.append(f"{label:<12}: {probability:.2f}%")

    print("\n".join(lines))


def load_predictor() -> ImagePredictor:
    """
    Create an image predictor.

    Returns
    -------
    ImagePredictor
    """
    return ImagePredictor()


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    """
    Execute CLI prediction.

    Returns
    -------
    int
        Process exit code.
    """

    arguments = parse_arguments()

    try:

        validate_image_path(
            arguments.image,
        )
        predictor = load_predictor()

        result = predictor.predict(
            arguments.image,
        )

        display_prediction(
            arguments.image,
            result,
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as error:

        logger.error(
            "Prediction failed: %s",
            error,
        )

        return 1

    except Exception:

        logger.exception("Unexpected prediction error.")

        return 1


if __name__ == "__main__":

    sys.exit(
        main(),
    )

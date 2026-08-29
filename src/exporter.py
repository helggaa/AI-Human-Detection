"""
Model Exporter
==============

Export trained TensorFlow models to optimized formats (such as TFLite)
for low-latency CPU and edge device deployment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import load_model

from src.config import (
    IMAGE_SIZE,
    MODEL_DIR,
    MODEL_FILENAME,
)
from src.logger import logger

TFLITE_FILENAME = "model.tflite"


def export_to_tflite(
    model_path: Path | str | None = None,
    output_path: Path | str | None = None,
    quantize: bool = False,
) -> Path:
    """
    Convert a saved Keras model to TensorFlow Lite (.tflite) format.

    Parameters
    ----------
    model_path : Path | str | None, default=None
        Path to the saved .keras model.
        If None, default best_model.keras is used.
    output_path : Path | str | None, default=None
        Destination file path for the .tflite model.
    quantize : bool, default=False
        If True, apply dynamic range quantization for reduced file size.

    Returns
    -------
    Path
        Path to the exported .tflite model.
    """
    if model_path is None:
        model_path = MODEL_DIR / MODEL_FILENAME
    else:
        model_path = Path(model_path)

    if output_path is None:
        output_path = MODEL_DIR / TFLITE_FILENAME
    else:
        output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("Loading model from %s for TFLite export.", model_path)
    model = load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info("Dynamic range quantization enabled.")

    logger.info("Converting model to TFLite format...")
    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(
        "TFLite model exported successfully: %s (%.2f MB)",
        output_path,
        size_mb,
    )

    return output_path


def verify_tflite_model(
    tflite_path: Path | str,
    test_input: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run test inference on an exported TFLite model to verify correctness.

    Parameters
    ----------
    tflite_path : Path | str
        Path to the .tflite file.
    test_input : np.ndarray | None, default=None
        Input array with shape (1, height, width, 3). If None, zeros are used.

    Returns
    -------
    np.ndarray
        Prediction output probabilities.
    """
    tflite_path = Path(tflite_path)
    if not tflite_path.exists():
        raise FileNotFoundError(f"TFLite file not found: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if test_input is None:
        test_input = np.zeros(
            (1, *IMAGE_SIZE, 3),
            dtype=np.float32,
        )

    interpreter.set_tensor(input_details[0]["index"], test_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    logger.info("TFLite verification inference successful: %s", output_data)

    return output_data


def main() -> None:
    """
    CLI entry point for model export.
    """
    exported_path = export_to_tflite()
    verify_tflite_model(exported_path)
    logger.info("Model export and verification completed.")


if __name__ == "__main__":
    main()

__all__ = [
    "TFLITE_FILENAME",
    "export_to_tflite",
    "verify_tflite_model",
]

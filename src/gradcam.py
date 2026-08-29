"""
Explainable AI - Grad-CAM
=========================

Gradient-weighted Class Activation Mapping (Grad-CAM) visualization
for EfficientNetV2 image classification models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib as mpl
import numpy as np
import tensorflow as tf
from PIL import Image

from src.logger import logger


@dataclass(slots=True)
class GradCAMResult:
    """
    Structured Grad-CAM explanation result.

    Parameters
    ----------
    heatmap : np.ndarray
        2D normalized heatmap values (0.0 to 1.0).
    overlay_image : Image.Image
        PIL Image with superimposed heatmap.
    target_class_index : int
        Predicted or explained class index.
    target_class_label : str
        Predicted or explained class name.
    """

    heatmap: np.ndarray
    overlay_image: Image.Image
    target_class_index: int
    target_class_label: str


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    preprocessed_image: np.ndarray,
    class_index: int | None = None,
    backbone_layer_name: str = "efficientnetv2-b0",
    conv_layer_name: str = "top_conv",
) -> np.ndarray:
    """
    Compute 2D Grad-CAM activation heatmap for a target class.

    Parameters
    ----------
    model : tf.keras.Model
        Loaded classification model.
    preprocessed_image : np.ndarray
        Preprocessed image tensor with shape (1, height, width, 3).
    class_index : int | None, default=None
        Target class index. If None, uses the model's top predicted class.
    backbone_layer_name : str, default="efficientnetv2-b0"
        Name of the backbone layer inside the model.
    conv_layer_name : str, default="top_conv"
        Name of the target convolutional layer in the backbone.

    Returns
    -------
    np.ndarray
        2D float32 heatmap normalized between 0.0 and 1.0.
    """
    backbone = model.get_layer(backbone_layer_name)
    conv_layer = backbone.get_layer(conv_layer_name)
    backbone_submodel = tf.keras.Model(
        inputs=backbone.input,
        outputs=conv_layer.output,
    )

    with tf.GradientTape() as tape:
        conv_outputs = backbone_submodel(preprocessed_image)
        tape.watch(conv_outputs)

        # Forward through the classification head layers
        x = conv_outputs
        for layer in model.layers[2:]:
            x = layer(x)
        predictions = x

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))

        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs_val = conv_outputs[0]
    heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only positive contributions
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> Image.Image:
    """
    Overlay a 2D heatmap on a PIL Image.

    Parameters
    ----------
    original_image : Image.Image
        Original RGB PIL Image.
    heatmap : np.ndarray
        2D normalized heatmap array.
    alpha : float, default=0.4
        Blending opacity for the heatmap overlay.
    colormap_name : str, default="jet"
        Matplotlib colormap name.

    Returns
    -------
    Image.Image
        PIL Image with superimposed colormap heatmap.
    """
    width, height = original_image.size

    # Resize heatmap to match image dimensions
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )
    resized_heatmap = np.array(heatmap_pil) / 255.0

    colormap = mpl.colormaps[colormap_name]
    colored_heatmap = colormap(resized_heatmap)[:, :, :3]
    colored_heatmap_uint8 = np.uint8(255 * colored_heatmap)

    original_array = np.array(original_image.convert("RGB"))
    blended = np.uint8(
        (1.0 - alpha) * original_array + alpha * colored_heatmap_uint8
    )

    return Image.fromarray(blended)


def generate_gradcam(
    model: tf.keras.Model,
    original_image: Image.Image,
    preprocessed_image: np.ndarray,
    class_labels: Mapping[int, str],
    class_index: int | None = None,
    alpha: float = 0.4,
    colormap_name: str = "jet",
) -> GradCAMResult:
    """
    Generate complete Grad-CAM explanation result.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    original_image : Image.Image
        Original PIL image.
    preprocessed_image : np.ndarray
        Preprocessed image tensor (1, 224, 224, 3).
    class_labels : Mapping[int, str]
        Class index to label mapping.
    class_index : int | None, default=None
        Target class index.
    alpha : float, default=0.4
        Heatmap overlay opacity.
    colormap_name : str, default="jet"
        Colormap name.

    Returns
    -------
    GradCAMResult
        Structured explanation output.
    """
    logger.info("Generating Grad-CAM visualization.")

    if class_index is None:
        predictions = model.predict(preprocessed_image, verbose=0)[0]
        class_index = int(np.argmax(predictions))

    class_label = class_labels.get(class_index, f"Class {class_index}")

    heatmap = compute_gradcam_heatmap(
        model=model,
        preprocessed_image=preprocessed_image,
        class_index=class_index,
    )

    overlay = overlay_heatmap(
        original_image=original_image,
        heatmap=heatmap,
        alpha=alpha,
        colormap_name=colormap_name,
    )

    logger.info("Grad-CAM visualization generated for %s.", class_label)

    return GradCAMResult(
        heatmap=heatmap,
        overlay_image=overlay,
        target_class_index=class_index,
        target_class_label=class_label,
    )


__all__ = [
    "GradCAMResult",
    "compute_gradcam_heatmap",
    "overlay_heatmap",
    "generate_gradcam",
]

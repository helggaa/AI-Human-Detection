"""
Streamlit Application
=====================

Interactive web application for the AI Human Detection project.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image

from src.config import SUPPORTED_IMAGE_EXTENSIONS
from src.human_detector import contains_human
from src.logger import logger
from src.predictor import (
    ImagePredictor,
    PredictionResult,
)

APP_TITLE = "AI Human Detection"

APP_DESCRIPTION = (
    "Detect whether a human image is authentic "
    "or AI-generated using a trained "
    "EfficientNetV2 model."
)


def configure_page() -> None:
    """
    Configure the Streamlit page.
    """

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_resource
def load_predictor() -> ImagePredictor:
    """
    Load the prediction model.

    Returns
    -------
    ImagePredictor
    """

    logger.info("Loading predictor for Streamlit.")

    return ImagePredictor()


def render_sidebar(
    predictor: ImagePredictor,
) -> None:
    """
    Render sidebar information.

    Parameters
    ----------
    predictor : ImagePredictor
    """

    info = predictor.information

    st.sidebar.title(APP_TITLE)

    st.sidebar.markdown("---")

    st.sidebar.subheader("Model")

    st.sidebar.write(info.model_name)

    st.sidebar.subheader("Version")

    st.sidebar.write(info.model_version)

    st.sidebar.subheader("Input Size")

    st.sidebar.write(f"{info.image_size[0]} × {info.image_size[1]}")

    st.sidebar.subheader("Classes")

    for label in info.class_labels:
        st.sidebar.write(f"• {label}")


def render_header() -> None:
    """
    Render application header.
    """

    st.title(APP_TITLE)

    st.caption(APP_DESCRIPTION)

    st.markdown("---")


def render_uploaded_image() -> Image.Image | None:
    """
    Upload an image.

    Returns
    -------
    Image.Image | None
    """

    uploaded = st.file_uploader(
        "Upload an image",
        type=[
            extension.removeprefix(".")
            for extension in SUPPORTED_IMAGE_EXTENSIONS
        ],
    )

    if uploaded is None:
        return None

    image = Image.open(uploaded).convert("RGB")

    image_array = np.array(image)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch",
    )

    if not contains_human(image_array):
        st.warning("⚠️ There's no human in this picture.")

        st.stop()

    return image


def render_prediction(
    result: PredictionResult,
) -> None:
    """
    Render prediction summary.

    Parameters
    ----------
    result : PredictionResult
        Prediction result.
    """

    st.subheader("Prediction")

    left_column, right_column = st.columns(
        2,
    )

    with left_column:

        if result.predicted_label == "Authentic":

            st.success("✅ Authentic Human")

        else:

            st.error("🤖 AI Generated Human")

    with right_column:

        st.metric(
            label="Confidence",
            value=f"{result.confidence:.2f}%",
        )

        st.metric(
            label="Inference Time",
            value=f"{result.inference_time_ms:.2f} ms",
        )


def render_probability_section(
    result: PredictionResult,
) -> None:
    """
    Render class probabilities.

    Parameters
    ----------
    result : PredictionResult
        Prediction result.
    """

    st.subheader("Class Probabilities")

    for label, probability in result.probabilities.items():

        st.write(f"**{label}** — {probability:.2f}%")

        st.progress(
            probability / 100.0,
        )


def render_footer() -> None:
    """
    Render page footer.
    """

    st.markdown("---")

    st.caption(
        "Built with Streamlit • " "Powered by TensorFlow and EfficientNetV2"
    )


def main() -> None:
    """
    Execute the Streamlit application.
    """

    configure_page()

    predictor = load_predictor()

    render_sidebar(
        predictor,
    )

    render_header()

    image = render_uploaded_image()

    if image is None:
        return

    try:

        with st.spinner("Running inference..."):

            result = predictor.predict_from_pil(
                image,
            )

        render_prediction(
            result,
        )

        render_probability_section(
            result,
        )

    except Exception:

        logger.exception("Prediction failed.")

        st.error("Unable to process the uploaded image.")

    render_footer()


if __name__ == "__main__":
    main()

"""
Streamlit Application
=====================

Interactive web application for the AI Human Detection project.
"""

from __future__ import annotations

import io

import numpy as np
import streamlit as st
from PIL import Image

from src.config import FEEDBACK_PATH, SUPPORTED_IMAGE_EXTENSIONS
from src.feedback import (
    FeedbackConfiguration,
    FeedbackError,
    FeedbackManager,
)
from src.human_detector import contains_human
from src.logger import logger
from src.predictor import (
    ImagePredictor,
    PredictionResult,
)

APP_TITLE = "AI Human Detection"

MAX_FILE_SIZE_MB = 20
MINIMUM_CONFIDENCE = 60.0

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


@st.cache_resource
def load_feedback_manager() -> FeedbackManager:
    """
    Load the feedback manager.

    Returns
    -------
    FeedbackManager
    """

    configuration = FeedbackConfiguration(
        bucket_name=st.secrets["supabase"]["bucket_name"],
        supabase_url=st.secrets["supabase"]["url"],
        supabase_key=st.secrets["supabase"]["key"],
        metadata_path=FEEDBACK_PATH,
    )

    return FeedbackManager(
        configuration=configuration,
    )


def image_to_jpeg_bytes(
    image: Image.Image,
) -> bytes:
    """
    Convert an image to JPEG bytes.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=95,
    )

    return buffer.getvalue()


def render_sidebar(
    predictor: ImagePredictor,
    feedback_manager: FeedbackManager,
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

    st.sidebar.subheader("Confidence Threshold")

    st.sidebar.write(f"{MINIMUM_CONFIDENCE:.0f}%")

    st.sidebar.subheader("Version")

    st.sidebar.write(info.model_version)

    st.sidebar.subheader("Input Size")

    st.sidebar.write(f"{info.image_size[0]} × {info.image_size[1]}")

    st.sidebar.subheader("Classes")

    for label in info.class_labels:
        st.sidebar.write(f"• {label}")

    st.sidebar.markdown("---")

    statistics = feedback_manager.feedback_statistics()

    st.sidebar.subheader("Feedback Statistics")

    st.sidebar.metric(
        "Total Feedback",
        statistics["total_feedback"],
    )

    st.sidebar.metric(
        "👍 Positive",
        statistics["positive_feedback"],
    )

    st.sidebar.metric(
        "👎 Negative",
        statistics["negative_feedback"],
    )


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

    if uploaded is not None:
        file_size_mb = uploaded.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:

            st.error("Maximum file size is 20 MB.")

            st.stop()

    if uploaded is None:
        return None

    current_file = uploaded.name

    if "last_uploaded_file" not in st.session_state:
        st.session_state["last_uploaded_file"] = current_file

    elif st.session_state["last_uploaded_file"] != current_file:
        st.session_state["feedback_submitted"] = False

        st.session_state["last_uploaded_file"] = current_file

    image = Image.open(uploaded).convert("RGB")
    image.thumbnail((1024, 1024))

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


def render_feedback_section(
    image: Image.Image,
    result: PredictionResult,
    feedback_manager: FeedbackManager,
) -> None:
    """
    Render feedback section.

    Parameters
    ----------
    image : Image.Image
        Uploaded image.

    result : PredictionResult
        Prediction result.

    feedback_manager : FeedbackManager
        Feedback manager instance.
    """

    st.markdown("---")

    st.subheader("Feedback")

    if st.session_state["feedback_submitted"]:

        st.success("✅ Thank you for your feedback.")

        return

    st.write("Was this prediction correct?")

    left_column, right_column = st.columns(2)

    with left_column:

        positive_clicked = st.button(
            "👍 Correct",
            use_container_width=True,
        )

    with right_column:

        negative_clicked = st.button(
            "👎 Wrong",
            use_container_width=True,
        )

    if not positive_clicked and not negative_clicked:
        return

    feedback = "positive" if positive_clicked else "negative"

    try:

        image_bytes = image_to_jpeg_bytes(
            image,
        )

        feedback_manager.submit_feedback(
            image_bytes=image_bytes,
            feedback=feedback,
            predicted_label=result.predicted_label,
            confidence=result.confidence,
        )

        st.session_state["feedback_submitted"] = True

        st.success("✅ Thank you for your feedback.")

    except FeedbackError:

        logger.exception("Failed to save feedback.")

        st.error("Unable to save feedback.")


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

    feedback_manager = load_feedback_manager()

    if "feedback_submitted" not in st.session_state:

        st.session_state["feedback_submitted"] = False

    if "current_image_id" not in st.session_state:

        st.session_state["current_image_id"] = None

    render_sidebar(
        predictor,
        feedback_manager,
    )

    render_header()

    image = render_uploaded_image()

    if image is None:
        return

    current_image_id = str(hash(image.tobytes()))

    previous_image_id = st.session_state.get(
        "current_image_id",
    )

    if previous_image_id != current_image_id:

        st.session_state["feedback_submitted"] = False

        st.session_state["current_image_id"] = current_image_id

    try:

        with st.spinner("Running inference..."):

            result = predictor.predict_from_pil(
                image,
            )
            if result.confidence < MINIMUM_CONFIDENCE:

                st.warning(
                    f"⚠️ The confidence is below "
                    f"{MINIMUM_CONFIDENCE:.0f}%. "
                    "The prediction may be inaccurate."
                )

        render_prediction(
            result,
        )

        render_probability_section(
            result,
        )

        render_feedback_section(
            image=image,
            result=result,
            feedback_manager=feedback_manager,
        )

    except Exception:

        logger.exception("Prediction failed.")

        st.error("Unable to process the uploaded image.")

    render_footer()


if __name__ == "__main__":
    main()

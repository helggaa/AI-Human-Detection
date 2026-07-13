from __future__ import annotations

import csv
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from supabase import Client, create_client

from src.config import (
    CLASS_LABELS,
    FEEDBACK_COLUMNS,
    FEEDBACK_IMAGE_EXTENSION,
)
from src.logger import logger
from src.utils import ensure_directory


class FeedbackError(Exception):
    """Raised when a feedback operation fails."""


@dataclass(slots=True)
class FeedbackConfiguration:
    """
    Configuration for the feedback system.

    Parameters
    ----------
    positive_folder_id: str
    negative_folder_id: str
    metadata_path: Path
    service_account_info: Mapping[str, Any]
    """

    bucket_name: str
    supabase_url: str
    supabase_key: str
    metadata_path: Path

    def __post_init__(self) -> None:

        if not self.bucket_name:
            raise ValueError("Bucket name cannot be empty.")

        if not self.supabase_url:
            raise ValueError("Supabase URL cannot be empty.")

        if not self.supabase_key:
            raise ValueError("Supabase key cannot be empty.")

        if self.metadata_path.suffix.lower() != ".csv":
            raise ValueError("Metadata path must point to a CSV file.")


@dataclass(slots=True)
class FeedbackRecord:
    """
    Represents a stored feedback record.

    Parameters
    ----------
    timestamp : str
        Submission timestamp.
    feedback : Literal["positive", "negative"]
        User feedback value.
    predicted_label : str
        Model prediction label.
    confidence : float
        Model confidence score.
    filename : str
        Generated image filename.
    drive_file_id : str
        Uploaded Google Drive file ID.
    drive_url : str
        Public or private Google Drive URL.
    """

    timestamp: str
    feedback: Literal["positive", "negative"]
    predicted_label: str
    confidence: float
    filename: str
    drive_file_id: str
    drive_url: str


@dataclass(slots=True)
class UploadResult:
    """
    Represents an uploaded image.

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    file_url : str
        Google Drive URL.
    filename : str
        Generated filename.
    """

    file_id: str
    file_url: str
    filename: str


class FeedbackManager:
    """
    Handles feedback submission and storage.
    """

    def __init__(
        self,
        configuration: FeedbackConfiguration,
    ) -> None:
        self.configuration = configuration

        self._storage_client: Client | None = None

        try:
            self._storage_client = self._build_storage_client()

        except Exception as error:
            logger.exception("Failed to initialize Google Drive client.")

            raise FeedbackError(
                "Failed to initialize feedback system."
            ) from error

    def _generate_filename(self) -> str:
        """
        Generate a unique feedback image filename.

        Returns
        -------
        str
            Unique filename.
        """

        timestamp = datetime.now(
            UTC,
        ).strftime("%Y%m%d_%H%M%S")

        identifier = uuid.uuid4().hex[:8]

        return f"{timestamp}_{identifier}" f"{FEEDBACK_IMAGE_EXTENSION}"

    def _append_metadata(
        self,
        record: FeedbackRecord,
    ) -> None:
        """
        Append feedback metadata to the CSV file.

        Parameters
        ----------
        record : FeedbackRecord
            Feedback record to persist.
        """

        metadata_path = self.configuration.metadata_path

        ensure_directory(metadata_path.parent)

        file_exists = metadata_path.exists()

        with metadata_path.open(
            mode="a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=FEEDBACK_COLUMNS,
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "timestamp": record.timestamp,
                    "feedback": record.feedback,
                    "predicted_label": record.predicted_label,
                    "confidence": record.confidence,
                    "filename": record.filename,
                    "drive_file_id": record.drive_file_id,
                    "drive_url": record.drive_url,
                }
            )

        logger.info(
            "Feedback metadata saved: %s",
            record.filename,
        )

    def load_feedback(self) -> pd.DataFrame:
        """
        Load feedback metadata.

        Returns
        -------
        pandas.DataFrame
            Feedback records.
        """

        metadata_path = self.configuration.metadata_path

        if not metadata_path.exists():
            return pd.DataFrame(
                columns=FEEDBACK_COLUMNS,
            )

        return pd.read_csv(
            metadata_path,
            dtype={
                "feedback": "string",
                "predicted_label": "string",
                "filename": "string",
                "drive_file_id": "string",
                "drive_url": "string",
                "confidence": "float64",
            },
            parse_dates=["timestamp"],
        )

    def feedback_statistics(
        self,
    ) -> dict[str, int | float]:
        """
        Compute feedback statistics.

        Returns
        -------
        dict[str, float]
            Feedback statistics.
        """

        dataframe = self.load_feedback()

        total_feedback = len(dataframe)

        if total_feedback == 0:
            return {
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
            }

        positive_feedback = int((dataframe["feedback"] == "positive").sum())

        negative_feedback = int((dataframe["feedback"] == "negative").sum())

        positive_ratio = (positive_feedback / total_feedback) * 100.0

        negative_ratio = (negative_feedback / total_feedback) * 100.0

        return {
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
        }

    def _build_storage_client(
        self,
    ) -> Client:
        """
        Build a Supabase client.

        Returns
        -------
        Client
        """

        return create_client(
            self.configuration.supabase_url,
            self.configuration.supabase_key,
        )

    def _upload_image(
        self,
        image_bytes: bytes,
        feedback: Literal["positive", "negative"],
    ) -> UploadResult:
        """
        Upload an image to Google Drive.

        Parameters
        ----------
        image_bytes : bytes
            JPEG-encoded image content.
        feedback : Literal["positive", "negative"]
            Feedback category.

        Returns
        -------
        UploadResult
            Upload metadata.

        Raises
        ------
        FeedbackError
            If the upload fails.
        """

        if self._storage_client is None:
            raise FeedbackError("Feedback system is not configured.")

        filename = self._generate_filename()

        storage_path = f"{feedback}/{filename}"

        try:

            self._storage_client.storage.from_(
                self.configuration.bucket_name
            ).upload(
                path=storage_path,
                file=image_bytes,
                file_options={
                    "content-type": "image/jpeg",
                },
            )

            file_url = storage_path

            logger.info(
                "Feedback image uploaded: %s",
                storage_path,
            )

            return UploadResult(
                file_id=storage_path,
                file_url=file_url,
                filename=filename,
            )

        except Exception as error:

            logger.exception("Failed to upload feedback image.")

            raise FeedbackError("Failed to upload feedback image.") from error

    def submit_feedback(
        self,
        image_bytes: bytes,
        feedback: Literal["positive", "negative"],
        predicted_label: str,
        confidence: float,
    ) -> FeedbackRecord:
        """
        Submit user feedback.

        Parameters
        ----------
        image_bytes : bytes
            JPEG-encoded image content.
        feedback : Literal["positive", "negative"]
            User feedback.
        predicted_label : str
            Predicted label.
        confidence : float
            Prediction confidence.

        Returns
        -------
        FeedbackRecord
            Persisted feedback record.

        Raises
        ------
        FeedbackError
            If the submission fails.
        """

        if predicted_label not in CLASS_LABELS.values():
            raise FeedbackError(f"Invalid predicted label: {predicted_label}")

        if not image_bytes:
            raise FeedbackError("Image content cannot be empty.")
        if not 0.0 <= confidence <= 100.0:
            raise FeedbackError("Confidence must be between 0 and 100.")

        if feedback not in {"positive", "negative"}:
            raise FeedbackError(f"Invalid feedback value: {feedback}")

        upload_result = self._upload_image(
            image_bytes=image_bytes,
            feedback=feedback,
        )

        record = FeedbackRecord(
            timestamp=datetime.now(UTC).isoformat(),
            feedback=feedback,
            predicted_label=predicted_label,
            confidence=confidence,
            filename=upload_result.filename,
            drive_file_id=upload_result.file_id,
            drive_url=upload_result.file_url,
        )

        self._append_metadata(
            record=record,
        )
        # TODO:
        # Remove uploaded file from Google Drive
        # if metadata persistence fails.

        logger.info(
            "Feedback submitted successfully: %s",
            record.filename,
        )

        return record


def _build_storage_client(
    self,
) -> Client:
    """
    Build a Supabase client.

    Returns
    -------
    Client
    """

    return create_client(
        self.configuration.supabase_url,
        self.configuration.supabase_key,
    )


__all__ = [
    "FeedbackError",
    "FeedbackConfiguration",
    "FeedbackRecord",
    "UploadResult",
    "FeedbackManager",
]

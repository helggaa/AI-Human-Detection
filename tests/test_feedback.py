from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.feedback import (
    FeedbackConfiguration,
    FeedbackError,
    FeedbackManager,
    FeedbackRecord,
    UploadResult,
)


@pytest.fixture
def feedback_configuration(
    tmp_path: Path,
) -> FeedbackConfiguration:

    return FeedbackConfiguration(
        bucket_name="feedback-images",
        supabase_url="https://example.supabase.co",
        supabase_key="fake-key",
        metadata_path=tmp_path / "feedback.csv",
    )


@pytest.fixture
def feedback_manager(
    feedback_configuration: FeedbackConfiguration,
) -> FeedbackManager:

    with patch.object(
        FeedbackManager,
        "_build_storage_client",
        return_value=MagicMock(),
    ):

        return FeedbackManager(
            feedback_configuration,
        )


def test_feedback_configuration_invalid_bucket_name(
    tmp_path: Path,
) -> None:

    with pytest.raises(ValueError):

        FeedbackConfiguration(
            bucket_name="",
            supabase_url="https://example.supabase.co",
            supabase_key="fake-key",
            metadata_path=tmp_path / "feedback.csv",
        )


def test_feedback_configuration_invalid_metadata_path(
    tmp_path: Path,
) -> None:

    with pytest.raises(ValueError):

        FeedbackConfiguration(
            bucket_name="feedback-images",
            supabase_url="https://example.supabase.co",
            supabase_key="fake-key",
            metadata_path=tmp_path / "feedback.txt",
        )


def test_generate_filename(
    feedback_manager: FeedbackManager,
) -> None:

    filename = feedback_manager._generate_filename()

    assert filename.endswith(".jpg")


def test_load_feedback_empty(
    feedback_manager: FeedbackManager,
) -> None:

    dataframe = feedback_manager.load_feedback()

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert dataframe.empty


def test_feedback_statistics_empty(
    feedback_manager: FeedbackManager,
) -> None:

    statistics = feedback_manager.feedback_statistics()

    assert statistics["total_feedback"] == 0
    assert statistics["positive_feedback"] == 0
    assert statistics["negative_feedback"] == 0


def test_submit_feedback_invalid_confidence(
    feedback_manager: FeedbackManager,
) -> None:

    with pytest.raises(FeedbackError):

        feedback_manager.submit_feedback(
            image_bytes=b"image",
            feedback="positive",
            predicted_label="Authentic Human",
            confidence=101.0,
        )


def test_submit_feedback_success(
    feedback_manager: FeedbackManager,
) -> None:

    upload_result = UploadResult(
        file_id="file-id",
        file_url="https://example.com",
        filename="image.jpg",
    )

    with (
        patch.object(
            feedback_manager,
            "_upload_image",
            return_value=upload_result,
        ) as upload_mock,
        patch.object(
            feedback_manager,
            "_append_metadata",
        ) as metadata_mock,
    ):

        result = feedback_manager.submit_feedback(
            image_bytes=b"image",
            feedback="positive",
            predicted_label="Authentic",
            confidence=95.0,
        )

    assert isinstance(
        result,
        FeedbackRecord,
    )

    upload_mock.assert_called_once()

    metadata_mock.assert_called_once()


def test_generate_filename_is_unique(
    feedback_manager: FeedbackManager,
) -> None:

    first = feedback_manager._generate_filename()

    second = feedback_manager._generate_filename()

    assert first != second


def test_load_feedback_existing_file(
    feedback_manager: FeedbackManager,
) -> None:

    record = FeedbackRecord(
        timestamp="2026-07-12T12:00:00+00:00",
        feedback="positive",
        predicted_label="Authentic Human",
        confidence=90.0,
        filename="image.jpg",
        drive_file_id="file-id",
        drive_url="https://example.com",
    )

    feedback_manager._append_metadata(
        record,
    )

    dataframe = feedback_manager.load_feedback()

    assert len(dataframe) == 1

    assert dataframe.iloc[0]["feedback"] == "positive"

    assert dataframe.iloc[0]["predicted_label"] == "Authentic Human"


def test_feedback_statistics_mixed_feedback(
    feedback_manager: FeedbackManager,
) -> None:

    positive_record = FeedbackRecord(
        timestamp="2026-07-12T12:00:00+00:00",
        feedback="positive",
        predicted_label="Authentic Human",
        confidence=95.0,
        filename="positive.jpg",
        drive_file_id="1",
        drive_url="https://example.com/1",
    )

    negative_record = FeedbackRecord(
        timestamp="2026-07-12T12:01:00+00:00",
        feedback="negative",
        predicted_label="AI Generated Human",
        confidence=80.0,
        filename="negative.jpg",
        drive_file_id="2",
        drive_url="https://example.com/2",
    )

    feedback_manager._append_metadata(
        positive_record,
    )

    feedback_manager._append_metadata(
        negative_record,
    )

    statistics = feedback_manager.feedback_statistics()

    assert statistics["total_feedback"] == 2

    assert statistics["positive_feedback"] == 1

    assert statistics["negative_feedback"] == 1

    assert statistics["positive_ratio"] == 50.0

    assert statistics["negative_ratio"] == 50.0


def test_submit_feedback_upload_failure(
    feedback_manager: FeedbackManager,
) -> None:

    with patch.object(
        feedback_manager,
        "_upload_image",
        side_effect=FeedbackError(
            "Upload failed.",
        ),
    ):

        with pytest.raises(
            FeedbackError,
            match="Upload failed.",
        ):

            feedback_manager.submit_feedback(
                image_bytes=b"image",
                feedback="positive",
                predicted_label="Authentic",
                confidence=95.0,
            )

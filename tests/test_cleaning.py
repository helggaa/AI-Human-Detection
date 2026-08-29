"""
Tests for cleaning.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.cleaning import CleaningStatistics, DatasetCleaner


def test_filename_generation():
    """Sequential filename generation should match expected pattern."""
    filename = DatasetCleaner._generate_filename(
        "AI",
        5,
    )
    assert filename == "ai_000005.jpg"


def test_filename_generation_large_index():
    """Large indices should format with leading zeros up to 6 digits."""
    filename = DatasetCleaner._generate_filename(
        "Authentic",
        1234,
    )
    assert filename == "authentic_001234.jpg"


def test_cleaning_statistics_default():
    """Default cleaning statistics should start at zero."""
    stats = CleaningStatistics()
    assert stats.total_images == 0
    assert stats.processed_images == 0
    assert stats.unreadable_images == 0
    assert stats.failed_images == 0


def test_process_single_image_success(tmp_path: Path):
    """
    Processing a valid image should produce a cleaned JPEG and report entry.
    """
    source_dir = tmp_path / "raw"
    dest_dir = tmp_path / "clean"
    source_dir.mkdir()
    dest_dir.mkdir()

    img_path = source_dir / "sample.png"
    Image.new("RGB", (100, 100), color="blue").save(img_path)

    cleaner = DatasetCleaner(source_dir=source_dir, destination_dir=dest_dir)
    out_path = dest_dir / "ai_000001.jpg"

    success = cleaner._process_single_image(
        image_path=img_path,
        output_path=out_path,
        class_name="AI",
    )

    assert success is True
    assert out_path.exists()
    assert len(cleaner.report_rows) == 1
    assert cleaner.report_rows[0]["status"] == "SUCCESS"
    assert cleaner.report_rows[0]["format"] == "PNG"


def test_process_single_image_unreadable(tmp_path: Path):
    """Processing an invalid corrupt image should record unreadable status."""
    source_dir = tmp_path / "raw"
    dest_dir = tmp_path / "clean"
    source_dir.mkdir()
    dest_dir.mkdir()

    corrupt_path = source_dir / "corrupt.jpg"
    corrupt_path.write_bytes(b"not an image data")

    cleaner = DatasetCleaner(source_dir=source_dir, destination_dir=dest_dir)
    out_path = dest_dir / "ai_000001.jpg"

    success = cleaner._process_single_image(
        image_path=corrupt_path,
        output_path=out_path,
        class_name="AI",
    )

    assert success is False
    assert not out_path.exists()
    assert cleaner.statistics.unreadable_images == 1

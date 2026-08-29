"""
Tests for utils.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.utils import (
    calculate_md5,
    convert_rgb,
    count_images,
    ensure_directory,
    image_info,
    is_image_readable,
    list_image_files,
    save_jpeg,
)


def test_ensure_directory(tmp_path: Path):
    """ensure_directory should create directories recursively."""
    directory = tmp_path / "nested" / "example"
    ensure_directory(directory)
    assert directory.exists()
    assert directory.is_dir()


def test_convert_rgb():
    """convert_rgb should convert grayscale (L) to RGB."""
    image = Image.new("L", (32, 32))
    rgb = convert_rgb(image)
    assert rgb.mode == "RGB"


def test_save_jpeg_and_readable(tmp_path: Path):
    """save_jpeg should create a valid readable JPEG."""
    img = Image.new("RGB", (64, 64), color="red")
    out_file = tmp_path / "test.jpg"
    save_jpeg(img, out_file, quality=90)

    assert out_file.exists()
    assert is_image_readable(out_file) is True


def test_is_image_readable_corrupt(tmp_path: Path):
    """Corrupt file should return False for is_image_readable."""
    corrupt_file = tmp_path / "bad.jpg"
    corrupt_file.write_bytes(b"corrupted binary content")
    assert is_image_readable(corrupt_file) is False


def test_image_info(tmp_path: Path):
    """image_info should return structured metadata."""
    img = Image.new("RGB", (100, 50), color="green")
    img_path = tmp_path / "test_info.jpg"
    img.save(img_path, format="JPEG")

    info = image_info(img_path)
    assert info["width"] == 100
    assert info["height"] == 50
    assert info["aspect_ratio"] == 2.0
    assert info["mode"] == "RGB"
    assert info["size_kb"] > 0


def test_calculate_md5(tmp_path: Path):
    """calculate_md5 should compute deterministic checksums."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello ai human detection", encoding="utf-8")
    md5_hash = calculate_md5(test_file)
    assert len(md5_hash) == 32
    assert md5_hash == calculate_md5(test_file)


def test_list_and_count_images(tmp_path: Path):
    """
    list_image_files and count_images should discover all supported formats.
    """
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.PNG").touch()
    (tmp_path / "img3.webp").touch()
    (tmp_path / "ignore.txt").touch()

    images = list_image_files(tmp_path)
    assert len(images) == 3
    assert count_images(tmp_path) == 3

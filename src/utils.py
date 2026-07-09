"""
Utility Functions
=================

Common helper functions used across the project.

"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import (
    Iterable,
    TypedDict,
)

from PIL import Image

from src.config import SUPPORTED_IMAGE_EXTENSIONS
from src.logger import logger


def ensure_directory(directory: Path) -> None:
    """
    Create a directory if it does not exist.

    Parameters
    ----------
    directory : Path
        Directory to create.
    """
    directory.mkdir(parents=True, exist_ok=True)


def list_image_files(directory: Path) -> list[Path]:
    """
    Get all supported image files in a directory.

    Parameters
    ----------
    directory : Path
        Directory containing images.

    Returns
    -------
    list[Path]
        Sorted list of supported image files.
    """
    files: set[Path] = set()

    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        files.update(directory.glob(f"*{extension}"))
        files.update(directory.glob(f"*{extension.upper()}"))

    return sorted(files)


def is_image_readable(image_path: Path) -> bool:
    """
    Check whether an image can be opened.

    Parameters
    ----------
    image_path : Path

    Returns
    -------
    bool
    """

    try:
        with Image.open(image_path) as img:
            img.verify()

        return True

    except Exception:

        logger.exception(
            "Cannot read image: %s",
            image_path.name,
        )

        return False


class ImageInfo(TypedDict):
    """
    Image metadata.
    """

    filename: str
    width: int
    height: int
    format: str | None
    mode: str
    aspect_ratio: float
    size_kb: float


def image_info(image_path: Path) -> ImageInfo:
    """
    Read image metadata.

    Parameters
    ----------
    image_path : Path

    Returns
    -------
    ImageInfo
        Image metadata.
    """
    with Image.open(image_path) as img:

        width, height = img.size

        aspect_ratio = round(width / height, 3) if height > 0 else 0.0

        size_kb = round(
            image_path.stat().st_size / 1024,
            2,
        )

        return {
            "filename": image_path.name,
            "width": width,
            "height": height,
            "format": img.format,
            "mode": img.mode,
            "aspect_ratio": aspect_ratio,
            "size_kb": size_kb,
        }


def convert_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB.

    Parameters
    ----------
    image : PIL.Image.Image

    Returns
    -------
    PIL.Image.Image
    """
    return image.convert("RGB")


def save_jpeg(
    image: Image.Image,
    output_path: Path,
    quality: int,
) -> None:
    """
    Save image as JPEG.

    Parameters
    ----------
    image : PIL.Image.Image

    output_path : Path

    quality : int
    """

    ensure_directory(output_path.parent)

    image.save(
        output_path,
        format="JPEG",
        quality=quality,
        optimize=True,
    )


def copy_image(
    source: Path,
    destination: Path,
) -> None:
    """
    Copy an image while preserving metadata.

    Parameters
    ----------
    source : Path
        Source image.

    destination : Path
        Destination image.
    """

    ensure_directory(destination.parent)

    shutil.copy2(
        src=source,
        dst=destination,
    )

    logger.info(
        "Copied image: %s -> %s",
        source,
        destination,
    )


def calculate_md5(
    file_path: Path,
    chunk_size: int = 8192,
) -> str:
    """
    Calculate MD5 checksum of a file.

    Parameters
    ----------
    file_path : Path

    chunk_size : int, default=8192

    Returns
    -------
    str
        MD5 hash.
    """

    hash_object = hashlib.md5()

    with file_path.open("rb") as file:

        while chunk := file.read(chunk_size):

            hash_object.update(chunk)

    return hash_object.hexdigest()


def count_images(directory: Path) -> int:
    """
    Count image files.

    Parameters
    ----------
    directory : Path

    Returns
    -------
    int
    """
    return len(list_image_files(directory))


def check_directories(
    directories: Iterable[Path],
) -> None:
    """
    Ensure all directories exist.

    Parameters
    ----------
    directories : Iterable[Path]
    """
    for directory in directories:
        ensure_directory(directory)
        logger.info(
            "Directory ready: %s",
            directory,
        )


# copy_image()
# calculate_md5()

__all__ = [
    "ImageInfo",
    "ensure_directory",
    "list_image_files",
    "is_image_readable",
    "image_info",
    "convert_rgb",
    "save_jpeg",
    "copy_image",
    "calculate_md5",
    "count_images",
    "check_directories",
]

"""
Tests for predict.py CLI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICT_SCRIPT = PROJECT_ROOT / "predict.py"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def cli_image(
    tmp_path: Path,
) -> Path:
    """
    Create a temporary RGB image.
    """

    image = Image.new(
        mode="RGB",
        size=(224, 224),
        color=(255, 255, 255),
    )

    image_path = tmp_path / "sample.jpg"

    image.save(
        image_path,
        format="JPEG",
    )

    return image_path


# =============================================================================
# CLI
# =============================================================================


def test_cli_success(
    cli_image: Path,
) -> None:
    """
    CLI should complete successfully.
    """

    result = subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            str(cli_image),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    output = result.stdout

    assert "Prediction" in output
    assert "Confidence" in output
    assert "Probabilities" in output


def test_cli_missing_file() -> None:
    """
    Missing image should return non-zero exit code.
    """

    result = subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            "missing.jpg",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0


def test_cli_invalid_image(
    tmp_path: Path,
) -> None:
    """
    Invalid image should return non-zero exit code.
    """

    invalid = tmp_path / "invalid.jpg"

    invalid.write_text(
        "not an image",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            str(invalid),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0


def test_cli_help() -> None:
    """
    --help should return success.
    """

    result = subprocess.run(
        [
            sys.executable,
            str(PREDICT_SCRIPT),
            "--help",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0

    assert "usage:" in result.stdout.lower()

from PIL import Image

from src.utils import (
    convert_rgb,
    ensure_directory,
)


def test_ensure_directory(
    tmp_path,
):
    directory = tmp_path / "example"

    ensure_directory(directory)

    assert directory.exists()


def test_convert_rgb():

    image = Image.new(
        "L",
        (32, 32),
    )

    rgb = convert_rgb(image)

    assert rgb.mode == "RGB"

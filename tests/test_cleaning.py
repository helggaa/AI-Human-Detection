from src.cleaning import DatasetCleaner


def test_filename_generation():

    filename = DatasetCleaner._generate_filename(
        "AI",
        5,
    )

    assert filename == "ai_000005.jpg"

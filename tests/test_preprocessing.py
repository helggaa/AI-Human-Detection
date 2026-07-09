from src.preprocessing import DatasetPreprocessor


def test_get_class_names():

    classes = DatasetPreprocessor.get_class_names()

    assert len(classes) == 2

    assert "AI" in classes

    assert "Authentic" in classes

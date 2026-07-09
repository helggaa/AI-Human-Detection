"""
Tests for model.py
"""

from __future__ import annotations

import pytest
import tensorflow as tf

from src.config import (
    FINE_TUNE_AT,
    NUM_CLASSES,
)
from src.model import (
    ModelArtifacts,
    ModelBuilder,
    ModelConfiguration,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def builder() -> ModelBuilder:
    """
    Create a model builder.
    """

    configuration = ModelConfiguration()

    return ModelBuilder(
        configuration=configuration,
    )


# =============================================================================
# BASE MODEL
# =============================================================================


def test_build_base_model(
    builder: ModelBuilder,
) -> None:
    """
    Test building the EfficientNet backbone.
    """

    base_model = builder.build_base_model()

    assert isinstance(
        base_model,
        tf.keras.Model,
    )

    assert builder.base_model is base_model

    assert base_model.trainable is False


# =============================================================================
# COMPLETE MODEL
# =============================================================================


def test_build_model(
    builder: ModelBuilder,
) -> None:
    """
    Test building the complete model.
    """

    model = builder.build_model()

    assert isinstance(
        model,
        tf.keras.Model,
    )

    assert builder.model is model

    assert model.output_shape[-1] == NUM_CLASSES


def test_run_returns_artifacts(
    builder: ModelBuilder,
) -> None:
    """
    run() should return ModelArtifacts.
    """

    artifacts = builder.run()

    assert isinstance(
        artifacts,
        ModelArtifacts,
    )

    assert artifacts.model is builder.model

    assert artifacts.base_model is builder.base_model


# =============================================================================
# FREEZE
# =============================================================================


def test_freeze_base_model(
    builder: ModelBuilder,
) -> None:
    """
    Base model should remain frozen.
    """

    builder.build_base_model()

    builder.freeze_base_model()

    assert builder.base_model is not None

    assert builder.base_model.trainable is False


def test_unfreeze_top_layers(
    builder: ModelBuilder,
) -> None:
    """
    Top layers should become trainable.
    """

    builder.build_base_model()

    builder.unfreeze_top_layers()

    assert builder.base_model is not None

    trainable_layers = sum(
        layer.trainable for layer in builder.base_model.layers
    )

    assert trainable_layers > 0

    assert trainable_layers <= FINE_TUNE_AT


# =============================================================================
# PARAMETERS
# =============================================================================


def test_total_parameters(
    builder: ModelBuilder,
) -> None:
    """
    Total parameter count.
    """

    builder.build_model()

    assert builder.total_parameters() > 0


def test_trainable_parameters(
    builder: ModelBuilder,
) -> None:
    """
    Trainable parameter count.
    """

    builder.build_model()

    assert builder.trainable_parameters() >= 0


def test_non_trainable_parameters(
    builder: ModelBuilder,
) -> None:
    """
    Non-trainable parameter count.
    """

    builder.build_model()

    assert builder.non_trainable_parameters() >= 0


# =============================================================================
# INFORMATION
# =============================================================================


def test_model_information(
    builder: ModelBuilder,
) -> None:
    """
    Model information dictionary.
    """

    builder.build_model()

    information = builder.model_information()

    assert information["total_parameters"] > 0

    assert information["trainable_parameters"] >= 0

    assert information["non_trainable_parameters"] >= 0


# =============================================================================
# STATE
# =============================================================================


def test_is_built_before_build(
    builder: ModelBuilder,
) -> None:
    """
    Model should not be built initially.
    """

    assert builder.is_built() is False


def test_is_built_after_build(
    builder: ModelBuilder,
) -> None:
    """
    Model should be built.
    """

    builder.build_model()

    assert builder.is_built() is True


# =============================================================================
# MAGIC METHODS
# =============================================================================


def test_len(
    builder: ModelBuilder,
) -> None:
    """
    __len__ returns total parameters.
    """

    builder.build_model()

    assert len(builder) == builder.total_parameters()


def test_repr(
    builder: ModelBuilder,
) -> None:
    """
    __repr__ should contain class name.
    """

    builder.build_model()

    representation = repr(
        builder,
    )

    assert "ModelBuilder" in representation

    assert builder.configuration.model_name in representation

"""
Dataset Preprocessing
=====================

This module prepares the cleaned dataset for model training.

Pipeline
--------
1. Load cleaned dataset
2. Build dataset dataframe
3. Perform stratified train/validation/test split
4. Create TensorFlow datasets
5. Resize images
6. Apply EfficientNetV2 preprocessing
7. Batch, cache, and prefetch datasets
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    TypeAlias,
    TypedDict,
)

import pandas as pd
import tensorflow as tf
from keras.applications.efficientnet_v2 import preprocess_input
from sklearn.model_selection import train_test_split

from src.config import (
    BATCH_SIZE,
    CACHE_DATASET,
    CLASS_DIRECTORIES,
    CLASS_LABELS,
    CLEAN_DATASET_DIR,
    DETERMINISTIC_DATASET,
    DROP_REMAINDER,
    IMAGE_SIZE,
    PREFETCH_DATASET,
    RANDOM_SEED,
    SHUFFLE_BUFFER_SIZE,
    SHUFFLE_DATASET,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from src.logger import logger
from src.utils import list_image_files

AUTOTUNE = tf.data.AUTOTUNE
INTERPOLATION = tf.image.ResizeMethod.BILINEAR
tf.keras.utils.set_random_seed(RANDOM_SEED)

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class DatasetRecord(TypedDict):
    """
    One dataset sample.
    """

    filepath: str
    label: int
    class_name: str


# =============================================================================
# TYPE ALIASES
# =============================================================================
TensorLabel: TypeAlias = tf.Tensor
TensorImage: TypeAlias = tf.Tensor
TensorDataset: TypeAlias = tf.data.Dataset


@dataclass(slots=True)
class DatasetBundle:
    """
    Bundle of dataframes, summaries, and TensorFlow datasets.
    """

    train_dataframe: pd.DataFrame
    validation_dataframe: pd.DataFrame
    test_dataframe: pd.DataFrame

    train_summary: pd.DataFrame
    validation_summary: pd.DataFrame
    test_summary: pd.DataFrame

    train_dataset: TensorDataset
    validation_dataset: TensorDataset
    test_dataset: TensorDataset

    class_names: list[str]


@dataclass(slots=True)
class DatasetStatistics:
    """
    Dataset split statistics.
    """

    total_images: int = 0
    train_images: int = 0
    validation_images: int = 0
    test_images: int = 0


@dataclass(slots=True)
class DatasetSplit:
    """
    Dataset split.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


# =============================================================================
# IMAGE DATASET
# =============================================================================


class DatasetPreprocessor:
    """
    Dataset preprocessing pipeline.
    """

    def __init__(
        self,
        dataset_directory: Path = CLEAN_DATASET_DIR,
    ) -> None:

        self.dataset_directory = dataset_directory

        self.dataframe = pd.DataFrame()
        self.statistics = DatasetStatistics()

        logger.info("=" * 70)
        logger.info("Dataset Preprocessing Initialized")
        logger.info("Dataset : %s", dataset_directory)
        logger.info("=" * 70)

    # =========================================================================
    # DATAFRAME
    # =========================================================================

    def load_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Build dataset dataframe.
        """

        records: list[DatasetRecord] = []

        for label, class_name in CLASS_LABELS.items():

            directory = self.dataset_directory / class_name.lower()

            image_files = list_image_files(directory)

            for image_path in image_files:
                if not image_path.exists():
                    raise FileNotFoundError(image_path)

                records.append(
                    {
                        "filepath": image_path,
                        "label": label,
                        "class_name": class_name,
                    }
                )

        dataframe = pd.DataFrame(records)

        dataframe = dataframe.sample(
            frac=1,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)

        self.dataframe = dataframe

        if dataframe.empty:
            raise ValueError("Dataset is empty.")

        if dataframe["label"].nunique() != len(CLASS_DIRECTORIES):
            raise ValueError(
                "Number of detected classes does not match CLASS_DIRECTORIES."
            )

        logger.info(
            "Dataset loaded : %d images",
            len(dataframe),
        )

        return dataframe

    # =========================================================================
    # DATASET SPLIT
    # =========================================================================

    def stratified_split(
        self,
    ) -> DatasetSplit:
        """
        Split dataset into train, validation, and test sets.

        Returns
        -------
        DatasetSplit
        """

        if self.dataframe.empty:
            self.load_dataframe()

        train_dataframe, temp_dataframe = train_test_split(
            self.dataframe,
            train_size=TRAIN_RATIO,
            stratify=self.dataframe["label"],
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        validation_size = VALIDATION_RATIO / (VALIDATION_RATIO + TEST_RATIO)

        validation_dataframe, test_dataframe = train_test_split(
            temp_dataframe,
            train_size=validation_size,
            stratify=temp_dataframe["label"],
            random_state=RANDOM_SEED,
            shuffle=True,
        )

        logger.info(
            "Train      : %d",
            len(train_dataframe),
        )

        logger.info(
            "Validation : %d",
            len(validation_dataframe),
        )

        logger.info(
            "Test       : %d",
            len(test_dataframe),
        )

        self.statistics.total_images = len(self.dataframe)
        self.statistics.train_images = len(train_dataframe)

        self.statistics.validation_images = len(validation_dataframe)

        self.statistics.test_images = len(test_dataframe)

        if len(train_dataframe) + len(validation_dataframe) + len(
            test_dataframe
        ) != len(self.dataframe):
            raise RuntimeError("Dataset split size mismatch.")

        return DatasetSplit(
            train=train_dataframe.reset_index(drop=True),
            validation=validation_dataframe.reset_index(drop=True),
            test=test_dataframe.reset_index(drop=True),
        )

    # =========================================================================
    # INFORMATION
    # =========================================================================

    @staticmethod
    def get_class_names() -> list[str]:
        """
        Return class names.
        """

        return list(CLASS_DIRECTORIES.keys())

    def dataset_summary(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Summarize dataset distribution.

        Parameters
        ----------
        dataframe : pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """

        summary = (
            dataframe.groupby("class_name").size().reset_index(name="images")
        )

        return summary.sort_values(by="class_name").reset_index(drop=True)

    def summary_statistics(
        self,
    ) -> DatasetStatistics:
        """
        Return dataset statistics.
        """

        return self.statistics

    def print_statistics(
        self,
    ) -> None:
        """
        Log dataset statistics.
        """

        logger.info("=" * 70)
        logger.info("DATASET STATISTICS")
        logger.info("=" * 70)

        logger.info(
            "Total Images : %d",
            self.statistics.total_images,
        )

        logger.info(
            "Train Images : %d",
            self.statistics.train_images,
        )

        logger.info(
            "Validation Images : %d",
            self.statistics.validation_images,
        )

        logger.info(
            "Test Images : %d",
            self.statistics.test_images,
        )

        logger.info("=" * 70)

    # =========================================================================
    # TENSORFLOW
    # =========================================================================

    @staticmethod
    def preprocess_image(
        image_path: TensorImage,
        label: TensorLabel,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess one image.
        """

        image = tf.io.read_file(image_path)

        image = tf.image.decode_jpeg(
            image,
            channels=3,
        )

        image = tf.image.resize(image, size=IMAGE_SIZE, method=INTERPOLATION)

        image = preprocess_input(image)

        return image, label

    def create_dataset(
        self,
        dataframe: pd.DataFrame,
        training: bool,
        cache_filename: str | None = None,
    ) -> TensorDataset:
        """
        Create TensorFlow dataset.
        """

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                dataframe["filepath"].astype(str).values,
                dataframe["label"].values,
            )
        )

        dataset = dataset.map(
            self.preprocess_image,
            num_parallel_calls=AUTOTUNE,
            deterministic=DETERMINISTIC_DATASET,
        )

        if training and SHUFFLE_DATASET:

            dataset = dataset.shuffle(
                SHUFFLE_BUFFER_SIZE,
                seed=RANDOM_SEED,
            )

        dataset = dataset.batch(
            batch_size=BATCH_SIZE,
            drop_remainder=DROP_REMAINDER,
        )

        if CACHE_DATASET:

            if cache_filename is None:
                dataset = dataset.cache()

            else:

                dataset = dataset.cache(cache_filename)

        if PREFETCH_DATASET:

            dataset = dataset.prefetch(
                AUTOTUNE,
            )

        return dataset

    # =========================================================================
    # DATASET PREPARATION
    # =========================================================================

    def build_dataset_bundle(
        self,
    ) -> DatasetBundle:
        """
        Prepare TensorFlow datasets.
        """

        split = self.stratified_split()

        train_dataframe = split.train

        validation_dataframe = split.validation

        test_dataframe = split.test

        train_summary = self.dataset_summary(
            train_dataframe,
        )

        validation_summary = self.dataset_summary(
            validation_dataframe,
        )

        test_summary = self.dataset_summary(
            test_dataframe,
        )

        train_dataset = self.create_dataset(
            dataframe=train_dataframe,
            training=True,
        )

        validation_dataset = self.create_dataset(
            dataframe=validation_dataframe,
            training=False,
        )

        test_dataset = self.create_dataset(
            dataframe=test_dataframe,
            training=False,
        )

        self.print_statistics()

        logger.info("TensorFlow datasets created successfully.")

        return DatasetBundle(
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            test_dataframe=test_dataframe,
            train_summary=train_summary,
            validation_summary=validation_summary,
            test_summary=test_summary,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            class_names=self.get_class_names(),
        )

    # =========================================================================
    # VERIFICATION
    # =========================================================================

    @staticmethod
    def verify_dataframe(
        dataframe: pd.DataFrame,
        name: str,
    ) -> None:
        """
        Verify dataframe integrity.
        """

        if dataframe.empty:
            raise ValueError(f"{name} dataframe is empty.")

        if not dataframe["filepath"].notna().all():
            raise ValueError(f"{name} contains missing file paths.")

        if not dataframe["label"].notna().all():
            raise ValueError(f"{name} contains missing labels.")

        if not dataframe["label"].isin(CLASS_LABELS.keys()).all():
            raise ValueError(f"{name} contains invalid labels.")

        if not dataframe["class_name"].notna().all():
            raise ValueError(f"{name} contains missing class names.")

        if not dataframe["filepath"].map(Path.exists).all():
            raise FileNotFoundError(f"{name} contains missing image files.")

        logger.info(
            "%s dataframe verified.",
            name,
        )

    def verify_bundle(
        self,
        bundle: DatasetBundle,
    ) -> None:
        """
        Verify generated dataset bundle.
        """

        self.verify_dataframe(
            bundle.train_dataframe,
            "Train",
        )

        self.verify_dataframe(
            bundle.validation_dataframe,
            "Validation",
        )

        self.verify_dataframe(
            bundle.test_dataframe,
            "Test",
        )

        logger.info("Dataset bundle verified.")

    # =========================================================================
    # INFORMATION
    # =========================================================================

    def class_distribution(
        self,
    ) -> pd.DataFrame:
        """
        Return full dataset distribution.
        """

        if self.dataframe.empty:
            self.load_dataframe()

        return self.dataset_summary(
            self.dataframe,
        )

    def number_of_classes(
        self,
    ) -> int:
        """
        Return number of classes.
        """

        return len(self.get_class_names())

    def dataset_size(
        self,
    ) -> int:
        """
        Return dataset size.
        """

        if self.dataframe.empty:
            self.load_dataframe()

        return len(self.dataframe)

    # =========================================================================
    # PIPELINE
    # =========================================================================

    def run(
        self,
    ) -> DatasetBundle:
        """
        Execute preprocessing pipeline.
        """

        logger.info("=" * 70)
        logger.info("START PREPROCESSING")
        logger.info("=" * 70)

        self.load_dataframe()

        bundle = self.build_dataset_bundle()

        self.verify_bundle(bundle)

        logger.info("=" * 70)
        logger.info("PREPROCESSING COMPLETED")
        logger.info("=" * 70)

        return bundle

    # =========================================================================
    # MAGIC METHODS
    # =========================================================================

    def __len__(
        self,
    ) -> int:
        """
        Return dataset size.
        """

        return self.dataset_size()

    def __repr__(
        self,
    ) -> str:

        return (
            "DatasetPreprocessor("
            f"images={self.dataset_size()}, "
            f"classes={self.number_of_classes()}"
            ")"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Execute preprocessing pipeline.
    """
    preprocessor = DatasetPreprocessor()
    bundle = preprocessor.run()

    logger.info(
        "Training Images : %d",
        len(bundle.train_dataframe),
    )

    logger.info(
        "Validation Images : %d",
        len(bundle.validation_dataframe),
    )

    logger.info(
        "Test Images : %d",
        len(bundle.test_dataframe),
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DatasetRecord",
    "DatasetStatistics",
    "DatasetSplit",
    "DatasetBundle",
    "TensorDataset",
    "TensorImage",
    "TensorLabel",
    "DatasetPreprocessor",
]

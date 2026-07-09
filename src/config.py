"""
Project Configuration
=====================

Central configuration for the AI Human Detection project.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATASET_DIR = PROJECT_ROOT / "dataset"
CLEAN_DATASET_DIR = PROJECT_ROOT / "dataset_clean"

MODEL_DIR = PROJECT_ROOT / "models"
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
LOG_DIR = REPORT_DIR / "logs"
METRIC_DIR = REPORT_DIR / "metrics"
TABLE_DIR = REPORT_DIR / "tables"

CLASSIFICATION_REPORT_FILENAME = "classification_report.csv"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.csv"
TRAINING_HISTORY_FILENAME = "training_history.csv"

CLASS_DIRECTORIES = {
    "Authentic": "authentic-human-images",
    "AI": "ai-generated-human-images",
}

CLASS_LABELS = {
    0: "Authentic",
    1: "AI",
}

NUM_CLASSES = len(CLASS_DIRECTORIES)

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

INPUT_SHAPE = (
    *IMAGE_SIZE,
    IMAGE_CHANNELS,
)

JPEG_QUALITY = 95


# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

SHUFFLE_DATASET = True

SHUFFLE_BUFFER_SIZE = 1000

CACHE_DATASET = True

PREFETCH_DATASET = True

SUPPORTED_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".avif",
)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9
assert TRAIN_RATIO > 0
assert VALIDATION_RATIO > 0
assert TEST_RATIO > 0

RANDOM_SEED = 42

SUPPORTED_MODELS = ("EfficientNetV2B0",)
DEFAULT_MODEL = SUPPORTED_MODELS[0]
MODEL_FILENAME = "best_model.keras"
FINAL_MODEL_FILENAME = "final_model.keras"

BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 1e-4
FINE_TUNE_LEARNING_RATE = 1e-5
FINE_TUNE_EPOCHS = 15

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.2
MIN_LEARNING_RATE = 1e-7

FIGURE_DPI = 300
SAVE_FIGURES = True

DROP_REMAINDER = False
DETERMINISTIC_DATASET = True

CLASSIFIER_DROPOUT_RATE = 0.2
FINE_TUNE_AT = 30

DEFAULT_FIGURE_SIZE = (8, 5)
CONFUSION_MATRIX_FIGURE_SIZE = (6, 6)
ROC_FIGURE_SIZE = (6, 6)

CLEANING_REPORT_FILENAME = "cleaning_report.csv"
MODEL_VERSION = "1.0.0"

for directory in (
    CLEAN_DATASET_DIR,
    MODEL_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    LOG_DIR,
    METRIC_DIR,
    TABLE_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

__all__ = [
    "PROJECT_ROOT",
    "RAW_DATASET_DIR",
    "CLEAN_DATASET_DIR",
    "MODEL_DIR",
    "NOTEBOOK_DIR",
    "REPORT_DIR",
    "FIGURE_DIR",
    "LOG_DIR",
    "METRIC_DIR",
    "TABLE_DIR",
    "CLASS_DIRECTORIES",
    "CLASS_LABELS",
    "NUM_CLASSES",
    "IMAGE_SIZE",
    "IMAGE_CHANNELS",
    "INPUT_SHAPE",
    "JPEG_QUALITY",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "TRAIN_RATIO",
    "VALIDATION_RATIO",
    "TEST_RATIO",
    "RANDOM_SEED",
    "SUPPORTED_MODELS",
    "DEFAULT_MODEL",
    "MODEL_FILENAME",
    "FINAL_MODEL_FILENAME",
    "BATCH_SIZE",
    "EPOCHS",
    "LEARNING_RATE",
    "FINE_TUNE_LEARNING_RATE",
    "FINE_TUNE_EPOCHS",
    "EARLY_STOPPING_PATIENCE",
    "REDUCE_LR_PATIENCE",
    "REDUCE_LR_FACTOR",
    "MIN_LEARNING_RATE",
    "SHUFFLE_DATASET",
    "SHUFFLE_BUFFER_SIZE",
    "CACHE_DATASET",
    "PREFETCH_DATASET",
    "CLASSIFICATION_REPORT_FILENAME",
    "CONFUSION_MATRIX_FILENAME",
    "TRAINING_HISTORY_FILENAME",
    "FIGURE_DPI",
    "SAVE_FIGURES",
    "DROP_REMAINDER",
    "DETERMINISTIC_DATASET",
    "CLASSIFIER_DROPOUT_RATE",
    "FINE_TUNE_AT",
    "DEFAULT_FIGURE_SIZE",
    "CONFUSION_MATRIX_FIGURE_SIZE",
    "ROC_FIGURE_SIZE",
    "MODEL_VERSION",
]

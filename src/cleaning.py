"""
Dataset Cleaning Pipeline
=========================

This module performs dataset cleaning before preprocessing.

Pipeline
--------
1. Validate image readability
2. Convert image to RGB
3. Save as JPEG
4. Rename sequentially
5. Generate cleaning report
6. Log every important process

NOTE
----
This module DOES NOT resize images.
Image resizing is handled in preprocessing.py.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import (
    CLASS_DIRECTORIES,
    CLEAN_DATASET_DIR,
    CLEANING_REPORT_FILENAME,
    JPEG_QUALITY,
    METRIC_DIR,
    RAW_DATASET_DIR,
)
from src.logger import logger
from src.utils import (
    calculate_md5,
    check_directories,
    convert_rgb,
    image_info,
    is_image_readable,
    list_image_files,
    save_jpeg,
)

# =============================================================================
# DATA CLASS
# =============================================================================


@dataclass(slots=True)
class CleaningStatistics:
    """
    Store dataset cleaning statistics.
    """

    total_images: int = 0
    processed_images: int = 0
    unreadable_images: int = 0
    failed_images: int = 0


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class CleaningReport(TypedDict):
    """
    Metadata describing a cleaned image.
    """

    original_filename: str
    new_filename: str
    class_name: str
    status: str
    width: int
    height: int
    format: str | None
    mode: str
    aspect_ratio: float
    size_kb: float
    md5: str


class CleaningSummary(TypedDict):
    """
    Summary of dataset cleaning.
    """

    total_images: int
    processed_images: int
    unreadable_images: int
    failed_images: int


# =============================================================================
# DATASET CLEANER
# =============================================================================


class DatasetCleaner:
    """
    Dataset Cleaning Pipeline.

    Parameters
    ----------
    source_dir : Path
        Original dataset directory.

    destination_dir : Path
        Clean dataset directory.
    """

    def __init__(
        self,
        source_dir: Path = RAW_DATASET_DIR,
        destination_dir: Path = CLEAN_DATASET_DIR,
    ) -> None:

        self.source_dir = source_dir
        self.destination_dir = destination_dir

        check_directories([self.destination_dir])

        self.statistics = CleaningStatistics()

        self.report_rows: list[CleaningReport] = []

        logger.info("=" * 70)
        logger.info("Dataset Cleaner Initialized")
        logger.info(
            "Source      : %s",
            self.source_dir,
        )

        logger.info(
            "Destination : %s",
            self.destination_dir,
        )
        logger.info("=" * 70)

    # =========================================================================
    # DIRECTORY
    # =========================================================================

    def _prepare_class_directory(
        self,
        class_name: str,
    ) -> Path:
        """
        Create destination class directory.

        Parameters
        ----------
        class_name : str

        Returns
        -------
        Path
        """

        output_dir = self.destination_dir / class_name

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return output_dir

    def _clear_output_directory(
        self,
        directory: Path,
    ) -> None:
        """
        Remove and recreate the output directory.

        Parameters
        ----------
        directory : Path
            Directory to clear.
        """
        if directory.exists():
            shutil.rmtree(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================================================================
    # FILE NAME
    # =========================================================================

    @staticmethod
    def _generate_filename(
        class_name: str,
        index: int,
    ) -> str:
        """
        Generate sequential filename.

        Examples
        --------
        authentic_000001.jpg

        ai_000001.jpg
        """

        prefix = class_name.lower()

        return f"{prefix}_{index:06d}.jpg"

    # =========================================================================
    # REPORT
    # =========================================================================

    def _append_report(
        self,
        report: CleaningReport,
    ) -> None:
        """
        Append one cleaning report.
        """

        self.report_rows.append(report)

    # =========================================================================
    # IMAGE PROCESSING
    # =========================================================================

    def _process_single_image(
        self,
        image_path: Path,
        output_path: Path,
        class_name: str,
    ) -> bool:
        """
        Process one image.

        Steps
        -----
        1. Validate image
        2. Convert RGB
        3. Save JPEG
        4. Store report

        Returns
        -------
        bool
        """

        if not is_image_readable(image_path):

            self.statistics.unreadable_images += 1

            logger.warning(
                "Unreadable image : %s",
                image_path.name,
            )

            return False

        try:

            info = image_info(image_path)

            with Image.open(image_path) as image:

                rgb_image = convert_rgb(image)

                save_jpeg(
                    image=rgb_image,
                    output_path=output_path,
                    quality=JPEG_QUALITY,
                )

                output_size = round(
                    output_path.stat().st_size / 1024,
                    2,
                )

                md5 = calculate_md5(output_path)

            self.statistics.processed_images += 1

            report: CleaningReport = {
                "original_filename": image_path.name,
                "new_filename": output_path.name,
                "class_name": class_name,
                "status": "SUCCESS",
                "width": info["width"],
                "height": info["height"],
                "format": info["format"],
                "mode": info["mode"],
                "aspect_ratio": info["aspect_ratio"],
                "size_kb": output_size,
                "md5": md5,
            }

            self._append_report(report)

            logger.info(
                "%s  ->  %s",
                image_path.name,
                output_path.name,
            )

            return True

        except Exception as error:

            self.statistics.failed_images += 1

            logger.exception(
                "Failed processing %s",
                image_path.name,
            )

            report: CleaningReport = {
                "original_filename": image_path.name,
                "new_filename": "",
                "class_name": class_name,
                "status": f"FAILED ({error})",
                "width": 0,
                "height": 0,
                "format": None,
                "mode": "",
                "aspect_ratio": 0.0,
                "size_kb": 0.0,
                "md5": "",
            }

            self._append_report(report)

            return False

    # =========================================================================
    # CLASS PROCESSING
    # =========================================================================

    def _process_class(
        self,
        class_name: str,
        source_folder: str,
    ) -> None:
        """
        Process all images belonging to one class.

        Parameters
        ----------
        class_name : str
            Display class name.

        source_folder : str
            Folder name inside RAW_DATASET_DIR.
        """

        input_directory = self.source_dir / source_folder

        if not input_directory.exists():

            logger.warning(
                "Directory not found : %s",
                input_directory,
            )
            return

        output_directory = self._prepare_class_directory(
            class_name.lower(),
        )

        image_files = list_image_files(input_directory)

        logger.info("=" * 70)
        logger.info("Processing class : %s", class_name)
        logger.info("Images found     : %d", len(image_files))
        logger.info("=" * 70)

        self.statistics.total_images += len(image_files)

        for index, image_path in enumerate(
            tqdm(
                image_files,
                desc=f"{class_name:>10}",
                unit="image",
                colour="green",
            ),
            start=1,
        ):

            output_filename = self._generate_filename(
                class_name=class_name,
                index=index,
            )

            output_path = output_directory / output_filename

            self._process_single_image(
                image_path=image_path,
                output_path=output_path,
                class_name=class_name,
            )

    # =========================================================================
    # DATASET PROCESSING
    # =========================================================================

    def clean_dataset(self) -> None:
        """
        Execute dataset cleaning.

        Pipeline
        --------
        1. Iterate each class
        2. Validate image
        3. Convert RGB
        4. Save JPEG
        """

        logger.info("=" * 70)
        logger.info("START DATASET CLEANING")
        logger.info("=" * 70)

        for class_name, folder_name in CLASS_DIRECTORIES.items():

            self._process_class(
                class_name=class_name,
                source_folder=folder_name,
            )

        logger.info("=" * 70)
        logger.info("DATASET CLEANING FINISHED")
        logger.info("=" * 70)

    # =========================================================================
    # REPORT
    # =========================================================================

    def get_report_dataframe(self) -> pd.DataFrame:
        """
        Return cleaning report.

        Returns
        -------
        pandas.DataFrame
        """

        if not self.report_rows:

            return pd.DataFrame()

        dataframe = pd.DataFrame(self.report_rows)

        dataframe = dataframe.sort_values(
            by=[
                "class_name",
                "new_filename",
            ]
        ).reset_index(drop=True)

        return dataframe

    def save_report(
        self,
        output_path: Path,
    ) -> None:
        """
        Save cleaning report to CSV.

        Parameters
        ----------
        output_path : Path
        """

        report = self.get_report_dataframe()

        report.to_csv(
            output_path,
            index=False,
        )

        logger.info(
            "Cleaning report saved : %s",
            output_path,
        )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def count_clean_images(self) -> dict[str, int]:
        """
        Count cleaned images for each class.

        Returns
        -------
        dict[str, int]
            Number of images stored in every cleaned class directory.
        """

        summary: dict[str, int] = {}

        for class_name in CLASS_DIRECTORIES.keys():

            directory = self.destination_dir / class_name.lower()

            if directory.exists():

                summary[class_name] = len(list_image_files(directory))

            else:
                summary[class_name] = 0

        return summary

    def verify_dataset(self) -> bool:
        """
        Verify cleaned dataset integrity.

        Returns
        -------
        bool
            True if processed image count matches
            images stored in destination directory.
        """

        cleaned_summary = self.count_clean_images()

        total_cleaned = sum(cleaned_summary.values())

        logger.info("=" * 70)
        logger.info("VERIFY CLEAN DATASET")
        logger.info("=" * 70)

        for class_name, total in cleaned_summary.items():

            logger.info(
                "%-12s : %5d images",
                class_name,
                total,
            )

        logger.info("-" * 70)
        logger.info(
            "Processed images : %d",
            self.statistics.processed_images,
        )

        logger.info(
            "Images on disk   : %d",
            total_cleaned,
        )

        valid = total_cleaned == self.statistics.processed_images

        if valid:

            logger.info("Dataset verification : PASSED")

        else:

            logger.warning("Dataset verification : FAILED")

        return valid

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def summary(
        self,
    ) -> CleaningSummary:
        """
        Return dataset cleaning summary.

        Returns
        -------
        CleaningSummary
        """

        return {
            "total_images": self.statistics.total_images,
            "processed_images": self.statistics.processed_images,
            "unreadable_images": self.statistics.unreadable_images,
            "failed_images": self.statistics.failed_images,
        }

    def print_summary(self) -> None:
        """
        Print cleaning summary.
        """

        stats = self.summary()

        logger.info("=" * 70)
        logger.info("CLEANING SUMMARY")
        logger.info("=" * 70)

        logger.info(
            "Total images      : %d",
            stats["total_images"],
        )

        logger.info(
            "Processed images  : %d",
            stats["processed_images"],
        )

        logger.info(
            "Unreadable images : %d",
            stats["unreadable_images"],
        )

        logger.info(
            "Failed images     : %d",
            stats["failed_images"],
        )

        logger.info("=" * 70)

    # =========================================================================
    # INFORMATION
    # =========================================================================

    def class_distribution(self) -> pd.DataFrame:
        """
        Return cleaned dataset distribution.

        Returns
        -------
        pandas.DataFrame
        """

        distribution = []

        counts = self.count_clean_images()

        for class_name, total in counts.items():

            distribution.append(
                {
                    "class": class_name,
                    "images": total,
                }
            )

        return pd.DataFrame(distribution)

    def show_distribution(self) -> None:
        """
        Display dataset distribution in console.
        """

        dataframe = self.class_distribution()

        logger.info("=" * 70)
        logger.info("DATASET DISTRIBUTION")
        logger.info("=" * 70)

        for _, row in dataframe.iterrows():

            logger.info(
                "%-12s : %5d",
                row["class"],
                row["images"],
            )

        logger.info("=" * 70)

    # =========================================================================
    # MAIN PIPELINE
    # =========================================================================

    def run(
        self,
        report_path: Path | None = None,
        clear_output: bool = False,
    ) -> pd.DataFrame:
        """
        Execute the complete dataset cleaning pipeline.

        Pipeline
        --------
        1. Clean dataset
        2. Generate report
        3. Verify cleaned dataset
        4. Print summary

        Parameters
        ----------
        report_path : Path | None, default=None
            Output CSV report location. If None, a default file inside
            reports/metrics will be used.

        Returns
        -------
        pandas.DataFrame
            Cleaning report.
        """

        try:

            logger.info("=" * 70)
            logger.info("START CLEANING PIPELINE")
            logger.info("=" * 70)

            start_time = time.perf_counter()

            if clear_output:
                self._clear_output_directory(self.destination_dir)

            self.clean_dataset()

            report = self.get_report_dataframe()

            if report_path is None:
                report_path = METRIC_DIR / CLEANING_REPORT_FILENAME

            self.save_report(report_path)
            if not self.verify_dataset():
                raise RuntimeError("Dataset verification failed.")
            self.show_distribution()
            self.print_summary()

            logger.info("=" * 70)

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Execution Time : %.2f seconds",
                elapsed,
            )
            logger.info("PIPELINE FINISHED SUCCESSFULLY")
            logger.info("=" * 70)

            return report

        except KeyboardInterrupt:

            logger.warning("Cleaning interrupted by user.")
            raise

        except Exception:

            logger.exception("Unexpected error during cleaning pipeline.")
            raise

    # =========================================================================
    # EXPORT
    # =========================================================================

    def export_summary(self) -> pd.DataFrame:
        """
        Export summary statistics as DataFrame.

        Returns
        -------
        pandas.DataFrame
        """

        return pd.DataFrame([self.summary()])

    def export_distribution(self) -> pd.DataFrame:
        """
        Export cleaned class distribution.

        Returns
        -------
        pandas.DataFrame
        """

        return self.class_distribution().copy()

    def export_all(
        self,
    ) -> dict[str, pd.DataFrame]:
        """
        Export every generated table.

        Returns
        -------
        dict[str, pandas.DataFrame]
        """

        return {
            "report": self.get_report_dataframe(),
            "summary": self.export_summary(),
            "distribution": self.export_distribution(),
        }

    # =========================================================================
    # MAGIC METHODS
    # =========================================================================

    def __len__(self) -> int:
        """
        Return processed image count.
        """

        return self.statistics.processed_images

    def __repr__(self) -> str:

        stats = self.summary()

        return (
            "DatasetCleaner("
            f"processed={stats['processed_images']}, "
            f"failed={stats['failed_images']}, "
            f"unreadable={stats['unreadable_images']}"
            ")"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Execute the dataset cleaning pipeline.

    This function can be called directly when this module is executed
    as a script.

    Example
    -------
    python src/cleaning.py
    """

    logger.info("AI HUMAN DETECTION")

    cleaner = DatasetCleaner()
    cleaner.run()

    logger.info("=" * 70)
    logger.info("CLEANING COMPLETED")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

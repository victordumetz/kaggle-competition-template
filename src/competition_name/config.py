"""Module providing constants used throughout the competition."""

from __future__ import annotations

from typing import Any, Literal

# extension of the raw data files
RAW_DATA_FILES_EXTENSION: Literal["csv", "parquet"] = "csv"

# file names of the train and test files
RAW_TRAIN_FILE_NAMES: list[str] = ["train"]
RAW_TEST_FILE_NAMES: list[str] = ["test"]

# additional arguments to pass to the loaders for the raw data
RAW_DATA_LOADER_KWARGS: dict[str, Any] = {"index_col": 0}

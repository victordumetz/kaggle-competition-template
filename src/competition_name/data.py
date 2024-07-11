"""Define functions for handling the data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

from .config import (
    RAW_DATA_FILES_EXTENSION,
    RAW_DATA_LOADER_KWARGS,
    RAW_TEST_FILE_NAMES,
    RAW_TRAIN_FILE_NAMES,
)

if TYPE_CHECKING:
    from pathlib import Path


def load_raw_data(root_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test datasets.

    Parameters
    ----------
    root_path : Path
        Path to the root directory of the project.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the train and test pandas dataframes.
    """
    raw_data_path = root_path / "data" / "raw"

    train = _pandas_multi_loader(
        raw_data_path,
        RAW_TRAIN_FILE_NAMES,
        RAW_DATA_FILES_EXTENSION,
        **RAW_DATA_LOADER_KWARGS,
    )
    test = _pandas_multi_loader(
        raw_data_path,
        RAW_TEST_FILE_NAMES,
        RAW_DATA_FILES_EXTENSION,
        **RAW_DATA_LOADER_KWARGS,
    )

    return train, test


def _pandas_multi_loader(
    files_path: Path,
    file_names: list[str],
    files_extension: Literal["csv", "parquet"],
    **kwargs,  # noqa: ANN003
) -> pd.DataFrame:
    """Load multiple files into a single pandas DataFrame.

    Parameters
    ----------
    files_path : Path
        Path to the directory containing the files.
    file_names : list[str]
        List of file names to be loaded (without extension).
    files_extension : Literal["csv", "parquet"]
        File extenstion ("csv" and "parquet" are supported).
    **kwargs
        Keyword arguments to be passed to the loaders.

    Raises
    ------
    FileExtensionNotSupportedError
        Exception raised if `files_extension` is not "csv" nor
        "parquet".

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the loaded files.
    """
    if files_extension == "csv":
        reader = pd.read_csv
    elif files_extension == "parquet":
        reader = pd.read_parquet
    else:
        raise FileExtensionNotSupportedError(RAW_DATA_FILES_EXTENSION)

    # make sure "filepath_or_buffer" and "path" are not included in
    # `kwargs`
    kwargs.pop("filepath_or_buffer", None)
    kwargs.pop("path", None)

    return pd.concat(
        [
            reader(files_path / f"{file_name}.{files_extension}", **kwargs)
            for file_name in file_names
        ]
    )


class FileExtensionNotSupportedError(Exception):
    """Exception raised for non-implemented data loaders."""

    def __init__(self, file_extension: str) -> None:
        message = f"File extension '{file_extension}' is not yet supported."
        super().__init__(message)

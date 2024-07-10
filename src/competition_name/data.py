"""Define functions for handling the data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

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
    train = pd.read_csv(raw_data_path / "train.csv", index_col=0)
    test = pd.read_csv(raw_data_path / "test.csv", index_col=0)

    return train, test

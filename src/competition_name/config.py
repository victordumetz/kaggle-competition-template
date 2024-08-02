"""Module providing constants used throughout the competition."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from collections.abc import Callable

    from sklearn.model_selection import _BaseKFold


# ==========
# RAW DATA
# ==========

# extension of the raw data files
RAW_DATA_FILES_EXTENSION: Literal["csv", "parquet"] = "csv"

# file names of the train and test files
RAW_TRAIN_FILE_NAMES: list[str] = ["train"]
RAW_TEST_FILE_NAMES: list[str] = ["test"]

# additional arguments to pass to the loaders for the raw data
RAW_DATA_LOADER_KWARGS: dict[str, Any] = {"index_col": 0}


# ==========
# RANDOM STATE
# ==========

# random state passed to the cross validators
RANDOM_STATE: int = 42


# ==========
# NUMBER OF JOBS
# ==========

# number of jobs
# if not used by the estimator, cross validation is parallelized
N_JOBS: int = -1


# ==========
# TARGET LABEL ENCODING
# ==========

# whether the target variable should be label encoded or not
LABEL_ENCODE_TARGET: bool = False


# ==========
# VALIDATION
# ==========

# number of splits
N_SPLITS: int = 5

# whether the data should be shuffled during validation or not
SHUFFLE: bool = True

# cross validator class (will be instantiated in the `ModelWrapper`)
CROSS_VALIDATOR: type[_BaseKFold] = KFold

# metrics
METRICS: dict[str, Callable | str] = {
    "RMSE": "neg_root_mean_squared_error",
}

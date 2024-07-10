"""Implement tests for the data module."""

import unittest
from pathlib import Path
from unittest.mock import call, patch

import pandas as pd

from src.competition_name.data import load_raw_data


class TestDataLoadingFunctions(unittest.TestCase):
    """Test data loading functions."""

    def test_load_raw_data(self) -> None:
        """Test the `load_raw_data` function."""
        train = pd.DataFrame(
            {
                "pred_1": [1, 2, 3, 4, 5],
                "pred_2": [1, 2, 3, 4, 5],
                "target": [1, 2, 3, 4, 5],
            }
        )
        test = pd.DataFrame(
            {
                "pred_1": [1, 2, 3, 4, 5],
                "pred_2": [1, 2, 3, 4, 5],
            }
        )

        with patch("src.competition_name.data.pd.read_csv") as load_mock:
            load_mock.side_effect = (train, test)
            loaded_train, loaded_test = load_raw_data(Path(".."))
            load_mock.assert_has_calls(
                [
                    call(Path("..", "data", "raw", "train.csv"), index_col=0),
                    call(Path("..", "data", "raw", "test.csv"), index_col=0),
                ]
            )

        pd.testing.assert_frame_equal(loaded_train, train)
        pd.testing.assert_frame_equal(loaded_test, test)

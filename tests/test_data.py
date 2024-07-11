"""Implement tests for the data module."""

import unittest
from pathlib import Path
from unittest.mock import call, patch

import pandas as pd

from src.competition_name.data import _pandas_multi_loader, load_raw_data


class TestDataLoadingFunctions(unittest.TestCase):
    """Test data loading functions."""

    def test__pandas_multi_loader(self):
        """Test the `_pandas_multi_loader` function."""
        df_1 = pd.DataFrame(
            {
                "pred_1": [1, 2, 3, 4, 5],
                "pred_2": [1, 2, 3, 4, 5],
                "target": [1, 2, 3, 4, 5],
            }
        )
        df_2 = pd.DataFrame(
            {
                "pred_1": [1, 2, 3, 4, 5],
                "pred_2": [1, 2, 3, 4, 5],
                "target": [1, 2, 3, 4, 5],
            }
        )

        # test the function with `files_extenstion = "csv"`
        with patch("src.competition_name.data.pd.read_csv") as load_mock:
            load_mock.side_effect = (df_1, df_2)
            loaded_df = _pandas_multi_loader(
                Path("data"), ["df_1", "df_2"], "csv", index_col=0
            )
            load_mock.assert_has_calls(
                [
                    call(Path("data", "df_1.csv"), index_col=0),
                    call(Path("data", "df_2.csv"), index_col=0),
                ]
            )

        pd.testing.assert_frame_equal(loaded_df, pd.concat([df_1, df_2]))

        # test the function with `files_extenstion = "parquet"`
        with patch("src.competition_name.data.pd.read_parquet") as load_mock:
            load_mock.side_effect = (df_1, df_2)
            loaded_df = _pandas_multi_loader(
                Path("data"), ["df_1", "df_2"], "parquet", index_col=0
            )
            load_mock.assert_has_calls(
                [
                    call(Path("data", "df_1.parquet"), index_col=0),
                    call(Path("data", "df_2.parquet"), index_col=0),
                ]
            )

        pd.testing.assert_frame_equal(loaded_df, pd.concat([df_1, df_2]))

    def test_load_raw_data_csv(self):
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

        with patch(
            "src.competition_name.data._pandas_multi_loader"
        ) as load_mock:
            load_mock.side_effect = (train, test)
            loaded_train, loaded_test = load_raw_data(Path(".."))
            load_mock.assert_has_calls(
                [
                    call(
                        Path("..", "data", "raw"),
                        ["train"],
                        "csv",
                        index_col=0,
                    ),
                    call(
                        Path("..", "data", "raw"),
                        ["test"],
                        "csv",
                        index_col=0,
                    ),
                ]
            )

        pd.testing.assert_frame_equal(loaded_train, train)
        pd.testing.assert_frame_equal(loaded_test, test)

"""Implements tests for the models module."""

import unittest
from typing import Self
from unittest.mock import call, patch

import pandas as pd

from src.competition_name.models import ModelWrapper

X = pd.DataFrame(
    {
        "pred_1": [1, 2, 3, 4, 5],
        "pred_2": [1, 2, 3, 4, 5],
    }
)

y = pd.Series([1, 2, 3, 4, 5], name="target")


class BaseModel:
    """A base model implementing `fit` and `predict`.

    Calling `fit` doesn't do anything and `predict` returns a constant
    dataframe.

    Methods
    -------
    fit(X, y, **kwargs)
        Does nothing.
    predict(X, **kwargs)
        Return a constant dataframe.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Self:  # noqa: ARG002, N803
        """Return self.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe of predictors.
        y : pandas.Series
            The target feature.
        **kwargs :
            Necessary for compatibility with the `ModelProtocol`.
        """
        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:  # noqa: ARG002, N803
        """Return a constant dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The predictor dataframe.
        **kwargs :
            Necessary for compatibility with the `ModelProtocol`.

        Returns
        -------
        pandas.Series
            A constant pandas series.
        """
        return y


class TestModelWrapperClass(unittest.TestCase):
    """Test `ModelWrapper` methods and properties."""

    def test_model_id(self):
        """Test the `model_id` property."""
        model = ModelWrapper("test_model", "A test model.", BaseModel())

        # fit the model and get first `model_id`
        model.fit(X, y)
        first_model_id = model.model_id
        # check that `model_id` is a string
        self.assertIsInstance(first_model_id, str)
        # check that `model_id` is constant as long as the model is not
        # refitted
        self.assertEqual(model.model_id, first_model_id)

        # refit the model and check that `model_id` has been updated
        model.fit(X, y)
        self.assertNotEqual(model.model_id, first_model_id)

    def test_fit(self):
        """Test the `fit` method."""
        base_model = BaseModel()
        model = ModelWrapper("test_model", "A test model.", base_model)
        with patch.object(BaseModel, "fit") as fit_mock:
            model.fit(X, y)
            model.fit(X, y, some_kwarg=42)
            fit_mock.assert_has_calls(
                [
                    call(X, y),
                    call(X, y, some_kwarg=42),
                ]
            )

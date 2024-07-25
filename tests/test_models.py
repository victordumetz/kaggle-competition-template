"""Implements tests for the models module."""

import unittest
from typing import Self
from unittest.mock import call, patch

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor

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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:  # noqa: ARG002, N803
        """Return self.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe of predictors.
        y : pandas.Series
            The target feature.
        """
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:  # noqa: ARG002, N803
        """Return a constant dataframe.

        Parameters
        ----------
        X : pandas.DataFrame
            The predictor dataframe.

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
        # test that `model_id` is a string
        self.assertIsInstance(first_model_id, str)
        # test that `model_id` is constant as long as the model is not
        # refitted
        self.assertEqual(model.model_id, first_model_id)

        # refit the model and test that `model_id` has been updated
        model.fit(X, y)
        self.assertNotEqual(model.model_id, first_model_id)

    def test_label_encoder(self):
        """Test the label encoding."""
        base_model = BaseModel()

        model = ModelWrapper(
            "classifier",
            "A classifier, requiring label encoding of the target",
            base_model,
        )

        # test that `LabelEncoder` is not used when
        # `LABEL_ENCODE_TARGET == False`
        with (
            patch(
                "src.competition_name.models.LABEL_ENCODE_TARGET", new=False
            ),
            patch(
                "src.competition_name.models.LabelEncoder.fit_transform"
            ) as fit_transform_mock,
        ):
            model.fit(X, y)
            fit_transform_mock.assert_not_called()

        # test that `LabelEncoder` is used when
        # `LABEL_ENCODE_TARGET == True`
        with (
            patch("src.competition_name.models.LABEL_ENCODE_TARGET", new=True),
            patch(
                "src.competition_name.models.LabelEncoder.fit_transform"
            ) as fit_transform_mock,
        ):
            model.fit(X, y)
            fit_transform_mock.assert_has_calls([call(y)])

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

    def test_sklearn_compatibility(self):
        """Test type compatibility with scikit-learn estimators."""
        dummy_regressor = DummyRegressor()
        regressor_model = ModelWrapper(
            "regressor_model", "A test model.", dummy_regressor
        )
        with patch.object(DummyRegressor, "fit") as fit_mock:
            regressor_model.fit(X, y)
            fit_mock.assert_has_calls([call(X, y)])

        dummy_classifier = DummyClassifier()
        classifier_model = ModelWrapper(
            "dummy_classifier", "A test model.", dummy_classifier
        )
        with patch.object(DummyClassifier, "fit") as fit_mock:
            classifier_model.fit(X, y)
            fit_mock.assert_has_calls([call(X, y)])

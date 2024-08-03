"""Implements tests for the models module."""

import unittest
from typing import Self
from unittest.mock import call, patch

import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.competition_name.config import METRICS
from src.competition_name.models import EstimatorNotFittedError, ModelWrapper

X_TEST = pd.DataFrame(
    {
        "pred_1": [1, 2, 3, 4, 5],
        "pred_2": [1, 2, 3, 4, 5],
    }
)
Y_TEST = pd.Series([1, 2, 3, 4, 5], name="target")
Y_TEST_LE = pd.Series(LabelEncoder().fit_transform(Y_TEST), name="target")


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

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

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

    def predict(self, X: pd.DataFrame) -> pd.Series:  # noqa: N803
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
        return pd.Series([0] * X.shape[0], name="prediction")

    def get_params(self, deep: bool = True) -> dict:  # noqa: ARG002, FBT001, FBT002
        """Return the estimator's parameters.

        Parameters
        ----------
        deep : bool
            For compatibility with the `EstimatorProtocol`.

        Returns
        -------
        dict
            A dictionary with parameter names as keys and values as
            values.
        """
        return {"n_jobs": self.n_jobs}

    def set_params(self, **params) -> Self:
        """Set the estimator's parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the parameters to set.

        Returns
        -------
        BaseModel
            An instance of self with the parameters set.
        """
        n_jobs = params.get("n_jobs", None)
        if n_jobs is not None:
            self.n_jobs = n_jobs

        return self


class TestModelWrapperClass(unittest.TestCase):
    """Test `ModelWrapper` methods and properties."""

    def test_model_id(self):
        """Test the `model_id` property."""
        model = ModelWrapper("test_model", "A test model.", BaseModel())

        # fit the model and get first `model_id`
        model.fit(X_TEST, Y_TEST)
        first_model_id = model.model_id
        # test that `model_id` is a string
        self.assertIsInstance(first_model_id, str)
        # test that `model_id` is constant as long as the model is not
        # refitted
        self.assertEqual(model.model_id, first_model_id)

        # refit the model and test that `model_id` has been updated
        model.fit(X_TEST, Y_TEST)
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
            model.fit(X_TEST, Y_TEST)
            fit_transform_mock.assert_not_called()

        # test that `LabelEncoder` is used when
        # `LABEL_ENCODE_TARGET == True`
        with (
            patch("src.competition_name.models.LABEL_ENCODE_TARGET", new=True),
            patch(
                "src.competition_name.models.LabelEncoder.fit_transform"
            ) as fit_transform_mock,
        ):
            fit_transform_mock.return_value = Y_TEST_LE

            model.fit(X_TEST, Y_TEST)
            fit_transform_mock.assert_has_calls([call(Y_TEST)])

    def test_fit_calls(self):
        """Test the `fit` method."""
        base_model = BaseModel()
        model = ModelWrapper("test_model", "A test model.", base_model)
        with patch.object(BaseModel, "fit") as fit_mock:
            model.fit(X_TEST, Y_TEST)
            model.fit(X_TEST, Y_TEST, some_kwarg=42)
            fit_mock.assert_has_calls(
                [
                    call(X_TEST, Y_TEST),
                    call(X_TEST, Y_TEST, some_kwarg=42),
                ]
            )

    def test_fit_cross_validate(self):
        """Test cross validation during fitting."""
        # test that cross validation is parallelized when the estimator
        # fitting is not
        with (
            patch("src.competition_name.models.N_JOBS", -1),
            patch(
                "src.competition_name.models.cross_validate"
            ) as cross_validate_mock,
        ):
            base_model = BaseModel()
            model = ModelWrapper("test_model", "A test model.", base_model)
            model.fit(X_TEST, Y_TEST)

            cross_validate_mock.assert_has_calls(
                [
                    call(
                        base_model,
                        X_TEST,
                        Y_TEST,
                        scoring=model._cv_metrics,
                        cv=model._cross_validator,
                        n_jobs=-1,
                    )
                ]
            )

        # test that cross validation is not parallelized when the
        # estimator fitting is
        with (
            patch("src.competition_name.models.N_JOBS", -1),
            patch(
                "src.competition_name.models.cross_validate"
            ) as cross_validate_mock,
        ):
            base_model_parallel = BaseModel(n_jobs=-1)
            model_parallel = ModelWrapper(
                "test_model", "A test model.", base_model_parallel
            )
            model_parallel.fit(X_TEST, Y_TEST)
            cross_validate_mock.assert_has_calls(
                [
                    call(
                        model_parallel.estimator,
                        X_TEST,
                        Y_TEST,
                        scoring=model_parallel._cv_metrics,
                        cv=model_parallel._cross_validator,
                        n_jobs=1,
                    )
                ]
            )

        # test that `train_metrics` keys contain the metrics names
        base_model = BaseModel()
        model = ModelWrapper("test_model", "A test model.", base_model)
        model.fit(X_TEST, Y_TEST)
        for metric in [*METRICS, "fit_time", "score_time"]:
            self.assertIn(metric, model.train_metrics)

    def test_predict(self):
        """Test the `predict` method."""
        base_model = BaseModel()
        model = ModelWrapper("test_model", "A test model.", base_model)

        # test that exception is raised when not fitted
        self.assertRaises(EstimatorNotFittedError, model.predict, X_TEST)

        with patch.object(BaseModel, "predict") as predict_mock:
            model.fit(X_TEST, Y_TEST)
            model.predict(X_TEST)
            predict_mock.assert_has_calls([call(X_TEST)])

    def test_validate(self):
        """Test the `validate` method."""
        base_model = BaseModel()
        model = ModelWrapper("test_model", "A test model.", base_model)

        # test that exception is raised when not fitted
        self.assertRaises(
            EstimatorNotFittedError, model.validate, X_TEST, Y_TEST
        )

        model.fit(X_TEST, Y_TEST)
        model.validate(X_TEST, Y_TEST)

        # test that all metrics have been computed
        self.assertEqual(model.validation_metrics.keys(), METRICS.keys())

    def test_validate_predict_proba(self):
        """Test the `validate` method on metric using probabilities."""
        metrics = {"ROC AUC": ("roc_auc_ovo", "predict_proba")}
        with (
            patch("src.competition_name.models.LABEL_ENCODE_TARGET", new=True),
            patch(
                "src.competition_name.models.METRICS",
                metrics,
            ),
        ):
            base_model = LogisticRegression()
            model = ModelWrapper("test_model", "A test model.", base_model)

            y = pd.Series(["a", "b", "c", "a", "b"])

            model.fit(X_TEST, y)
            model.validate(X_TEST, y)

            # test that all metrics have been computed
            self.assertEqual(model.validation_metrics.keys(), metrics.keys())

    def test_sklearn_compatibility(self):
        """Test type compatibility with scikit-learn estimators."""
        dummy_regressor = DummyRegressor()
        regressor_model = ModelWrapper(
            "regressor_model", "A test model.", dummy_regressor
        )
        with patch.object(DummyRegressor, "fit") as fit_mock:
            regressor_model.fit(X_TEST, Y_TEST)
            fit_mock.assert_has_calls([call(X_TEST, Y_TEST)])

        dummy_classifier = DummyClassifier()
        classifier_model = ModelWrapper(
            "dummy_classifier", "A test model.", dummy_classifier
        )
        with patch.object(DummyClassifier, "fit") as fit_mock:
            classifier_model.fit(X_TEST, Y_TEST)
            fit_mock.assert_has_calls([call(X_TEST, Y_TEST)])

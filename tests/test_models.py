"""Implements tests for the models module."""

import pickle
import unittest
from io import BytesIO
from pathlib import Path
from typing import Self
from unittest.mock import call, patch

import joblib
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


class BaseEstimator:
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
        model = ModelWrapper("test_model", "A test model.", BaseEstimator())

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
        estimator = BaseEstimator()

        model = ModelWrapper(
            "classifier",
            "A classifier, requiring label encoding of the target",
            estimator,
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
        estimator = BaseEstimator()
        model = ModelWrapper("test_model", "A test model.", estimator)
        with patch.object(BaseEstimator, "fit") as fit_mock:
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
            estimator = BaseEstimator()
            model = ModelWrapper("test_model", "A test model.", estimator)
            model.fit(X_TEST, Y_TEST)

            cross_validate_mock.assert_has_calls(
                [
                    call(
                        estimator,
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
            estimator_parallel = BaseEstimator(n_jobs=-1)
            model_parallel = ModelWrapper(
                "test_model", "A test model.", estimator_parallel
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
        estimator = BaseEstimator()
        model = ModelWrapper("test_model", "A test model.", estimator)
        model.fit(X_TEST, Y_TEST)
        for metric in [*METRICS, "fit_time", "score_time"]:
            self.assertIn(metric, model.train_metrics)

    def test_predict(self):
        """Test the `predict` method."""
        estimator = BaseEstimator()
        model = ModelWrapper("test_model", "A test model.", estimator)

        # test that exception is raised when not fitted
        self.assertRaises(EstimatorNotFittedError, model.predict, X_TEST)

        with patch.object(BaseEstimator, "predict") as predict_mock:
            model.fit(X_TEST, Y_TEST)
            model.predict(X_TEST)
            predict_mock.assert_has_calls([call(X_TEST)])

    def test_validate(self):
        """Test the `validate` method."""
        estimator = BaseEstimator()
        model = ModelWrapper("test_model", "A test model.", estimator)

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
            estimator = LogisticRegression()
            model = ModelWrapper("test_model", "A test model.", estimator)

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

    def test_save(self):
        """Test the `save` method."""
        estimator = BaseEstimator()
        model = ModelWrapper("test_model", "A test model.", estimator)
        model.fit(X_TEST, Y_TEST).validate(X_TEST, Y_TEST)

        # test the case where stats file does not exist
        with (
            patch("src.competition_name.models.pd.read_csv") as load_mock,
            patch(
                "src.competition_name.models.pd.DataFrame.to_csv"
            ) as write_mock,
            patch("src.competition_name.models.joblib.dump") as pickle_mock,
        ):
            load_mock.side_effect = FileNotFoundError()

            # test that warning is thrown
            with self.assertWarns(Warning):
                model.save(Path())

            # test that stats file is loaded
            load_mock.assert_has_calls(
                [call(Path("models", "models_stats.csv"))]
            )
            # test that stats file is written
            write_mock.assert_has_calls(
                [call(Path("models", "models_stats.csv"), index=False)]
            )
            # test that the model is pickled
            pickle_mock.assert_has_calls(
                [
                    call(
                        model,
                        Path("models", f"{model.model_id}_{model.name}.pkl"),
                    )
                ]
            )

        # test the case where stats file exists
        with (
            patch("src.competition_name.models.pd.read_csv") as load_mock,
            patch(
                "src.competition_name.models.pd.DataFrame.to_csv",
                autospec=True,
            ) as write_mock,
            patch("src.competition_name.models.joblib.dump") as pickle_mock,
        ):
            load_mock.return_value = pd.DataFrame(
                {
                    "model_id": "123",
                    "name": "old_model",
                    "description": "An old model.",
                    "fit_datetime": None,
                    "train_fit_time": 0.1,
                    "train_score_time": 0.1,
                    "train_RMSE": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                    "validation_RMSE": 3.0,
                }
            )

            model.save(Path())

            # test that new row is appended
            saved_df = write_mock.call_args_list[0][0][0]
            self.assertEqual(2, saved_df.shape[0])

    def test_from_pickle(self):
        """Test the `from_pickle` class method."""
        estimator = BaseEstimator()
        model = ModelWrapper("test_model", "A test model.", estimator)
        model.fit(X_TEST, Y_TEST).validate(X_TEST, Y_TEST)

        with (
            patch("src.competition_name.models.next") as next_mock,
            patch("src.competition_name.models.Path.glob") as glob_mock,
        ):
            bytes_container = BytesIO()
            joblib.dump(model, bytes_container)

            next_mock.return_value = bytes_container

            loaded_model = ModelWrapper.from_pickle(Path(), "123")

            # test `Path.glob` call
            glob_mock.assert_has_calls([call("123_*.pkl")])

            # test loaded and saved model equality
            self.assertEqual(model.model_id, loaded_model.model_id)
            self.assertEqual(model.name, loaded_model.name)
            self.assertEqual(model.description, loaded_model.description)
            self.assertEqual(
                pickle.dumps(model.estimator),
                pickle.dumps(loaded_model.estimator),
            )  # proxy for testing the estimators equality

    def test_generate_submission(self):
        """Test the `generate_submission` method."""
        estimator = BaseEstimator()
        model = ModelWrapper("test_model", "A test model.", estimator)
        model.fit(X_TEST, Y_TEST)

        with (
            patch("src.competition_name.models.load_raw_data") as load_mock,
            patch(
                "src.competition_name.models.pd.DataFrame.to_csv"
            ) as write_mock,
        ):
            load_mock.return_value = (X_TEST, X_TEST)
            model.generate_submission(Path())

            load_mock.assert_has_calls([call(Path())])
            write_mock.assert_has_calls(
                [
                    call(
                        Path(
                            "submissions", f"{model.model_id}_{model.name}.csv"
                        )
                    )
                ]
            )

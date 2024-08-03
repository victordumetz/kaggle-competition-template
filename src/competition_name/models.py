"""Define a wrapper class for managing models."""

import datetime
from collections.abc import Callable
from typing import Concatenate, Protocol, Self

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

from .config import (
    CROSS_VALIDATOR,
    LABEL_ENCODE_TARGET,
    METRICS,
    N_JOBS,
    N_SPLITS,
    RANDOM_STATE,
    SHUFFLE,
)


class EstimatorProtocol(Protocol):
    """Class defining a protocol for estimators.

    Methods
    -------
    fit(X, y, ...)
        A fit method that takes at least `X` and `y` as arguments and
        returns the class instance.
    predict(X, ...)
        A predict method that takes at least `X` as argument and returns
        a pandas dataframe or series, a numpy array-like, or a SciPy CSC
        matrix.
    get_params(deep)
        A getter for the estimator parameters that returns a dictionary
        with parameter names as keys and their values as values.
    set_params(**params)
        A setter for the estilator parameters that takes a dictionary of
        parameters names and values as parameter and returns the
        estimator.
    """

    @property
    def fit(  # noqa: D102
        self,
    ) -> Callable[Concatenate[pd.DataFrame, pd.Series, ...], Self]: ...

    @property
    def predict(  # noqa: D102
        self,
    ) -> Callable[
        Concatenate[pd.DataFrame, ...],
        pd.DataFrame | pd.Series | ArrayLike | csc_matrix,
    ]: ...

    @property
    def get_params(self) -> Callable[[bool], dict]: ...  # noqa: D102

    @property
    def set_params(self) -> Callable[..., Self]: ...  # noqa: D102


class ModelWrapper:
    """Class providing tools for managing models.

    Attributes
    ----------
    model_id : str
        A unique automatically generated ID for the model.
    name : str
        The name of the model.
    description : str
        A description for the model.
    estimator : EstimatorProtocol
        The estimator for the model.

    Methods
    -------
    fit(X, y, **kwargs)
        Fit the estimator on the data.
    """

    def __init__(
        self, name: str, description: str, estimator: EstimatorProtocol
    ) -> None:
        """Initialise the class instance.

        Parameters
        ----------
        name : str
            The name of the model.
        description : str
            A description of the model.
        estimator : EstimatorProtocol
            The estimator for the model.
        """
        self.name = name
        self.description = description
        self.estimator = estimator

        self._label_encoder = LabelEncoder()

        # dictionary of shape {metric_name: (metric, response_method)}
        self._metrics = {
            name: (get_scorer(metric), response_method)
            for name, (metric, response_method) in METRICS.items()
        }
        # dictionary of shape {metric_name: metric} used for CV
        self._cv_metrics = {
            name: metric for name, (metric, _) in self._metrics.items()
        }

        # set of response methods
        self._unique_metric_response_methods = {
            response_method for _, response_method in self._metrics.values()
        }

        self._n_jobs_cross_validator = (
            N_JOBS if getattr(self.estimator, "n_jobs", 1) == 1 else 1
        )
        self._cross_validator = CROSS_VALIDATOR(
            n_splits=N_SPLITS,
            shuffle=SHUFFLE,
            random_state=(
                RANDOM_STATE if SHUFFLE else None
            ),  # prevent error when `SHUFFLE == False`
        )

        self._fitted = None
        self._fit_datetime = None

    @property
    def model_id(self) -> str:
        """The model ID property."""
        return self._generate_model_id()

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Self:  # noqa: ANN003, N803
        """Fit the model to the data and compute train metrics.

        Parameters
        ----------
        X : pandas.DataFrame
            Dataframe containing the independent variables.
        y : pandas.Series
            Series containing the dependent variable.

        Returns
        -------
        ModelWrapper
            Self return to allow chaining.
        """
        if LABEL_ENCODE_TARGET:
            y = pd.Series(self._label_encoder.fit_transform(y), name=y.name)

        self.estimator.fit(X, y, **kwargs)

        # compute train metrics
        self.train_metrics = cross_validate(
            self.estimator,
            X,
            y,
            scoring=self._cv_metrics,
            cv=self._cross_validator,
            n_jobs=self._n_jobs_cross_validator,
        )
        # remove "test_" prefix from metrics' names and cast to floats
        self.train_metrics = {
            key.replace("test_", ""): value
            for key, value in self.train_metrics.items()
        }

        # set the `_fitted` and `_fit_datetime` attributes
        self._fitted = True
        self._fit_datetime = datetime.datetime.now(datetime.UTC)

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:  # noqa: N803
        """Predict the target for the input data.

        Parameters
        ----------
        X : pandas.DataFrame
            Pandas dataframe containing the indenpendent variables.

        Returns
        -------
        pandas.Series
            Series containing the predictions.

        Raises
        ------
        EstimatorNotFittedError
            Exception raised if the estimator has not been fitted
        """
        if not self._fitted:
            raise EstimatorNotFittedError

        return pd.Series(
            np.array(self.estimator.predict(X)).reshape(
                -1
            )  # ensure the predictions are 0-dimensional
        )

    def validate(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:  # noqa: N803
        """Validate the model and compute validation metrics.

        Parameters
        ----------
        X_val : pandas.DataFrame
            DataFrame frame containing the independent variables of the
            validation set.
        y_val ; pandas.Series
            Series containing the dependent variable of the validation
            set.
        """
        if not self._fitted:
            raise EstimatorNotFittedError

        if LABEL_ENCODE_TARGET:
            y_val = pd.Series(
                self._label_encoder.fit_transform(y_val), name=y_val.name
            )

        # compute the predictions for each response method
        y_pred = {}
        for response_method in self._unique_metric_response_methods:
            y_pred[response_method] = getattr(self.estimator, response_method)(
                X_val
            )

        # compute the metrics
        self.validation_metrics = {
            name: float(
                metric._score_func(  # noqa: SLF001
                    y_val,
                    y_pred[response_method],
                    **metric._kwargs,  # noqa: SLF001
                )
            )
            for name, (metric, response_method) in self._metrics.items()
        }

        self._validated = True

    def _generate_model_id(self) -> str:
        """Generate the model ID.

        The ID is generated as a hash of the model `name` and the
        model's `_fit_datetime`.

        Returns
        -------
        str
            The generated model ID.
        """
        return str(hash((self.name, self._fit_datetime)))


class EstimatorNotFittedError(Exception):
    """Exception raised when an estimator has not yet been fitted."""

    def __init__(self) -> None:
        """Initialise the class."""
        message = (
            "The model has not yet been fitted. First call the model's `fit` "
            "method."
        )
        super().__init__(message)

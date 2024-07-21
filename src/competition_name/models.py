"""Define a wrapper class for managing models."""

import datetime
from collections.abc import Callable
from typing import Concatenate, Protocol, Self

import pandas as pd
from numpy.typing import ArrayLike
from scipy.sparse import csc_matrix


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

        self._fitted = None
        self._fit_datetime = None

    @property
    def model_id(self) -> str:
        """The model ID property."""
        return self._generate_model_id()

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Self:  # noqa: ANN003, N803
        """Fit the model to the data.

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
        self.estimator.fit(X, y, **kwargs)

        # set the `_fitted` and `_fit_datetime` attributes
        self._fitted = True
        self._fit_datetime = datetime.datetime.now(datetime.UTC)

        return self

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

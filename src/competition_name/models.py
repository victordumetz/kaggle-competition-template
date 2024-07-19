"""Define a wrapper class for managing models."""

import datetime
from typing import Protocol, Self

import pandas as pd


class EstimatorProtocol(Protocol):
    """Class defining a protocol for estimators.

    Methods
    -------
    fit(X, y, **kwargs)
        A fit method that takes at least `X` and `y` as arguments.
    predict(X, **kwargs)
        A predict method that takes at least `X` as argument and returns
        a pandas dataframe.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Self:  # noqa: ANN003, N803
        """Define the signature of a `fit` method.

        The `fit` method should take at least `X` and `Y` as arguments.
        It can take some additional arguments as kwargs.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe of predictors.
        y : pandas.Series
            The target feature.
        **kwargs
            Any additional arguments.

        Returns
        -------
        EstimatorProtocol
            Self return to allow chaining.
        """
        ...

    def predict(self, X: pd.DataFrame, **kwargs) -> pd.Series:  # noqa: ANN003, N803
        """Define the signature of a `predict` method.

        The `predict` method should take at least `X` as an argument. It
        can take some additional arguments as kwargs.

        Parameters
        ----------
        X : pandas.DataFrame
            The dataframe of predictors.
        **kwargs
            Any additional arguments.

        Returns
        -------
        pandas.Series
            Pandas series containing the predicitions.
        """
        ...


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

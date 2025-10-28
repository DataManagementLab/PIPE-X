import inspect
from warnings import simplefilter

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class SuperSetColumnSelector:
    """Create a callable to select columns to be used with
    :class:`ColumnTransformer`.

    Parameters
    ----------
    columns : list[str]
        List of column names to be selected if they are present in the DataFrame.

    Returns
    -------
    selector : callable
        Callable for column selection to be used by a
        :class:`ColumnTransformer`.
    """

    def __init__(self, columns):
        self.column_superset = columns

    def __call__(self, df):
        """Callable for column selection to be used by a
        :class:`ColumnTransformer`.

        Parameters
        ----------
        df : dataframe of shape (n_features, n_samples)
            DataFrame to select columns from.
        """
        if not hasattr(df, "iloc"):
            raise ValueError(
                    "make_column_selector can only be applied to pandas dataframes"
            )
        return [column for column in self.column_superset if column in df.columns]


class PIPEXTransformer(BaseEstimator, TransformerMixin):
    # noinspection PyPep8Naming, PyUnusedLocal, PyMissingOrEmptyDocstring
    def fit(self, X: DataFrame, y: np.ndarray = None) -> 'PIPEXTransformer':
        return self

    # noinspection PyPep8Naming, PyUnusedLocal, PyMissingOrEmptyDocstring, PyMethodMayBeStatic
    def transform(self, X: DataFrame) -> DataFrame:
        return DataFrame(X)


class CustomTransformer(BaseEstimator):
    def fit(self, df: pd.DataFrame, y: pd.DataFrame = None) -> 'CustomTransformer':
        """ Fit the transformer on the input data """
        return self

    def transform(self, df: pd.DataFrame, y: pd.DataFrame = None) -> pd.DataFrame:
        """ Apply the transformation and return transformed DataFrame """
        return df

    def fit_transform(self, df: pd.DataFrame, y: pd.DataFrame = None, **fit_params) -> pd.DataFrame:
        self.fit(df, y, **fit_params)
        sig = inspect.signature(self.transform)
        if len(sig.parameters) >= 2:
            return self.transform(df, y)
        else:
            return self.transform(df)

    def explain(self) -> str:
        """ Return a human-readable explanation of the transformation """
        return "No explanation implemented yet."


class CustomFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name, feature_function) -> None:
        """
        @param feature_name: str, Name of the new feature to be created
        @param feature_function: callable, Function that takes a DataFrame and returns a Series for the new feature
        """
        # Initialize any parameters you need here
        self.feature_name = feature_name
        self.feature_function = feature_function

    def fit(self, X, y=None) -> 'CustomFeatureTransformer':
        # Fit the transformer to the data (e.g. compute any necessary statistics)
        return self

    def transform(self, X) -> pd.DataFrame:
        # Create the custom features
        X[self.feature_name] = self.feature_function(X)
        return X


# For imblearn pipelines:

def noop(x, y):
    """
    No operation transformer that returns inputs unchanged.
    """
    return x, y

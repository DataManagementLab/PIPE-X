""" Custom transformers that can be used in a pipeline.
"""
from typing import Any

# noinspection PyUnresolvedReferences
from .custom_transformers_FPP import *

import numpy as np
from numpy.random import Generator
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


# Sanity Check Experiments

class DestroyerAdult(BaseEstimator, TransformerMixin):
    """
    Custom transformer that sets all values in a DataFrame to a constant value with a given probability.

    :param seed: int, Seed for the random state
    :param probability: float, Probability that the field will be set to the specified value
    :param value: float, Value to set the field to
    """
    seed: Generator
    probability: float = 0
    value: Any = None

    def __init__(self, seed: Generator, probability: float, value) -> None:
        """
        Initialize the transformer.

        :param seed: Generator, random number generator
        :param probability: float, Probability that the field will be set to the value, should be between 0 and 1
        :param value: Any, constant to set the value to
        """
        self.seed = seed
        self.probability = probability
        self.value = value

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'DestroyerAdult':
        """
        Fit the transformer.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        """
        return self

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> DataFrame:
        """
        Set values of certain columns in the DataFrame to self.value with a chance of self.probability.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: DataFrame, transformed data
        """
        df = DataFrame(X)
        n = len(df)

        for i, col in enumerate(['education-num', 'capital-gain', 'sex', 'age', 'hours-per-week']):
            rng = np.random.default_rng(23 + i)
            shuffled_index = rng.permutation(n)
            column_number = df.columns.get_loc(col)
            df.iloc[shuffled_index[:int(self.probability * n)], column_number] = self.value

            df.loc[(df.sample(int(len(df.index) * self.probability), random_state=23 + i)).index, col] = self.value

        return df


class MaritalStatusTransformer(BaseEstimator, TransformerMixin):
    """ Custom transformer that replaces marital-status values with 'married' or 'not married'.
    """

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'MaritalStatusTransformer':
        """
        Fit the transformer.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        """
        return self

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal, PyMethodMayBeStatic
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> DataFrame:
        """
        Replace marital-status values with 'married' or 'not married'.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        """

        X = DataFrame(X)
        if 'marital-status' in X.columns:
            X['marital-status'] = X['marital-status'].replace(
                    ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent',
                     'Never-married', 'Separated', 'Widowed'],
                    ['not married', 'married', 'married', 'married', 'not married', 'not married', 'not married'],
            )

        return X

""" Example functions to create pickled pipelines along our sanity check experiments.
"""
from typing import Any

from numpy import nan
from numpy.random import Generator
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from data.adult.setup import ADULT_CATEGORICAL_COLUMNS, ADULT_CATEGORICAL_NOISY, ADULT_NUMERICAL_COLUMNS, \
    ADULT_NUMERICAL_NOISY
from PIPE_X.pipeline import SklearnPipelineWrapper
from PIPE_X.transformer import SuperSetColumnSelector
from pipelines import custom_transformers


def get_categorical_column_transformer(imputer=True) -> ColumnTransformer:
    """
    Creates a `ColumnTransformer` for processing categorical columns.

    This transformer applies an `OrdinalEncoder` to encode categorical features.
    Optionally, a `KNNImputer` can be added to handle missing values.

    Args:
        imputer (bool): If True, includes a `KNNImputer` step for imputing missing values.
                        Defaults to True.

    Returns:
        ColumnTransformer: A `ColumnTransformer` that processes categorical columns.
    """

    steps = [('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan))]
    if imputer: steps += [('categorical_imputer', KNNImputer())]

    return ColumnTransformer(transformers=[
        ('categorical_transformers', SklearnPipelineWrapper(steps=steps), selector(dtype_exclude="number")),
    ])


def get_numerical_column_transformer() -> ColumnTransformer:
    """
    Creates a `ColumnTransformer` for processing numerical columns.

    This transformer applies the following steps to numerical features:
    1. `KNNImputer`: Imputes missing values using the k-nearest neighbors algorithm.
    2. `StandardScaler`: Scales numerical features to have zero mean and unit variance.

    Returns:
        ColumnTransformer: A `ColumnTransformer` that processes numerical columns.
    """
    numerical_transformer = SklearnPipelineWrapper(steps=[
        ('numerical_imputer', KNNImputer()),
        ('numerical_scaler', StandardScaler()),
    ])

    return ColumnTransformer(transformers=[
        ('numerical_transformers', numerical_transformer, selector(dtype_include="number")),
    ])


def get_adult_noisy_column_transformer(replace_marital_status) -> ColumnTransformer:
    """
    Creates a `ColumnTransformer` for processing noisy columns in the Adult dataset.

    This transformer handles both numerical and categorical noisy columns:
    - For numerical columns, it applies a `KNNImputer` to impute missing values.
    - For categorical columns, it applies an `OrdinalEncoder` followed by a `KNNImputer`.
    - Optionally, it can replace the `marital-status` column with a custom transformation.

    Args:
        replace_marital_status (bool): If True, applies a custom transformation to the `marital-status` column.

    Returns:
        ColumnTransformer: A `ColumnTransformer` that processes noisy columns and passes through the rest.
    """
    transformers = []

    for column_name in ADULT_NUMERICAL_NOISY:
        transformers.append((column_name + '_imputer', KNNImputer(), selector(column_name)))

    for column_name in ADULT_CATEGORICAL_NOISY:
        column_pipeline = SklearnPipelineWrapper(steps=[
            (column_name + '_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan)),
            (column_name + '_imputer', KNNImputer()),
        ])
        transformers.append((column_name + '_pipeline', column_pipeline, selector(column_name)))

    if replace_marital_status:
        transformers.append(
                ('replace_marital-status', custom_transformers.MaritalStatusTransformer(), selector('marital-status'))
        )

    return ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
    )


# Sanity Check Experiments


def get_sc_imputer_pipeline(redundant: bool, marital_status: bool) -> SklearnPipelineWrapper:
    """
    Creates a pipeline for imputing and transforming the adult dataset.

    Args:
        redundant (bool): If True, includes redundant imputation for both categorical and numerical columns.
        marital_status (bool): If True, applies a custom transformation to the `marital-status` column.

    Returns:
        SklearnPipelineWrapper: A pipeline that applies the specified transformations to the dataset.
    """
    steps = [('column_transformer', get_adult_noisy_column_transformer(marital_status))]

    if redundant:
        steps += [
            ('categorical_transformers', get_categorical_column_transformer()),
            ('numerical_transformers', get_numerical_column_transformer())
        ]
    else:
        clean_numerical_columns = list(set(ADULT_NUMERICAL_COLUMNS) - set(ADULT_NUMERICAL_NOISY))
        clean_categorical_columns = list(set(ADULT_CATEGORICAL_COLUMNS) - set(ADULT_CATEGORICAL_NOISY))

        numerical_imputer_transformer = ColumnTransformer(transformers=[
            ('numerical_imputer', SimpleImputer(strategy='mean'), SuperSetColumnSelector(clean_numerical_columns)),
        ])

        categorical_imputer_transformer = ColumnTransformer(transformers=[
            ('categorical_imputer', SimpleImputer(strategy='most_frequent'),
             SuperSetColumnSelector(clean_numerical_columns)),
        ])

        categorical_transformers = ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude="number")),
        ])

        numerical_transformers = ColumnTransformer(transformers=[
            ('numerical_scaler', StandardScaler(),
             selector(dtype_include="number")),
        ])

        steps += [
            ('numerical_imputer_transformer', numerical_imputer_transformer),
            ('categorical_imputer_transformer', categorical_imputer_transformer),
            ('categorical_transformers', categorical_transformers),
            ('numerical_transformers', numerical_transformers),
        ]

    return SklearnPipelineWrapper(steps=steps)


def get_sc_destroyer_pipeline(rng: Generator, destroyer_probability: float, destroyer_value: Any,
                              marital_status: bool, scaled: bool) -> SklearnPipelineWrapper:
    """
    Creates an `SklearnPipelineWrapper` pipeline that applies transformations around artificial noise.

    This pipeline includes the following steps:
    1. Optionally replaces the values in the `marital-status` column with binned values.
    2. Applies transformations to categorical columns.
    3. Introduces noise using a destroyer step.
    4. Processes numerical columns.

    The order of the destroyer and numerical transformers depends on the `scaled` parameter,
    so destroyer constant may or may not be scaled.

    Args:
        rng (Generator): A random number generator used for introducing noise.
        destroyer_probability (float): The probability of applying the destroyer transformation.
        destroyer_value (Any): The constant value to use for the destroyer transformation.
        marital_status (bool): If True, applies a custom transformation to the `marital-status` column.
        scaled (bool): If True, scales numerical columns after applying the destroyer step.

    Returns:
        SklearnPipelineWrapper: A pipeline that applies the specified transformations to the dataset.
    """
    steps = []

    if marital_status:
        steps.append(('replace_marital_status', custom_transformers.MaritalStatusTransformer()))

    steps.append(('categorical_transformers', get_categorical_column_transformer()))

    destroyer = ('destroyer', custom_transformers.DestroyerAdult(rng, destroyer_probability, destroyer_value))
    numerical_transformers = ('numerical_transformers', get_numerical_column_transformer())

    steps.extend([destroyer, numerical_transformers] if scaled else [numerical_transformers, destroyer])

    return SklearnPipelineWrapper(steps=steps)

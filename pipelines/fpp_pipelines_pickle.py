""" Pipelines from Fair Preprocessing Paper (FPP)
"""

import numpy as np
from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN
from numpy import nan
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler

import pipelines.custom_transformers_FPP as FPP_ct
from PIPE_X.pipeline import ImblearnPipelineWrapper, SklearnPipelineWrapper
from PIPE_X.transformer import SuperSetColumnSelector
from pipelines import custom_transformers


# fair preprocessing paper pipelines

def drop_rows_with_any_nan(x, y):
    """
    Removes rows from the input DataFrame `X` and the corresponding rows in the target `y`
    where any value in `X` is NaN (missing).

    :param x: pd.DataFrame, The input feature DataFrame.
    :param y: pd.DataFrame or pd.Series, The target values corresponding to `X`.

    :return: Tuple[pd.DataFrame, pd.DataFrame or pd.Series], A tuple containing `X` with rows containing NaN values
    removed. `y` with the corresponding rows removed to maintain alignment with `X`.
    """
    mask = ~x.isna().any(axis=1)
    return x.loc[mask], y.loc[mask]


## adult dataset pipelines

def get_ac1_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC1 pipeline for preprocessing the adult dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Encodes specified categorical columns using an ordinal encoder.
    - Binarizes the 'race' column.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline.
    """
    columns_to_encode = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                         'native-country']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_ac2_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC2 pipeline for preprocessing the adult dataset. The pipeline performs the following steps:
    - Imputes missing values in specific columns ('workclass', 'occupation', 'native-country') with constant values.
    - Encodes categorical columns using an ordinal encoder.
    - Binarizes the 'race' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                         'native-country']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('imputer', ColumnTransformer(transformers=[
            ('workclass_imputer', SimpleImputer(strategy='constant', fill_value='X'), selector('workclass')),
            ('occupation_imputer', SimpleImputer(strategy='constant', fill_value='X'), selector('occupation')),
            ('native-country_imputer', SimpleImputer(strategy='constant', fill_value='United-States'),
             selector('native-country')),
        ], remainder='passthrough')),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
    ])


def get_ac3_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC3 pipeline for preprocessing the adult dataset. The pipeline performs the following steps:
    - Defines the features to keep and drops the rest.
    - Drops rows with NaN values.
    - Eliminates unnecessary features based on the defined list.
    - One-hot encodes specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    features = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation', 'relationship',
                'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                'native-country', 'income-per-year']
    features_to_keep = ['workclass', 'education-num', 'marital-status', 'race', 'sex', 'relationship', 'capital-gain',
                        'capital-loss', 'income-per-year']
    features_to_drop = [feat for feat in features if feat not in features_to_keep]

    columns_to_encode = ['sex', 'workclass', 'marital-status', 'relationship']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('drop_columns', FPP_ct.FeatureEliminator(features_to_drop)),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
    ])


def get_ac4_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC4 pipeline for preprocessing the adult dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Encodes specified categorical columns using an ordinal encoder.
    - Binarizes the 'race' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.
    - Reduces dimensionality using PCA with 14 components.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                         'native-country']
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
        # ('pca', PCA(n_components=14)),
    ])


def get_ac5_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC5 pipeline for preprocessing the dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Encodes specified categorical columns using an ordinal encoder.
    - Binarizes the 'race' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                         'native-country']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
    ])


def get_ac6_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC6 pipeline for preprocessing the dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Replaces marital status values with binary categories using a custom transformer.
    - Drops specified columns from the dataset.
    - Applies one-hot encoding to specified columns.
    - Binarizes the 'race' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    features_to_drop = ['workclass', 'education', 'occupation', 'relationship', 'native-country']
    columns_to_encode = ['age', 'hours-per-week', 'sex', 'race']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('replace_marital_status', FPP_ct.MaritalStatusBinarizer()),
        ('drop_columns', FPP_ct.FeatureEliminator(features_to_drop)),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
    ])


def get_ac7_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC7 pipeline for preprocessing the dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Maps education levels to broader categories using a custom mapping.
    - Maps marital status values to broader categories using a custom mapping.
    - Bins the 'age' column into 20 equal-width bins.
    - Bins the 'hours-per-week' column into 10 equal-width bins.
    - Encodes specified categorical columns using an ordinal encoder.
    - Binarizes the 'race' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.
    - Reduces dimensionality using PCA with 2 components.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    education_mapping = [
        (['Preschool', '10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th'], 'dropout'),
        (['Some-college', 'Assoc-acdm', 'Assoc-voc'], 'CommunityCollege'),
        (['HS-Grad', 'HS-grad'], 'HighGrad'),
        (['Masters', 'Prof-school'], 'Masters')
    ]

    marital_status_mapping = {
        'Never-married': 'NotMarried',
        'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married',
        'Married-spouse-absent': 'NotMarried',
        'Separated': 'Separated',
        'Divorced': 'Separated',
        'Widowed': 'Widowed'
    }

    columns_to_encode = ['age', 'hours-per-week', 'sex', 'workclass', 'education', 'marital-status', 'occupation',
                         'relationship', 'native-country']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('map_education', FPP_ct.ValueMapper('education', education_mapping)),
        ('map_marital', FPP_ct.ValueMapper('marital-status', marital_status_mapping)),
        ('bin_age', FPP_ct.FeatureBinner(column='age', bins=20)),
        ('bin_hours', FPP_ct.FeatureBinner(column='hours-per-week', bins=10)),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
        # ('pca', PCA(n_components=2)),
    ])


def get_ac8_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC8 pipeline for preprocessing the dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Transforms the 'marital-status' column using a custom transformer.
    - One-hot encodes specified columns.
    - Binarizes the 'race' column separately.
    - Applies an ordinal encoder to remaining categorical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline.
    """
    columns_to_encode = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                         'native-country']

    # Create the pipeline
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('replace_marital_status', custom_transformers.MaritalStatusTransformer()),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_ac9_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC9 pipeline for preprocessing the dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Applies one-hot encoding to specified categorical columns.
    - Binarizes the 'race' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                         'native-country']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('ohe', FPP_ct.OneHotEncoder(columns=columns_to_encode, prefix_sep='=')),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_ac10_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the AC10 pipeline for preprocessing the dataset. The pipeline performs the following steps:
    - Imputes missing values in specific columns ('workclass', 'occupation', 'native-country') with their most frequent
    values.
    - Applies one-hot encoding to specified categorical columns.
    - Binarizes the 'race' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship',
                         'native-country']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('imputer', ColumnTransformer(transformers=[
            ('workclass_imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan),
             selector('workclass')),
            ('occupation_imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan),
             selector('occupation')),
            ('native-country_imputer', SimpleImputer(strategy='most_frequent', missing_values=np.nan),
             selector('native-country')),
        ], remainder='passthrough')),
        ('ohe', FPP_ct.OneHotEncoder(columns=columns_to_encode, prefix_sep='=')),
        ('race_encoder', FPP_ct.RaceBinarizer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
    ])


## bank dataset pipelines

def get_bm1_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the BM1 pipeline for preprocessing the bank dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Binarizes the 'age' column using a custom transformer.
    - Encodes specified categorical columns using an ordinal encoder.
    - Applies a custom transformer to encode the 'p_outcome' column.
    - Transforms the 'duration' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('age_transformer', FPP_ct.AgeBinarizer()),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('p_outcome_encoder', custom_transformers.POutcomeEncoder()),
        ('duration_transformer', FPP_ct.DurationTransformer()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_bm2_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the BM2 pipeline for preprocessing the bank dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Binarizes the 'age' column using a custom transformer.
    - Encodes specified categorical columns using an ordinal encoder.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                         'poutcome']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('age_transformer', FPP_ct.AgeBinarizer()),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
    ])


def get_bm3_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the BM3 pipeline for preprocessing the bank dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Binarizes the 'age' column using a custom transformer.
    - Encodes specified categorical columns using an ordinal encoder.
    - Transforms the 'duration' column using a quantile-based custom transformer.
    - Encodes the 'p_days' column using a custom transformer.
    - Transforms the 'euribor_3m' column using a custom transformer.
    - Encodes the 'p_outcome' column using a custom transformer.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('age_transformer', FPP_ct.AgeBinarizer()),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('duration_transformer', FPP_ct.DurationQuantileTransformer()),
        ('p_days_transformer', FPP_ct.PDaysEncoder()),
        ('euribor_3_m_transformer', custom_transformers.Euribor3mTransformer()),
        ('p_outcome_transformer', custom_transformers.POutcomeEncoder()),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_bm4_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the BM4 pipeline for preprocessing the bank dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Binarizes the 'age' column using a custom transformer.
    - Applies one-hot encoding to specified categorical columns.
    - Drops the 'duration' column from the dataset.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                         'poutcome']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('age_transformer', FPP_ct.AgeBinarizer()),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('drop_columns', FPP_ct.FeatureEliminator(['duration'])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_bm5_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the BM5 pipeline for preprocessing the bank dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Binarizes the 'age' column using a custom transformer.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome', 'default',
                         'housing', 'loan']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('age_transformer', FPP_ct.AgeBinarizer()),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_bm6_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the BM6 pipeline for preprocessing the bank dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Binarizes the 'age' column using a custom transformer.
    - Drops the 'marital' and 'education' columns from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['job', 'default', 'housing', 'contact', 'month', 'day_of_week', 'poutcome']
    columns_to_drop = ['marital', 'education']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('age_transformer', FPP_ct.AgeBinarizer()),
        ('drop_columns', FPP_ct.FeatureEliminator(columns_to_drop)),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_bm7_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the BM7 pipeline for preprocessing the bank dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Binarizes the 'age' column using a custom transformer.
    - Drops the 'pdays' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
                         'poutcome']
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('age_transformer', FPP_ct.AgeBinarizer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['pdays'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
    ])


def get_bm8_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the BM8 pipeline for preprocessing the bank dataset. The pipeline performs the following steps:
    - Drops rows with NaN values.
    - Binarizes the 'age' column using a custom transformer.
    - Drops the 'marital' and 'education' columns from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['job', 'default', 'housing', 'contact', 'month', 'day_of_week', 'poutcome']
    columns_to_drop = ['marital', 'education']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('age_transformer', FPP_ct.AgeBinarizer()),
        ('drop_columns', FPP_ct.FeatureEliminator(columns_to_drop)),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
    ])

    # german credit dataset pipelines


GERMAN_COLUMNS_TO_ENCODE = ['status', 'credit_history', 'purpose', 'savings', 'employment', 'other_debtors', 'property',
                            'installment_plans', 'housing', 'skill_level', 'telephone', 'foreign_worker']


def get_gc1_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC1 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('pca', PCA(n_components=3)),
    ])


def get_gc2_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC2 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Binarizes the 'age' feature into 26 bins using a custom transformer.
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.
    - Balances the dataset using SMOTE (Synthetic Minority Oversampling Technique).

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('age_transformer', FPP_ct.AgeBinarizer(n=26)),
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
    ])


def get_gc3_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC3 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Binarizes the 'age' feature into 26 bins using a custom transformer.
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('age_transformer', FPP_ct.AgeBinarizer(n=26)),
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('pca', PCA(n_components=16)),
    ])


def get_gc4_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC4 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Binarizes the 'age' feature into 26 bins using a custom transformer.
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Encodes specified categorical columns using an ordinal encoder.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('age_transformer', FPP_ct.AgeBinarizer(n=26)),
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_gc5_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC5 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Binarizes the 'age' feature into 26 bins using a custom transformer.
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('age_transformer', FPP_ct.AgeBinarizer(n=26)),
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_gc6_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC6 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Binarizes the 'age' feature into 26 bins using a custom transformer.
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Encodes specified categorical columns using an ordinal encoder.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('age_transformer', FPP_ct.AgeBinarizer(n=26)),
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=None)),
    ])


def get_gc7_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC7 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.
    - Reduces dimensionality using PCA with 3 components.
    - Selects the top 2 features using SelectKBest.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=3)),
        ('select_k_best', SelectKBest(k=2)),
    ])


def get_gc8_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC8 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Binarizes the 'age' feature into 26 bins using a custom transformer.
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('age_transformer', FPP_ct.AgeBinarizer(n=26)),
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_gc9_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC9 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Binarizes the 'age' feature into 26 bins using a custom transformer.
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Balances the dataset using SMOTE (Synthetic Minority Oversampling Technique).

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('age_transformer', FPP_ct.AgeBinarizer(n=26)),
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('smote', SMOTE(random_state=42)
         ),
    ])


def get_gc10_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the GC10 pipeline for preprocessing the German credit dataset. The pipeline performs the following steps:
    - Binarizes the 'age' feature into 26 bins using a custom transformer.
    - Transforms the 'sex' feature using a custom transformer.
    - Transforms the 'credit history' feature using a custom transformer.
    - Transforms the 'savings' feature using a custom transformer.
    - Transforms the 'employment' feature using a custom transformer.
    - Transforms the 'status' feature using a custom transformer.
    - Relabels the target variable from 1, 2 to 1, 0 using a custom function.
    - Drops the 'personal_status' column from the dataset.
    - Applies one-hot encoding to specified categorical columns.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Balances the dataset using AllKNN (All k-Nearest Neighbors).

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    return ImblearnPipelineWrapper.make_pipeline_with_default_steps([
        ('age_transformer', FPP_ct.AgeBinarizer(n=26)),
        ('sex_feature', FPP_ct.SexTransformer()),
        ('credit_history_transformer', FPP_ct.CreditHistoryTransformer()),
        ('savings_transformer', FPP_ct.SavingsTransformer()),
        ('employment_transformer', FPP_ct.EmploymentTransformer()),
        ('status_transformer', FPP_ct.StatusTransformer()),
        ('drop_columns', FPP_ct.FeatureEliminator(['personal_status'])),
        ('ohe_encoder', ColumnTransformer(transformers=[
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             SuperSetColumnSelector(GERMAN_COLUMNS_TO_ENCODE)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('allKNN', AllKNN()),
    ])


# titanic dataset pipelines

def get_tt1_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the TT1 pipeline for preprocessing the Titanic dataset. The pipeline performs the following steps:
    - Binarizes the 'Sex' column using a custom transformer.
    - Imputes missing values in the 'Age', 'Embarked', and 'Fare' columns using median and most frequent strategies.
    - Drops rows with NaN values using a custom function.
    - Adds a feature indicating whether a passenger is alone using a custom transformer.
    - Groups rare titles in the 'Title' column into a single category using a custom transformer.
    - Drops unnecessary columns such as 'PassengerId', 'Cabin', 'Ticket', 'Name', and 'Parch'.
    - Bins the 'Fare' column into 4 quantile-based bins.
    - Bins the 'Age' column into 5 equal-width bins.
    - Encodes specified categorical columns using an ordinal encoder.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_drop = ['PassengerId', 'Cabin', 'Ticket', 'Name', 'Parch']
    columns_to_encode = ['Embarked', 'Title', 'Age', 'Fare']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('sex_binarizer', FPP_ct.SexBinarizer()),
        ('imputer', ColumnTransformer(transformers=[
            ('age_imputer', SimpleImputer(strategy='median'), selector('Age')),
            ('Embarked_imputer', SimpleImputer(strategy='most_frequent'), selector('Embarked')),
            ('Fare_imputer', SimpleImputer(strategy='median'), selector('Fare')),
        ], remainder='passthrough')),
        ('drop_nan', FunctionSampler(func=drop_rows_with_any_nan, validate=False)),
        ('family_is_alone', FPP_ct.FamilyTransformer()),
        ('title_rare_grouper', FPP_ct.TitleRareGrouperTransformer(min_count=10, rare_label="Misc")),
        ('drop_columns', FPP_ct.FeatureEliminator(columns_to_drop)),
        ('bin_fare', FPP_ct.FeatureQuantileBinner(column='Fare', bins=4)),
        ('bin_age', FPP_ct.FeatureBinner(column='Age', bins=5)),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
    ])


def get_tt7_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the TT7 pipeline for preprocessing the Titanic dataset. The pipeline performs the following steps:
    - Binarizes the 'Sex' column using a custom transformer.
    - Imputes missing values in the 'Embarked' column with a constant value ('S').
    - Imputes missing values in the 'Fare' column using group-based median imputation.
    - Bins the 'Fare' column into 13 quantile-based bins.
    - Extracts the 'Deck' feature from the 'Cabin' column using a custom transformer.
    - Bins the 'Age' column using median-based binning.
    - Generates family size-related features using a custom transformer.
    - Counts the frequency of ticket numbers using a custom transformer.
    - Adds features related to marital status and family relationships using a custom transformer.
    - Encodes specified categorical columns using an ordinal encoder.
    - Drops unnecessary columns from the dataset.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales numerical features using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['Age', 'Embarked', 'Deck', 'Title', 'Family', 'Family_Size_Grouped']
    columns_to_drop = ['PassengerId', 'Cabin', 'Name', 'Ticket']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('sex_binarizer', FPP_ct.SexBinarizer()),
        ('imputer', ColumnTransformer(transformers=[
            ('embarked_imputer', SimpleImputer(strategy='constant', fill_value='S'), selector('Embarked')),
        ], remainder='passthrough')),
        ('fare_grp_imputer', FPP_ct.FareGroupMedianImputer()),
        ('fare_binner', FPP_ct.FareQuantileBinner(n_bins=13)),
        ('deck_extractor', FPP_ct.DeckTransformer()),
        ('age_binner', FPP_ct.AgeMedianBinner()),
        ('family_size_features', FPP_ct.FamilySizeFeatures()),
        ('ticket_freq', FPP_ct.TicketFrequencyCounter()),
        ('title_marry_family', FPP_ct.TitleIsMarriedAndFamily()),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('drop_columns', FPP_ct.FeatureEliminator(columns_to_drop)),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', StandardScaler()),
    ])


def get_tt8_pipeline() -> ImblearnPipelineWrapper:
    """
    Constructs the TT8 pipeline for preprocessing the Titanic dataset. The pipeline performs the following steps:
    - Binarizes the 'Sex' column using a custom transformer.
    - Fills missing values in the 'Embarked' column with 'C' using a custom transformer.
    - Imputes missing values in the 'Fare' column using targeted median imputation.
    - Extracts the 'Deck' feature from the 'Cabin' column using a custom transformer.
    - Groups family sizes into buckets using a custom transformer.
    - Bins the lengths of names into specified intervals using a custom transformer.
    - Extracts ticket numbers using a custom transformer.
    - Normalizes and extracts titles from names using a custom transformer.
    - Encodes specified categorical columns using an ordinal encoder.
    - Imputes missing values in the 'Age' column using a random forest-based imputer.
    - Drops unnecessary columns from the dataset.
    - Applies an ordinal encoder to remaining numerical columns as a fallback strategy.
    - Scales the 'Age' and 'Fare' columns using StandardScaler.

    Returns:
        ImblearnPipelineWrapper: The constructed pipeline with default preprocessing steps.
    """
    columns_to_encode = ['Embarked', 'Sex', "Title", "FsizeD", "NlengthD", 'Deck']

    return ImblearnPipelineWrapper.make_pipeline_with_default_steps(steps=[
        ('sex_binarizer', FPP_ct.SexBinarizer()),
        ('embarked_fill_c', FPP_ct.EmbarkedFillCTransformer()),
        ('fare_targeted_impute', FPP_ct.FareTargetedMedianImputer()),
        ('deck_from_cabin', FPP_ct.DeckFromCabinTT8()),
        ('family_size_bucket', FPP_ct.FamilySizeBucketTransformer()),
        ('name_len_bins', FPP_ct.NameLengthAndBinsTransformer(bins=(0, 20, 40, 57, 85))),
        ('ticket_number', FPP_ct.TicketNumberExtractorTransformer()),
        ('title_normalize', FPP_ct.TitleExtractorNormalizerTT8()),
        ('encoder', ColumnTransformer(transformers=[
            ('categorical_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             SuperSetColumnSelector(columns_to_encode)),
        ])),
        ('age_impute_rf', FPP_ct.AgeImputerRFTransformer(n_estimators=400, random_state=42)),
        ('drop_columns', FPP_ct.FeatureEliminator(['PassengerId', 'Cabin', 'Ticket', 'Name', 'Parch'])),
        ('default_encoder', ColumnTransformer(transformers=[
            ('fallback_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan),
             selector(dtype_exclude='number')),
        ])),
        ('scaler', ColumnTransformer(transformers=[
            ('age_fare_scaler', StandardScaler(),
             SuperSetColumnSelector(['Age', 'Fare'])),
        ])),
    ])


# compas dataset pipelines


def get_cp1_pipeline():
    """ SklearnPipelineWrapper for compas CP1 dataset.
    """
    impute1_and_onehot = SklearnPipelineWrapper([
        ('imputer1', SimpleImputer(strategy='mean')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    featurizer1 = ColumnTransformer(transformers=[
        ('impute1_and_onehot', impute1_and_onehot, ['is_recid'])
    ], remainder='passthrough')

    impute2_and_bin = SklearnPipelineWrapper([
        ('imputer2', SimpleImputer(strategy='mean')),
        ('discretizer', KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'))
    ])

    featurizer2 = ColumnTransformer(transformers=[
        ('impute2_and_bin', impute2_and_bin, ['age'])
    ], remainder='passthrough')

    pipeline = SklearnPipelineWrapper(steps=[
        ('recode', FPP_ct.RecodeTransformer()),
        ('filter', FPP_ct.FilterTransformer()),
        ('featurizer1', featurizer1),
        ('featurizer2', featurizer2),
        ('score_text_transformer', FPP_ct.ScoreTextTransformer())
    ])

    return pipeline

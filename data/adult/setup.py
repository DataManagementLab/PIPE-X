#!/usr/bin/env python3
""" Get and configure the UCI Adult dataset for usage.
"""

import os

import numpy as np
import pandas as pd

from data.utils import download_files

DATASET = 'adult'
URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
]  # data as csv + meta info
NAMES = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income',
]  # extracted manually from adult.names in this specific case
TARGET_NAME = 'income'  # name of the label attribute in the list above
NA_CHAR = '?'  # character (sequence) to indicate missing values in the csv files

# Define constants for Adult dataset
ADULT_CATEGORICAL_NOISY = ['race']  # categorical columns in which to introduce noise
ADULT_NUMERICAL_NOISY = ['education-num', 'capital-gain']  # numerical columns in which to introduce noise
ADULT_NUMERICAL_COLUMNS = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
]
ADULT_CATEGORICAL_COLUMNS = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]


def process_data(split: str, path: str, seed: int = None, numerical_only: bool = False,
                 drop_proba: float = None) -> None:
    """
    Process the data files for the Adult dataset.

    split (str): The split of the data to process (either 'train' or 'test').
    path (str): The path to save the processed data files to.
    numerical_only (bool): Whether to only keep numerical columns.
    drop_proba (float): The probability of dropping a value in the dataset.
    seed (int): The seed to use for the random number generator.
    :return: None
    """

    # read data file providing names manually, ignoring whitespace, set N/A value
    # the Adult test file is formatted badly; thus skipping the first line and removing dots and the end of the line
    data = pd.read_csv(os.path.join(path, 'adult', f'{DATASET}.{"data" if split == "train" else split}'),
                       names=NAMES,
                       skipinitialspace=True,
                       na_values=NA_CHAR,
                       skip_blank_lines=True,
                       skiprows=1 if split == 'test' else None  # skip line not containing sample in test file
                       )

    # remove . at end of each line in the test file
    data[TARGET_NAME] = data[TARGET_NAME].str.rstrip('.')

    # remove 'education' attribute, as same information is encoded in 'education_num' in numerical form
    # data.drop('education', axis=1, inplace=True)

    # add prefix to indicate target column
    data.rename(columns={TARGET_NAME: 'target_' + TARGET_NAME}, inplace=True)

    # map target column to binary values
    data = data.assign(target_income=pd.Categorical(data['target_income']).codes)

    if numerical_only:
        # remove columns of type object in example_data
        data = data.select_dtypes(exclude=['object'])

    if drop_proba is None:
        data = data.dropna()  # drop rows with missing values
    else:
        # with probability drop_proba replace values from important columns with NaN
        n = len(data)
        for i, column in enumerate(ADULT_CATEGORICAL_NOISY + ADULT_NUMERICAL_NOISY):
            rng = np.random.default_rng(seed + i)
            shuffled_index = rng.permutation(n)
            column_number = data.columns.get_loc(column)
            data.iloc[shuffled_index[:int(drop_proba * n)], column_number] = pd.NA

    # write to csv, including column names, to be read with pandas read_csv with default parameters
    if numerical_only:
        data.to_csv(os.path.join(path, 'adult_numerical', f'{DATASET}_numerical.{split}.csv'), index=False)
    elif drop_proba is not None:
        data.to_csv(os.path.join(path, 'adult_noisy', f'{DATASET}_{drop_proba:05f}.{split}.csv'), index=False)
    else:
        data.to_csv(os.path.join(path, 'adult', f'{DATASET}.{split}.csv'), index=False)
    # Note: We will assume that categorical columns are in strings


def adult_dataset(path: str):
    """
    Process the data files for the Adult dataset.

    path (str): The path to save the processed data files to.
    """
    # download source files from UCI repository
    download_files(URLS, os.path.join(path, 'adult'))

    # process training and test data for the Adult dataset
    process_data('train', path)
    process_data('test', path)


def adult_numerical_dataset(path: str):
    """
    Process the data files for the Adult dataset with only numerical columns.

    path (str): The path to save the processed data files to.
    """
    # download source files from UCI repository
    download_files(URLS, os.path.join(path, 'adult'))

    # process training and test data for a purely numerical version of the Adult dataset
    process_data('train', path, numerical_only=True)
    process_data('test', path, numerical_only=True)


def adult_noisy_dataset(probabilities: list, path: str):
    """
    Process the data files for the Adult dataset with missing values.

    probabilities (list): The probabilities of dropping a value in each generated dataset.
    path (str): The path to save the processed data files to.
    """
    # download source files from UCI repository
    download_files(URLS, os.path.join(path, 'adult'))

    # process training and test data for a noisy version of the Adult dataset
    for drop_proba in probabilities:
        process_data('train', path, seed=42, drop_proba=drop_proba)


def adult_noisy_target_dataset(drop_proba: float, path: str, seed: int = 42):
    """
      masks with a drop_proba-value  target_income-values to NaN.
    """

    download_files(URLS, os.path.join(path, 'adult'))
    process_data('train', path)
    process_data('test', path)

    # 2) for both splits mask target_income
    for split in ('train', 'test'):
        fn = os.path.join(path, 'adult', f'{DATASET}.{split}.csv')
        df = pd.read_csv(fn)
        n = len(df)
        rng = np.random.default_rng(seed)
        k = int(drop_proba * n)
        idx = rng.choice(df.index, size=k, replace=False)
        df.loc[idx, 'target_income'] = pd.NA

        out_dir = os.path.join(path, 'adult_noisy_target')
        os.makedirs(out_dir, exist_ok=True)
        out_fn = os.path.join(
                out_dir,
                f'{DATASET}_target_{drop_proba:05f}.{split}.csv'
        )
        df.to_csv(out_fn, index=False)


def adult_noisy_input_target_dataset(drop_proba: float, path: str, seed: int = 42):
    """
    masks with a drop_proba-value  target_income-values to NaN.
    """

    download_files(URLS, os.path.join(path, 'adult'))
    process_data('train', path)
    process_data('test', path)

    # 2) for both splits mask target_income and corresponding age and workclass input column
    for split in ('train', 'test'):
        fn = os.path.join(path, 'adult', f'{DATASET}.{split}.csv')
        df = pd.read_csv(fn)
        n = len(df)
        rng = np.random.default_rng(seed)
        k = int(drop_proba * n)
        idx = rng.choice(df.index, size=k, replace=False)
        df.loc[idx, 'marital-status'] = pd.NA
        df.loc[idx, 'workclass'] = pd.NA

        out_dir = os.path.join(path, 'adult_noisy_input_and_target')
        os.makedirs(out_dir, exist_ok=True)
        out_fn = os.path.join(
                out_dir,
                f'{DATASET}_input_and_target_{drop_proba:05f}.{split}.csv'
        )
        df.to_csv(out_fn, index=False)

"""
This module contains the DataSetWrapper class, which is a wrapper around a Dataset but includes
the raw data, fully processed data and all intermediate Data Snapshots needed for explanation.

It also contains a DataSnapshot class inheriting from sklearn Bunch to store train, test and targets per snapshot.
"""
import logging
from dataclasses import dataclass

import pandas as pd
from numpy.random import Generator
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from PIPE_X.metrics import Metrics


def split_x_y(data: DataFrame, target: str) -> tuple[DataFrame, DataFrame]:
    """
    Split data into features and target variable.

    :param data: DataFrame
    :param target: str, name of the target attribute
    :return: DataFrame input, DataFrame target
    """
    y = DataFrame(data[target])
    x = data.drop(columns=[target], inplace=False)
    return x, y


@dataclass
class DataSnapshot:
    """ Data Snapshot class to store training and test data and targets at a given step.
    """
    train_data: pd.DataFrame
    train_target: pd.DataFrame
    test_data: pd.DataFrame
    test_target: pd.DataFrame

    def get_sample(self, sample_id: int, test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ Return a single sample from the data.

        :param test: bool, whether to return a sample from the test data, defaults to False (training data)
        :param sample_id: int, ID of the sample to return
        :return: tuple, (DataFrame, DataFrame) with features and target
        """
        if test:
            data, target = self.test_data, self.test_target
        else:
            data, target = self.train_data, self.train_target

        # Handle case where sample_id doesn't exist (e.g., after undersampling)
        if sample_id not in data.index:
            # Return the first available sample instead
            # TODO: improve handling/prevent occurrence
            if len(data) > 0:
                first_id = data.index[0]
                return data.loc[[first_id]], target.loc[[first_id]]
            else:
                # If no data available, raise a more descriptive error
                raise ValueError(f"No samples available in {'test' if test else 'train'} data")

        return data.loc[[sample_id]], target.loc[[sample_id]]


class DataWrapper:
    """ Wrapper around several data snapshots generated from the same raw data and a given preprocessing pipeline.
    """

    snapshot_by_score: dict[Metrics, list[DataSnapshot]]
    target_attribute: str = None
    rng: Generator

    def __init__(self, data_path: DataFrame, rng, split_seed, test_size: float = 0.2, additional_features=None) -> None:
        """
        Initialize a data wrapper with raw data.

        :param data_path: str, path to the raw data csv file
        :param rng: Generator, random number generator
        :param split_seed: int, seed for splitting the data
        :param test_size: float, proportion of data to include in the test split
        :param additional_features: list, list of dummy columns to generate that will be filled by the pipeline
        """
        logging.debug("Instantiating DataWrapper...")

        self.rng = rng

        self.snapshot_by_score: dict[Metrics, list[DataSnapshot]] = {metric: [] for metric in Metrics}

        # load raw data from file
        self.load_data(data_path, split_seed, test_size, additional_features=additional_features)

    def load_data(self, dataframe: DataFrame, split_seed, test_size, additional_features=None) -> None:
        """
        Load raw data from csv file and split it into train and test sets.

        :param dataframe: DataFrame, raw data csv file
        :param split_seed: int, seed for splitting the data
        :param test_size: float, proportion of data to include in the test split
        :param additional_features: list, list of dummy columns to generate that will be filled by the pipeline
        """
        logging.debug(f"Load data: {dataframe}.")
        logging.info(f"Data loaded with shape {dataframe.shape}.")

        logging.debug("Identifying target attribute by column name starting with 'target_'")
        try:
            self.target_attribute = [column for column in dataframe.columns if column.startswith('target_')][0]
        except IndexError:
            logging.error("No target attribute found in the data.")
            raise ValueError("""No target attribute found in the data. 
                                Please provide a target attribute starting with 'target_'.""")
        logging.info("Target attribute is {self.target_attribute}.")

        # Create columns with dummy values for additional features if specified
        if additional_features is not None:
            for feature_name, feature_default_value in additional_features.items():
                dataframe[feature_name] = feature_default_value

        # Create columns with dummy values for additional features if specified
        if additional_features is not None:
            for feature_name, feature_default_value in additional_features.items():
                dataframe[feature_name] = feature_default_value

        logging.debug("Store raw data as first snapshot.")
        # Split data into features and target variable
        x, y = split_x_y(dataframe, self.target_attribute)

        logging.debug("Split data into train and test sets and store as data snapshot.")
        # split x and y into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size,
                                                            random_state=split_seed)

        # create data snapshot from raw data
        raw_snapshot = DataSnapshot(train_data=x_train,
                                    train_target=y_train,
                                    test_data=x_test,
                                    test_target=y_test)

        # reset snapshot lists and store raw as first snapshot for immediate impact snapshots
        self.snapshot_by_score[Metrics.IMMEDIATE] = [raw_snapshot]
        self.snapshot_by_score[Metrics.LEAVE_OUT] = []

    def get_raw(self) -> DataSnapshot:
        """
        Return raw data.

        :return: DataSnapshot
        """
        return self.snapshot_by_score[Metrics.IMMEDIATE][0]

    def get_fully_processed(self) -> DataSnapshot | None:
        """
        Return fully preprocessed data.

        :return: DataSnapshot
        """
        if len(self.snapshot_by_score[Metrics.IMMEDIATE]) == 1:
            logging.warning("Attempted to retrieve fully processed data, but only raw data is saved.")
            return None
        return self.snapshot_by_score[Metrics.IMMEDIATE][-1]

    def store_snapshots(self, train: list, train_target: list, test: list, test_target: list,
                        metric_name: Metrics) -> None:
        """
        Store snapshots of the data after each step in the pipeline.

        :param train: list of DataSnapshots, one after each step of a preprocessing pipeline (in order) for train data
        :param train_target: list of target labels snapshots for train data
        :param test: list of DataSnapshots, one after each step of a preprocessing pipeline (in order) for test data
        :param test_target: list of target labels snapshots for test data
        :param metric_name: Metrics, name of the metric these snapshots are for
        """
        if not isinstance(metric_name, Metrics):
            logging.error(f"Unknown snapshot list {metric_name}.")
            raise ValueError(f"Unknown snapshot list {metric_name}. Please use 'one_by_one' or 'leave_out'.")
        snapshots = []

        logging.info(f"Storing {len(train)} snapshots in {str(metric_name)} list.")
        for i in range(len(train)):
            snapshots.append(DataSnapshot(train_data=train[i],
                                          train_target=train_target[i],
                                          test_data=test[i],
                                          test_target=test_target[i]))

        self.snapshot_by_score[metric_name] = snapshots

    def is_valid(self) -> bool:
        """
        Check if data contains non-transformed categorical columns.

        :return: bool, whether the data is valid
        """
        for metric in Metrics:
            for snapshot in self.snapshot_by_score[metric]:
                if (snapshot.train_data.select_dtypes(include='object').shape[1] > 0 or
                        snapshot.train_target.select_dtypes(include='object').shape[1] > 0 or
                        snapshot.test_data.select_dtypes(include='object').shape[1] > 0 or
                        snapshot.test_target.select_dtypes(include='object').shape[1] > 0):
                    return False

        if len(self.snapshot_by_score[Metrics.IMMEDIATE]) > 0:
            logging.info(f"Number of one_by_one snapshots: {len(self.snapshot_by_score[Metrics.IMMEDIATE])}")
            logging.info(f"Features: train shape: {self.snapshot_by_score[Metrics.IMMEDIATE][-1].train_data.shape}; "
                         f"test shape: {self.snapshot_by_score[Metrics.IMMEDIATE][-1].test_data.shape}")
        logging.info("Snapshot shapes for 'leave_out':")
        if len(self.snapshot_by_score[Metrics.LEAVE_OUT]) > 0:
            logging.info(f"Number of leave_out snapshots: {len(self.snapshot_by_score[Metrics.LEAVE_OUT])}")
            logging.info(
                    f"Features: train shape: {self.snapshot_by_score[Metrics.LEAVE_OUT][-1].train_data.shape}; "
                    f"test shape: {self.snapshot_by_score[Metrics.LEAVE_OUT][-1].test_data.shape}")

        return True

"""
Utility functions for the PIPE_X explainer.

Includes functions for loading configurations, data, and setting up logging.
"""
import logging
import os
from datetime import datetime

import pandas as pd
import yaml


def setup_logging(model: str, sample_size: int, directory: str = "logs") -> None:
    """
    Sets up logging for the application.

    - Creates a `logs` directory if it does not exist.
    - Configures logging to write messages to a file named with the current timestamp.
    - Logs messages in the format: timestamp, log level, and message.
    """
    os.makedirs(directory, exist_ok=True)

    logging.basicConfig(
            format='%(asctime)s %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=os.path.join(directory, f'{model}_{sample_size}_{datetime.now():%Y-%m-%d_%H%M%S}.log')
    )


def generate_yaml(model_names: list[str],
                  datasets: list[tuple[str, str]],
                  metrics: list,
                  path: str,
                  filename: str,
                  experiments: list[dict],
                  additional_config_items: dict = None) -> None:
    """
    Generate a yaml file with the given data.

    :param model_names: list of model names
    :param datasets: list of tuples with dataset names and paths
    :param metrics: list of score names
    :param path: path to save the yaml file
    :param filename: name of the yaml file
    :param experiments: list of dictionaries with the following keys
        - name: name of the experiment
        - datasets: list of dataset names
        - pipeline: path to the pipeline pickle file
        - models: list of model names (optional)
    :param additional_config_items: additional items to add to the first section of the yaml file
    """

    data = dict([
        ('default_models', model_names),
        ('datasets', dict(datasets)),
        ('metrics', metrics)
    ])
    data.update(additional_config_items or {})

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'w') as file:
        yaml.dump_all([data] + experiments, file, default_flow_style=False)


def load_yaml_config(config_path: str) -> tuple[dict, list]:
    """
    Load the configuration file in YAML format.

    This function reads a YAML file containing experiment configurations and returns
    the general configuration and a list of experiment-specific configurations.

    :param config_path: str, Path to the YAML configuration file.
    :return: tuple, A tuple containing:
        - dict: General configuration dictionary (first document in the YAML file).
        - list: List of experiment-specific configurations (subsequent documents in the YAML file).
    """

    with open(config_path, 'r') as file:
        configs = list(yaml.safe_load_all(file))
        return configs[0], configs[1:]


def load_data_from_csv(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    :param path: str, The file path to the CSV file to be loaded.
    :return: pd.DataFrame, A pandas DataFrame containing the data from the CSV file.
    """
    return pd.read_csv(path)

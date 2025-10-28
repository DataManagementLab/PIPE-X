#!/usr/bin/env python3
"""
Example for the usage of PIPE-X

This script demonstrates the usage of the PIPE-X explainer.
It parses command line arguments and then calls the appropriate functions to load the data,
preprocess the data, train the model, and generate explanations.

Example:
$ python run.py --data_path ./data/adult/adult.train.csv
                --pipeline_path ./pipelines/pickles/sanity_check_destroyer/sanity_check_destroyer_minimal_0.500000.pkl
                --model DT
                --sample 4
"""

import argparse
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from PIPE_X.explainer import Explainer
from PIPE_X.pipeline import AbstractPipeline, unpickle_pipeline
from PIPE_X.utilities import setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the script.

    :return: An `argparse.Namespace` object containing the parsed arguments.
    Arguments:
        --data_path: Path to the input data file (CSV format).
        --pipeline_path: Path to the pickled sklearn pipeline.
        --model: Name of the classifier architecture to use.
        --sample: ID of the sample from the dataset to explain.
        --seed: (Optional) Seed for reproducibility.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to the data file.')
    parser.add_argument('--pipeline_path', help='path to pickled SKlearn Pipeline.')
    parser.add_argument('--model', help='Name of the classifier architecture to be used.')
    parser.add_argument('--sample', type=int, help='ID of the sample from the dataset to explain.')
    parser.add_argument('--seed', type=int, help='Seed for reproducibility.')
    return parser.parse_args()


def main(
        data: DataFrame,
        pipeline: AbstractPipeline,
        model: str,
        sample: int,
        seed: int = None
) -> None:
    """
    Example Process for using PIPE_X.

    :param data: DataFrame, The input data as a pandas DataFrame.
    :param pipeline: Pipeline, The preprocessing pipeline object.
    :param model: str, The name of the classifier architecture to use.
    :param sample: int, The ID of the sample from the dataset to explain.
    :param seed: int, (Optional) Seed for reproducibility.

    :return: A dictionary with impacts, vectors, and PCA results if `api` is True, otherwise None.
    """

    setup_logging(model, 1)
    logging.info(f''''Starting PIPE_X, with
                        data: {data},
                        pipeline: {pipeline},
                        classifier architecture: {model},
                        sample id: {sample},
                        seed: {seed}''')

    # Create seeded random number generator to pass on
    rng = np.random.default_rng(seed)

    # Instantiate, setup and run explainer
    logging.debug('Setting up Explainer...')
    explainer = Explainer(data, pipeline, model, rng)
    explainer.setup()

    logging.debug(f'Running Explainer on sample {sample}...')
    explainer.run(sample)

    logging.info('Done.')
    return None


if __name__ == '__main__':
    cli_arguments = parse_arguments()

    raw_data = pd.read_csv(cli_arguments.data_path)
    prep_pipeline = unpickle_pipeline(cli_arguments.pipeline_path)

    main(raw_data, prep_pipeline, cli_arguments.model, cli_arguments.sample, seed=cli_arguments.seed)

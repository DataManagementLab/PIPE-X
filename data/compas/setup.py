#!/usr/bin/env python3
""" Get and configure the compas dataset for usage.
"""
import os

import pandas as pd

from data.utils import download_files

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET = 'compas'
URLS = [
    'https://raw.githubusercontent.com/propublica/compas-analysis/refs/heads/master/compas-scores-two-years.csv']

TARGET_NAME = 'two_year_recid'


def download_and_prepare_compas_data() -> None:
    download_files(URLS, f"{FILE_DIR}")

    # read csv file
    data = pd.read_csv(f'{FILE_DIR}/compas-scores-two-years.csv')

    # add prefix to indicate target column
    data.rename(columns={TARGET_NAME: 'target_' + TARGET_NAME}, inplace=True)
    # write to csv, including column names, to be read with pandas read_csv with default parameters
    data.to_csv(f'{FILE_DIR}/{DATASET}.csv', index=False)
    os.remove(f'{FILE_DIR}/compas-scores-two-years.csv')


if __name__ == '__main__':
    # The following is only executed if this file is called directly.
    # Perform all necessary setup steps for the compas dataset as an exemplary workflow.

    # download the compas dataset from the UCI repository
    download_and_prepare_compas_data()

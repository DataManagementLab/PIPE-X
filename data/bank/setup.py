#!/usr/bin/env python3
""" Get and configure the Bank Marketing(Additional) dataset for Fair Pre-Processing Pipelines.

This script does the following:
    1. Downloads the Bank Marketing dataset from the UCI Machine Learning Repository.
    2. Unzips it's nested zip structure to fetch only the bank-additional-full.csv file.
    3. Processes only bank-additional-full.csv file to update delimiter, rename the target column and convert its values,
    4. Saves the processed data to a new CSV file bank.csv, and removes all intermediate files and folders.
"""

import os
import shutil

import pandas as pd

from data.utils import download_zip_file, extract_zip_file

FILE_DIR = os.path.abspath(os.path.dirname(__file__))
URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def fetch_bank_additional_data() -> None:
    # only download if file does not already exist
    if os.path.exists(f'{FILE_DIR}/bank.csv'):
        print("Bank marketing dataset already exists. Skipping download and processing.")
        return

    download_zip_file(URL, f'{FILE_DIR}/bank_marketing.zip')
    extract_zip_file(f'{FILE_DIR}/bank_marketing.zip', f'{FILE_DIR}', target_filename='bank-additional.zip')
    extract_zip_file(f'{FILE_DIR}/bank-additional.zip', f'{FILE_DIR}',
                     target_filename='bank-additional/bank-additional-full.csv')
    # Remove extracted zip and CSV after processing
    os.remove(f'{FILE_DIR}/bank_marketing.zip')
    os.remove(f'{FILE_DIR}/bank-additional.zip')

    # # Rename target column, binarize values and convert delimiter
    src_csv = f'{FILE_DIR}/bank-additional/bank-additional-full.csv'
    dst_csv = f'{FILE_DIR}/bank.csv'

    df = pd.read_csv(src_csv, sep=';', na_values='unknown')
    df.rename(columns={'y': 'target_y'}, inplace=True)
    df['target_y'] = df['target_y'].map({'yes': 1, 'no': 0})

    df.to_csv(dst_csv, index=False, sep=',')
    os.remove(src_csv)

    shutil.rmtree(f'{FILE_DIR}/bank-additional', ignore_errors=True)


if __name__ == '__main__':
    # The following is only executed if this file is called directly.
    # Perform all necessary setup steps for the bank marketing dataset as an exemplary workflow.
    fetch_bank_additional_data()

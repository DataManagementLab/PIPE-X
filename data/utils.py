""" Utility functions for loading data, to avoid code duplication.
"""
import os
import zipfile

import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from ucimlrepo.dotdict import dotdict


def download_zip_file(url: str, download_path: str) -> str:
    """
    Download a zip file from a URL to a specified path.
    :param url: URL of the zip file
    :param download_path: Path to save the downloaded zip file
    :return: Path to the downloaded zip file"""
    print("Downloading...")
    r = requests.get(url, stream=True)
    with open(download_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {download_path}")
    return download_path


def extract_zip_file(archive_path: str, extract_path: str, target_filename: str) -> str:
    """
    Extract a specific CSV file from a zip archive.
    :param archive_path: Path to the zip archive
    :param extract_path: Path to extract the contents to
    :param target_filename: Name of the target CSV file to extract
    :return: Path to the extracted CSV file
    """
    os.makedirs(extract_path, exist_ok=True)
    extracted_csv = None

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, 'r') as z:
            print(z.namelist())
            if target_filename in z.namelist():
                # Extract to temp path
                z.extract(target_filename, extract_path)
                final_path = os.path.join(extract_path, target_filename)

                extracted_csv = final_path

    if not extracted_csv:
        raise FileNotFoundError("CSV file not found in the archive.")

    print(f"CSV file located: {extracted_csv}")
    return extracted_csv


def download_files(urls: list, path: str, replace: bool = False) -> None:
    """
    Download files from the given paths to the current working directory, if they do not exist yet.

    :param urls: List of Strings containing URLs of desired files
    :param path: Path to save the downloaded files to
    :param replace: Boolean indicating whether to replace existing files
    :return: None
    """

    for url in urls:
        name = os.path.basename(url)
        if not replace and os.path.exists(os.path.join(path, name)):
            continue
        response = requests.get(url)
        with open(os.path.join(path, name), 'w') as downloaded_file:
            downloaded_file.write(response.content.decode('utf-8'))


def process_uci_data(name: str, dataset: dotdict, path: str, drop_na: bool = False, column_names=None) -> None:
    """
    Process a dataset from the UCI machine learning repository.

    check that there is a single target column and mark with prefix
    save as csv

    :param name: name of the dataset
    :param dataset: dotdict containing the dataset
    :param path: path to save the processed data to
    :param drop_na: bool, whether to drop rows with missing values
    :param column_names: list, Optional list of verbose column names
    :return: None
    """
    # check if more than one target and raise error if so
    if dataset.data.targets.columns.size > 1:
        raise ValueError(
                'More than one target column detected. This setup script does not support multiple targets.')

    # Check if the values are not numerical and encode them numerically
    if not pd.api.types.is_numeric_dtype(dataset.data.original[dataset.data.targets.columns[0]]):
        dataset.data.original[dataset.data.targets.columns[0]] = LabelEncoder().fit_transform(
                dataset.data.original[dataset.data.targets.columns[0]])

    # Check if target variable is encoded [0, 1] and convert if necessary
    unique_values = dataset.data.original[dataset.data.targets.columns[0]].unique()
    if set(unique_values) == {1, 2}:
        dataset.data.original[dataset.data.targets.columns[0]] = dataset.data.original[
            dataset.data.targets.columns[0]].map({1: 0, 2: 1})

    # add prefix to indicate target column
    dataset.data.original.rename(columns={dataset.data.targets.columns[0]: f'target_{dataset.data.targets.columns[0]}'},
                                 inplace=True)

    if drop_na:
        # drop rows with missing values
        dataset.data.original.dropna(inplace=True)

    # Ensure folder exists
    os.makedirs(os.path.join(path, name), exist_ok=True)

    # rename columns if column names are provided
    if column_names:
        dataset.data.original.columns = column_names

    # write to csv to be read with pandas read_csv with default parameters
    dataset.data.original.to_csv(os.path.join(path, name, f'{name}.csv'), index=False)

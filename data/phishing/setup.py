#!/usr/bin/env python3
""" Get and configure the Phishing dataset for usage.
"""

from ucimlrepo import fetch_ucirepo

from data.utils import process_uci_data

DATASET = 'phishing'
UCI_ID = 327


def process_phishing_data(path='data') -> None:
    """
    Process the Phishing dataset from the UCI machine learning repository.

    :param path: path to save the processed data to
    :return: None
    """
    phishing = fetch_ucirepo(id=UCI_ID)
    process_uci_data(DATASET, phishing, path=path, drop_na=True)


if __name__ == '__main__':
    # The following is only executed if this file is called directly.
    # Perform all necessary setup steps for the german credit dataset as an exemplary workflow.

    # download the german credit dataset from the UCI repository
    process_phishing_data()

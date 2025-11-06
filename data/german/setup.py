#!/usr/bin/env python3
""" Get and configure the German Credit dataset for usage.
"""

from ucimlrepo import fetch_ucirepo

from data.utils import process_uci_data

EXAMPLE_DATASET = 'german'
EXAMPLE_UCI_ID = 144


def german_dataset(path: str):
    """ Perform all necessary setup steps for the german credit dataset as an exemplary workflow.
    """

    # download the german credit dataset from the UCI repository
    german = fetch_ucirepo(id=EXAMPLE_UCI_ID)

    # process and prepare for use, save as csv
    process_uci_data(EXAMPLE_DATASET, german, path, column_names=['status', 'month', 'credit_history',
                                                                  'purpose', 'credit_amount', 'savings', 'employment',
                                                                  'investment_as_income_percentage', 'personal_status',
                                                                  'other_debtors', 'residence_since', 'property', 'age',
                                                                  'installment_plans', 'housing', 'number_of_credits',
                                                                  'skill_level', 'people_liable_for', 'telephone',
                                                                  'foreign_worker', 'target_credit'])

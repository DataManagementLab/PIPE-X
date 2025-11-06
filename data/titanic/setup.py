#!/usr/bin/env python3
"""Download and prepare the Titanic dataset (OpenML 40945) for our pipelines."""
from pathlib import Path

from openml.datasets import get_dataset

EXAMPLE_DATASET = 'titanic'
EXAMPLE_OPENML_ID = 40945
EXAMPLE_TARGET_NAME = 'survived'


def main():
    # target path: ../data/titanic/titanic.csv (relative to this file)
    here = Path(__file__).resolve().parent
    out_dir = here
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'{EXAMPLE_DATASET}.csv'

    # fetch data
    ds = get_dataset(EXAMPLE_OPENML_ID, download_data=False,
                     download_qualities=False, download_features_meta_data=False)
    df, *_ = ds.get_data()  # pandas DataFrame

    # rename target -> 'target_survived'
    if EXAMPLE_TARGET_NAME in df.columns:
        df.rename(columns={EXAMPLE_TARGET_NAME: f'target_{EXAMPLE_TARGET_NAME}'}, inplace=True)

    # normalize common column names
    rename_map = {
        'sex': 'Sex',
        'embarked': 'Embarked',
        'fare': 'Fare',
        'pclass': 'Pclass',
        'parch': 'Parch',
        'sibsp': 'SibSp',
        'cabin': 'Cabin',
        'age': 'Age',
        'ticket': 'Ticket',
        'name': 'Name',
    }
    have = set(df.columns.str.lower())
    apply_map = {k: v for k, v in rename_map.items() if k in have}
    if apply_map:
        # build actual mapping with the original spellings present in df
        real_map = {}
        lower2orig = {c.lower(): c for c in df.columns}
        for k_lower, v in apply_map.items():
            real_map[lower2orig[k_lower]] = v
        df.rename(columns=real_map, inplace=True)

    # match Titanic kaggle competition
    # ensure PassengerId exists and fill with 1..N if not present
    if 'PassengerId' not in df.columns:
        df.insert(0, 'PassengerId', range(1, len(df) + 1))

    # drop columns boat, body, home.dest if present
    for col in ['boat', 'body', 'home.dest']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # write
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with shape {df.shape} and columns: {list(df.columns)[:8]}...")


if __name__ == '__main__':
    main()

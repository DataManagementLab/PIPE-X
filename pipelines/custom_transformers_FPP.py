"""
Custom transformers for data preprocessing in PIPE_X pipelines.
Each transformer inherits from CustomTransformer and implements fit() and transform() methods.
These transformers handle tasks such as imputation, scaling, encoding,
and feature engineering.
"""
import logging
import re
import string

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder

from PIPE_X.transformer import PIPEXTransformer


# adult dataset transformers

class MaritalStatusBinarizer(PIPEXTransformer):
    """ Replaces 'marital-status' with binary values: Married=0, Single=1
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if 'marital-status' in X.columns:
            married = {'Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'}
            single = {'Never-married', 'Divorced', 'Separated', 'Widowed'}

            X['marital-status'] = (
                X['marital-status']
                .replace(list(married), 'Married')
                .replace(list(single), 'Single')
                .map({'Married': 0.0, 'Single': 1.0})
            )
        return X


class RaceBinarizer(PIPEXTransformer):
    """ Replaces 'race' with binary values: White=1, All other=0
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if 'race' in X.columns:
            X['race'] = (X['race'].map(lambda value: 1.0 if value == 'White' else 0.0))
        return X


class FeatureEliminator(PIPEXTransformer):
    """
    Drop a given list of columns/features from the data
    """

    def __init__(self, columns, errors='ignore'):
        self.columns = [columns] if isinstance(columns, str) else list(columns)
        self.errors = errors

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        before_shape = X_out.shape

        X_out.drop(columns=self.columns, inplace=True)

        after_shape = X_out.shape
        logging.info(f"dropped {len(self.columns)} cols | shape {before_shape} -> {after_shape}")

        return X_out


class OneHotEncoder(PIPEXTransformer):
    """
    Performs one-hot encoding on selected columns.
    Ensures consistent column output across fit/transform with fallback for missing categories.
    """

    def __init__(self, columns, prefix_sep="="):
        self.columns = list(columns)
        self.prefix_sep = prefix_sep
        self.categories_ = None
        self.columns_out_ = None

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        X_enc = pd.get_dummies(X[self.columns], prefix_sep=self.prefix_sep)
        self.columns_out_ = sorted(list(X_enc.columns.map(str)))
        return self

    def transform(self, X: pd.DataFrame):
        x_out = X.copy()
        x_ohe = pd.get_dummies(x_out[self.columns], prefix_sep=self.prefix_sep)

        # Reindex to enforce same column schema
        x_ohe = x_ohe.reindex(columns=self.columns_out_, fill_value=0.0)

        # Drop original categorical columns
        x_out = x_out.drop(columns=self.columns, errors='ignore')

        # Add encoded columns
        x_out = pd.concat([x_out, x_ohe], axis=1)
        x_out.columns = x_out.columns.map(str)
        x_out = x_out.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        logging.info(f'encoded {len(self.columns)} cols -> {len(self.columns_out_)} new cols')
        return x_out


class ValueMapper(PIPEXTransformer):
    """ Maps values in a specified column using a provided dictionary or list of replacements.
    """

    def __init__(self, column: str, mappings: dict | list[tuple[list[str], str]]):
        self.column = column
        self.mappings = mappings

    def transform(self, X):
        X = X.copy()
        if self.column not in X.columns:
            return X

        if isinstance(self.mappings, dict):
            # Direct dict mapping
            X[self.column] = X[self.column].replace(self.mappings)
        elif isinstance(self.mappings, list):
            # List of ([values], new_value)
            for values, new_value in self.mappings:
                X[self.column] = X[self.column].replace(values, new_value)
        else:
            raise ValueError("Mappings must be dict or list of (list, str) tuples.")

        return X


class FeatureBinner(PIPEXTransformer):
    """
    Splits a single continuous column into discrete intervals ("bins").
    Uses pandas.cut to replace numerical values with interval categories.
    """

    def __init__(self, column, bins):
        self.column = column
        self.bins = bins

    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = pd.cut(X[self.column], bins=self.bins)
        return X


# bank dataset transformers


class AgeBinarizer(PIPEXTransformer):
    """ Custom transformer that replaces age values with 0 or 1.
    """

    def __init__(self, n=25):
        super().__init__()
        self.n = n

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Replace age values with 0 or 1.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :return: numpy array of shape [n_samples, n_features_new], Transformed array.
        """
        X = DataFrame(X)
        X['age'] = X['age'].apply(lambda x: np.float16(x >= self.n))
        return X


class DurationTransformer(PIPEXTransformer):
    """ Custom transformer that replaces duration values with 1, 2, 3, 4, or 5.
    """

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> DataFrame:
        """
        Replace duration values with 1, 2, 3, 4, or 5.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: numpy array of shape [n_samples, n_features_new], Transformed array.
        """
        X = DataFrame(X)

        X.loc[X['duration'] <= 102, 'duration'] = 1
        X.loc[(X['duration'] > 102) & (X['duration'] <= 180), 'duration'] = 2
        X.loc[(X['duration'] > 180) & (X['duration'] <= 319), 'duration'] = 3
        X.loc[(X['duration'] > 319) & (X['duration'] <= 644.5), 'duration'] = 4
        X.loc[X['duration'] > 644.5, 'duration'] = 5

        return X


class DurationQuantileTransformer(PIPEXTransformer):
    """ Custom transformer that replaces duration values with 1, 2, 3, 4.
    """

    def transform(self, X: np.ndarray) -> DataFrame:
        """
        Replace duration values with 1, 2, 3, 4.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :return: numpy array of shape [n_samples, n_features_new], Transformed array.
        """
        X = DataFrame(X)

        q1 = X['duration'].quantile(0.25)
        q2 = X['duration'].quantile(0.50)
        q3 = X['duration'].quantile(0.75)
        X.loc[(X['duration'] <= q1), 'duration'] = 1
        X.loc[(X['duration'] > q1) & (X['duration'] <= q2), 'duration'] = 2
        X.loc[(X['duration'] > q2) & (X['duration'] <= q3), 'duration'] = 3
        X.loc[(X['duration'] > q3), 'duration'] = 4

        return X


class POutcomeEncoder(PIPEXTransformer):
    """ Custom transformer that replaces POutcome values with 1, 2, or 3.
    """

    def transform(self, X: np.ndarray) -> DataFrame:
        """ Replace POutcome values with 1, 2, or 3.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :return: numpy array of shape [n_samples, n_features_new], Transformed array.
        """
        X = DataFrame(X)

        X['poutcome'] = X['poutcome'].map({'nonexistent': 1, 'failure': 2, 'success': 3})

        return X


class PDaysEncoder(PIPEXTransformer):
    """ Custom transformer that replaces pdays values with 1, 2, 3, or 4.
    """

    def transform(self, X: np.ndarray) -> DataFrame:
        """ Replace pdays values with 1, 2, 3, or 4.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :return: numpy array of shape [n_samples, n_features_new], Transformed array.
        """
        X = DataFrame(X)

        X.loc[(X['pdays'] == 999), 'pdays'] = 1
        X.loc[(X['pdays'] > 0) & (X['pdays'] <= 10), 'pdays'] = 2
        X.loc[(X['pdays'] > 10) & (X['pdays'] <= 20), 'pdays'] = 3
        X.loc[(X['pdays'] > 20) & (X['pdays'] != 999), 'pdays'] = 4

        return X


class Euribor3mTransformer(PIPEXTransformer):
    """ Custom transformer that replaces euribor3m values with 1, 2, 3, 4, or 5.
    """

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> DataFrame:
        """ Replace euribor3m values with 1, 2, 3, 4, or 5.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: numpy array of shape [n_samples, n_features_new], Transformed array.
        """
        X = DataFrame(X)

        X.loc[(X['euribor3m'] < 1), 'euribor3m'] = 1
        X.loc[(X['euribor3m'] > 1) & (X['euribor3m'] <= 2), 'euribor3m'] = 2
        X.loc[(X['euribor3m'] > 2) & (X['euribor3m'] <= 3), 'euribor3m'] = 3
        X.loc[(X['euribor3m'] > 3) & (X['euribor3m'] <= 4), 'euribor3m'] = 4
        X.loc[(X['euribor3m'] > 4), 'euribor3m'] = 5

        return X


# german dataset transformers

class SexTransformer(PIPEXTransformer):

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        status_map = {'A91': 1, 'A93': 1, 'A94': 1, 'A92': 0, 'A95': 0}
        X['sex'] = X['personal_status'].map(status_map)
        return X


class CreditHistoryTransformer(PIPEXTransformer):

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        feature_map = {'A30': 'None/Paid',
                       'A31': 'None/Paid',
                       'A32': 'None/Paid',
                       'A34': 'Other',
                       }
        X['credit_history'] = X['credit_history'].apply(lambda x: feature_map.get(x, 'NA'))

        return X


class EmploymentTransformer(PIPEXTransformer):

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        feature_map = {'A71': 'Unemployed',
                       'A72': '1-4 years',
                       'A73': '1-4 years',
                       'A74': '4+ years',
                       'A75': '4+ years',
                       }
        X['employment'] = X['employment'].apply(lambda x: feature_map.get(x, 'NA'))

        return X


class SavingsTransformer(PIPEXTransformer):

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        feature_map = {'A61': '<500',
                       'A62': '<500',
                       'A63': '500+',
                       'A64': '500+',
                       'A65': 'Unknown/None',
                       }
        X['savings'] = X['savings'].apply(lambda x: feature_map.get(x, 'NA'))

        return X


class StatusTransformer(PIPEXTransformer):

    def transform(self, X: DataFrame) -> DataFrame:
        X = X.copy()
        feature_map = {'A11': '<200',
                       'A12': '<200',
                       'A13': '200+',
                       'A14': 'None',
                       }
        X['status'] = X['status'].apply(lambda x: feature_map.get(x, 'NA'))

        return X


class CreditEncoder(PIPEXTransformer):

    def transform(self, X) -> DataFrame:
        X = DataFrame(X)
        X['target_credit'] = X['target_credit'].replace([1, 2], [1, 0])
        return X


# titanic dataset transformers

class SexBinarizer(PIPEXTransformer):
    """
    Make 'Sex' (or 'sex') numeric 0.0/1.0 early so even the raw snapshot is safe.
    female→0.0, male→1.0. Leaves other values as NaN.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Convert 'Sex'/'sex' to 0.0/1.0; leave unknowns as NaN.

        :param X: DataFrame or array-like.
        :return:  DataFrame with converted column(s).
        """
        df = pd.DataFrame(X).copy()
        for col in ['Sex', 'sex']:
            if col in df.columns:
                s = df[col].astype(str).str.lower()
                df[col] = s.map({'female': 0.0, 'male': 1.0})
        return df


class FamilyTransformer(PIPEXTransformer):
    """
    Add:
      - FamilySize = SibSp + Parch + 1
      - IsAlone = 1 if FamilySize == 1 else 0
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Compute FamilySize and IsAlone. Missing SibSp/Parch as 0.

        :param X: DataFrame or array-like with optional 'SibSp' and 'Parch'.
        :return:  DataFrame with added 'FamilySize' and 'IsAlone' columns.
        """
        df = pd.DataFrame(X).copy()
        sib = pd.to_numeric(df.get('SibSp', 0), errors='coerce').fillna(0)
        par = pd.to_numeric(df.get('Parch', 0), errors='coerce').fillna(0)
        df['FamilySize'] = sib + par + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        return df


class TitleRareGrouperTransformer(PIPEXTransformer):
    """
    Title from 'Name'; collapse rare titles (count < min_count) to rare_label.
    """

    def __init__(self, min_count: int = 10, rare_label: str = "Misc"):
        self.min_count = min_count
        self.rare_label = rare_label
        self._rare_titles = set()

    @staticmethod
    def _extract_title(s: str) -> str:
        # 'Last, Title. First' -> 'Title'
        return str(s).split(', ', 1)[-1].split('.', 1)[0] if ',' in str(s) and '.' in str(s) else ""

    def fit(self, X, y=None) -> 'TitleRareGrouperTransformer':
        """
        Learn which titles are rare in the training data (count < min_count).

        :param X: DataFrame or array-like; uses 'Name' if present.
        :param y: Optional target; ignored.
        :return:  self
        """
        df = pd.DataFrame(X)
        if 'Name' in df.columns:
            titles = df['Name'].astype(str).apply(self._extract_title)
            vc = titles.value_counts(dropna=False)
            self._rare_titles = set(vc[vc < self.min_count].index.tolist())
        else:
            self._rare_titles = set()
        return self

    def transform(self, X):
        """
        Add normalized 'Title' with rare titles collapsed to rare_label.

        :param X: DataFrame or array-like; uses 'Name' if present.
        :return:  DataFrame with added 'Title' column.
        """
        df = pd.DataFrame(X).copy()
        if 'Name' in df.columns:
            title = df['Name'].astype(str).apply(self._extract_title)
            # collapse rare
            if self._rare_titles:
                title = title.where(~title.isin(self._rare_titles), self.rare_label)
            df['Title'] = title
        else:
            df['Title'] = self.rare_label  # stable default
        return df


class FeatureQuantileBinner(PIPEXTransformer):
    """
    Splits a single continuous column into discrete intervals ("bins").
    Uses pandas.qcut to replace numerical values with interval categories based on quantiles.
    """

    def __init__(self, column, bins):
        self.column = column
        self.bins = bins

    def transform(self, X):
        X = X.copy()
        if self.column in X.columns:
            X[self.column] = pd.qcut(X[self.column], self.bins)
        return X


class FareGroupMedianImputer(PIPEXTransformer):
    """
    Fit: store group medians of Fare by (PClass,Parch,SibSp).
    Transform: fill Fare with group median; fallback to global median.
    """
    grp_ = None

    def fit(self, X, y=None):
        """
        Learn group medians of Fare.

        :param X: DataFrame with PClass, Parch, SibSp, Fare.
        :param y: target values
        :return:  self
        """
        df = pd.DataFrame(X).copy()
        if {'Pclass', 'Parch', 'SibSp', 'Fare'}.issubset(df.columns):
            self.grp_ = (
                df.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()
                .rename('Fare_median')
            )
        else:
            self.grp_ = None
        return self

    def transform(self, X):
        """
        Fill 'Fare' with learned group medians; then global median.

        :param X: DataFrame to impute.
        :return:  DataFrame with 'Fare' imputed.
        """
        df = pd.DataFrame(X).copy()
        if self.grp_ is not None:
            df = df.merge(self.grp_, how='left', left_on=['Pclass', 'Parch', 'SibSp'], right_index=True)
            df['Fare'] = df['Fare'].fillna(df['Fare_median'])
            df.drop(columns=['Fare_median'], inplace=True)
        if 'Fare' in df.columns:
            df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        return df


class FareQuantileBinner(PIPEXTransformer):
    """Bin Fare into N quantile bins -> numeric codes 1..N (float)"""

    def __init__(self, n_bins=13):
        self.n_bins = n_bins

    def transform(self, X):
        """
        Replace 'Fare' with quantile-bin codes.

        :param X: DataFrame or array-like.
        :return:  DataFrame with binned 'Fare' as float.
        """
        df = pd.DataFrame(X).copy()
        if 'Fare' in df.columns:
            try:
                df['Fare'] = pd.qcut(df['Fare'], self.n_bins, labels=list(range(1, self.n_bins + 1)))
            except Exception:
                ranks = df['Fare'].rank(method='average')
                df['Fare'] = pd.qcut(ranks, self.n_bins, labels=list(range(1, self.n_bins + 1)))
            df['Fare'] = df['Fare'].astype(float)
        return df


class DeckTransformer(PIPEXTransformer):
    """
    Cabin → Deck first letter, map T→A, collapse:
    {A,B,C}→'ABC', {D,E}→'DE', {F,G}→'FG', fill 'M'. Leaves as string (categorical).
    """
    age_median_: float = None

    def transform(self, X):
        """
        Create 'Deck' based on 'Cabin'.

        :param X: DataFrame or array-like.
        :return:  DataFrame with new 'Deck' column.
        """
        df = pd.DataFrame(X).copy()
        if 'Cabin' in df.columns:
            deck = df['Cabin'].astype(str).str[0]
            deck = deck.where(df['Cabin'].notna(), 'M')
            deck = deck.replace({'T': 'A'})
            deck = deck.replace({'A': 'ABC', 'B': 'ABC', 'C': 'ABC', 'D': 'DE', 'E': 'DE', 'F': 'FG', 'G': 'FG'})
            df['Deck'] = deck.astype(str)
        else:
            df['Deck'] = 'M'
        return df


class AgeMedianBinner(PIPEXTransformer):
    """Fill Age median and bin into 10 equal-width bins -> numeric codes 0..9 (float)."""

    age_median_: float = None

    def fit(self, X, y=None):
        """
        Store the training median of 'Age' (if available).

        :param X: DataFrame to read median from.
        :param y: target values
        :return:  self
        """
        df = pd.DataFrame(X).copy()
        self.age_median_ = df['Age'].median() if 'Age' in df.columns else None
        return self

    def transform(self, X):
        """
        Impute 'Age' by median, then bin into 10 equal-width bins.

        :param X: DataFrame to transform.
        :return:  DataFrame with binned 'Age'.
        """
        df = pd.DataFrame(X).copy()
        if 'Age' in df.columns:
            med = self.age_median_ if self.age_median_ is not None else df['Age'].median()
            df['Age'] = df['Age'].fillna(med)
            df['Age'] = pd.cut(df['Age'].astype(float), 10, labels=False).astype(float)
        return df


class FamilySizeFeatures(PIPEXTransformer):
    """
    Family_Size = SibSp + Parch + 1 (numeric)
    Family_Size_Grouped = {'Alone','Small','Medium','Large'} (string)
    """

    def transform(self, X):
        """
        Compute family size and a simple grouped label.

        :param X: DataFrame with SibSp/Parch (if available).
        :return:  DataFrame with 'Family_Size' and 'Family_Size_Grouped'.
        """
        df = pd.DataFrame(X).copy()
        if {'SibSp', 'Parch'}.issubset(df.columns):
            df['Family_Size'] = df['SibSp'].fillna(0) + df['Parch'].fillna(0) + 1
            fam_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small',
                       5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
            df['Family_Size_Grouped'] = df['Family_Size'].map(fam_map).fillna('Small')
        else:
            df['Family_Size'] = np.nan
            df['Family_Size_Grouped'] = 'Small'
        return df


class TicketFrequencyCounter(PIPEXTransformer):
    """Ticket_Frequency = count per Ticket (numeric). If Ticket missing, set 1."""

    def transform(self, X):
        """
        Count frequency per ticket string.

        :param X: DataFrame or array-like.
        :return:  DataFrame with 'Ticket_Frequency'.
        """
        df = pd.DataFrame(X).copy()
        if 'Ticket' in df.columns:
            df['Ticket_Frequency'] = df.groupby('Ticket')['Ticket'].transform('count')
        else:
            df['Ticket_Frequency'] = 1
        return df


class TitleIsMarriedAndFamily(PIPEXTransformer):
    """
    From Name: Title (string, consolidated), Is_Married (0/1), Family (surname string).
    If Name absent, defaults: Title='Unknown', Is_Married=0, Family='Unknown'
    """

    def transform(self, X):
        """
        Extract title, married flag, and family surname from 'Name'.

        :param X: DataFrame or array-like.
        :return:  DataFrame with 'Title', 'Is_Married', 'Family'.
        """
        df = pd.DataFrame(X).copy()
        if 'Name' in df.columns:
            title = (
                df['Name'].astype(str)
                .str.split(', ', expand=True)[1]
                .str.split('.', expand=True)[0]
            )
            df['Title'] = title
            df['Is_Married'] = (df['Title'] == 'Mrs').astype(int)
            df['Title'] = df['Title'].replace(
                    ['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms'
            )
            df['Title'] = df['Title'].replace(
                    ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy'
            )
            fam = df['Name'].astype(str).str.split('(', n=1, expand=True)[0].str.split(',', n=1, expand=True)[0]
            for c in string.punctuation:
                fam = fam.str.replace(c, '', regex=False).str.strip()
            df['Family'] = fam
        else:
            df['Title'] = 'Unknown'
            df['Is_Married'] = 0
            df['Family'] = 'Unknown'
        # ensure these stay categorical
        df['Title'] = df['Title'].astype(str)
        df['Family'] = df['Family'].astype(str)
        return df


class EmbarkedFillCTransformer(PIPEXTransformer):
    """
    Fill missing 'Embarked' with 'C' (kept as string).
    """

    def transform(self, X):
        """
        Fill missing 'Embarked' with 'C' (if column exists).

        :param X: DataFrame or array-like.
        :return:  DataFrame with 'Embarked' imputed to 'C'.
        """
        df = pd.DataFrame(X).copy()
        if 'Embarked' in df.columns:
            df['Embarked'] = df['Embarked'].fillna('C')
        return df


class FareTargetedMedianImputer(PIPEXTransformer):
    """
    Impute 'Fare' in two steps:
      1) For (PClass==3 & Embarked=='S') fill with that group's median (from train).
      2) Fill remaining NaNs with global Fare median (from train).
    """

    def __init__(self):
        self.median_fare_pclass3_S_ = None
        self.fare_global_median_ = None

    def fit(self, X, y=None) -> 'FareTargetedMedianImputer':
        """
        Learn medians for targeted and global 'Fare' imputation from training data.

        :param X: DataFrame with columns 'Fare', 'Pclass', 'Embarked'.
        :param y: Optional target; ignored.
        :return:  self
        """
        df = pd.DataFrame(X).copy()
        if all(c in df.columns for c in ['Fare', 'Pclass', 'Embarked']):
            self.median_fare_pclass3_S_ = df.loc[
                (df['Pclass'] == 3) & (df['Embarked'] == 'S'), 'Fare'
            ].median()
            self.fare_global_median_ = df['Fare'].median()
        return self

    def transform(self, X):
        """
        Impute 'Fare' using learned medians: first for (Pclass==3 & 'S'),
        then remaining NaNs via global median.

        :param X: DataFrame to transform.
        :return:  DataFrame with 'Fare' imputed (no test leakage).
        """
        df = pd.DataFrame(X).copy()
        if all(c in df.columns for c in ['Fare', 'Pclass', 'Embarked']):
            if self.median_fare_pclass3_S_ is not None:
                mask = df['Fare'].isna() & (df['Pclass'] == 3) & (df['Embarked'] == 'S')
                df.loc[mask, 'Fare'] = self.median_fare_pclass3_S_
            if self.fare_global_median_ is not None:
                df['Fare'] = df['Fare'].fillna(self.fare_global_median_)
        return df


class DeckFromCabinTT8(PIPEXTransformer):
    """
    Create 'Deck' from first char of 'Cabin'; missing -> 'Z'. No grouping.
    """

    def transform(self, X):
        """
        Create 'Deck' from first char of 'Cabin'; use 'Z' when missing.

        :param X: DataFrame or array-like.
        :return:  DataFrame with added 'Deck' (string).
        """
        df = pd.DataFrame(X).copy()
        if 'Cabin' in df.columns:
            deck = df['Cabin'].astype(str).str[0]
            deck = deck.where(df['Cabin'].notna(), 'Z')
            df['Deck'] = deck.astype(str)
        else:
            df['Deck'] = 'Z'
        return df


class FamilySizeBucketTransformer(PIPEXTransformer):
    """
    Add:
      - FamilySize = SibSp + Parch + 1
      - FsizeD in {'singleton','small','large'}
    """

    def transform(self, X):
        """
        Add 'FamilySize' and bucket 'FsizeD' (singleton/small/large).

        :param X: DataFrame or array-like (SibSp/Parch optional).
        :return:  DataFrame with 'FamilySize' and 'FsizeD'.
        """
        df = pd.DataFrame(X).copy()
        sib = pd.to_numeric(df.get('SibSp', 0), errors='coerce').fillna(0)
        par = pd.to_numeric(df.get('Parch', 0), errors='coerce').fillna(0)
        df['FamilySize'] = sib + par + 1

        def _bucket(n):
            try:
                n = float(n)
            except Exception:
                return 'singleton'
            if n == 1: return 'singleton'
            if 1 < n < 5: return 'small'
            return 'large'

        df['FsizeD'] = df['FamilySize'].apply(_bucket)
        return df


class NameLengthAndBinsTransformer(PIPEXTransformer):
    """
    Add:
      - NameLength = length of 'Name'
      - NlengthD bins over NameLength using given cuts (default: 0,20,40,57,85)
        -> labels: short / okay / good / long
    """

    def __init__(self, bins=(0, 20, 40, 57, 85)):
        self.bins = tuple(bins)

    def transform(self, X):
        """
        Add 'NameLength' and binned 'NlengthD' with stable labels.

        :param X: DataFrame or array-like.
        :return:  DataFrame with 'NameLength' and 'NlengthD'.
        """
        df = pd.DataFrame(X).copy()
        df['NameLength'] = df['Name'].astype(str).apply(len).astype(float) if 'Name' in df.columns else 20.0
        labels = ['short', 'okay', 'good', 'long'][:max(0, len(self.bins) - 1)]
        try:
            df['NlengthD'] = pd.cut(df['NameLength'], self.bins, labels=labels, include_lowest=True)
        except Exception:
            df['NlengthD'] = 'short'
        return df


class TicketNumberExtractorTransformer(PIPEXTransformer):
    """
    TicketNumber: extract numeric block (>=2 digits) from 'Ticket' as float.
    Fill missing with training median.
    """

    def __init__(self):
        self.median_ = None

    @staticmethod
    def _extract_num(s) -> float:
        m = re.search(r'(\d{2,})', str(s))
        return float(m.group(1)) if m else float('nan')

    def fit(self, X, y=None) -> 'TicketNumberExtractorTransformer':
        """
        Learn median of extracted 'TicketNumber' on training data for later fill.

        :param X: DataFrame with 'Ticket' if available.
        :param y: Optional target; ignored.
        :return:  self
        """
        df = pd.DataFrame(X).copy()
        if 'Ticket' in df.columns:
            vals = df['Ticket'].apply(self._extract_num)
            self.median_ = pd.to_numeric(vals, errors='coerce').median()
        return self

    def transform(self, X):
        """
        Extract numeric block from 'Ticket' to 'TicketNumber' and
        fill missing with the stored median.

        :param X: DataFrame or array-like.
        :return:  DataFrame with 'TicketNumber' (float).
        """
        df = pd.DataFrame(X).copy()
        if 'Ticket' in df.columns:
            df['TicketNumber'] = df['Ticket'].apply(self._extract_num)
            if self.median_ is not None:
                df['TicketNumber'] = pd.to_numeric(df['TicketNumber'], errors='coerce').fillna(self.median_)
        else:
            df['TicketNumber'] = self.median_ if self.median_ is not None else np.nan
        return df


class TitleExtractorNormalizerTT8(PIPEXTransformer):
    """
    Title from 'Name' regex; normalize:
      - Mlle/Ms -> Miss; Mme -> Mrs
      - rare set -> 'Rare Title'
    """
    RARE = {'Dona', 'Lady', 'Countess', 'Capt', 'Col', 'Don', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dr'}

    @staticmethod
    def _get_title(name: str) -> str:
        m = re.search(r' ([A-Za-z]+)\.', str(name))
        return m.group(1) if m else ""

    def transform(self, X):
        """
        Extract 'Title' from 'Name'; normalize and map rare titles to 'Rare Title'.

        :param X: DataFrame or array-like.
        :return:  DataFrame with normalized 'Title'.
        """
        df = pd.DataFrame(X).copy()
        if 'Name' in df.columns:
            t = df['Name'].apply(self._get_title).astype(str)
        else:
            t = pd.Series('Rare Title', index=df.index)
        t = t.replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
        t = t.where(~t.isin(self.RARE), 'Rare Title')
        df['Title'] = t
        return df


class AgeImputerRFTransformer(PIPEXTransformer):
    """
    Predict missing 'Age' with a RandomForestRegressor trained on:
    ['Embarked','Fare','Parch','SibSp','TicketNumber','Title','Pclass',
     'FamilySize','FsizeD','NameLength','NlengthD','Deck'].
    Categorical features are encoded to integer codes learned on train.
    """

    def __init__(self, n_estimators=400, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.age_model_ = None
        self.cat_cols_ = ['Embarked', 'Title', 'FsizeD', 'NlengthD', 'Deck']
        self.age_feature_cols_ = ['Embarked', 'Fare', 'Parch', 'SibSp', 'TicketNumber', 'Title',
                                  'Pclass', 'FamilySize', 'FsizeD', 'NameLength', 'NlengthD', 'Deck']
        self.cat_maps_ = {}  # {col: {cat->int}}
        self.col_medians_ = None

    def _encode_matrix(self, df_feat: pd.DataFrame, cols) -> np.ndarray:
        """
        Build a numeric matrix for the RF using stored categorical maps
        and per-column medians for NaN fill.

        :param df_feat: DataFrame containing model features.
        :return:        2D numpy array (n_samples, n_features).
        """
        feat = pd.DataFrame({c: df_feat[c] if c in df_feat.columns else np.nan for c in cols}, index=df_feat.index)
        x_mat = []
        for c in cols:
            if c in self.cat_cols_:
                mapping = self.cat_maps_.get(c, {})
                codes = feat[c].astype(str).map(mapping).astype('Int64').fillna(-1).astype(int)
                x_mat.append(codes.to_numpy())
            else:
                x_mat.append(pd.to_numeric(feat[c], errors='coerce').to_numpy())
        x_mat = np.vstack(x_mat).T
        # Fill NaNs with train medians (per column).
        if np.isnan(x_mat).any() and self.col_medians_ is not None:
            indexes = np.where(np.isnan(x_mat))
            x_mat[indexes] = np.take(self.col_medians_, indexes[1])
        return x_mat

    def fit(self, X, y=None) -> 'AgeImputerRFTransformer':
        """
        Learn categorical encodings, compute column medians, and fit a
        RandomForest on rows with 'Age' present.

        :param X: DataFrame containing 'Age' and feature columns.
        :param y: Optional target; ignored.
        :return:  self
        """
        from sklearn.ensemble import RandomForestRegressor
        df = pd.DataFrame(X).copy()

        age_feature_cols_ = [c for c  in self.age_feature_cols_ if c in df.columns]

        # Build cat maps from training data.
        for c in self.cat_cols_:
            vals = df.get(c, pd.Series([], dtype=object)).astype(str).fillna('NA').unique().tolist()
            self.cat_maps_[c] = {v: i for i, v in enumerate(vals)}

        # Prepare training set.
        if 'Age' in df.columns:
            train = df[df['Age'].notna()][['Age'] + age_feature_cols_].copy()
            if len(train) > 0:
                X_train = self._encode_matrix(train[age_feature_cols_], age_feature_cols_)
                y_train = pd.to_numeric(train['Age'], errors='coerce').to_numpy()
                mask = ~pd.isna(y_train)
                X_train, y_train = X_train[mask], y_train[mask]
                # Column medians for NaN fill at inference time.
                self.col_medians_ = np.nanmedian(X_train, axis=0) if X_train.size else None
                if len(y_train) > 0:
                    self.age_model_ = RandomForestRegressor(
                            n_estimators=self.n_estimators, n_jobs=-1, random_state=self.random_state
                    ).fit(X_train, y_train)
        return self

    def transform(self, X):
        """
        Predict and fill missing 'Age' using the trained RF model.
        Leaves 'Age' unchanged if the model is not available.

        :param X: DataFrame to transform.
        :return:  DataFrame with 'Age' imputed where missing.
        """
        df = pd.DataFrame(X).copy()

        age_feature_cols_ = [c for c in self.age_feature_cols_ if c in df.columns]

        if (self.age_model_ is not None) and ('Age' in df.columns):
            miss_mask = df['Age'].isna()
            if miss_mask.any():
                X_test = self._encode_matrix(df.loc[miss_mask, age_feature_cols_], age_feature_cols_)
                df.loc[miss_mask, 'Age'] = self.age_model_.predict(X_test)
        return df


# compas dataset transformers


# Miscellaneous

class RecodeTransformer(PIPEXTransformer):
    """Custom transformer that recodes the 'sex', 'c_charge_degree', and 'race' columns.
    """

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'RecodeTransformer':
        """
        Fit the transformer.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: RecodeTransformer
        """
        return self

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal, PyMethodMayBeStatic
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> DataFrame:
        """
        Recode the 'sex', 'c_charge_degree', and 'race' columns.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: DataFrame, Transformed DataFrame
        """
        X = DataFrame(X)
        X['sex'] = X['sex'].replace({'Female': 1.0, 'Male': 0.0})
        X['c_charge_degree'] = X['c_charge_degree'].replace({'F': 1.0, 'M': 0.0})
        X['race'] = X['race'].apply(lambda x: 1.0 if x == "Caucasian" else 0.0)
        return X


class FilterTransformer(PIPEXTransformer):
    """Custom transformer that filters the DataFrame based on specific conditions."""

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'FilterTransformer':
        """
        Fit the transformer.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: FilterTransformer
        """
        return self

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal, PyMethodMayBeStatic
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> DataFrame:
        """
        Filter the DataFrame based on specific conditions.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: DataFrame, Filtered DataFrame
        """
        X = DataFrame(X)
        X = X.loc[(X['days_b_screening_arrest'] <= 30)]
        X = X.loc[(X['days_b_screening_arrest'] >= -30)]
        X = X.loc[(X['is_recid'] != -1)]
        X = X.loc[(X['c_charge_degree'] != "O")]
        X = X.loc[(X['score_text'] != 'N/A')]
        return X


class ScoreTextTransformer(PIPEXTransformer):
    """Custom transformer that replaces values in the 'score_text' column and applies label encoding."""

    encoder: LabelEncoder = None

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'ScoreTextTransformer':
        """
        Fit the transformer.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: ScoreTextTransformer
        """
        X = DataFrame(X)
        self.encoder = LabelEncoder().fit(X['score_text'].replace('Medium', 'Low'))
        return self

    # parameters are defined by scikit-learn
    # noinspection PyPep8Naming, PyUnusedLocal, PyMethodMayBeStatic
    def transform(self, X: np.ndarray, y: np.ndarray = None) -> DataFrame:
        """
        Replace values in the 'score_text' column and apply label encoding.

        :param X: numpy array of shape [n_samples, n_features], Training set
        :param y: numpy array of shape [n_samples], Target values
        :return: DataFrame, Transformed DataFrame
        """
        X = DataFrame(X)
        X['score_text'] = self.encoder.transform(X['score_text'].replace('Medium', 'Low'))
        return X

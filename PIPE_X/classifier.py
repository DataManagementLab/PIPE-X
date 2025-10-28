"""
Module to handle classifiers and architectures, including Abstract Base Class and a method to register new classes.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Type

import pandas as pd
from numpy.random import Generator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearnex.neighbors import KNeighborsClassifier
from sklearnex.svm import SVC
from xgboost import XGBClassifier

from PIPE_X.data import DataWrapper
from PIPE_X.metrics import Metrics


class ModelABC(ABC):
    """ Abstract Base Class for classification models.
    """

    model = None

    @abstractmethod
    def __init__(self, rng):
        """ Initialize the model.
        """
        self.rng = rng

    def train(self, x_train: pd.DataFrame, y_train) -> None:
        """ Train the model.

        :param x_train: DataFrame, the training data
        :param y_train: , the training labels
        """
        self.model.fit(x_train.values, y_train)

    def predict(self, samples: Any) -> Any:
        """ Predict single output using the model.

        :param samples: array-like of shape (n_samples, n_features), the data to predict
        :return: array-like of shape (n_samples, n_classes), probability of the sample for each class in the model,
                where classes are ordered as they are in
        """
        return self.model.predict_proba(samples)

    def evaluate(self, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        """ Evaluate the model.

        :param x_test: DataFrame, the test data
        :param y_test: DataFrame, the test labels
        :return: float, the evaluation score
        """
        y_predictions = self.model.predict(x_test.values)
        score = f1_score(y_test, y_predictions)
        return score


MODELS = {}


def register_model(name: str, model: Type[ModelABC]) -> None:
    """
    Register new model architecture class as available to PIPE_X, so that it can be accessed by name.

    :param name: str, name of the model architecture
    :param model: class instantiating ModelWrapper
    :return: None
    """
    MODELS[name] = model


class LRClassifier(ModelABC):  # Anchors/LimeTabular baseline

    def __init__(self, rng):
        self.model = LogisticRegression(solver='liblinear', warm_start=True, random_state=rng.integers(1000))


register_model('LR', LRClassifier)


class DTClassifier(ModelABC):  # baseline, NaN tolerant

    def __init__(self, rng):
        self.model = DecisionTreeClassifier(random_state=rng.integers(1000))


register_model('DT', DTClassifier)


class GBTClassifier(ModelABC):  # Anchors/LimeTabular baseline, NaN compatible

    def __init__(self, rng):
        self.model = XGBClassifier(n_estimators=400, nthread=-1, random_state=rng.integers(1000))


register_model('GBT', GBTClassifier)


class KNNClassifier(ModelABC):  # baseline

    def __init__(self, rng):
        logging.getLogger(name='sklearnex').setLevel(logging.WARNING)
        self.model = KNeighborsClassifier(n_jobs=-1)
        self.rng = rng


register_model('KNN', KNNClassifier)


class HGBClassifier(ModelABC):  # NaN tolerant

    def __init__(self, rng):
        self.model = HistGradientBoostingClassifier(random_state=rng.integers(1000))


register_model('HGB', HGBClassifier)


class NNClassifier(ModelABC):  # Anchors/LimeTabular baseline

    def __init__(self, rng):
        self.model = MLPClassifier(hidden_layer_sizes=(50, 50),
                                   max_iter=500,
                                   early_stopping=True,
                                   warm_start=True,
                                   random_state=rng.integers(1000))


register_model('NN', NNClassifier)


class SVCClassifier(ModelABC):

    def __init__(self, rng):
        logging.getLogger(name='sklearnex').setLevel(logging.WARNING)
        self.model = SVC(probability=True)
        self.rng = rng


register_model('SVC', SVCClassifier)


class ModelWrapper:
    """
    Wrapper around a classification model, handling training, prediction, evaluation and explanation.
    """
    data: DataWrapper
    classifier: Type[ModelABC]
    models_by_score: dict[Metrics, list[float]] = {metric: [] for metric in Metrics}
    rng: Generator

    def __init__(self, architecture: str, data: DataWrapper, rng: Generator) -> None:
        self.models_by_score: dict[Metrics, list[ModelABC]] = {metric: [] for metric in Metrics}
        logging.debug("Instantiating ModelWrapper...")

        self.data = data
        self.classifier = MODELS[architecture]
        self.rng = rng

    def train_all(self) -> None:
        """ Train all classifier models on the appropriate snapshots.
        """
        for score in self.data.snapshot_by_score.keys():
            if not self.models_by_score[score]:
                for i, snapshot in enumerate(self.data.snapshot_by_score[score]):
                    model = self.classifier(self.rng)
                    if snapshot.train_data.isnull().values.any():
                        logging.warning(f"Training model {i} for {score} on NaN values")
                    model.train(snapshot.train_data, snapshot.train_target.values.ravel())
                    self.models_by_score[score].append(model)

    def train_single(self, train_data, target):
        """
        Train a single classifier model on the provided data.

        :param train_data: DataFrame, the training data
        :param target: DataFrame, the training labels
        """
        model = self.classifier(self.rng)
        model.train(train_data, target.values.ravel())
        return model

"""
Feature attribution module for the PIPE_X project.
"""
from typing import Any

from lime.lime_tabular import LimeTabularExplainer
from numpy import array, dtype, ndarray
from numpy.random import Generator

from PIPE_X.classifier import ModelWrapper
from PIPE_X.data import DataWrapper
from PIPE_X.metrics import Metrics


def lime_to_vector(explanation):
    """
    Convert a LIME explanation to a vector of feature weights.
    """
    explanation.sort(key=lambda x: x[0])
    weight_vector = array([value for _, value in explanation])

    return weight_vector


def run_fa(sample_id, data, model, columns, fa) -> tuple[ndarray[Any, dtype[Any]], Any]:
    """
    Run the feature attribution for a single sample.

    :param sample_id: int, Sample ID
    :param data: DataWrapper, Data to process
    :param model: ModelWrapper, Model to use for prediction
    :param columns: list[str], List of all columns of the pipeline
    :param fa: LimeTabularExplainer, Feature attribution explainer
    """
    sample = data.get_sample(sample_id)[0]

    exp = fa.explain_instance(sample.values.ravel(), model.predict, num_features=len(sample.columns))
    exp_list = exp.as_list()

    # Ensure all columns from the pipeline are represented in the explanation,
    # even those currently not present in the data
    exp_all_columns = {col: 0.0 for col in columns}
    for col, weight in exp_list:
        if col not in exp_all_columns:
            raise ValueError(f"Column {col} from LIME explanation not found in expected columns.")
        exp_all_columns[col] = weight
    exp_list = list(exp_all_columns.items())

    return lime_to_vector(exp_list), exp_list


class FeatureAttributionWrapper:
    """     Wrapper around a feature attribution explainer, providing explanations for individual samples.
    """
    data: DataWrapper
    model: ModelWrapper
    feature_attributors: dict[Metrics, list]
    raw_experiments: dict[Metrics, list]
    rng: Generator
    columns: list[str]

    def __init__(self, data: DataWrapper, model: ModelWrapper, rng: Generator) -> None:
        """
        Initialize the feature attribution wrapper with data and model.

        :param data: DataWrapper, Data to process
        :param model: ModelWrapper, Model to use for prediction
        """
        self.data = data
        self.model = model
        self.feature_attributors = {metric: [] for metric in Metrics}
        self.raw_experiments = {metric: [] for metric in Metrics}

        self.rng = rng

    def setup(self, columns: list[str]) -> None:
        """
        Set up the feature attribution wrapper by training the models.

        :param columns: list of str, Columns to use for feature attribution
        (superset of the columns in the individual snapshots)
        """
        self.columns = columns

        categorical_features = self.data.get_raw().train_data.select_dtypes(include=['object']).columns

        for score in self.feature_attributors.keys():
            for i in range(len(self.data.snapshot_by_score[score])):
                data = self.data.snapshot_by_score[score][i].train_data
                self.feature_attributors[score].append(LimeTabularExplainer(training_data=data,
                                                                            feature_names=data.columns,
                                                                            categorical_features=categorical_features,
                                                                            kernel_width=3,
                                                                            discretize_continuous=False,
                                                                            random_state=self.rng.integers(1000)))

    def run_all(self, sample_id):
        """
        Run the feature attribution for all models and snapshots.

        :param sample_id:
        :return:
        """
        scores: dict[Metrics, list[ndarray[Any, dtype[Any]]]] = {metric: [] for metric in Metrics}
        explanations: dict[Metrics, list[Any]] = {metric: [] for metric in Metrics}

        for score in Metrics:
            for i, fa in enumerate(self.feature_attributors[score]):
                exp_vect, exp_list = run_fa(sample_id, self.data.snapshot_by_score[score][i],
                                            self.model.models_by_score[score][i], self.columns, fa)
                scores[score].append(exp_vect)
                explanations[score].append(exp_list)

        return scores, explanations

"""
This module contains the Explainer class and its functions for setting up data, pipeline and models,
as well as the explanation process for individual samples.
"""
import logging
from typing import Any

import numpy as np
from numpy import floating
from numpy.random import Generator
from numpy.typing import NDArray
from pandas import DataFrame
from sklearn.decomposition import PCA

from PIPE_X.classifier import ModelWrapper
from PIPE_X.data import DataWrapper
from PIPE_X.feature_attribution import FeatureAttributionWrapper
from PIPE_X.metrics import Metrics
from PIPE_X.pipeline import AbstractPipeline, PipelineWrapper


def vector_projection(b: NDArray, a: NDArray) -> tuple[NDArray, float]:
    """
    Calculate the projection of b on vector a

    :param b: np.array, vector to project
    :param a: np.array, vector to project on
    :return: np.array, projected vector
    """
    # projection = ((a*b) / (a*a)) * a
    projection = (np.dot(a, b) / np.dot(a, a)) * a

    # Determine the direction of the projected vector relative to a
    direction = np.sign(np.dot(a, b))

    return projection, direction


def get_normalized_length(vector: NDArray, norm: NDArray) -> floating[Any]:
    """
    Normalize a vector relative to a norm vector.

    :param vector: np.array, vector to normalize
    :param norm: np.array, vector to normalize by
    :return: float, normalized length of the vector
    """
    # Calculate the length of the projection and normalize relative to the length of a
    return np.linalg.norm(vector) / np.linalg.norm(norm)


def get_immediate_impacts(weights: list[NDArray]) -> list[float]:
    """
    Calculate the immediate impact of a step of the preprocessing pipeline on model behavior.

    :param weights: list, feature weight vectors at each step of the pipeline
    :return: list[float], immediate impacts
    """
    logging.debug(f"Calculating immediate impacts for {len(weights) - 1} steps...")

    raw_impacts = [weights[i] - weights[i - 1] for i in range(1, len(weights))]
    full_pipeline_impact = weights[-1] - weights[0]

    projection_vectors = []
    projection_directions = []
    for i in range(0, len(raw_impacts)):
        projection, direction = vector_projection(raw_impacts[i], full_pipeline_impact)
        projection_vectors.append(projection)
        projection_directions.append(direction)

    immediate_impacts = []
    for i in range(0, len(raw_impacts)):
        immediate_impacts.append(
                projection_directions[i] * get_normalized_length(projection_vectors[i], full_pipeline_impact))

    return immediate_impacts


def get_leave_out_impacts(weights: list[NDArray], weights_full: NDArray, weights_raw: NDArray) -> list[float]:
    """
    Calculate the immediate impact of a step of the preprocessing pipeline on model behavior.

    :param weights: list, feature weight vectors with one step of the pipeline left out
    :param weights_full: np.array, feature weight vector of the full pipeline run
    :param weights_raw: np.array, feature weight vector of the raw data run
    :return: list[float], leave-out impacts
    """
    logging.debug(f"Calculating leave-out impacts for {len(weights)} steps...")

    impact_full = weights_full - weights_raw

    leave_out_impacts = []
    for i in range(len(weights)):
        # Calculate diff of full pipeline including the step and the one without and project it on full pipeline vector
        # weights_full - weights[i] is semantically equivalent to (weights_full-weights_raw)-(weights[i]-weights_raw)
        projection, direction = vector_projection(weights_full - weights[i], impact_full)
        impact = direction * get_normalized_length(projection, impact_full)
        leave_out_impacts.append(impact)

    return leave_out_impacts


def pca_reduction(one_by_one_weights: list, leave_out_weights: list) -> tuple[NDArray, NDArray]:
    """
    Reduces the input into two-dimensional vectors using the PCA

    :param one_by_one_weights: list, feature weight vectors at each step of the pipeline
    :param leave_out_weights: list,  feature weight vectors with one step of the pipeline left out
    :return: (np.ndarray, np.ndarray), the reduced feature weights
    """
    n_components = 2
    pca = PCA(n_components=n_components)
    reduced_weights_one_by_one = pca.fit_transform(one_by_one_weights) if len(
            one_by_one_weights) >= n_components else None
    reduced_weights_leave_out = pca.fit_transform(leave_out_weights) if len(leave_out_weights) >= n_components else None
    return reduced_weights_one_by_one, reduced_weights_leave_out


class Explainer:
    """
    The Explainer class provides interaction with data, pipeline and models,
    and allows for explanation of individual samples and visualization of the results.
    """

    data: DataWrapper
    pipeline: PipelineWrapper
    model: ModelWrapper
    sample: DataFrame = None
    impacts: dict[Metrics, list[float]] = {metric: [] for metric in Metrics}
    # summary_dictionary: Summary_dictionary
    rng: Generator
    raw_explanations: dict[Metrics, list[float]] = {metric: [] for metric in Metrics}

    def __init__(self, dataframe: DataFrame, pipeline_obj: AbstractPipeline, architecture: str, rng: Generator = None,
                 predefined_essential_steps: list = None, split_seed: int = 42, additional_features=None) -> None:
        """
        Initialize the explainer with data, pipeline and model.

        :param dataframe: DataFrame, data file
        :param pipeline_obj: Pipeline object, pickled pipeline
        :param architecture: str, name of the classifier architecture to be used
        :param additional_features: list, list of dummy columns to generate that will be filled by the pipeline
        """
        self.rng = rng
        self.data = DataWrapper(dataframe, self.rng, split_seed, additional_features=additional_features)
        self.model = ModelWrapper(architecture, self.data, self.rng)
        self.pipeline = PipelineWrapper(pipeline_obj, self.data, self.model,
                                        predefined_essential_steps=predefined_essential_steps or [])
        self.fa = FeatureAttributionWrapper(self.data, self.model, self.rng)

    def setup(self) -> None:
        """ Set up the explainer by preprocessing the data, storing variations and training the models.
        """
        self.pipeline.process()
        logging.info(f"""Data preprocessed successfully. Produced 
                         {len(self.data.snapshot_by_score[Metrics.IMMEDIATE])} immediate impact snapshots and
                         {len(self.data.snapshot_by_score[Metrics.LEAVE_OUT])} leave-out impact snapshots.""")
        self.model.train_all()
        self.fa.setup(self.pipeline.columns)

    def affected_data_by_step(self):
        """
            Get the columns and row counts affected by each pipeline step.
        """
        pipeline_steps_reduced = [
            step for step, is_essential in
            zip(self.pipeline.step_names, self.pipeline.is_essential_list)
            if not is_essential
        ]

        affected_row_counts = []
        affected_columns = []

        for i, name in enumerate(pipeline_steps_reduced):
            # Snapshot before the step
            old_df = self.data.snapshot_by_score[Metrics.IMMEDIATE][i].train_data
            old_df = old_df.reindex(sorted(old_df.columns), axis=1)

            # Snapshot after the step
            new_df = self.data.snapshot_by_score[Metrics.IMMEDIATE][i + 1].train_data
            new_df = new_df.reindex(sorted(new_df.columns), axis=1)

            # Align both DataFrames on the intersection of rows and columns
            old_aligned, new_aligned = old_df.align(new_df, join='inner', axis=0)
            old_aligned, new_aligned = old_aligned.align(new_aligned, join='inner', axis=1)

            # Find columns that were added or dropped
            old_cols = set(old_df.columns)
            new_cols = set(new_df.columns)
            removed_cols = old_cols - new_cols
            added_cols = new_cols - old_cols

            # Compare only the aligned portions
            compared_df = new_aligned.compare(old_aligned)

            affected_row_counts.append(compared_df.shape[0] + abs(old_df.shape[0] - new_df.shape[0]))
            columns_with_value_changes = compared_df.columns.get_level_values(0).unique().tolist()
            affected_columns_str_list = list(set(columns_with_value_changes))
            affected_columns_str_list.extend(f"+{col}" for col in added_cols)
            affected_columns_str_list.extend(f"-{col}" for col in removed_cols)

            # Extract the names of the top-level columns that changed
            affected_columns.append(affected_columns_str_list)

        return affected_columns, affected_row_counts

    def run(self, sample_id: int) -> None:
        """
        Run the explainer for a chosen sample, explaining it, and providing impact measures for pipeline steps.

        :param sample_id: int, ID of the sample to explain
        :return: None
        """
        logging.debug(f"Running explainer for sample {sample_id}...")

        # here we calculate fa attributions scores
        scores, raw_explanations = self.fa.run_all(sample_id)
        one_by_one_weights = scores[Metrics.IMMEDIATE]
        leave_out_weights = scores[Metrics.LEAVE_OUT]
        self.raw_explanations[Metrics.IMMEDIATE] = raw_explanations[Metrics.IMMEDIATE]
        self.raw_explanations[Metrics.LEAVE_OUT] = raw_explanations[Metrics.LEAVE_OUT]
        self.impacts[Metrics.IMMEDIATE] = get_immediate_impacts(one_by_one_weights)
        self.impacts[Metrics.LEAVE_OUT] = get_leave_out_impacts(leave_out_weights, one_by_one_weights[-1],
                                                                one_by_one_weights[0])

        # apply pca
        # TODO Check: Why don't we use the full pipeline weights for leave out here (but for the impacts)?
        # reduced_weights_one_by_one, reduced_weights_leave_out = pca_reduction(one_by_one_weights,
        #                                                                       leave_out_weights)

        # Always convert to DataFrame, using original weights if PCA is not applicable
        # reduced_weights_one_by_one = pd.DataFrame(
        #         reduced_weights_one_by_one if reduced_weights_one_by_one is not None else one_by_one_weights)
        # reduced_weights_leave_out = pd.DataFrame(
        #         reduced_weights_leave_out if reduced_weights_leave_out is not None else leave_out_weights)

        # last_row_one_by_one = reduced_weights_one_by_one.tail(1)
        # append full pipeline weights for comparison
        # reduced_weights_leave_out = pd.concat([reduced_weights_leave_out, last_row_one_by_one],
        #                                       ignore_index=True)
        # self.summary_dictionary = Summary_dictionary(self.impacts, self.pipeline.step_names,
        #                                              reduced_weights_one_by_one, reduced_weights_leave_out,
        #                                              self.pipeline.is_essential_list)
        # save pickle file for notebook
        # pickle_summary_dictionary("test.pkl", self.summary_dictionary)
        # self.summary_dictionary.calculate_vectors()

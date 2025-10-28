"""
This module contains the PipelineWrapper class, which is a wrapper around the sklearn Pipeline class.
It provides a method to fit and transform the data while providing snapshots of the data at each step of the pipeline.
"""
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Tuple, Union

from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline as IPipeline
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import DataConversionWarning
from sklearn.pipeline import Pipeline

import PIPE_X.transformer
# noinspection PyUnresolvedReferences
from PIPE_X.classifier import ModelWrapper
from PIPE_X.data import DataWrapper
from PIPE_X.metrics import Metrics
from pipelines.custom_transformers import *


class AbstractPipeline(ABC):
    """
    Abstract base class defining the interface for pipeline-like objects.
    """
    steps: List
    essential_steps: Iterable[str] = ()

    # noinspection PyMissingOrEmptyDocstring, PyPep8Naming
    @abstractmethod
    def fit(self, X: pd.DataFrame, y=None):
        pass

    # noinspection PyMissingOrEmptyDocstring, PyPep8Naming
    @abstractmethod
    def transform(self, X: pd.DataFrame):
        pass

    # noinspection PyMissingOrEmptyDocstring, PyPep8Naming
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame, y=None):
        pass

    # noinspection PyMissingOrEmptyDocstring, PyPep8Naming
    @abstractmethod
    def make_pipeline(self, steps):
        pass

    def get_essential_steps(self) -> List[str]:
        """
        Get the names of the essential steps in the pipeline.
        """
        return list(self.essential_steps)


class SklearnPipelineWrapper(Pipeline, AbstractPipeline):
    # Just subclassing sklearn Pipeline to work with Pandas pipeline.

    def make_pipeline(self, steps) -> 'SklearnPipelineWrapper':
        """ Create a new SklearnPipelineWrapper with the given steps.
        """
        return SklearnPipelineWrapper(steps=steps)


class ImblearnPipelineWrapper(IPipeline, AbstractPipeline):
    # Just subclassing Imblearn Pipeline to work with Pandas pipeline.

    def make_pipeline(self, steps) -> 'ImblearnPipelineWrapper':
        """ Create a new ImblearnPipelineWrapper with the given steps.
        """
        return ImblearnPipelineWrapper(steps=steps)

    @staticmethod
    def make_pipeline_with_default_steps(steps):
        """
        Create a new ImblearnPipelineWrapper with the given steps and default steps for ending the pipeline in a
        sampler and schema alignment.
        """
        steps.extend(
                [
                    ('noop', FunctionSampler(func=PIPE_X.transformer.noop, validate=False)),
                ]
        )
        pipeline = ImblearnPipelineWrapper(steps=steps)
        pipeline.essential_steps = ["categorical_encoder", "noop"]
        return pipeline


class PandasPipeline(AbstractPipeline, BaseEstimator):
    """A pipeline that works with pandas DataFrames and custom transformers.

    This pipeline implementation is designed to work with both scikit-learn compatible
    transformers and custom pandas-based transformers. It maintains the DataFrame
    structure throughout the pipeline and provides additional functionality for
    tracking transformations and generating snapshots.
    """

    def __init__(self, steps: List[Tuple[str, Union[TransformerMixin, PIPE_X.transformer.CustomTransformer]]]):
        """Initialize the pipeline with a list of (name, transformer) tuples.

        Args:
            steps: List of (name, transformer) tuples. Transformers can be either
                  scikit-learn compatible or custom pandas transformers.
        """
        self.steps = steps
        self._fitted = False

    def _repr_html_(self):
        from IPython.display import display
        temp_pipeline = Pipeline(steps=self.steps)
        set_config(display='diagram')
        display(temp_pipeline)

    # noinspection PyPep8Naming
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PandasPipeline':
        """Fit all transformers in the pipeline.

        Args:
            X: Input DataFrame
            y: Optional target Series

        Returns:
            self: The fitted pipeline
        """
        self._fitted = True
        return self

    # noinspection PyPep8Naming
    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Transform the data through all steps in the pipeline.

        Args:
            X: Input DataFrame
            y: Optional target Series

        Returns:
            Transformed DataFrame
        """
        if not self._fitted:
            raise ValueError("Pipeline must be fitted before transform")

        x_transformed = X.copy()
        y_transformed = y.copy() if y is not None else None

        for name, transformer in self.steps:
            if isinstance(transformer, PIPE_X.transformer.CustomTransformer):
                transformed_data = transformer.transform(x_transformed, y_transformed)
                if isinstance(transformed_data, tuple):
                    x_transformed, y_transformed = transformed_data
                else:
                    x_transformed = transformed_data
            else:
                x_transformed = transformer.transform(x_transformed)

        # check if y and y transformed are same, then return only X_transformed
        if y is None or y_transformed is None or y.equals(y_transformed):
            return x_transformed
        else:
            return x_transformed, y_transformed

    # noinspection PyPep8Naming
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit and transform the data in one step.

        Args:
            X: Input DataFrame
            y: Optional target Series

        Returns:
            Transformed DataFrame
        """
        X_transformed = X.copy()
        y_transformed = y.copy() if y is not None else None
        for name, transformer in self.steps:
            if isinstance(transformer, PIPE_X.transformer.CustomTransformer):
                transformed_data = transformer.fit_transform(X_transformed, y_transformed)
                if isinstance(transformed_data, tuple):
                    X_transformed, y_transformed = transformed_data
                else:
                    X_transformed = transformed_data
            else:
                if y is not None:
                    X_transformed = transformer.fit_transform(X_transformed, y_transformed)
                else:
                    X_transformed = transformer.fit_transform(X_transformed)

        if y is None or y_transformed is None or y.equals(y_transformed):
            return X_transformed
        else:
            return X_transformed, y_transformed

    def make_pipeline(self, steps) -> 'PandasPipeline':
        """ Create a new PandasPipeline with the given steps.
        """
        return PandasPipeline(steps=steps)


def pickle_pipeline(pipeline: AbstractPipeline, path: str, filename: str) -> None:
    """
    store pipeline using pickle

    :param pipeline: sklearn.Pipeline.pipeline to store
    :param path: directory in which to store the pickled pipeline
    :param filename: file in which to store the pickled pipeline
    :return: None
    """

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump(pipeline, f)


def unpickle_pipeline(path: str) -> AbstractPipeline:
    """
    load sklearn pipeline using pickle

    :param path: path under which the pickled object is stored
    :return: Pipeline
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


class PipelineWrapper:
    """
    Wrapper around the AbstractPipeline class. It provides a method to fit and transform the data and provide
    snapshots of the data at each step of the pipeline.
    """
    pipeline: AbstractPipeline
    step_names: list[str]
    essential_pipline: AbstractPipeline
    data: DataWrapper
    is_essential_list: list[bool]
    model: ModelWrapper
    columns: list[str]

    def __init__(self, pipeline_obj: AbstractPipeline, data: DataWrapper, model: ModelWrapper,
                 predefined_essential_steps: list = None) -> None:
        """
        Initialize the pipeline wrapper with a pipeline and data.

        :param pipeline_obj: AbstractPipeline, pickled pipeline
        :param data: DataWrapper, Data to process
        :param model: ModelWrapper, Model to use
        :param predefined_essential_steps: list, list of essential steps (steps with names in this list will be directly
            considered as essential without actively checking)
        """
        logging.debug("Instantiating PipelineWrapper...")

        logging.debug("Setting sklearn config to produce pandas DataFrames")
        set_config(transform_output="pandas")

        logging.debug(f"Loading pickled pipeline...")
        self.pipeline = pipeline_obj
        logging.info(f"Successfully loaded sklearn/pandas pipeline from {pipeline_obj}.")

        self.step_names = []

        self.data = data
        self.model = model
        self.is_essential_list = []
        self.predefined_essential_steps = set(predefined_essential_steps or [])

    def generate_step_by_step_snapshots(self, pipeline, data, targets, test_data, store_step_names=False):
        """
        Generate snapshots of the data at each (non-essential) step of the pipeline.
        Essential steps will be applied to all snapshots, hence for essential steps, no new snapshot is created.
        The first entry of the snapshot list will be the raw data with only essential steps applied.
        """
        state = {
            'current_index': -1
        }

        def _process_step_by_step(element, data, targets, test_data, snapshot_list, snapshot_targets, snapshot_test,
                                  name='', selector=None):
            """
            Recursively go through all steps in a nested pipeline, create new snapshots for each new non-essential step
            and apply the essential steps to all snapshots.

            :param element: AbstractPipeline, ColumnTransformer or Pipeline Step
            :param data: DataFrame
            :param snapshot_list: list of DataFrames
            :param name: name of the transformation step
            :return: DataFrame
            """

            def apply_step(step_process, data, targets, test_data, snapshot_list, snapshot_targets, snapshot_test,
                           name, selector):
                """
                Apply step

                :param step_process: Step to apply
                :param data: DataFrame
                :param targets: target Series
                :param snapshot_list: list of DataFrames
                :param snapshot_targets: list of target Series
                :param name: name of the transformation step
                :param selector: Column selector function
                :return: DataFrame
                """
                state['current_index'] += 1

                def _apply(data, targets, test_data):
                    # Restrict operation to certain columns if needed
                    # The remaining columns are concatenated back to the data after the operation
                    data_concat = None
                    test_concat = None
                    selector_columns = None
                    if selector:
                        selector_columns = selector if isinstance(selector, list) else selector(data)

                        if len(selector_columns) == 0:
                            return data, targets, test_data

                        logging.debug(f"Restricting step to columns {selector_columns} based on selector")
                        data_concat = data.drop(columns=selector_columns, inplace=False)
                        data = data[selector_columns]
                        test_concat = test_data.drop(columns=selector_columns, inplace=False)
                        test_data = test_data[selector_columns]

                    # Apply transformation step
                    if hasattr(step_process, 'fit_transform'):
                        data = step_process.fit_transform(data, y=targets)
                        test_data = step_process.transform(test_data)
                    elif hasattr(step_process, 'fit_resample'):
                        data, targets = step_process.fit_resample(data, targets)
                    else:
                        raise ValueError(f"Step {name} does not have fit_transform or fit_resample method.")

                    # Rejoin unchanged columns (if any)
                    # only for features, not for targets. Modify if needed
                    if selector_columns:
                        data = pd.concat([data, data_concat], axis=1)
                        test_data = pd.concat([test_data, test_concat], axis=1)

                    return data, targets, test_data

                # copy data to avoid modifying the original data
                data = data.copy()
                targets = targets.copy()
                test_data = test_data.copy()
                data, targets, test_data = _apply(data, targets, test_data)

                # is this step essential?
                if self.is_essential_list[state['current_index']]:
                    # Apply it to all snapshots already produced
                    for i, (snapshot, target, test_snapshot) in enumerate(
                            zip(snapshot_list, snapshot_targets, snapshot_test)):
                        snapshot_list[i], snapshot_targets[i], snapshot_test[i] = _apply(snapshot, target,
                                                                                         test_snapshot)
                    # Don't store a new snapshot, since there will be no difference to the previous one
                else:
                    # Non-essential step? Create new snapshot from the data just computed
                    snapshot_list.append(data)
                    snapshot_targets.append(targets)
                    snapshot_test.append(test_data)

                if store_step_names:
                    self.step_names.append(name)

                return data, targets, test_data

            if isinstance(element, ColumnTransformer):
                # Column transformer? Go into each transformer operation
                for transformer_name, transformer_process, columns_selector in element.transformers:
                    logging.debug(f"ColumnTransformer for {transformer_name}")
                    data, targets, test_data = _process_step_by_step(transformer_process, data, targets, test_data,
                                                                     snapshot_list, snapshot_targets, snapshot_test,
                                                                     transformer_name, columns_selector)
                return data, targets, test_data
            elif isinstance(element, AbstractPipeline):
                # Pipeline? Go into each step
                for step_name, step_process in element.steps:
                    if isinstance(step_process, AbstractPipeline):
                        logging.debug(f"Entering nested pipeline: {step_name}")
                        data, targets, test_data = _process_step_by_step(step_process, data, targets, test_data,
                                                                         snapshot_list,
                                                                         snapshot_targets, snapshot_test)
                    elif isinstance(step_process, ColumnTransformer):
                        logging.debug(f"Entering ColumnTransformer: {step_name}")
                        step_process.remainder = 'passthrough'
                        data, targets, test_data = _process_step_by_step(step_process, data, targets, test_data,
                                                                         snapshot_list,
                                                                         snapshot_targets, snapshot_test)
                    else:
                        logging.debug(f"Applying {step_name}")
                        data, targets, test_data = apply_step(step_process, data, targets, test_data, snapshot_list,
                                                              snapshot_targets, snapshot_test, step_name,
                                                              selector)
            else:
                # Not a pipeline but directly a step? Create snapshot
                data, targets, test_data = apply_step(element, data, targets, test_data, snapshot_list,
                                                      snapshot_targets, snapshot_test, name, selector)

            return data, targets, test_data

        # Store a copy of the raw data as first snapshot
        snapshots_list = [data.copy()]
        snapshot_targets = [targets.copy()]
        snapshot_test = [test_data.copy()]

        # Generate further snapshots recursively
        _process_step_by_step(pipeline, data, targets, test_data, snapshots_list, snapshot_targets, snapshot_test)

        return snapshots_list, snapshot_targets, snapshot_test

    def copy_pipeline(self, element):
        """
        Copy a pipeline recursively.

        :param element: Pipeline, Transformer or Pipeline Step
        :return: AbstractPipeline
        """
        copied_steps = []
        # TODO Add Handling of other pipeline structures (nested transformers etc.)
        for step_name, step_process in element.steps:
            if isinstance(step_process, AbstractPipeline):
                logging.debug(f"Copying nested pipeline: {step_name}")
                copied_steps.append((step_name, self.copy_pipeline(step_process)))
            elif isinstance(step_process, ColumnTransformer):
                logging.debug(f"Copying ColumnTransformer: {step_name}")
                copied_transformers = [(name, self.copy_pipeline(transformer), columns)
                                       for name, transformer, columns in step_process.transformers]
                copied_steps.append((step_name, ColumnTransformer(copied_transformers)))
            else:
                logging.debug(f"Copying step: {step_name}")
                copied_steps.append((step_name, step_process))

        return self.pipeline.make_pipeline(steps=copied_steps)

    def generate_leave_one_out_pipelines(self) -> Tuple[list[AbstractPipeline], list[str]]:
        """
        Generate pipelines with one step left out at a time.

        :return: list of Pipelines
        """
        state = {
            'current_index': -1,
            'max_index': -1,
            'leave_one_out_index': 0,
            'left_out_step_names': []
        }

        def _generate(element: AbstractPipeline | ColumnTransformer,
                      name="") -> AbstractPipeline | None | ColumnTransformer:
            """
            Generate a copy of the pipeline with one step left out.

            :param element: Pipeline, Transformer or Pipeline Step to process
            :return: Pipeline
            """

            def apply(step_process, step_name):
                """
                Really apply the step (unless it is the one that is to be left out)

                @param step_process: Step to apply
                @param step_name: Name of that step
                """
                state['current_index'] += 1
                if state['max_index'] < state['current_index']:
                    state['max_index'] += 1

                if state['current_index'] == state['leave_one_out_index']:
                    logging.debug(f"Leaving out step: {step_name} ({state['current_index']})")
                    state['left_out_step_names'].append(step_name)
                    return None

                logging.debug(f"Copying step: {step_name} ({state['current_index']})")
                return step_process

            copied_steps = []

            if isinstance(element, ColumnTransformer):
                # Encountered a ColumnTransformer instead of normal pipeline (step)?
                copied_transformers = []
                for name, transformer, columns in element.transformers:
                    copied_transformers.append(
                            (name, new_pipeline, columns)
                            if (new_pipeline := _generate(transformer, name)) is not None
                            else ('noop', 'passthrough', columns)
                    )
                return ColumnTransformer(transformers=copied_transformers, verbose_feature_names_out=False,
                                         remainder='passthrough')
            elif isinstance(element, AbstractPipeline):
                # Normal pipeline element -- work on all pipeline steps
                for step_name, step_process in element.steps:
                    if isinstance(step_process, AbstractPipeline):
                        logging.debug(f"Copying nested pipeline: {step_name}")
                        if (new_pipeline := _generate(step_process)) is not None:
                            copied_steps.append((step_name, new_pipeline))

                    elif isinstance(step_process, ColumnTransformer):
                        logging.debug(f"Copying ColumnTransformer: {step_name}")
                        copied_steps.append(
                                (step_name, _generate(step_process))
                        )

                    else:
                        if (copied_step := apply(step_process, step_name)) is not None:
                            copied_steps.append((step_name, copied_step))
                if len(copied_steps) > 0:
                    return element.make_pipeline(steps=copied_steps)
                return None
            else:
                # Not a full pipeline but directly a step? Directly apply it
                return apply(element, name)

        pipelines = []

        while True:
            state['current_index'] = -1
            pipelines.append(_generate(self.pipeline))
            state['leave_one_out_index'] += 1
            if state['leave_one_out_index'] > state['max_index']:
                break

        return pipelines, state['left_out_step_names']

    def process(self):
        """ Process the data using the pipeline and store snapshots of the data at each step.
        """
        logging.debug("Run data through preprocessing pipeline...")

        # Ingore warnings for y shape
        warnings.simplefilter("ignore", DataConversionWarning)

        logging.debug("Generating leave-one-out pipelines and snapshots...")
        leave_out_pipelines, step_names = self.generate_leave_one_out_pipelines()

        snapshots_x = []
        snapshots_y = []
        snapshots_test_x = []
        snapshots_test_y = []

        models = []

        self.is_essential_list = []
        for num, (step_name, pipeline) in enumerate(zip(step_names, leave_out_pipelines)):
            if step_name in self.predefined_essential_steps:
                logging.debug(f"Step {num} ({step_name}) is predefined as essential.")
                self.is_essential_list.append(True)
                continue
            try:
                if hasattr(pipeline, 'fit_resample'):
                    transformed_data_train = pipeline.fit_resample(self.data.get_raw().train_data.copy(),
                                                                   y=self.data.get_raw().train_target.copy())
                else:
                    transformed_data_train = pipeline.fit_transform(self.data.get_raw().train_data.copy(),
                                                                    y=self.data.get_raw().train_target.copy())

                snapshot_train_features, snapshot_train_target = (
                    (transformed_data_train, self.data.get_raw().train_target)
                    if not isinstance(transformed_data_train, tuple)
                    else (transformed_data_train[0], transformed_data_train[1])
                )

                model = self.model.train_single(snapshot_train_features, snapshot_train_target)

                if isinstance(pipeline, ImblearnPipelineWrapper):
                    pipeline.set_params(noop='passthrough')  # Set noop step to passthrough to avoid issues during transform
                transformed_data_test = pipeline.transform(self.data.get_raw().test_data.copy())

                # handle case where transformed data is a tuple (features, target) in case of custom transformers
                snapshot_test_features, snapshot_test_target = (
                    (transformed_data_test, self.data.get_raw().test_target)
                    if not isinstance(transformed_data_test, tuple)
                    else (transformed_data_test[0], transformed_data_test[1])
                )

                snapshots_x.append(snapshot_train_features)
                snapshots_y.append(snapshot_train_target)
                snapshots_test_x.append(snapshot_test_features)
                snapshots_test_y.append(snapshot_test_target)
                models.append(model)

                self.is_essential_list.append(False)
            except ValueError as e:
                print(f"Could not transform pipeline, marking step {num} as essential. Reason: {e}")
                self.is_essential_list.append(True)

        # Get a list of all columns present at any time during the pipeline execution
        columns = set()
        for i, snapshot in enumerate(snapshots_x):
            columns.update(snapshot.columns)

        self.data.store_snapshots(snapshots_x, snapshots_y, snapshots_test_x,
                                  snapshots_test_y, Metrics.LEAVE_OUT)

        self.model.models_by_score[Metrics.LEAVE_OUT] = models

        logging.debug("Generate immediate impact snapshots")

        snapshots_x, snapshots_y, snapshots_test_x = self.generate_step_by_step_snapshots(self.pipeline,
                                                                                          self.data.get_raw().train_data,
                                                                                          self.data.get_raw().train_target,
                                                                                          self.data.get_raw().test_data,
                                                                                          store_step_names=True)
        snapshots_test_y = [self.data.get_raw().test_target] * len(snapshots_test_x)

        for i, snapshot in enumerate(snapshots_x):
            columns.update(snapshot.columns)
        self.columns = list(columns)

        essential_steps_string = ', '.join(
                [name for name, is_essential in zip(self.step_names, self.is_essential_list) if is_essential])
        inspect_steps_string = ', '.join(
                [name for name, is_essential in zip(self.step_names, self.is_essential_list) if not is_essential])
        if essential_steps_string:
            print(f"Essential steps: {essential_steps_string}")
        else:
            print("No essential steps")
        print(f"Steps that can be inspected: {inspect_steps_string}")

        # Store immediate impacts
        self.data.store_snapshots(snapshots_x, snapshots_y, snapshots_test_x, snapshots_test_y, Metrics.IMMEDIATE)

        # Validator: Ensure that every categorical data in the data is transformed
        if not self.data.is_valid():
            raise ValueError("Data contains non-transformed categorical columns. Please preprocess data first.")

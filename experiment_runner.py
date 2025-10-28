#!/usr/bin/env python3
"""
This script runs experiments with different configurations automatically.
The configuration is expected as a YAML file.

Example:
    $ python experiment_runner.py
        --config ./pipelines/experiments.yaml
        --output ./experiments/
        --sample_size 5
        [--replace]
        [--same_samples_across_experiments]
"""

import argparse
import json
import logging
import os
import random
import time
import traceback
from datetime import datetime

import numpy as np
from tqdm.auto import tqdm

from PIPE_X.data import DataWrapper
from PIPE_X.explainer import Explainer
from PIPE_X.metrics import Metrics
from PIPE_X.pipeline import unpickle_pipeline
from PIPE_X.utilities import load_data_from_csv, load_yaml_config, setup_logging


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the experiment runner script.

    :return: An argparse.Namespace object containing the parsed command-line arguments.

    Command-line arguments:
        --config: str, Path to the configuration file (YAML format) that defines the experiments.
        --output: str, Path to the directory where the experiment results will be saved.
        --sample_size: int, Number of samples to process in each experiment.
        --replace: bool, If specified, existing results with the same sample size will be overwritten.
        --same_samples_across_experiments: bool, If specified, the same samples will be used across all experiments for
                                                 each dataset.
        --no_affected: bool, If specified, disables tracking of affected columns and row counts for non-essential
                             pipeline steps.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the config file.')
    parser.add_argument('--output', required=True, help='Path to the output directory.')
    parser.add_argument('--sample_size', type=int, required=True, help='Number of samples to explain.')
    parser.add_argument('--replace', action='store_true',
                        help='''Overwrite existing results (with same sample size). 
                        Otherwise, only new experiments will be performed.''')
    parser.add_argument('--same_samples_across_experiments', action='store_true',
                        help='Use the same samples across experiments on the same dataset.')
    parser.add_argument('--no_affected', action='store_true',
                        help='Skip tracking affected columns and row counts for each non-essential pipeline step.')
    parser.add_argument('--no_f1', action='store_true',
                        help='Skip checking performance of the model variations.')
    return parser.parse_args()


def reorder_experiments(experiments: list) -> list:
    """
    Reorder experiment configurations if there are more than 3.

    This function ensures that the first and last experiments in the list are always executed first,
    and the remaining experiments are shuffled randomly.
    This is useful for scanning intermediate results while experiments are still running.

    :param experiments: list, A list of experiment configurations to be reordered.
    :return: list, The reordered list of experiment configurations.
    """

    if len(experiments) > 3:
        experiments[:] = [experiments[0], experiments[-1]] + random.sample(experiments[1:-1], len(experiments) - 2)
    return experiments


def run_experiment(name: str, dataset: str, sample_size: int, model: str, sample_ids: list[int], general_config: dict,
                   result_path: str, args: argparse.Namespace, explainer: Explainer, times: dict) -> None:
    """
    Run a single experiment and save results to a JSON file.

    :param name: str, Name of the experiment.
    :param dataset: str, Name of the dataset used in the experiment.
    :param sample_size: int, Number of samples to process in the experiment.
    :param model: str, Name of the model used in the experiment.
    :param sample_ids: list, List of sample IDs to process.
    :param general_config: dict, General configuration dictionary containing experiment settings.
    :param result_path: str, Path to save the experiment results as a JSON file.
    :param args: argparse.Namespace, Command-line arguments passed to the script.
    :param explainer: Explainer, An instance of the Explainer class used to run the experiment.

    :return: None
    """
    results = {
        'name': name,
        'dataset': dataset,
        'model': model,
        'sample_size': sample_size,
        'sample_ids': sample_ids,
        'pipeline_steps': explainer.pipeline.step_names,
        'pipeline_is_essential': explainer.pipeline.is_essential_list,
        'impacts': {},
        'aggregated_impacts': {},
        'raw_explanations': {},
        'time_measures': times,
    }

    if not args.no_affected:
        results['affected_columns'], results['affected_row_counts'] = explainer.affected_data_by_step()

    t_0_c = time.process_time()
    t_0_s = time.perf_counter()
    for sample in tqdm(sample_ids, desc=f'{model} for {name} on {dataset}', dynamic_ncols=True):
        logging.info(f'Running explainer for sample {sample}...')
        explainer.run(sample)

        for metric in general_config['metrics']:
            metric = Metrics(metric)
            results['impacts'].setdefault(sample, {})[metric] = explainer.impacts[metric]
            results['raw_explanations'].setdefault(sample, {})[metric] = explainer.raw_explanations[metric]

    t_d_s = time.perf_counter() - t_0_s
    t_d_c = time.process_time() - t_0_c
    results['time_measures']['avg_cycles_per_run'] = t_d_c / sample_size
    results['time_measures']['avg_seconds_per_run'] = t_d_s / sample_size
    results['time_measures']['run_cycles'] = t_d_c
    results['time_measures']['run_seconds'] = t_d_s

    results['aggregated_impacts'] = {
        Metrics(metric): np.mean([impacts[Metrics(metric)] for impacts in results['impacts'].values()], axis=0).tolist()
        for metric in general_config['metrics']
    }

    if not args.no_f1:
        # Check variation model performance
        results['f1_scores'] = {}
        for metric in general_config['metrics']:
            metric = Metrics(metric)
            results['f1_scores'][metric] = []
            for model, snapshot in zip(explainer.model.models_by_score[metric],
                                       explainer.data.snapshot_by_score[metric]):
                f1_score = model.evaluate(snapshot.test_data, snapshot.test_target)
                results['f1_scores'][metric].append(f1_score)

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)


def main(general_config: dict, experiment_configs: list[dict], args: argparse.Namespace) -> None:
    """
    Run all experiments based on the provided configurations.

    :param general_config: dict, Configuration dictionary
    :param experiment_configs: list, List of experiment configurations
    :param args: argparse.Namespace, Command line arguments

    :return: None
    """

    default_models = general_config['default_models']
    samples_per_dataset = {}

    reorder_experiments(experiment_configs)

    for i, experiment in enumerate(experiment_configs):
        print(f'Starting experiment {i + 1} of {len(experiment_configs)}')
        name = experiment['name']
        models = experiment.get('models', default_models)
        datasets = experiment['datasets'] if 'datasets' in experiment else [experiment['dataset']]
        seed, split_seed = experiment.get('seed', 42), experiment.get('split_seed', 23)
        additional_features = experiment.get('additional_features')
        essential_steps = experiment.get('essential_steps', [])

        # TODO Allow to set this? -> currently defaults handled in PIPE_X
        # test_size = experiment_configuration['test_size'] if 'test_size' in experiment_configuration else 0.2

        for dataset in datasets:
            try:
                data = load_data_from_csv(general_config['datasets'][dataset])
                pipeline = unpickle_pipeline(experiment['pipeline'])
                essential_steps_including_pipeline_defaults = essential_steps + pipeline.get_essential_steps()

                output_directory = os.path.join(args.output, f'{name}_{dataset}')
                os.makedirs(output_directory, exist_ok=True)

                for model in models:
                    # Setup logging to store log into file (remove/close old target if needed)
                    for handler in logging.root.handlers[:]:
                        logging.root.removeHandler(handler)
                    setup_logging(model, args.sample_size, directory=os.path.join('logs', f'{name}_{dataset}'))

                    result_path = os.path.join(output_directory, f'{model}_{args.sample_size}.json')
                    if os.path.exists(result_path) and not args.replace:
                        print(f'Skipping {model} for {name} on {dataset} as results already exist.')
                        logging.info(f'Skipping {model} for {name} on {dataset} as results already exist.')
                        continue

                    print(f'{model} for {name} on {dataset}')
                    print(datetime.now())
                    logging.info(f'Starting PIPE-X, data: {dataset}, pipeline: {pipeline}, model: {model}')

                    try:
                        rng = np.random.default_rng(seed)

                        logging.debug('Setting up Explainer...')
                        t_0_s = time.perf_counter()
                        t_0_c = time.process_time()
                        explainer = Explainer(data, pipeline, model, rng,
                                              predefined_essential_steps=essential_steps_including_pipeline_defaults,
                                              split_seed=split_seed, additional_features=additional_features)
                        t_1_s = time.perf_counter()
                        t_1_c = time.process_time()
                        explainer.setup()
                        t_2_s = time.perf_counter()
                        t_2_c = time.process_time()

                        time_dict = {
                            'setup_cycles': t_2_c - t_1_c,
                            'setup_seconds': t_2_s - t_1_s,
                            'init_cycles': t_1_c - t_0_c,
                            'init_seconds': t_1_s - t_0_s,
                        }

                        logging.debug('Running Explainer...')

                        dataset_for_sampling = experiment.get('dataset_for_sampling', dataset)

                        if args.same_samples_across_experiments and dataset_for_sampling in samples_per_dataset:
                            sample_ids = samples_per_dataset[dataset_for_sampling]
                        else:
                            data_for_sampling = DataWrapper(
                                    load_data_from_csv(config['datasets'][dataset_for_sampling]), rng,
                                    split_seed) if dataset_for_sampling != dataset else explainer.data

                            # Get sample-size-many samples from valid indices of the dataframe that do not contain NaN
                            valid_indices = data_for_sampling.get_raw().train_data.dropna().index.tolist()
                            # Ensure that sample ids are ints, not int64, as JSON cannot serialize int64
                            sample_ids = sorted([int(i) for i in rng.choice(valid_indices, size=args.sample_size)])

                            if args.same_samples_across_experiments:
                                samples_per_dataset[dataset_for_sampling] = sample_ids

                        run_experiment(name, dataset, args.sample_size, model, sample_ids, general_config, result_path,
                                       args, explainer, time_dict)

                    except Exception as e:
                        print(f'Error in experiment {name} on dataset {dataset} with model {model}: {e}')
                        print(traceback.format_exc())
                        logging.warning(f'Error in experiment {name} on dataset {dataset} with model {model}: {e}')
            except Exception as e:
                print(f'Error in experiment {name} on {dataset} (independent of model): {e}')
                print(traceback.format_exc())
                logging.warning(f'Error in experiment {name} on {dataset} (independent of model): {e}')
        print(datetime.now())


if __name__ == '__main__':
    cli_arguments = parse_arguments()

    config, experiment_configurations = load_yaml_config(cli_arguments.config)

    print(f'Running {len(experiment_configurations)} experiment configurations in config file.')
    main(config, experiment_configurations, cli_arguments)

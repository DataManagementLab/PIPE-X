""" This module provides a class to analyze the results of experiments
"""
import base64
import fnmatch
import io
import json
import os
from json import JSONDecodeError

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, HTML

from experiment_runner import load_yaml_config
from PIPE_X.metrics import Metrics


def split_matching_results(matching_results):
    """ Split matching results and ensure only one result is returned
    """
    if len(matching_results) > 1:
        raise Exception(
                "Multiple results found for the given experiment name (prefix), dataset, model and sample size")

    result = matching_results[0]
    experiment_config = result['config']

    pipeline_steps = result['pipeline_steps']
    pipeline_is_essential = result['pipeline_is_essential']
    pipeline_steps_reduced = [step for step, is_essential in zip(pipeline_steps, pipeline_is_essential) if
                              not is_essential]

    return result, experiment_config, pipeline_steps, pipeline_is_essential, pipeline_steps_reduced


class Analyzer:
    """ Class to analyze the results of experiments
    """
    results: list = []

    def __init__(self, base_path, experiment_config, result_path):
        self.BASE_PATH = base_path
        self.EXPERIMENT_CONFIG = experiment_config
        self.RESULT_PATH = result_path
        self.config, self.experiment_configs = load_yaml_config(str(os.path.join(base_path, experiment_config)))
        self.measures = self.config['metrics']
        print(f"Loaded {experiment_config}")
        print(f"Measures: {', '.join(self.measures)}")
        # Make sure table columns are wide enough
        pd.set_option("display.max_colwidth", None)

    # For all experiments in experiment_configs
    # check whether there is a corresponding result file and load it or mark that there is no corresponding result file
    def load_results(self):
        """ Load the results of the experiments
        """
        self.results = []
        for experiment in self.experiment_configs:
            experiment_name = experiment['name']

            if 'datasets' in experiment:
                datasets = experiment['datasets']
            else:
                datasets = [experiment['dataset']]

            if 'models' in experiment:
                models = experiment['models']
            else:
                models = self.config['default_models']

            # For each dataset possible for this experiment
            for dataset in datasets:
                for model in models:
                    # Find all files starting with the model name in the result folder
                    no_results = True
                    result_path = str(os.path.join(self.BASE_PATH, self.RESULT_PATH, f"{experiment_name}_{dataset}"))
                    if os.path.exists(result_path):
                        for result_file in os.listdir(result_path):
                            if fnmatch.fnmatch(result_file, f"{model}*.json"):
                                with open(os.path.join(result_path, result_file), 'r') as f:
                                    try:
                                        result = json.load(f)
                                        result['config'] = experiment
                                        result['experiment'] = experiment_name
                                        result['dataset'] = dataset
                                        result['model'] = model
                                        if 'sample_size' not in result:
                                            result['sample_size'] = len(result['sample_ids'])
                                        result['status'] = 'OK'
                                        self.results.append(result)
                                        no_results = False
                                    except JSONDecodeError as e:
                                        print(f"Error loading {os.path.join(result_path, result_file)}: {e}")
                    if no_results:
                        self.results.append({'experiment': experiment_name, 'dataset': dataset, 'model': model,
                                             'config': experiment,
                                             'error': 'No result file found', 'status': 'ERROR'})

    # Create a flowchart like figure visualizing given steps and whether they are marked as essential
    # noinspection PyMethodMayBeStatic
    def _visualize_pipeline(self, pipeline_steps, pipeline_is_essential, aggregated_impacts=None):
        if aggregated_impacts is None:
            aggregated_impacts = {}

        """
        g = nx.DiGraph()

        for i, step in enumerate(pipeline_steps):
            g.add_node(i, label=step, essential=pipeline_is_essential[i])
            if i > 0:
                g.add_edge(i - 1, i)

        # Custom layout to position all nodes in one row
        pos = {i: (i, 0) for i in g.nodes}
        colors = ['gray' if g.nodes[i]['essential'] else 'green' for i in g.nodes]

        agg_imp_index = 0
        labels = {}
        for i in g.nodes:
            if pipeline_is_essential[i]:
                labels[i] = f"{g.nodes[i]['label']}\n[Essential]"
            else:
                agg_str = "\n".join(
                        f"{k[0]}: {aggregated_impacts[k][agg_imp_index]:4f}" for k in aggregated_impacts.keys())
                labels[i] = f"{g.nodes[i]['label']}\n{agg_str}"
                agg_imp_index += 1

        fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(len(pipeline_steps) * 2, 4),
                                       gridspec_kw={'height_ratios': [1, 1.5]})

        nx.draw(g, pos, labels=labels, node_color=colors, node_shape='s', with_labels=True, node_size=5000,
                font_size=10,
                font_color='white', ax=ax1)
        """

        fig, ax = plt.subplots(1, 1, figsize=(len(pipeline_steps) * 2, 2))

        min_value = min(min(vs) for vs in aggregated_impacts.values())
        max_value = max(max(vs) for vs in aggregated_impacts.values())
        range = max_value - min_value

        # Draw the line graph for aggregated impacts
        first_row = True
        for measure, values in aggregated_impacts.items():
            c = ax._get_lines.get_next_color()
            agg_index = 0
            values_with_placeholders = [0, ]
            markers = [None, ]
            for step in pipeline_steps:
                if step == 'noop':
                    continue
                if not pipeline_is_essential[pipeline_steps.index(step)]:
                    values_with_placeholders.append(values[agg_index])
                    agg_index += 1
                    markers.append('o')
                else:
                    values_with_placeholders.append(None)
                    markers.append(None)
            values_with_placeholders.append(0)
            markers.append(None)
            # ax2.plot(pipeline_steps, values_with_placeholders, label=measure, marker='o')

            xs = np.arange(len(values_with_placeholders))
            series = np.array(values_with_placeholders).astype(np.double)
            s_mask = np.isfinite(series)

            for x, y, m, s in zip(xs, series, np.array(markers),
                                  [''] + [p for p in pipeline_steps if p != 'noop'] + ['']):
                ax.plot(x, y, linestyle='-', marker=m, color=c, label="_nolegend_")
                if m is not None:
                    ax.text(x, y + 0.3 * (-1 if not first_row else 1) * range, f"{y:.4f}", ha='center', va='bottom',
                            fontsize=10, color=c)
                if first_row:
                    s_name = s
                    if m is None and s != '':
                        s_name = f"[{s}]"
                    ax.text(x, min_value - 0.4 * range - 0.2 * range * (x % 2), s_name, ha='center', va='top',
                            fontsize=10)

            ax.plot(xs[s_mask], series[s_mask], label=measure, linestyle='-', marker=None, color=c)

            first_row = False

        # ax2.set_xlabel('Pipeline Steps')
        ax.set_ylabel('Impacts')
        # Hide the x-axis labels
        ax.set_xticks([])
        ax.legend()
        ax.yaxis.grid(False)
        # No lines in the background
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.subplots_adjust(hspace=0.01)
        # plt.tight_layout()
        plt.show()

    # Display the result as a table (rendered in jupyter notebook)
    # The table should list the experiment name, the dataset, the sample size.
    # If no corresponding result file is found, the table should indicate this.
    def display_results(self):
        """ Display the results as a table
        """
        df = pd.DataFrame(self.results)
        columns_to_display = ['experiment', 'dataset', 'model', 'sample_size', 'status', 'aggregated_impacts']
        return df[columns_to_display]

    def _get_matching_results(self, dataset, model, sample_size, experiment_name=None, param_min=None, param_max=None):
        """
        Identify all results matching the given parameters

        :param dataset: str, Dataset name
        :param model: str, Model name
        :param sample_size: int, Sample size
        :param experiment_name: str, Experiment name, will be treated as prefix (optional)
        :param param_min: float, Minimum range for parameter value (optional)
        :param param_max: float, Maximum range for parameter value (optional)
        """
        matching_results = []
        for r in self.results:
            if r['status'] == "OK" and r['model'] == model and r['dataset'].startswith(dataset) and r[
                'sample_size'] == sample_size and \
                    (experiment_name is None
                     or (isinstance(experiment_name, str) and r['experiment'].startswith(experiment_name))
                     or (isinstance(experiment_name, list) and r['experiment'] in experiment_name)) and \
                    (param_min is None or ('param' in r['config'] and r['config']['param'] >= param_min)) and \
                    (param_max is None or ('param' in r['config'] and r['config']['param'] <= param_max)):
                matching_results.append(r)

        if len(matching_results) == 0:
            raise Exception("No matching results found")

        return matching_results

    # Display details for a given experiment
    # This includes a graphic describing the pipeline, the configuration (dataset, nodel), the sample size,
    # the status and the aggregated impacts
    def display_experiment_details(self, experiment_name, dataset, model, sample_size):
        """ Display the details of a given experiment

        :param experiment_name: str, Experiment name
        :param dataset: str, Dataset name
        :param model: str, Model name
        :param sample_size: int, Sample size
        """
        result, experiment_config, pipeline_steps, pipeline_essential, pipeline_steps_reduced = split_matching_results(
                self._get_matching_results(dataset, model, sample_size, experiment_name))

        impacts_by_step = {
            measure:
                {step:
                     [v[measure][i] for v in result['impacts'].values()]
                 for i, step in enumerate(pipeline_steps_reduced)}
            for measure in result['aggregated_impacts'].keys()
        }

        sample_ids = result['sample_ids']
        sample_ids.sort()

        # Display the configuration
        print(f"Name: {experiment_name}")
        print(f"Dataset: {dataset}")
        print(f"Model: {model}")
        print(f"Pipeline: {experiment_config['pipeline']}\n")
        print(f"Sample size: {result['sample_size']}")
        print(f"Samples: {result['sample_ids']}")
        print(f"Pipeline steps: {', '.join(result['pipeline_steps'])}\n")
        print(f"Aggregated impacts: {result['aggregated_impacts']}")
        if 'f1_scores' in result:
            for key, values in result['f1_scores'].items():
                print(f"F1 {key}: {', '.join(str(v) for v in values)}")
        if 'time_measures' in result:
            time_measures = [
                {'What?': 'Init', 'CPU Cycles': result['time_measures']['init_cycles'],
                 'Seconds': result['time_measures']['init_seconds']},
                {'What?': 'Setup', 'CPU Cycles': result['time_measures']['setup_cycles'],
                 'Seconds': result['time_measures']['setup_seconds']},
                {'What?': 'Avg per Run', 'CPU Cycles': result['time_measures']['avg_cycles_per_run'],
                 'Seconds': result['time_measures']['avg_seconds_per_run']},
                {'What?': 'Total Run', 'CPU Cycles': result['time_measures']['run_cycles'],
                 'Seconds': result['time_measures']['run_seconds']},
            ]
            df_time = pd.DataFrame(time_measures, columns=['What?', 'CPU Cycles', 'Seconds'])
            display(df_time)

        pipeline_steps = result['pipeline_steps']
        pipeline_essential = result['pipeline_is_essential']

        # Visualize the pipeline
        self._visualize_pipeline(pipeline_steps, pipeline_essential, result['aggregated_impacts'])

        # Use whole width for the plot
        fig, axes = plt.subplots(1, len(result['aggregated_impacts'].keys()), figsize=(12, 6))
        fig.suptitle(f'Impacts per step\nExperiment: {experiment_name}, Dataset: {dataset}, Model: {model}')
        # Plot the impact distributions per step and measure
        for i, measure in enumerate(result['aggregated_impacts'].keys()):
            # Extract keys and values
            keys = list(impacts_by_step[measure].keys())
            values = list(impacts_by_step[measure].values())

            # Create a boxplot
            axes[i].boxplot(values)

            # Set x-tick labels to the dictionary keys
            axes[i].set_xticks(range(1, len(keys) + 1), keys, rotation=45)

            axes[i].set_title(f"Metric: {measure}")
            axes[i].set_xlabel('Steps')
            axes[i].set_ylabel('Impacts')

        # Show the plot
        plt.show()

        if 'affected_columns' in result:
            rows = []
            for step, affected_columns, affected_row_count in zip(pipeline_steps_reduced, result['affected_columns'],
                                                                  result['affected_row_counts']):
                affected_columns_str = ", ".join(affected_columns) if len(affected_columns) > 0 else "-"
                rows.append({'Step': step, 'Which columns?': affected_columns_str, 'No of rows': affected_row_count})
            df_effects = pd.DataFrame(rows, columns=['Step', 'Which columns?', 'No of rows'])
            print("Changes caused by each step:")
            display(df_effects)

    # Display details for a given experiment
    # This includes a graphic describing the pipeline, the configuration (dataset, nodel), the sample size,
    # the status and the aggregated impacts
    def display_experiment_details_for_all_models(self, experiment_name, dataset, models, sample_size):
        """ Display the details of a given experiment

        :param experiment_name: str, Experiment name
        :param dataset: str, Dataset name
        :param model: str, Model name
        :param sample_size: int, Sample size
        """
        for i, model in enumerate(models):
            result, experiment_config, pipeline_steps, pipeline_essential, pipeline_steps_reduced = split_matching_results(
                    self._get_matching_results(dataset, model, sample_size, experiment_name))
            if i == 0:
                # Display the configuration
                print(f"Name: {experiment_name}")
                print(f"Dataset: {dataset}")
                print(f"Model: {model}")
                print(f"Pipeline: {experiment_config['pipeline']}\n")
                print(f"Sample size: {result['sample_size']}")
                print(f"Samples: {result['sample_ids']}")
                print(f"Pipeline steps: {', '.join(result['pipeline_steps'])}\n")
                print(f"Aggregated impacts: {result['aggregated_impacts']}")
                if 'f1_scores' in result:
                    for key, values in result['f1_scores'].items():
                        print(f"F1 {key}: {', '.join(str(v) for v in values)}")
                if 'time_measures' in result:
                    time_measures = [
                        {'What?': 'Init', 'CPU Cycles': result['time_measures']['setup_cycles'], 'Seconds': result['time_measures']['setup_seconds']},
                        {'What?': 'Setup', 'CPU Cycles': result['time_measures']['init_cycles'], 'Seconds': result['time_measures']['init_seconds']},
                        {'What?': 'Avg per Run', 'CPU Cycles': result['time_measures']['avg_cycles_per_run'],
                         'Seconds': result['time_measures']['avg_seconds_per_run']},
                        {'What?': 'Total Run', 'CPU Cycles': result['time_measures']['run_cycles'], 'Seconds': result['time_measures']['run_seconds']},
                    ]
                    df_time = pd.DataFrame(time_measures, columns=['What?', 'CPU Cycles', 'Seconds'])
                    display(df_time)

                if 'affected_columns' in result:
                    rows = []
                    for step, affected_columns, affected_row_count in zip(pipeline_steps_reduced,
                                                                          result['affected_columns'],
                                                                          result['affected_row_counts']):
                        affected_columns_str = ", ".join(affected_columns) if len(affected_columns) > 0 else "-"
                        rows.append({'Step': step, 'Which columns?': affected_columns_str,
                                     'No of rows': affected_row_count})
                    df_effects = pd.DataFrame(rows, columns=['Step', 'Which columns?', 'No of rows'])
                    print("Changes caused by each step:")
                    display(df_effects)

            print(f"Model: {model}")

            # Visualize the pipeline
            self._visualize_pipeline(pipeline_steps, pipeline_essential, result['aggregated_impacts'])

    # Display feature attribution for a given experiment
    def display_feature_attribution(self, experiment_name, dataset, model, sample_size, show_tables=False):
        """ Display the details of a given experiment

        :param experiment_name: str, Experiment name
        :param dataset: str, Dataset name
        :param model: str, Model name
        :param sample_size: int, Sample size
        :param show_tables: bool, Whether to show the tables of feature attributions
        """
        result, experiment_config, pipeline_steps, pipeline_essential, pipeline_steps_reduced = split_matching_results(
                self._get_matching_results(dataset, model, sample_size, experiment_name))

        # Load the dataset itself
        data = pd.read_csv(str(os.path.join(self.BASE_PATH, self.config['datasets'][dataset])))

        sample_ids = result['sample_ids']
        sample_ids.sort()

        if "raw_explanations" in result:
            steps_by_measure = {}
            max_step_count = 0
            for measure in result['aggregated_impacts'].keys():
                if measure == Metrics.IMMEDIATE:
                    steps_by_measure[measure] = ['raw'] + pipeline_steps_reduced
                else:
                    steps_by_measure[measure] = pipeline_steps_reduced
                max_step_count = max(max_step_count, len(steps_by_measure[measure]))

            # Define markers and colors for each unique step
            markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_']
            colors = plt.get_cmap('tab10', max_step_count)

            # Show and plot the feature attribution distributions per step and measure
            for i, measure in enumerate(result['aggregated_impacts'].keys()):
                explanations = []
                for sample_id, values in result['raw_explanations'].items():
                    for step, exp in zip(steps_by_measure[measure], values[measure]):
                        sd = {'step': step, 'sample_id': sample_id}
                        d = {**sd, **dict(exp)}
                        explanations.append(d)

                df = pd.DataFrame(explanations)
                if show_tables:
                    title = f'<h4>Feature Attribution. '
                    title += f'Experiment: {experiment_name}, Dataset: {dataset}, Model: {model}, Metric: {measure}</h4'
                    display(HTML(title))
                    # Show step as first column
                    display(HTML(df.sort_values(by='sample_id', ascending=False).to_html(index=False)))

                fig, ax = plt.subplots(figsize=(12, 6))

                columns_to_plot = df.columns.difference(['step', 'sample_id'])
                steps_in_legend = set()

                # Plot each step with a different color and marker
                for j, step in enumerate(steps_by_measure[measure]):
                    step_data = df[df['step'] == step]  # .drop(['sample_id', 'step'])
                    for sample_id in step_data['sample_id']:
                        row = step_data[step_data['sample_id'] == sample_id]
                        label = step if step not in steps_in_legend else "_nolegend_"
                        ax.scatter(columns_to_plot, row[columns_to_plot].values.flatten(),
                                   marker=markers[j % len(markers)],
                                   color=colors(j), label=label, alpha=0.8)
                        steps_in_legend.add(step)

                # Set plot title and labels
                ax.set_title(
                        f'Feature Attribution per Step\n'
                        f'Experiment: {experiment_name}, Dataset: {dataset}, Model: {model}, Metric: {measure}')
                ax.set_ylabel('FA Value')
                ax.legend(title='Step')
                # Set y values from -0.1 to 0.3
                ax.set_ylim(-0.1, 0.3)
                plt.xticks(rotation=45)

                # Show the plot
                plt.show()

        if show_tables:
            # Display the samples as table (including the associated impacts)
            columns = [f"{step} [{measure[0]}]" for measure in result['aggregated_impacts'].keys() for step in
                       pipeline_steps_reduced]

            # Load impacts for each step and metric for a given sample row
            def load_impacts_for_samples(sample_row):
                """ Load impacts for each step and metric for samples
                """
                return pd.Series(
                        result['impacts'][str(sample_row.name)][impact][k] for impact in
                        result['aggregated_impacts'].keys()
                        for
                        k in
                        range(len(pipeline_steps_reduced)))

            samples_with_impacts = data.loc[sample_ids]
            samples_with_impacts[columns] = samples_with_impacts.apply(load_impacts_for_samples, axis=1)
            return samples_with_impacts
        return None

    # Inspect the same step over multiple experiments
    def get_impacts_across_experiments(self, step, model, dataset, sample_size, experiment_names=None):
        """ Get impacts for a given step across multiple experiments
        """
        impacts_by_experiment = {}
        raw_explanations_by_experiment = {}
        affected_column_counts = []
        affected_row_counts = []

        matching_results = self._get_matching_results(dataset, model, sample_size, experiment_names)

        for result in matching_results:
            pipeline_steps = result['pipeline_steps']
            pipeline_is_essential = result['pipeline_is_essential']
            pipeline_steps_reduced = [step for step, is_essential in zip(pipeline_steps, pipeline_is_essential)
                                      if
                                      not is_essential]

            # Get index of step to inspect
            step_index = pipeline_steps_reduced.index(step)

            if 'affected_columns' in result:
                affected_column_counts.append(len(result['affected_columns'][step_index]))
                affected_row_counts.append(result['affected_row_counts'][step_index])
            else:
                affected_column_counts.append(None)
                affected_row_counts.append(None)

            for measure in self.measures:
                s_index = step_index
                if measure not in impacts_by_experiment:
                    impacts_by_experiment[measure] = {}
                    raw_explanations_by_experiment[measure] = {}

                if 'param' in result['config']:
                    name = f"{result['config']['param']:.5f}"
                else:
                    name = f"{result['experiment']}_{result['dataset']}"
                # Get impacts for this step
                impacts_by_experiment[measure][name] = [v[measure][s_index] for v in
                                                        result['impacts'].values()]
                if "raw_explanations" in result:
                    if measure == "ONE_BY_ONE":
                        # Get index of step to inspect
                        s_index = step_index + 1
                    raw_explanations_by_experiment[measure][name] = [dict(v[measure][s_index]) for v
                                                                     in
                                                                     result[
                                                                         'raw_explanations'].values()]

        return impacts_by_experiment, raw_explanations_by_experiment, affected_column_counts, affected_row_counts

    def _expand_experiment_names(self, experiment_names: str | list[str]):
        """
        Expand the experiment names in case the user specified a placeholder
        :param experiment_names: list of experiment names or placeholder
        """
        if experiment_names == '*':
            experiment_names = [n['name'] for n in self.experiment_configs]
            experiment_names.sort()
        elif isinstance(experiment_names, str):
            experiment_names = [n['name'] for n in self.experiment_configs if n['name'].startswith(experiment_names)]
            experiment_names.sort()
        return experiment_names

    def inspect_step_across_experiments(self, step, model, dataset, sample_size, experiment_names=None):
        """ Inspect a given step across multiple experiments
        """
        impacts_by_experiment, raw_explanations_by_experiment, affected_column_counts, affected_row_counts = \
            self.get_impacts_across_experiments(
                    step,
                    model,
                    dataset,
                    sample_size,
                    experiment_names)

        for measure in self.measures:
            # Extract keys and values
            keys = list(impacts_by_experiment[measure].keys())
            values = list(impacts_by_experiment[measure].values())

            # Calculate means and standard deviations
            means = [np.mean(v) for v in values]
            std_devs = [np.std(v) for v in values]
            alignment = [2 * abs(np.sum([1 for value in v if value >= 0]) / len(v) - 0.5) for v in values]

            # consistency = [1 - s for s in std_devs]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'Step: {step}, Model: {model}, Dataset: {dataset}, Metric: {measure}')

            # Create a boxplot
            ax1.boxplot(values)

            # Set x-tick labels to the dictionary keys
            ax1.set_xticklabels(keys, rotation=90)

            ax1.set_xlabel('Experiments')
            ax1.set_ylabel('Impacts')

            # Create the line plot for means and standard deviations
            # Plot lightly
            ax2.plot(keys, std_devs, label='Standard Deviation', marker='x', alpha=0.3)
            ax2.plot(keys, alignment, label='Alignment', marker='.', alpha=0.3)
            # ax2.plot(keys, consistency, label='Consistency', marker='.', alpha=0.3)
            ax2.plot(keys, means, label='Average', marker='o')
            ax2.set_xticks(range(len(keys)))
            ax2.set_xticklabels(keys, rotation=90)
            ax2.set_xlabel('Experiments')
            ax2.legend()

            # Show the plots
            plt.tight_layout()
            plt.show()

            tabs = widgets.Tab()
            children = []
            titles = []

            first_exp = next(iter(raw_explanations_by_experiment[measure].values()))[0]

            for attr in first_exp.keys():
                # Create HTML content for the tab
                exps = []
                values = []
                str_rep = []
                for exp_name, exp_explanations in raw_explanations_by_experiment[measure].items():
                    exps.append(exp_name)
                    values.append([e[attr] for e in exp_explanations])
                    str_rep.append(f"{exp_name}: {', '.join(str(v) for v in values[-1])}")

                # noinspection PyUnusedLocal
                content = f"""
                <h3>Attribute: {attr}</h3>
                {'<br>'.join(str_rep)}
                """

                plt.figure(figsize=(10, 5))
                plt.boxplot(values, tick_labels=exps)
                plt.title(f'Attribute: {attr} [{measure}]')
                plt.xlabel('Experiments')
                plt.ylabel('FA Values')
                plt.xticks(rotation=90)

                # Save the plot to a BytesIO object
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()

                img_str = base64.b64encode(buf.read()).decode('utf-8')
                img_html = f'<img src="data:image/png;base64,{img_str}" />'

                # Create a widget for the tab content
                tab_content = widgets.HTML(value=img_html)
                children.append(tab_content)
                titles.append(attr)

            tabs.children = children
            for i, title in enumerate(titles):
                tabs.set_title(i, title)

            display(tabs)

        # Try to visualize affected column and row counts

        # Show two plot next to each other
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f'Affected number of Columns and Rows\nStep: {step}, Model: {model}, Dataset: {dataset}')

        # Plot the affected column counts as line plot
        keys = list(impacts_by_experiment[self.measures[0]].keys())

        ax1.plot(keys, affected_column_counts, label='Affected Columns', marker='o')
        ax1.tick_params(axis='x', rotation=90)
        ax1.set_ylabel('# of affected columns')

        ax2.plot(keys, affected_row_counts, label='Affected Rows', marker='o')
        ax2.tick_params(axis='x', rotation=90)
        ax2.set_ylabel('# of affected rows')

        plt.show()

    def inspect_averages_across_experiments(self, model, dataset, sample_size, experiment_names=None, param_min=None,
                                            param_max=None):
        """ Inspect averages across multiple experiments
        """
        print(f"Model: {model}, Dataset: {dataset}, Sample Size: {sample_size}, "
              f"Experiment: {experiment_names}, Param: {param_min}-{param_max}")

        matching_results = self._get_matching_results(dataset, model, sample_size,
                                                      experiment_name=experiment_names, param_min=param_min,
                                                      param_max=param_max)

        fig, axes = plt.subplots(1, len(self.measures)+1, figsize=(18, 6))
        fig.suptitle(f'Model: {model}, Dataset: {dataset}, Sample Size: {sample_size}')

        for measure, ax in zip(self.measures, axes):
            keys = []
            impacts = {}
            for result in matching_results:
                if 'param' in result['config']:
                    name = f"{result['config']['param']:.3f}"
                else:
                    name = f"{result['experiment']}_{result['dataset']}"
                keys.append(name)

                pipeline_steps = result['pipeline_steps']
                pipeline_is_essential = result['pipeline_is_essential']
                pipeline_steps_reduced = [step for step, is_essential in zip(pipeline_steps, pipeline_is_essential)
                                          if
                                          not is_essential]

                for step_index, step in enumerate(pipeline_steps_reduced):
                    if not step in impacts:
                        impacts[step] = []
                    impacts[step].append([v[measure][step_index] for v in result['impacts'].values()])

            for step, values in impacts.items():
                ax.plot(keys, [np.mean(v) for v in values], label=step)

            ax.set_xticks(range(len(keys)))
            ax.tick_params(axis='x', rotation=90)
            ax.set_xlabel('Experiments')
            ax.set_ylabel('Avg. Impacts')
            ax.set_title(f'Metric: {measure}')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)


        # Visualize leave-out f1 scores
        if 'f1_scores' in matching_results[0]:
            # Plot f1 scores as line graph
            keys = []
            f1s = {}
            for result in matching_results:
                if 'param' in result['config']:
                    name = f"{result['config']['param']:.3f}"
                else:
                    name = f"{result['experiment']}_{result['dataset']}"
                keys.append(name)

                pipeline_steps = result['pipeline_steps']
                pipeline_is_essential = result['pipeline_is_essential']
                pipeline_steps_reduced = [step for step, is_essential in zip(pipeline_steps, pipeline_is_essential)
                                          if
                                          not is_essential]

                for step_index, step in enumerate(pipeline_steps_reduced):
                    if not step in f1s:
                        f1s[step] = []
                    f1s[step].append(result['f1_scores']['leave_out_impact'][step_index])

            for step, values in f1s.items():
                axes[-1].plot(keys, [np.mean(v) for v in values], label=step)
            axes[-1].set_ylabel('F1 Score')
            axes[-1].set_xticks(range(len(keys)))
            axes[-1].tick_params(axis='x', rotation=90)
            axes[-1].grid(False)
            axes[-1].set_title(f'F1 scores (leave-out)')
            axes[-1].legend()

        # Show the plots
        plt.tight_layout()
        plt.show()
        plt.close()


    def inspect_effects_across_experiments(self, model, dataset, sample_size, experiment_names=None, param_min=None,
                                           param_max=None):
        """ Inspect step effects across experiments
        """
        print(f"Model: {model}, Dataset: {dataset}, Sample Size: {sample_size}, "
              f"Experiment: {experiment_names}, Param: {param_min}-{param_max}")

        matching_results = self._get_matching_results(dataset, model, sample_size,
                                                      experiment_name=experiment_names, param_min=param_min,
                                                      param_max=param_max)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Model: {model}, Dataset: {dataset}, Sample Size: {sample_size}')

        keys = []
        affected_column_counts_per_step = {}
        affected_row_counts_per_step = {}

        result = matching_results[0]
        pipeline_steps = result['pipeline_steps']
        pipeline_is_essential = result['pipeline_is_essential']
        pipeline_steps_reduced = [step for step, is_essential in zip(pipeline_steps, pipeline_is_essential)
                                  if
                                  not is_essential]

        for step in pipeline_steps_reduced:
            affected_column_counts_per_step[step] = []
            affected_row_counts_per_step[step] = []

        for result in matching_results:
            if 'param' in result['config']:
                name = f"{result['config']['param']:.5f}"
            else:
                name = f"{result['experiment']}_{result['dataset']}"
            keys.append(name)

            for step_index, step in enumerate(pipeline_steps_reduced):
                if 'affected_columns' in result:
                    affected_column_counts_per_step[step].append(len(result['affected_columns'][step_index]))
                    affected_row_counts_per_step[step].append(result['affected_row_counts'][step_index])
                else:
                    affected_column_counts_per_step[step].append(None)
                    affected_row_counts_per_step[step].append(None)

        for step, values in affected_column_counts_per_step.items():
            ax1.plot(keys, values, label=step)

        ax1.tick_params(axis='x', rotation=90)
        ax1.set_ylabel('# of affected columns')
        ax1.set_title(f'Columns')
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

        for step, values in affected_row_counts_per_step.items():
            ax2.plot(keys, values, label=step)

        ax2.tick_params(axis='x', rotation=90)
        ax2.set_ylabel('# of affected rows')
        ax2.set_title(f'Rows')
        ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

        # Show the plots
        plt.tight_layout()
        plt.show()

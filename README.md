# PIPE-X

## About this Project

This repository contains code for PIPE-X, a system providing impact measures of preprocessing steps.
For a detailed description of the system, please refer to the accompanying paper:
> "Towards Extending XAI for Full Data Science Pipelines" (Geisler et al., HILDA@SIGMODD 2024)

### Cite this work

IF you use this code or the underlying contributions in your research, please cite the following paper:

```bibtex
@inproceedings{10.1145/3665939.3665967,
author = {Geisler, Nadja and Binnig, Carsten},
title = {Towards Extending XAI for Full Data Science Pipelines},
year = {2024},
isbn = {9798400706936},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3665939.3665967},
doi = {10.1145/3665939.3665967},
booktitle = {Proceedings of the 2024 Workshop on Human-In-the-Loop Data Analytics},
pages = {1–7},
numpages = {7},
location = {Santiago, AA, Chile},
series = {HILDA  24}
}
```

## Requirements

This code was developed and tested under Python 3.12.
The required Python packages are listed in the `requirements.txt` file, which can be installed using pip.
If any packages cause issues, check the `requirements.txt` file for a comment on what a package is mainly used for.
Depending on your use case, you might be able to work without it.

## Setup

These instructions assume you have git, Python3.12 and virtualenv installed on your system and a bash-compatible shell.

1. Clone the repository and navigate to the project directory:

    ```bash
    git clone git@github.com:DataManagementLab/PIPE-X.git && cd PIPE-X
    ```

2. Setup Python environment, either using the setup bash script or by manually performing the steps within.
   The script will abort upon error, you can simply rerun after you take care of the issue.

    ```bash
    ./utils/setup.sh
    ```

3. Activate the virtual Python environment (virtualenv).

    ```bash
    source venv/bin/activate
    ```

4. You are now ready to run PIPE-X. For details see below.

## Quickstart

1. Download and preprocess dataset (see experiments.ipynb)
2. Generate and store pipeline  (see experiments.ipynb)
3. Run the main script with the generated pipeline with your chosen model, on your chosen sample

   ```bash
    python run.py --data_path <path-to-csv> \
                   --pipeline_path <path-to-pckl> \
                   --model GBT \
                   --sample 42
    ```

## Usage

Run `main.py` as script as above, see in-code comments for more information on script arguments.
Alternatively, use `main.py` or `experiment_runner.py` as inspiration to integrate the system into your own code.
For running experiments, see the `experiments.ipynb` notebook for examples.

## Inputs

In addition to the model architecture and sample index arguments, the script requires the following files to be present:

### Dataset

The dataset should be stored in a CSV file, with the target variable name prefixed by `target_` and otherwise
appropriate for default pandas csv import.
Several data setup scripts are included in this repository (see `data` directory) for reference.

The Bank Marketing, German Credit and Phishing Websites datasets demonstrate direct import from the UCI ML repository.
The Adult Census dataset demonstrates use of files, such as those in the UCI repository but not accessible via python.
The Titanic dataset demonstrates use of a dataset from openML.
The COMPAS dataset demonstrates use of a dataset from GitHub.

### Pipeline

PIPE-X supports scikit-learn pipelines out-of-the-box.
Additionally, a custom pipeline interface based on pandas dataframes is provided and supported.
It is largely compatible with sklearn but allows for additional operations.
This includes modifications of the target variable, which allows for dropping rows and other index manipulations.
The pipeline should be stored in a pickle file, examples are generated using the `pipeline_pickle.py` file.

Any custom transformers used in the pipeline should be stored/imported in `pipelines/custom_transformers.py`.

## Experiment runner

To run a bunch of experiments, use the experiment runner, as demonstrated in `experiments.ipynb`.
This script runs the system with different configurations automatically.
The configuration is expected as a yaml file, an example can be found in `experiments/experiments.yaml.bck`.
The generation of `.yaml` files is demonstrated in `experiments.ipynb`.

The experiment runner accepts the following arguments

   ```bash
    $ python experiment_runner.py
        --config ./experiments/experiments.yaml
        --output ./experiments/
        --sample_size 5
        [--replace]
        [--same_samples_across_experiments]
   ```

At least, a config file, an output directory and a sample size need to be provided.

Normally, experiments are only run when there is no existing result with the same sample size.
To overwrite those results, use the `--replace` flag.

`--same_samples_across_experiments` allows to use the same samples for all experiments on a given dataset.

The experiment runner uses `tqdm` to display progress bars.
They can be disabled by setting the environment variable `TQDM_DISABLE` to `1`.

## Experiments
Most experiment results can be found in the evaluation jupyter notebooks.
Those who want to reproduce numbers from scratch, uncomment the according lines in `run_all.sh` and run the script.

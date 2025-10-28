#!/usr/bin/env bash
# run all experiments on a new machine
# run setup.sh first!
# this script assumes a python runtime with requirements installed, etc.
# downloads and preps datasets, generates example pipelines
# execute as ./utils/run_all.sh

# for the rest of the script, abort on error, print executed commands
set -ex

# activate virtualenv if not already active
if [ -z ${VIRTUAL_ENV+x} ]; then
    source venv/bin/activate
fi

# add current directory to python path
# shellcheck disable=SC2155
export PYTHONPATH=$PYTHONPATH:$(pwd)

# disable tqdm progress bars
# export TQDM_DISABLE=1

# remove all datasets, pipelines and logs
./utils/clean_all.sh

# download datasets, generate pipelines and config files
jupyter nbconvert --to notebook --execute notebooks/experiments_sanity_checks.ipynb
jupyter nbconvert --to notebook --execute notebooks/experiments_FPP.ipynb

# run experiments

## sanity check
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_destroyer_unscaled.yaml' --output './experiments/' --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_destroyer_scaled.yaml' --output './experiments/' --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_destroyer_unscaled_minimal.yaml' --output './experiments/' --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_destroyer_scaled_minimal.yaml' --output './experiments/' --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_destroyer_unscaled_missing.yaml' --output './experiments/' --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_destroyer_scaled_missing.yaml' --output './experiments/' --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_destroyer_unscaled_minimal_missing.yaml' --output './experiments/' --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_destroyer_scaled_minimal_missing.yaml' --output './experiments/' --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_imputer_redundant.yaml' --output './experiments/' --sample_size 100  --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_imputer_nonredundant.yaml' --output './experiments/' --sample_size 100  --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_imputer_redundant_minimal.yaml' --output './experiments/' --sample_size 100  --same_samples_across_experiments --replace 2>&1 | tee out.log
# python './experiment_runner.py' --config './experiments/yaml_files/exp_sc_imputer_nonredundant_minimal.yaml' --output './experiments/' --sample_size 100  --same_samples_across_experiments --replace 2>&1 | tee out.log

## fair preprocessing pipelines
# python ./experiment_runner.py --config ./experiments/yaml_files/exp_FPP_adult.yaml --output ./experiments/ --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python ./experiment_runner.py --config ./experiments/yaml_files/exp_FPP_bank.yaml --output ./experiments/ --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python ./experiment_runner.py --config ./experiments/yaml_files/exp_FPP_german.yaml --output ./experiments/ --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log
# python ./experiment_runner.py --config ./experiments/yaml_files/exp_FPP_titanic.yaml --output ./experiments/ --sample_size 100 --same_samples_across_experiments --replace 2>&1 | tee out.log

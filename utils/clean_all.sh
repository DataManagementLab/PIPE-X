#!/usr/bin/env bash
# clean all outputs and logs
# this script assumes a python runtime with requirements installed, etc.
# this script will remove all datasets, intermediate results and logs
# execute as ./utils/clean_all.sh

# 0. Prep

# Display Warning and ask user to confirm y/n, abort on n
echo "This script will remove all datasets, intermediate results and logs in order to generate new ones."
read -p "Are you sure you want to continue? (y/n) " -n 1 -r

# If user input was not y, abort
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Aborted."
    exit 1
fi

# for the rest of the script, abort on error, print executed commands
set -ex

# activate virtualenv if not already active
if [ -z ${VIRTUAL_ENV+x} ]; then
    source venv/bin/activate
fi

# add current directory to python path
# shellcheck disable=SC2155
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 1. Remove residual/old versions of datasets
# Base directory for datasets
BASE_DIR="data"

# Loop through all subdirectories of BASE_DIR
for DIR in "$BASE_DIR"/*/; do
    # Find and delete all files except setup.py and .gitkeep
    find "$DIR" -type f ! -name 'setup.py' ! -name '.gitkeep' -delete
done

# 2. Remove residual/old versions of pipelines
# Base directory for pipelines
BASE_DIR="pipelines/pickles"

# remove all subdirectories of BASE_DIR
find "${BASE_DIR:?}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

# 3. Remove residual generated yaml files
# Base directory for config files
BASE_DIR="experiments/yaml_files"

# remove all files in BASE_DIR ending in '.yaml'
find "${BASE_DIR:?}" -type f -name '*.yaml' -delete

# 4. Remove residual old experiment outputs
# Base directory for outputs
BASE_DIR="experiments"

# remove all subdirectories of BASE_DIR except for yaml_files
find "${BASE_DIR:?}" -mindepth 1 -maxdepth 2 -type d ! -name 'yaml_files' -exec rm -rf {} +

# 5. Remove residual outputs of notebook execution
# Base directory for notebooks
BASE_DIR="notebooks"

# remove all files in BASE_DIR ending in '.nbconvert.ipynb'
find "${BASE_DIR:?}" -type f -name '*.nbconvert.ipynb' -delete


# 6. Remove residual generated json files
# Base directory for config files
BASE_DIR="experiments"

# remove all files in BASE_DIR ending in '.json'
find "${BASE_DIR:?}" -type f -name '*.json' -delete

# 7. remove out.log file if present
rm -f out.log

# 8. Remove residual old experiment logs
# Base directory for logs
BASE_DIR="logs"

# remove all subdirectories of BASE_DIR
find "${BASE_DIR:?}" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +

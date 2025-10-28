#!/usr/bin/env bash
# setup environment
# execute as ./utils/setup.sh

# abort on error, print executed commands
set -ex

# remove old virtualenv
rm -rf venv/

# Setup Python Environment
# Requires: Virtualenv, appropriate Python installation
virtualenv venv -p python3.12
source venv/bin/activate
pip install --upgrade setuptools pip wheel

# install Python requirements
pip install -r requirements.txt

deactivate

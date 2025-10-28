#!/usr/bin/env bash
# update environment
# execute as ./utils/update.sh

# abort on error, print executed commands
set -ex

# activate virtualenv if not already active
if [ -z ${VIRTUAL_ENV+x} ]; then
    source venv/bin/activate
fi

# update files to standard remote's main branch
git pull

# update Python requirements
pip install --upgrade setuptools pip wheel
pip install --upgrade -r requirements.txt

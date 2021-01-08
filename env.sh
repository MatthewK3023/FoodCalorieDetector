#!/bin/sh
export GITROOT=$(git rev-parse --show-toplevel)

if [ ! -e "$GITROOT/.env" ]; then
  if command -v pypy3 >/dev/null 2>/dev/null; then
    PYTHON=pypy3
  else
    PYTHON=python3
  fi
  if "$PYTHON" -m venv "$GITROOT/.env"; then
    . "$GITROOT/.env/bin/activate"
    pip install --upgrade pip wheel setuptools
    pip install -r "$GITROOT/requirements.txt"
    export PYTHONPATH="$GITROOT:$PYTHONPATH"
    export PATH="$GITROOT/src:$PATH"
  else
    echo "Unable to create a python3 venv" 1>&2
    false
  fi
else
  if [ -e "$GITROOT/.env" ]; then
    . "$GITROOT/.env/bin/activate"
  fi
  export PYTHONPATH="$GITROOT:$PYTHONPATH"
  export PATH="$GITROOT/src:$PATH"
fi

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BASE_PYTHON=${BASE_PYTHON:-/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python}
"$BASE_PYTHON" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-deps transformers==5.9.0 accelerate==1.14.0
.venv/bin/python -m unittest discover -s tests -v

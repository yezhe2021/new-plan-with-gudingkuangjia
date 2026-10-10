#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
PY=/hy-tmp/yezhe/data/miniconda3/envs/attnkv/bin/python
"$PY" test_multi_query_strict.py
"$PY" -u multi_query_strict.py --smoke
"$PY" -u multi_query_strict.py

#!/bin/bash
# Reproduce the Task 3 benchmark: downloads llama.cpp + models if missing,
# then runs both settings (F16 and Q4_K_M) on the same 10 prompts.
# Needs only Python 3 (stdlib). Run from this directory.
set -e
cd "$(dirname "$0")"
python3 bench.py "$@"

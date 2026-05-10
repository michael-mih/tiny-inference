#!/usr/bin/env bash
set -euo pipefail

python src/benchmark.py \
  --device cuda \
  --scenario optimized \
  --base-only \
  --warmup-runs 3 \
  --benchmark-runs 10 \
  --max-new-tokens 256 \
  "$@"

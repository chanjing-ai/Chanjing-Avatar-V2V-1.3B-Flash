#!/usr/bin/env bash
set -euo pipefail

uv run --extra train python training/train.py \
  --data_dir ./data/processed \
  --base_model ./models/base-1.3b \
  --output_dir ./outputs/training \
  --strategy auto

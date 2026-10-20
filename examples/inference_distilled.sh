#!/usr/bin/env bash
set -euo pipefail

uv run python inference.py \
  --video_path ./inputs/source.mp4 \
  --audio_path ./inputs/driving.wav \
  --model_variant distilled \
  --vae_path ./models/Wan2.2_VAE.pth \
  --wav2vec_dir ./models/chinese-wav2vec2-base \
  --lmk_weight_path ./models/landmark.onnx \
  --output_path ./outputs/result_distilled.mp4

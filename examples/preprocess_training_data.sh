#!/usr/bin/env bash
set -euo pipefail

uv run --extra train python training/preprocess.py \
  --data_dir ./data/raw \
  --save_dir ./data/processed \
  --vae_path ./models/Wan2.2_VAE.pth \
  --audio_encoder_path ./models/chinese-wav2vec2-base \
  --landmark_model_path ./models/landmark.onnx \
  --num_gpus 1

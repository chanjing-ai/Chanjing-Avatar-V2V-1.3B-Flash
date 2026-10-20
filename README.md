# Chanjing-Avatar V2V 1.3B Flash

[简体中文](README_zh.md) | English

Training and inference code for **Chanjing-Avatar V2V 1.3B Flash**, a compact
audio-driven video-to-video avatar model from [Chanjing](https://www.chanjing.cc/).
The pipeline preserves the source video's identity and scene while driving
facial motion from a new audio track.

> **Code-only release:** model weights, VAE weights, audio encoder weights,
> face detector weights, and example media are not included. Chanjing-Avatar
> V2V 1.3B Flash model weights are not currently released. End-to-end inference
> requires a compatible checkpoint obtained separately.

The model repository is available on
[Hugging Face](https://huggingface.co/cicada-ai/Chanjing-Avatar-V2V-1.3B-Flash),
but weights have not been uploaded yet.

## Model family

- [Chanjing-Avatar](https://github.com/chanjing-ai/Chanjing-Avatar): image-to-video digital human model.
- [Chanjing-Avatar V2V 5B](https://github.com/chanjing-ai/Chanjing-Avatar-V2V-5B): higher-capacity video-to-video model.
- [Chanjing-Avatar V2V 1.3B](https://github.com/chanjing-ai/Chanjing-Avatar-V2V-1.3B): long-form 1.3B video-to-video model.
- **Chanjing-Avatar V2V 1.3B Flash**: this compact codebase, with training and a three-stage inference pipeline.

## What is included

```text
inference.py                  # preprocess -> latent inference -> decode
chanjing_avatar/              # model, VAE, audio, face, and pipeline modules
training/preprocess.py        # multi-GPU training-data preprocessing
training/train.py             # Lightning training and safetensors delta export
configs/model_1_3b_flash.json # 1.3B Flash architecture (no parameters)
examples/                     # command examples only; no private filenames/media
tests/                        # config, CLI, and dataset-contract tests
```

The released configuration uses 30 transformer layers, hidden dimension 1536,
96 input channels, 48 output channels, and a `(4, 16, 16)` VAE stride. Audio
features condition every transformer block directly; there is no text prompt.

## Installation

Linux, Python 3.13, CUDA 12.8, and an NVIDIA GPU are the tested baseline.
`ffmpeg` must be installed and available on `PATH`.

```bash
uv sync
```

For training:

```bash
uv sync --extra train
```

YOLOv8 face-box support and distributed sequence parallelism are optional:

```bash
uv sync --extra yolo --extra distributed
```

PyTorch scaled-dot-product attention is used when FlashAttention is absent.
For FlashAttention, install it after the base environment is ready:

```bash
uv sync --extra build
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## Local model layout

Set `CHANJING_AVATAR_MODEL_DIR` or pass every asset path explicitly:

```text
models/
├── Chanjing-Avatar-V2V-1.3B-Flash/
│   ├── original/                    # original: 14 steps, CFG 7
│   │   ├── model-00001-of-00002.safetensors
│   │   └── model-00002-of-00002.safetensors
│   └── cfg-distilled/               # audio-CFG distilled: 6-step inference, CFG 1
│       ├── model-00001-of-00002.safetensors
│       └── model-00002-of-00002.safetensors
├── Wan2.2_VAE.pth
├── chinese-wav2vec2-base/
└── landmark.onnx
```

These files are intentionally ignored by Git. Their licenses may differ from
this repository's code license; verify the terms of every asset you use.

## Inference

Run preflight first. It checks the input, architecture, checkpoints, auxiliary
assets, and `ffmpeg` without loading a model or allocating GPU memory.

```bash
uv run python inference.py \
  --video_path ./inputs/source.mp4 \
  --audio_path ./inputs/driving.wav \
  --model_variant original \
  --validate_only
```

Then run all three stages:

```bash
bash examples/inference.sh
```

The default uses 81-frame clips, 9 motion frames, and 14 sampling steps. Output
intermediates are placed under `outputs/cache/`. Use `--help` for detector,
sampling, device, codec, and overwrite options.

Select the audio-CFG-distilled model with `--model_variant distilled`. This
chooses the `cfg-distilled` directory and applies `sample_steps=6`,
`sample_shift=7.0`, and `audio_guide_scale=1.0` automatically:

```bash
bash examples/inference_distilled.sh
```

The distillation objective absorbs the teacher's audio CFG response into one
conditional forward pass. Six steps is the recommended inference preset; the
model was not trained with four-step trajectory or DMD distillation.

## Training data

Each raw video must be 25 FPS and have a same-stem WAV file:

```text
data/raw/person_001.mp4
data/raw/person_001.wav
```

The preprocessing command generates landmarks when a same-stem `.npy` file is
absent, then emits complete sample triplets:

```text
latents_001.pth
audio_001.pth
latents_y_001.pth
```

```bash
bash examples/preprocess_training_data.sh
```

Validate a prepared dataset without loading the DiT:

```bash
uv run --extra train python training/train.py \
  --data_dir ./data/processed \
  --base_model ./models/base-1.3b \
  --validate_only
```

Start training with `bash examples/train.sh`. The default `--trainable unloaded`
loads shape-compatible parameters from the base checkpoint, freezes them, and
trains newly initialized or incompatible Flash parameters. Use `--trainable all`
for full fine-tuning. Checkpoints contain only trainable parameters and are
written as safetensors; `--init_from_scratch` must be explicit.

## Tests

```bash
uv run --extra train --extra test pytest
uv run python -m compileall -q inference.py chanjing_avatar training tests
```

Tests and preflight do not prove visual quality without the unreleased model
checkpoint. They verify the public code, configuration, imports, and data
contract.

## Responsible use

Use only media for which you have the necessary rights and consent. Clearly
label synthetic content where appropriate. Do not use this project for fraud,
impersonation, harassment, or other unlawful activity.

## Acknowledgements

This implementation builds on ideas and components from
[Wan2.2](https://github.com/Wan-Video/Wan2.2),
[Hugging Face Transformers](https://github.com/huggingface/transformers), and
[Diffusers](https://github.com/huggingface/diffusers).

Code in this repository is released under the [Apache License 2.0](LICENSE).

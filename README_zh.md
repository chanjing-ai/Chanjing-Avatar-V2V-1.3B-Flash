# 蝉镜数字人 V2V 1.3B Flash

简体中文 | [English](README.md)

**蝉镜数字人 V2V 1.3B Flash** 是[蝉镜](https://www.chanjing.cc/)推出的轻量级音频驱动
视频到视频数字人模型。本项目提供训练与推理代码，在保留源视频人物身份和场景的同时，
使用新音频驱动面部运动。

> **当前仅开放代码：**仓库不包含主模型、VAE、音频编码器、人脸检测器等权重，也不包含
> 示例媒体。蝉镜数字人 V2V 1.3B Flash 的模型权重暂未发布；完整推理需要另行取得兼容权重。

模型仓库已在 [Hugging Face](https://huggingface.co/cicada-ai/Chanjing-Avatar-V2V-1.3B-Flash)
创建，权重暂未上传。

## 模型家族

- [Chanjing-Avatar](https://github.com/chanjing-ai/Chanjing-Avatar)：图生视频数字人模型。
- [Chanjing-Avatar V2V 5B](https://github.com/chanjing-ai/Chanjing-Avatar-V2V-5B)：更高容量的视频到视频模型。
- [Chanjing-Avatar V2V 1.3B](https://github.com/chanjing-ai/Chanjing-Avatar-V2V-1.3B)：面向长视频的 1.3B 视频到视频模型。
- **蝉镜数字人 V2V 1.3B Flash**：本仓库，包含轻量版训练代码和三阶段推理流程。

## 仓库内容

```text
inference.py                  # 预处理 -> latent 推理 -> 解码
chanjing_avatar/              # 模型、VAE、音频、人脸和流水线模块
training/preprocess.py        # 多卡训练数据预处理
training/train.py             # Lightning 训练及 safetensors 增量权重导出
configs/model_1_3b_flash.json # 1.3B Flash 网络结构，不含参数
examples/                     # 脱敏后的命令示例
tests/                        # 配置、命令行和数据格式测试
```

公开配置采用 30 层 Transformer、1536 隐藏维度、96 输入通道、48 输出通道和
`(4, 16, 16)` VAE stride。音频特征直接进入每层交叉注意力，训练和推理均不使用文本提示词。

## 安装

已验证的基础环境为 Linux、Python 3.13、CUDA 12.8 和 NVIDIA GPU。系统还需要安装
`ffmpeg`，并确保它在 `PATH` 中。

```bash
uv sync
```

训练环境：

```bash
uv sync --extra train
```

YOLOv8 人脸框和分布式序列并行是可选功能：

```bash
uv sync --extra yolo --extra distributed
```

没有安装 FlashAttention 时会回退到 PyTorch 原生注意力。如需安装：

```bash
uv sync --extra build
uv pip install flash-attn==2.8.3 --no-build-isolation
```

## 本地模型目录

可设置 `CHANJING_AVATAR_MODEL_DIR`，也可以在命令中逐项指定：

```text
models/
├── Chanjing-Avatar-V2V-1.3B-Flash/
│   ├── original/                    # 原模型：14 步，CFG 7
│   │   ├── model-00001-of-00002.safetensors
│   │   └── model-00002-of-00002.safetensors
│   └── cfg-distilled/               # Audio CFG 蒸馏：6 步推理，CFG 1
│       ├── model-00001-of-00002.safetensors
│       └── model-00002-of-00002.safetensors
├── Wan2.2_VAE.pth
├── chinese-wav2vec2-base/
└── landmark.onnx
```

这些文件均被 Git 忽略。外部模型可能采用不同许可证，请自行核对所用资产的授权条件。

## 推理

先执行无权重加载预检。它会检查输入、模型结构、权重分片、辅助模型和 `ffmpeg`，但不会
加载模型或占用显存：

```bash
uv run python inference.py \
  --video_path ./inputs/source.mp4 \
  --audio_path ./inputs/driving.wav \
  --model_variant original \
  --validate_only
```

确认通过后运行完整三阶段流程：

```bash
bash examples/inference.sh
```

默认使用 81 帧片段、9 帧运动上下文和 14 步采样。中间文件写入 `outputs/cache/`。
人脸检测、采样、设备、编码器和覆盖参数可通过 `--help` 查看。

Audio CFG 蒸馏版使用 `--model_variant distilled`，它会自动选择 `cfg-distilled` 目录，
并设置 `sample_steps=6`、`sample_shift=7.0`、`audio_guide_scale=1.0`：

```bash
bash examples/inference_distilled.sh
```

该训练的蒸馏目标是把 Teacher 的 Audio CFG 效果吸收到一次有条件前向中。6 步是
当前推荐的少步推理配置，该模型没有进行 4 步轨迹蒸馏或 DMD 蒸馏。

## 训练数据

每个原始视频必须为 25 FPS，并配有同名 WAV 音频：

```text
data/raw/person_001.mp4
data/raw/person_001.wav
```

同名 `.npy` 不存在时，预处理程序会自动生成 landmark，随后为每个片段写出：

```text
latents_001.pth
audio_001.pth
latents_y_001.pth
```

```bash
bash examples/preprocess_training_data.sh
```

可先检查数据与路径，不加载 DiT：

```bash
uv run --extra train python training/train.py \
  --data_dir ./data/processed \
  --base_model ./models/base-1.3b \
  --validate_only
```

运行 `bash examples/train.sh` 开始训练。默认的 `--trainable unloaded` 会加载并冻结基础权重中
形状兼容的参数，只训练新初始化或形状不兼容的 Flash 参数；全参数微调使用
`--trainable all`。训练结果仅保存可训练参数，并统一导出为 safetensors。若不使用基础权重，
必须显式传入 `--init_from_scratch`。

## 测试

```bash
uv run --extra train --extra test pytest
uv run python -m compileall -q inference.py chanjing_avatar training tests
```

由于主模型权重暂未发布，测试和预检不能替代视觉效果验证；它们覆盖公开代码、配置、导入和
训练数据契约。

## 使用边界

请只处理已获得必要权利和本人授权的素材，并在适当场景明确标注合成内容。禁止用于欺诈、
冒充、骚扰及其他违法用途。

## 致谢

本项目参考并使用了 [Wan2.2](https://github.com/Wan-Video/Wan2.2)、
[Hugging Face Transformers](https://github.com/huggingface/transformers) 和
[Diffusers](https://github.com/huggingface/diffusers) 的相关工作。

本仓库代码采用 [Apache License 2.0](LICENSE) 开源。

import os
import gc
import math
import json
import random
import logging
import warnings
import argparse
import hashlib

from tqdm import tqdm
from pathlib import Path
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

warnings.filterwarnings("ignore")

import librosa
import numpy as np
import pyloudnorm as pyln
import torch
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None

from chanjing_avatar.preprocess import preprocess_video
from chanjing_avatar.audio.wav2vec2 import Wav2Vec2Model
from chanjing_avatar.models.dit import WanModelAudioProject


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
PINGPONG_LOOP_SCHEDULE_VERSION = "pingpong_loop_v1"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )


def torch_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def ensure_exists(path: str, name: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def choose_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is unavailable, fallback to cpu.")
        return torch.device("cpu")
    return torch.device(device)


def choose_dtype(dtype: str, device: torch.device) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        if device.type == "cuda":
            return torch.bfloat16
        logging.warning("bf16 on cpu is not recommended, fallback to fp32.")
        return torch.float32
    if dtype == "fp16":
        if device.type == "cuda":
            return torch.float16
        logging.warning("fp16 on cpu is unsupported, fallback to fp32.")
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stable_hash(payload: Dict[str, Any], length: int = 12) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def _normalize_path_for_hash(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path).resolve())


def build_preprocess_signature(
    face_box_source: str = "landmark",
    frame_num: int = 41,
    motion_frame: int = 5,
    margin: int = 32,
    lmk_weight_path: Optional[str] = None,
    yolov8_weight_path: Optional[str] = None,
    yolov8_conf: float = 0.5,
    yolov8_iou: float = 0.45,
    yolov8_imgsz: int = 640,
    yolov8_max_det: int = 300,
    yolov8_half: bool = False,
    yolov8_face_box_scale: float = 1.25,
    schedule_version: str = PINGPONG_LOOP_SCHEDULE_VERSION,
    loop_start_mode: str = "head",
) -> str:
    payload = {
        "schedule_version": str(schedule_version),
        "loop_start_mode": str(loop_start_mode),
        "face_box_source": face_box_source,
        "frame_num": int(frame_num),
        "motion_frame": int(motion_frame),
        "margin": int(margin),
        "lmk_weight_path": _normalize_path_for_hash(lmk_weight_path),
        "yolov8_weight_path": _normalize_path_for_hash(yolov8_weight_path),
        "yolov8_conf": float(yolov8_conf),
        "yolov8_iou": float(yolov8_iou),
        "yolov8_imgsz": int(yolov8_imgsz),
        "yolov8_max_det": int(yolov8_max_det),
        "yolov8_half": bool(yolov8_half),
        "yolov8_face_box_scale": float(yolov8_face_box_scale),
    }
    return _stable_hash(payload)


def default_preprocessed_path(
    video_path: str,
    face_box_source: str = "landmark",
    frame_num: int = 41,
    motion_frame: int = 5,
    margin: int = 32,
    lmk_weight_path: Optional[str] = None,
    yolov8_weight_path: Optional[str] = None,
    yolov8_conf: float = 0.5,
    yolov8_iou: float = 0.45,
    yolov8_imgsz: int = 640,
    yolov8_max_det: int = 300,
    yolov8_half: bool = False,
    yolov8_face_box_scale: float = 1.25,
    schedule_version: str = PINGPONG_LOOP_SCHEDULE_VERSION,
    loop_start_mode: str = "head",
) -> str:
    video = Path(video_path)
    tag = build_preprocess_signature(
        face_box_source=face_box_source,
        frame_num=frame_num,
        motion_frame=motion_frame,
        margin=margin,
        lmk_weight_path=lmk_weight_path,
        yolov8_weight_path=yolov8_weight_path,
        yolov8_conf=yolov8_conf,
        yolov8_iou=yolov8_iou,
        yolov8_imgsz=yolov8_imgsz,
        yolov8_max_det=yolov8_max_det,
        yolov8_half=yolov8_half,
        yolov8_face_box_scale=yolov8_face_box_scale,
        schedule_version=schedule_version,
        loop_start_mode=loop_start_mode,
    )
    return str(
        video.with_name(
            f"{video.stem}_preprocessed_{face_box_source}_{schedule_version}_{tag}.pt"
        )
    )


# -----------------------------
# audio utils
# -----------------------------
def loudness_norm(audio_array: np.ndarray, sr: int = 16000, lufs: float = -23.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    return pyln.normalize.loudness(audio_array, loudness, lufs)


def load_audio_array(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    ensure_exists(audio_path, "audio")
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    return loudness_norm(audio, sr)


def load_audio_encoder(wav2vec_dir: str, device: torch.device):
    ensure_exists(wav2vec_dir, "wav2vec_dir")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        wav2vec_dir,
        local_files_only=True,
    )
    encoder = Wav2Vec2Model.from_pretrained(
        wav2vec_dir,
        local_files_only=True,
    ).to(device)
    encoder.eval()
    if hasattr(encoder, "feature_extractor") and hasattr(encoder.feature_extractor, "_freeze_parameters"):
        encoder.feature_extractor._freeze_parameters()
    return feature_extractor, encoder


@torch.no_grad()
def extract_audio_embedding(
    audio_array: np.ndarray,
    feature_extractor: Wav2Vec2FeatureExtractor,
    encoder: Wav2Vec2Model,
    device: torch.device,
    sample_rate: int = 16000,
) -> torch.Tensor:
    audio_duration = len(audio_array) / sample_rate
    video_length = max(1, int(round(audio_duration * 25)))

    audio_feature = np.squeeze(
        feature_extractor(audio_array, sampling_rate=sample_rate).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().unsqueeze(0).to(device)

    outputs = encoder(audio_feature, seq_len=video_length, output_hidden_states=True)
    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None or len(outputs.hidden_states) <= 1:
        raise RuntimeError("Failed to extract audio embedding: empty hidden_states.")

    audio_emb = torch.stack(outputs.hidden_states[1:], dim=1).squeeze(0)
    if torch.isnan(audio_emb).any() or torch.isinf(audio_emb).any():
        raise RuntimeError("Audio embedding contains NaN/Inf.")
    audio_emb = rearrange(audio_emb, "b s d -> s b d")
    return audio_emb.detach().cpu()


def pad_audio_embedding_flip_tail(
    full_audio_emb: torch.Tensor,
    audio_end_idx: int,
    radius: int = 2,
) -> torch.Tensor:
    if full_audio_emb.ndim != 3:
        raise ValueError(
            f"Expected audio embedding shape [T, B, D], got {tuple(full_audio_emb.shape)}"
        )
    if full_audio_emb.shape[0] <= 0:
        raise ValueError("Audio embedding length must be positive.")
    if radius < 0:
        raise ValueError(f"radius must be >= 0, got {radius}")
    if audio_end_idx <= 0:
        return full_audio_emb

    required_len = audio_end_idx + radius + 1

    padded = full_audio_emb
    while padded.shape[0] < required_len:
        miss_length = required_len - padded.shape[0]
        take = min(max(miss_length, 1), padded.shape[0])
        add_audio_emb = torch.flip(padded[-take:], dims=[0])
        padded = torch.cat([padded, add_audio_emb], dim=0)

    return padded


def build_audio_context(
    full_audio_emb: torch.Tensor,
    frame_start: int,
    frame_num: int,
    radius: int = 2,
    device: Optional[torch.device] = None,
    tail_pad_mode: str = "flip_tail",
) -> torch.Tensor:
    if full_audio_emb.ndim != 3:
        raise ValueError(
            f"Expected audio embedding shape [T, B, D], got {tuple(full_audio_emb.shape)}"
        )

    if tail_pad_mode not in {"flip_tail", "clamp_last"}:
        raise ValueError(f"Unsupported tail_pad_mode: {tail_pad_mode}")

    source_audio = full_audio_emb
    if tail_pad_mode == "flip_tail":
        audio_end_idx = frame_start + frame_num
        source_audio = pad_audio_embedding_flip_tail(
            full_audio_emb=full_audio_emb,
            audio_end_idx=audio_end_idx,
            radius=radius,
        )

    num_audio_frames = source_audio.shape[0]

    frame_ids = torch.arange(frame_start, frame_start + frame_num, dtype=torch.long)
    offsets = torch.arange(-radius, radius + 1, dtype=torch.long)
    gather_idx = frame_ids.unsqueeze(1) + offsets.unsqueeze(0)
    gather_idx = gather_idx.clamp_(0, num_audio_frames - 1)

    clip_audio = source_audio[gather_idx]
    clip_audio = clip_audio.unsqueeze(0)

    if device is not None:
        clip_audio = clip_audio.to(device)

    return clip_audio


# -----------------------------
# preprocess utils
# -----------------------------
def normalize_source_frame_indices(
    item: Dict[str, Any],
    record_idx: int,
    expected_frame_num: Optional[int],
) -> List[int]:
    source_frame_indices = item.get("source_frame_indices", None)

    if source_frame_indices is None:
        if "source_start_idx" in item:
            start = int(item["source_start_idx"])
        else:
            start = int(item["start_idx"])

        if expected_frame_num is None:
            end = int(item.get("source_end_idx", item["end_idx"]))
            source_frame_indices = list(range(start, end + 1))
        else:
            source_frame_indices = list(range(start, start + int(expected_frame_num)))

    if isinstance(source_frame_indices, torch.Tensor):
        source_frame_indices = source_frame_indices.detach().cpu().tolist()
    elif isinstance(source_frame_indices, np.ndarray):
        source_frame_indices = source_frame_indices.tolist()

    if not isinstance(source_frame_indices, (list, tuple)):
        raise ValueError(
            f"Record[{record_idx}] source_frame_indices must be list/tuple, "
            f"got {type(source_frame_indices)}"
        )

    source_frame_indices = [int(x) for x in source_frame_indices]

    if expected_frame_num is not None and len(source_frame_indices) != expected_frame_num:
        raise ValueError(
            f"Record[{record_idx}] source_frame_indices length mismatch: "
            f"expected {expected_frame_num}, got {len(source_frame_indices)}"
        )

    total_source_frames = item.get("total_source_frames", None)
    if total_source_frames is not None:
        total_source_frames = int(total_source_frames)
        for source_idx in source_frame_indices:
            if source_idx < 0 or source_idx >= total_source_frames:
                raise ValueError(
                    f"Record[{record_idx}] source index out of range: "
                    f"idx={source_idx}, total_source_frames={total_source_frames}"
                )

    return source_frame_indices


def copy_preprocess_metadata(
    item: Dict[str, Any],
    source_frame_indices: List[int],
) -> Dict[str, Any]:
    out = {
        "bbox": item.get("bbox", None),
        "start_idx": int(item["start_idx"]),
        "end_idx": int(item["end_idx"]),
        "latent": item["latent"].contiguous(),
        "latent_y": item["latent_y"].contiguous(),
    }

    optional_keys = [
        "schedule_version",
        "virtual_start_idx",
        "virtual_end_idx",
        "source_start_idx",
        "source_end_idx",
        "total_source_frames",
        "clip_index_in_cycle",
        "cycle_segment_count",
        "loop_period",
        "loop_start_offset",
        "effective_start_idx",
        "effective_end_idx",
        "effective_span",
        "discard",
        "stride",
        "is_reversed",
        "loop_plan",
    ]

    for key in optional_keys:
        if key in item:
            out[key] = item[key]

    out["source_frame_indices"] = [int(x) for x in source_frame_indices]
    out["source_start_idx"] = int(source_frame_indices[0])
    out["source_end_idx"] = int(source_frame_indices[-1])

    return out


def validate_preprocessed_records(
    records: Any,
    expected_frame_num: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("Preprocessed file must be a non-empty list of clip records.")

    validated: List[Dict[str, Any]] = []
    detected_schedule_versions = set()

    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"Record[{idx}] is not a dict.")

        for key in ("latent", "latent_y", "start_idx", "end_idx"):
            if key not in item:
                raise ValueError(f"Record[{idx}] missing key: {key}")

        latent = item["latent"]
        latent_y = item["latent_y"]
        start_idx = int(item["start_idx"])
        end_idx = int(item["end_idx"])

        if not torch.is_tensor(latent) or latent.ndim != 4:
            raise ValueError(f"Record[{idx}]['latent'] must be a 4D tensor [C,T,H,W].")
        if not torch.is_tensor(latent_y) or latent_y.ndim != 4:
            raise ValueError(f"Record[{idx}]['latent_y'] must be a 4D tensor [C,T,H,W].")
        if latent.shape != latent_y.shape:
            raise ValueError(
                f"Record[{idx}] latent/latent_y shape mismatch: "
                f"{latent.shape} vs {latent_y.shape}"
            )
        if torch.isnan(latent).any() or torch.isinf(latent).any():
            raise ValueError(f"Record[{idx}] latent contains NaN/Inf.")
        if torch.isnan(latent_y).any() or torch.isinf(latent_y).any():
            raise ValueError(f"Record[{idx}] latent_y contains NaN/Inf.")

        this_frame_num = end_idx - start_idx + 1
        if expected_frame_num is not None and this_frame_num != expected_frame_num:
            raise ValueError(
                f"Record[{idx}] frame count mismatch: "
                f"expected {expected_frame_num}, got {this_frame_num}"
            )

        source_frame_indices = normalize_source_frame_indices(
            item=item,
            record_idx=idx,
            expected_frame_num=expected_frame_num,
        )

        schedule_version = str(item.get("schedule_version", "linear_legacy"))
        detected_schedule_versions.add(schedule_version)

        if schedule_version == PINGPONG_LOOP_SCHEDULE_VERSION:
            required_keys = [
                "source_frame_indices",
                "virtual_start_idx",
                "virtual_end_idx",
                "loop_plan",
            ]
            for key in required_keys:
                if key not in item:
                    raise ValueError(
                        f"Record[{idx}] schedule_version={schedule_version} "
                        f"requires key: {key}"
                    )

        validated.append(
            copy_preprocess_metadata(
                item=item,
                source_frame_indices=source_frame_indices,
            )
        )

    ref_shape = validated[0]["latent"].shape
    for idx, item in enumerate(validated[1:], start=1):
        if item["latent"].shape != ref_shape:
            raise ValueError(
                f"All records must share the same latent shape. "
                f"Record[0]={ref_shape}, Record[{idx}]={item['latent'].shape}"
            )

    if len(detected_schedule_versions) > 1:
        logging.warning(
            f"Multiple preprocess schedule versions detected: "
            f"{sorted(detected_schedule_versions)}"
        )

    return validated


def ensure_preprocessed(
    video_path: str,
    preprocessed_path: str,
    vae_path: str,
    device: str,
    frame_num: int,
    motion_frame: int,
    margin: int,
    face_box_source: str = "landmark",
    lmk_weight_path: Optional[str] = None,
    yolov8_weight_path: Optional[str] = None,
    yolov8_conf: float = 0.5,
    yolov8_iou: float = 0.45,
    yolov8_imgsz: int = 640,
    yolov8_max_det: int = 300,
    yolov8_half: bool = False,
    yolov8_face_box_scale: float = 1.25,
    loop_start_mode: str = "head",
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    need_preprocess = overwrite or (not os.path.exists(preprocessed_path))

    if not need_preprocess:
        try:
            records = torch.load(preprocessed_path, map_location="cpu", weights_only=True)
            validated = validate_preprocessed_records(records, expected_frame_num=frame_num)

            first_schedule = str(validated[0].get("schedule_version", "linear_legacy"))
            if first_schedule != PINGPONG_LOOP_SCHEDULE_VERSION:
                logging.warning(
                    f"Cached preprocessed file uses schedule_version={first_schedule}, "
                    f"expected={PINGPONG_LOOP_SCHEDULE_VERSION}. Will regenerate."
                )
                need_preprocess = True
            else:
                return validated

        except Exception as e:
            logging.warning(f"Invalid preprocessed file detected, will regenerate: {e}")
            need_preprocess = True

    if need_preprocess:
        logging.info("Running preprocess_video because cached preprocessed file is missing or invalid...")
        preprocess_video(
            video_path=video_path,
            out_pt=preprocessed_path,
            vae_path=vae_path,
            device=device,
            frame_num=frame_num,
            motion_frame=motion_frame,
            margin=margin,
            face_box_source=face_box_source,
            lmk_weight_path=lmk_weight_path,
            yolov8_weight_path=yolov8_weight_path,
            yolov8_conf=yolov8_conf,
            yolov8_iou=yolov8_iou,
            yolov8_imgsz=yolov8_imgsz,
            yolov8_max_det=yolov8_max_det,
            yolov8_half=yolov8_half,
            yolov8_face_box_scale=yolov8_face_box_scale,
            loop_start_mode=loop_start_mode,
        )

    records = torch.load(preprocessed_path, map_location="cpu", weights_only=True)
    return validate_preprocessed_records(records, expected_frame_num=frame_num)


def unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint is not a dict-like state_dict.")
    return obj


def strip_common_prefixes(state_dict: Dict[str, torch.Tensor], model_keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    model_keys = set(model_keys)
    candidates = ["", "model."]

    best_state_dict = state_dict
    best_score = -1

    for prefix in candidates:
        if prefix == "":
            candidate = state_dict
        else:
            candidate = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    candidate[k[len(prefix):]] = v

        score = len(model_keys.intersection(candidate.keys()))
        if score > best_score:
            best_score = score
            best_state_dict = candidate

    return best_state_dict


def load_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ensure_exists(path, "checkpoint")

    if path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("safetensors is required to load .safetensors checkpoints.")
        return safe_load_file(path, device="cpu")

    obj = torch.load(path, map_location="cpu", weights_only=True)
    return unwrap_state_dict(obj)


def load_merged_state_dict(paths: Sequence[str]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for p in paths:
        p = p.strip()
        if not p:
            continue
        part = load_checkpoint_state_dict(p)
        overlap = set(merged.keys()).intersection(part.keys())
        if overlap:
            logging.warning(
                f"Found overlapping keys while merging checkpoints, later keys overwrite earlier ones. "
                f"overlap={len(overlap)}"
            )
        merged.update(part)
    if not merged:
        raise RuntimeError("Merged state_dict is empty.")
    return merged


def build_and_load_model(
    config_path: str,
    model_path: Optional[str],
    device: torch.device,
    param_dtype: torch.dtype,
) -> WanModelAudioProject:
    with open(config_path, "r", encoding="utf-8") as f:
        model_kwargs = json.load(f)

    model = WanModelAudioProject(**model_kwargs).to(device)
    model.eval()

    if device.type == "cuda":
        model = model.to(dtype=param_dtype)

    if model_path:
        ckpt_paths = [p.strip() for p in model_path.split(",") if p.strip()]
        state_dict = load_merged_state_dict(ckpt_paths)
    else:
        raise ValueError("model_path must be provided.")

    state_dict = strip_common_prefixes(state_dict, model.state_dict().keys())
    model_state = model.state_dict()
    shape_mismatches = {
        key: (tuple(value.shape), tuple(model_state[key].shape))
        for key, value in state_dict.items()
        if key in model_state and value.shape != model_state[key].shape
    }
    if shape_mismatches:
        preview = ", ".join(
            f"{key}: checkpoint {actual}, model {expected}"
            for key, (actual, expected) in list(shape_mismatches.items())[:5]
        )
        raise RuntimeError(f"Checkpoint shape mismatch: {preview}")
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state}

    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            f"Checkpoint is incomplete; missing {len(incompatible.missing_keys)} model keys"
        )
    if incompatible.unexpected_keys:
        logging.warning(f"Unexpected keys while loading DiT: {len(incompatible.unexpected_keys)}")

    return model


# -----------------------------
# scheduler / denoise
# -----------------------------
def timestep_transform(
    t: torch.Tensor,
    shift: float = 5.0,
    num_timesteps: int = 1000,
) -> torch.Tensor:
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


def build_timesteps_exact_like_coupled(
    sampling_steps: int,
    num_timesteps: int = 1000,
    sample_shift: Optional[float] = 5.0,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    if sample_shift is None:
        sample_shift = 5.0

    timesteps = list(np.linspace(num_timesteps, 1, sampling_steps, dtype=np.float32))
    timesteps.append(0.0)
    timesteps = [torch.tensor([t], device=device, dtype=torch.float32) for t in timesteps]
    timesteps = [timestep_transform(t, shift=sample_shift, num_timesteps=num_timesteps) for t in timesteps]
    return timesteps


def inject_noise_into_latent(
    base_latent: torch.Tensor,
    motion_latent: torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    latent = base_latent.clone()
    motion_len = motion_latent.shape[1]

    latent[:, :motion_len] = motion_latent

    if motion_len < latent.shape[1]:
        if latent.shape[-2] > 4 and latent.shape[-1] > 4:
            noise = torch.randn(
                latent[:, motion_len:, 2:-2, 2:-2].shape,
                device=latent.device,
                dtype=latent.dtype,
                generator=generator,
            )
            latent[:, motion_len:, 2:-2, 2:-2] = noise
        else:
            noise = torch.randn(
                latent[:, motion_len:].shape,
                device=latent.device,
                dtype=latent.dtype,
                generator=generator,
            )
            latent[:, motion_len:] = noise

    return latent


def write_back_conditions_(
    latent: torch.Tensor,
    original_latent: torch.Tensor,
    motion_latent: torch.Tensor,
) -> None:
    motion_len = motion_latent.shape[1]
    latent[:, :motion_len] = motion_latent

    if motion_len >= latent.shape[1]:
        return

    if latent.shape[-2] > 4 and latent.shape[-1] > 4:
        latent[:, motion_len:, :2, :] = original_latent[:, motion_len:, :2, :]
        latent[:, motion_len:, -2:, :] = original_latent[:, motion_len:, -2:, :]
        latent[:, motion_len:, :, :2] = original_latent[:, motion_len:, :, :2]
        latent[:, motion_len:, :, -2:] = original_latent[:, motion_len:, :, -2:]


@torch.no_grad()
def denoise_one_clip(
    model: WanModelAudioProject,
    base_latent: torch.Tensor,
    cond_latent_y: torch.Tensor,
    audio_context: torch.Tensor,
    timesteps: List[torch.Tensor],
    num_train_timesteps: int,
    audio_guide_scale: float,
    motion_latent: torch.Tensor,
    param_dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    latent = inject_noise_into_latent(base_latent, motion_latent, generator=generator)
    null_context = torch.zeros_like(audio_context)

    progress_wrap = partial(tqdm, total=len(timesteps) - 1)
    for i in progress_wrap(range(len(timesteps) - 1)):
        timestep = timesteps[i]

        write_back_conditions_(latent, base_latent, motion_latent)

        noise_pred = model(
            x=latent.unsqueeze(0),
            timestep=timestep,
            context=audio_context,
            y=cond_latent_y.unsqueeze(0),
        )[0]

        if audio_guide_scale > 1.0: #audio_guide_scale > 1.0 表示使用音频引导，蒸馏模型设置为1
            if i % 2 == 0 or i == len(timesteps) - 2:
                null_noise_pred = model(
                    x=latent.unsqueeze(0),
                    timestep=timestep,
                    context=null_context,
                    y=cond_latent_y.unsqueeze(0),
                )[0]
                noise_pred = null_noise_pred + audio_guide_scale * (noise_pred - null_noise_pred)

        if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
            raise RuntimeError(f"noise_pred contains NaN/Inf at step {i}")

        noise_pred = -noise_pred

        dt = timesteps[i] - timesteps[i + 1]
        dt = dt / float(num_train_timesteps)

        latent = latent + noise_pred * dt[:, None, None, None]
        latent = latent.to(dtype=param_dtype)

        if torch.isnan(latent).any() or torch.isinf(latent).any():
            raise RuntimeError(f"latent contains NaN/Inf at step {i}")

    return latent


def build_saved_clip_metadata(
    clip_idx: int,
    record_idx: int,
    record: Dict[str, Any],
    frame_start: int,
    frame_end: int,
    valid_frame_end: int,
    valid_frame_count: int,
    latent: torch.Tensor,
) -> Dict[str, Any]:
    source_frame_indices = record.get("source_frame_indices", None)
    if source_frame_indices is not None:
        if isinstance(source_frame_indices, torch.Tensor):
            source_frame_indices = source_frame_indices.detach().cpu().tolist()
        elif isinstance(source_frame_indices, np.ndarray):
            source_frame_indices = source_frame_indices.tolist()
        source_frame_indices = [int(x) for x in source_frame_indices]

    source_start_idx = int(
        record.get(
            "source_start_idx",
            source_frame_indices[0] if source_frame_indices else record["start_idx"],
        )
    )
    source_end_idx = int(
        record.get(
            "source_end_idx",
            source_frame_indices[-1] if source_frame_indices else record["end_idx"],
        )
    )

    item = {
        "clip_index": int(clip_idx),
        "record_index": int(record_idx),

        "frame_start": int(frame_start),
        "frame_end": int(frame_end),
        "valid_frame_end": int(valid_frame_end),
        "valid_frame_count": int(valid_frame_count),

        "bbox": record.get("bbox", None),

        "source_start_idx": int(source_start_idx),
        "source_end_idx": int(source_end_idx),
        "source_frame_indices": source_frame_indices,

        "preprocess_start_idx": int(record["start_idx"]),
        "preprocess_end_idx": int(record["end_idx"]),

        "latent": latent.detach().cpu(),
    }

    optional_keys = [
        "schedule_version",
        "virtual_start_idx",
        "virtual_end_idx",
        "total_source_frames",
        "clip_index_in_cycle",
        "cycle_segment_count",
        "loop_period",
        "loop_start_offset",
        "effective_start_idx",
        "effective_end_idx",
        "effective_span",
        "discard",
        "stride",
        "is_reversed",
        "loop_plan",
    ]

    for key in optional_keys:
        if key in record:
            item[key] = record[key]

    return item


# -----------------------------
# main inference engine
# -----------------------------
class LatentOnlyInferenceEngine:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = choose_device(args.device)
        self.audio_device = choose_device(args.audio_device or args.device)
        self.param_dtype = choose_dtype(args.dtype, self.device)

        self.num_train_timesteps = 1000
        self.dit_config_path = args.dit_config_path

        self.model = build_and_load_model(
            config_path=self.dit_config_path,
            model_path=args.model_path,
            device=self.device,
            param_dtype=self.param_dtype,
        )

        self.audio_feature_extractor, self.audio_encoder = load_audio_encoder(
            args.wav2vec_dir,
            self.audio_device,
        )

        if self.device.type == "cuda":
            torch.set_grad_enabled(False)

    def run(self) -> Dict[str, Any]:
        args = self.args

        ensure_exists(args.video_path, "video_path")
        ensure_exists(args.audio_path, "audio_path")

        face_box_source = getattr(args, "face_box_source", "landmark")
        lmk_weight_path = getattr(args, "lmk_weight_path", None)
        yolov8_weight_path = getattr(args, "yolov8_weight_path", None)
        yolov8_conf = float(getattr(args, "yolov8_conf", 0.5))
        yolov8_iou = float(getattr(args, "yolov8_iou", 0.45))
        yolov8_imgsz = int(getattr(args, "yolov8_imgsz", 640))
        yolov8_max_det = int(getattr(args, "yolov8_max_det", 300))
        yolov8_half = bool(getattr(args, "yolov8_half", False))
        yolov8_face_box_scale = float(getattr(args, "yolov8_face_box_scale", 1.25))
        loop_start_mode = str(getattr(args, "loop_start_mode", "head"))

        preprocessed_path = args.preprocessed_path or default_preprocessed_path(
            video_path=args.video_path,
            face_box_source=face_box_source,
            frame_num=args.frame_num,
            motion_frame=args.motion_frame,
            margin=args.preprocess_margin,
            lmk_weight_path=lmk_weight_path,
            yolov8_weight_path=yolov8_weight_path,
            yolov8_conf=yolov8_conf,
            yolov8_iou=yolov8_iou,
            yolov8_imgsz=yolov8_imgsz,
            yolov8_max_det=yolov8_max_det,
            yolov8_half=yolov8_half,
            yolov8_face_box_scale=yolov8_face_box_scale,
            schedule_version=PINGPONG_LOOP_SCHEDULE_VERSION,
            loop_start_mode=loop_start_mode,
        )

        records = ensure_preprocessed(
            video_path=args.video_path,
            preprocessed_path=preprocessed_path,
            vae_path=args.vae_path,
            device=args.preprocess_device or args.device,
            frame_num=args.frame_num,
            motion_frame=args.motion_frame,
            margin=args.preprocess_margin,
            face_box_source=face_box_source,
            lmk_weight_path=lmk_weight_path,
            yolov8_weight_path=yolov8_weight_path,
            yolov8_conf=yolov8_conf,
            yolov8_iou=yolov8_iou,
            yolov8_imgsz=yolov8_imgsz,
            yolov8_max_det=yolov8_max_det,
            yolov8_half=yolov8_half,
            yolov8_face_box_scale=yolov8_face_box_scale,
            loop_start_mode=loop_start_mode,
            overwrite=args.overwrite_preprocess,
        )

        logging.info("Loading audio and extracting embedding...")
        audio_array = load_audio_array(args.audio_path, sample_rate=args.audio_sample_rate)
        full_audio_emb = extract_audio_embedding(
            audio_array=audio_array,
            feature_extractor=self.audio_feature_extractor,
            encoder=self.audio_encoder,
            device=self.audio_device,
            sample_rate=args.audio_sample_rate,
        )

        total_audio_frames = int(full_audio_emb.shape[0])
        if total_audio_frames <= 0:
            raise RuntimeError("Audio embedding length is zero.")

        stride = args.frame_num - args.motion_frame
        if stride <= 0:
            raise ValueError(
                f"frame_num must be larger than motion_frame, got {args.frame_num=} {args.motion_frame=}"
            )

        num_required_clips = max(
            1,
            math.ceil(max(total_audio_frames - args.frame_num, 0) / stride) + 1,
        )

        logging.info(
            f"audio_frames={total_audio_frames}, frame_num={args.frame_num}, "
            f"motion_frame={args.motion_frame}, stride={stride}, "
            f"num_required_clips={num_required_clips}"
        )
        logging.info(
            f"preprocess_source={face_box_source}, "
            f"preprocessed_path={preprocessed_path}, "
            f"loop_start_mode={loop_start_mode}"
        )

        timesteps = build_timesteps_exact_like_coupled(
            sampling_steps=args.sample_steps,
            num_timesteps=self.num_train_timesteps,
            sample_shift=args.sample_shift,
            device=self.device,
        )

        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(args.seed)

        motion_latent_len_next = int(1 + (args.motion_frame - 1) // 4)
        prev_motion_latent: Optional[torch.Tensor] = None
        saved_clips: List[Dict[str, Any]] = []

        for clip_idx in range(num_required_clips):
            frame_start = clip_idx * stride
            frame_end = frame_start + args.frame_num - 1
            valid_frame_end = min(frame_end, total_audio_frames - 1)
            valid_frame_count = valid_frame_end - frame_start + 1

            record_idx = clip_idx % len(records)
            record = records[record_idx]

            base_latent = record["latent"].to(device=self.device, dtype=self.param_dtype)
            cond_latent_y = record["latent_y"].to(device=self.device, dtype=self.param_dtype)

            current_motion_frame = 1 if clip_idx == 0 else args.motion_frame
            current_motion_latent_len = int(1 + (current_motion_frame - 1) // 4)

            if clip_idx == 0:
                motion_latent = base_latent[:, :current_motion_latent_len].clone()
            else:
                if prev_motion_latent is None:
                    raise RuntimeError("prev_motion_latent is None for non-first clip.")
                motion_latent = prev_motion_latent

            audio_context = build_audio_context(
                full_audio_emb=full_audio_emb,
                frame_start=frame_start,
                frame_num=args.frame_num,
                radius=args.audio_radius,
                device=self.device,
                tail_pad_mode="flip_tail",
            )

            logging.info(
                f"[clip {clip_idx + 1}/{num_required_clips}] "
                f"record_idx={record_idx}, frame_range=[{frame_start}, {frame_end}], "
                f"valid_frames={valid_frame_count}, "
                f"schedule={record.get('schedule_version', 'linear_legacy')}, "
                f"is_reversed={record.get('is_reversed', False)}"
            )

            latent = denoise_one_clip(
                model=self.model,
                base_latent=base_latent,
                cond_latent_y=cond_latent_y,
                audio_context=audio_context,
                timesteps=timesteps,
                num_train_timesteps=self.num_train_timesteps,
                audio_guide_scale=args.audio_guide_scale,
                motion_latent=motion_latent,
                param_dtype=self.param_dtype,
                generator=generator,
            )

            prev_motion_latent = latent[:, -motion_latent_len_next:].detach()

            saved_clips.append(
                build_saved_clip_metadata(
                    clip_idx=clip_idx,
                    record_idx=record_idx,
                    record=record,
                    frame_start=frame_start,
                    frame_end=frame_end,
                    valid_frame_end=valid_frame_end,
                    valid_frame_count=valid_frame_count,
                    latent=latent,
                )
            )

            del base_latent, cond_latent_y, audio_context, latent
            torch_gc()

        first_record = records[0]
        first_loop_plan = first_record.get("loop_plan", None)
        preprocess_schedule_version = str(
            first_record.get("schedule_version", "linear_legacy")
        )

        output = {
            "meta": {
                "video_path": args.video_path,
                "audio_path": args.audio_path,
                "preprocessed_path": preprocessed_path,
                "frame_num": int(args.frame_num),
                "motion_frame": int(args.motion_frame),
                "stride": int(stride),
                "audio_radius": int(args.audio_radius),
                "audio_sample_rate": int(args.audio_sample_rate),
                "audio_frames": int(total_audio_frames),
                "sample_steps": int(args.sample_steps),
                "sample_shift": float(args.sample_shift) if args.sample_shift is not None else None,
                "audio_guide_scale": float(args.audio_guide_scale),
                "num_train_timesteps": int(self.num_train_timesteps),
                "dtype": args.dtype,
                "seed": int(args.seed),
                "num_required_clips": int(num_required_clips),
                "source_record_count": int(len(records)),
                "latent_motion_len": int(motion_latent_len_next),
                "scheduler_impl": "chanjing_avatar.euler_flow_match",
                "target_fps": 25.0,
                "model_spec": getattr(args, "model_path", None),

                "face_box_source": face_box_source,
                "preprocess_margin": int(args.preprocess_margin),
                "preprocess_schedule_version": preprocess_schedule_version,
                "loop_start_mode": loop_start_mode,
                "loop_plan": first_loop_plan,
                "preprocess_loop_period": first_record.get("loop_period", None),
                "preprocess_effective_span": first_record.get("effective_span", None),
                "preprocess_effective_start_idx": first_record.get("effective_start_idx", None),
                "preprocess_effective_end_idx": first_record.get("effective_end_idx", None),
                "preprocess_discard": first_record.get("discard", None),
                
                "lmk_weight_path": lmk_weight_path,
                "yolov8_weight_path": yolov8_weight_path,
                "yolov8_conf": yolov8_conf,
                "yolov8_iou": yolov8_iou,
                "yolov8_imgsz": yolov8_imgsz,
                "yolov8_max_det": yolov8_max_det,
                "yolov8_half": yolov8_half,
                "yolov8_face_box_scale": yolov8_face_box_scale,
                "preprocess_signature": build_preprocess_signature(
                    face_box_source=face_box_source,
                    frame_num=args.frame_num,
                    motion_frame=args.motion_frame,
                    margin=args.preprocess_margin,
                    yolov8_weight_path=yolov8_weight_path,
                    yolov8_conf=yolov8_conf,
                    yolov8_iou=yolov8_iou,
                    yolov8_imgsz=yolov8_imgsz,
                    yolov8_max_det=yolov8_max_det,
                    yolov8_half=yolov8_half,
                    yolov8_face_box_scale=yolov8_face_box_scale,
                    schedule_version=PINGPONG_LOOP_SCHEDULE_VERSION,
                    loop_start_mode=loop_start_mode,
                ),
            },
            "clips": saved_clips,
        }
        return output

import os
import gc
import json
import argparse
import logging
import tempfile
import subprocess
import hashlib

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from chanjing_avatar.models.vae import Wan2_2_VAE


PINGPONG_LOOP_SCHEDULE_VERSION = "pingpong_loop_v1"


# ============================================================
# basic
# ============================================================
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )


def torch_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_exists(path: str, name: str) -> None:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"{name} not found: {path}")


def choose_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA unavailable, fallback to cpu.")
        return torch.device("cpu")
    return torch.device(device)


def choose_dtype(dtype: str, device: torch.device) -> torch.dtype:
    dtype = dtype.lower()
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype == "fp16":
        return torch.float16 if device.type == "cuda" else torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


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

# ============================================================
# latent / preprocess loading
# ============================================================
def load_latent_result(latent_path: str) -> Dict[str, Any]:
    ensure_exists(latent_path, "latent_path")
    result = torch.load(latent_path, map_location="cpu", weights_only=True)

    if not isinstance(result, dict):
        raise ValueError("Latent result must be a dict.")
    if "meta" not in result or "clips" not in result:
        raise ValueError("Latent result must contain 'meta' and 'clips'.")
    if not isinstance(result["clips"], list) or len(result["clips"]) == 0:
        raise ValueError("Latent result 'clips' must be a non-empty list.")
    if not isinstance(result["meta"], dict):
        raise ValueError("Latent result 'meta' must be a dict.")
    return result


def load_preprocessed_records(preprocessed_path: str) -> List[Dict[str, Any]]:
    ensure_exists(preprocessed_path, "preprocessed_path")
    records = torch.load(preprocessed_path, map_location="cpu", weights_only=True)
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("Invalid preprocessed file: must be a non-empty list.")
    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid preprocessed record[{idx}]: must be a dict.")
    return records


def infer_total_output_frames(meta: Dict[str, Any], clips: List[Dict[str, Any]]) -> int:
    if "audio_frames" in meta:
        return int(meta["audio_frames"])

    max_idx = -1
    motion_frame = int(meta.get("motion_frame", 5))
    for i, clip in enumerate(clips):
        frame_start = int(clip["frame_start"])
        valid_frame_count = int(
            clip.get(
                "valid_frame_count",
                int(clip["valid_frame_end"]) - frame_start + 1,
            )
        )
        local_start = 0 if i == 0 else motion_frame
        if valid_frame_count > local_start:
            max_idx = max(max_idx, frame_start + valid_frame_count - 1)
    return max_idx + 1


def resolve_path_with_meta(
    cli_value: Optional[str],
    meta_value: Optional[str],
    field_name: str,
    required: bool = True,
) -> Optional[str]:
    resolved = cli_value if cli_value else meta_value
    if required and not resolved:
        raise ValueError(
            f"{field_name} is required, but neither CLI nor latent meta provides it."
        )
    if cli_value and meta_value and os.path.normpath(cli_value) != os.path.normpath(meta_value):
        logging.warning(
            f"{field_name} overridden by CLI.\n"
            f"  meta: {meta_value}\n"
            f"  cli : {cli_value}"
        )
    return resolved


def normalize_int_list(value: Any, name: str) -> Optional[List[int]]:
    if value is None:
        return None

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be list/tuple/ndarray/tensor, got {type(value)}")

    return [int(x) for x in value]


def get_record_from_preprocessed(
    preprocessed_records: Optional[List[Dict[str, Any]]],
    record_index: int,
) -> Optional[Dict[str, Any]]:
    if preprocessed_records is None or len(preprocessed_records) == 0:
        return None
    return preprocessed_records[int(record_index) % len(preprocessed_records)]


# ============================================================
# geometry helpers
# ============================================================
def pick_target_size_from_box(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x_min, y_min, x_max, y_max = box
    frame_w = int(x_max - x_min)
    frame_h = int(y_max - y_min)

    candidates = [(512, 512), (448, 576), (576, 448)]
    orig_ratio = frame_w / max(frame_h, 1)

    target_w, target_h = min(
        candidates,
        key=lambda wh: abs((wh[0] / wh[1]) - orig_ratio),
    )
    return target_h, target_w


def normalize_box_tuple(box: Any) -> Tuple[int, int, int, int]:
    if isinstance(box, torch.Tensor):
        box = box.detach().cpu().tolist()
    elif isinstance(box, np.ndarray):
        box = box.tolist()

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"Invalid bbox: {box}")

    return tuple(int(round(float(x))) for x in box)


def clamp_box_to_frame(
    box: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x_min, y_min, x_max, y_max = box

    x_min = max(0, min(int(x_min), width))
    x_max = max(0, min(int(x_max), width))
    y_min = max(0, min(int(y_min), height))
    y_max = max(0, min(int(y_max), height))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Invalid bbox after clamp: {(x_min, y_min, x_max, y_max)}, "
            f"image_size=({width}, {height})"
        )
    return x_min, y_min, x_max, y_max


def inverse_letterbox_to_box_size(
    decoded_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    x_min, y_min, x_max, y_max = box
    frame_w = int(x_max - x_min)
    frame_h = int(y_max - y_min)

    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"Invalid box size: {box}")

    target_h, target_w = decoded_bgr.shape[:2]

    scale = min(target_w / frame_w, target_h / frame_h)
    new_w = int(round(frame_w * scale))
    new_h = int(round(frame_h * scale))

    x_off = max(0, (target_w - new_w) // 2)
    y_off = max(0, (target_h - new_h) // 2)
    x1 = min(target_w, x_off + new_w)
    y1 = min(target_h, y_off + new_h)

    crop = decoded_bgr[y_off:y1, x_off:x1]
    if crop.size == 0:
        raise RuntimeError(
            f"Empty crop after inverse letterbox: decoded_shape={decoded_bgr.shape}, box={box}"
        )

    restored = cv2.resize(crop, (frame_w, frame_h), interpolation=cv2.INTER_CUBIC)
    return restored


# ============================================================
# color correction
# ============================================================
def color_transfer_lab(
    src_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    strength: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return src_bgr

    if src_bgr.size == 0 or ref_bgr.size == 0:
        return src_bgr

    src = src_bgr.astype(np.uint8)
    ref = ref_bgr.astype(np.uint8)

    src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_mean = src_lab.reshape(-1, 3).mean(axis=0)
    src_std = src_lab.reshape(-1, 3).std(axis=0) + eps
    ref_mean = ref_lab.reshape(-1, 3).mean(axis=0)
    ref_std = ref_lab.reshape(-1, 3).std(axis=0) + eps

    matched = (src_lab - src_mean) * (ref_std / src_std) + ref_mean
    matched = np.clip(matched, 0, 255).astype(np.uint8)
    matched_bgr = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)

    if strength < 1.0:
        return cv2.addWeighted(
            matched_bgr,
            strength,
            src,
            1.0 - strength,
            0.0,
        )
    return matched_bgr


# ============================================================
# blending
# ============================================================
def build_backway_like_mask(
    frame_w: int,
    frame_h: int,
    target_w: int,
    target_h: int,
) -> Tuple[np.ndarray, int, int]:
    tx_offset = int((frame_h // max(target_h, 1)) * 24)
    ty_offset = int((frame_w // max(target_w, 1)) * 24)

    tx_offset = max(0, min(tx_offset, max(frame_w // 4, 0)))
    ty_offset = max(0, min(ty_offset, max(frame_h // 4, 0)))

    inner_w = frame_w - tx_offset * 2
    inner_h = frame_h - ty_offset * 2

    if inner_w <= 2 or inner_h <= 2:
        tx_offset = 0
        ty_offset = 0
        inner_w = frame_w
        inner_h = frame_h

    n_y = int(frame_h / max(target_h, 1) * 12)
    n_x = int(frame_w / max(target_w, 1) * 12)

    n_y = max(1, min(n_y, max(inner_h // 2 - 1, 1)))
    n_x = max(1, min(n_x, max(inner_w // 2 - 1, 1)))

    mask = np.zeros((inner_h, inner_w), dtype=np.uint8)
    if inner_h > 2 * n_y and inner_w > 2 * n_x:
        mask[n_y:inner_h - n_y, n_x:inner_w - n_x] = 255
    else:
        mask[:, :] = 255

    nate = frame_w / 512.0
    kernel_size = max(3, int(round(3 * nate)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.dilate(mask, kernel, iterations=3)

    blur_ks = 81
    if blur_ks % 2 == 0:
        blur_ks += 1

    if hasattr(cv2, "stackBlur"):
        mask = cv2.stackBlur(mask, (blur_ks, blur_ks))
    else:
        mask = cv2.GaussianBlur(mask, (blur_ks, blur_ks), 0)

    mask = (mask.astype(np.float32) / 255.0)[..., None]
    mask = np.repeat(mask, 3, axis=-1)
    return mask, tx_offset, ty_offset


def paste_patch_with_blend(
    base_frame_bgr: np.ndarray,
    patch_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    x_min, y_min, x_max, y_max = box
    frame_w = int(x_max - x_min)
    frame_h = int(y_max - y_min)

    if patch_bgr.shape[1] != frame_w or patch_bgr.shape[0] != frame_h:
        patch_bgr = cv2.resize(
            patch_bgr,
            (frame_w, frame_h),
            interpolation=cv2.INTER_CUBIC,
        )

    target_h, target_w = pick_target_size_from_box(box)
    mask, tx_offset, ty_offset = build_backway_like_mask(
        frame_w=frame_w,
        frame_h=frame_h,
        target_w=target_w,
        target_h=target_h,
    )

    x0 = x_min + tx_offset
    x1 = x_max - tx_offset
    y0 = y_min + ty_offset
    y1 = y_max - ty_offset

    if x1 <= x0 or y1 <= y0:
        out = base_frame_bgr.copy()
        out[y_min:y_max, x_min:x_max] = patch_bgr
        return out

    patch_inner = patch_bgr[ty_offset:frame_h - ty_offset, tx_offset:frame_w - tx_offset]
    roi = base_frame_bgr[y0:y1, x0:x1]

    if patch_inner.shape[:2] != roi.shape[:2]:
        patch_inner = cv2.resize(
            patch_inner,
            (roi.shape[1], roi.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
        mask = cv2.resize(mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
        if mask.ndim == 2:
            mask = mask[..., None]
        if mask.shape[-1] == 1:
            mask = np.repeat(mask, 3, axis=-1)

    out = base_frame_bgr.copy()
    blended = patch_inner.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)
    out[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


# ============================================================
# video io
# ============================================================
def probe_video_info(video_path: str) -> Tuple[int, int, int, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 1e-6:
        fps = 25.0
    return width, height, total_frames, fps


class IndexedVideoReader:
    def __init__(self, video_path: str, enable_cache: bool = True):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.width, self.height, self.total_frames, self.fps = probe_video_info(video_path)
        self.last_index = -1
        self.enable_cache = enable_cache and self.total_frames <= 3000
        self.cache: Dict[int, np.ndarray] = {}

    def read_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(
                f"source frame index out of range: idx={frame_idx}, "
                f"total_frames={self.total_frames}"
            )

        if self.enable_cache and frame_idx in self.cache:
            return self.cache[frame_idx].copy()

        if frame_idx == self.last_index + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")

        self.last_index = frame_idx

        if self.enable_cache:
            self.cache[frame_idx] = frame.copy()
        return frame

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FFmpegRawVideoWriter:
    def __init__(
        self,
        out_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "libx264",
        preset: str = "medium",
        quality: int = 18,
    ):
        self.out_path = out_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", codec,
        ]

        if codec == "h264_nvenc":
            cmd += ["-preset", preset, "-cq", str(quality)]
        elif codec == "libx264":
            cmd += ["-preset", preset, "-crf", str(quality)]
        else:
            cmd += ["-preset", preset]

        cmd += [
            "-pix_fmt", "yuv420p",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-color_range", "tv",
            out_path,
        ]

        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame_bgr: np.ndarray):
        if frame_bgr.dtype != np.uint8:
            frame_bgr = np.clip(frame_bgr, 0, 255).astype(np.uint8)
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame shape mismatch, expected {(self.height, self.width)}, "
                f"got {frame_bgr.shape[:2]}"
            )
        if self.proc.stdin is None:
            raise RuntimeError("FFmpeg stdin is closed.")
        self.proc.stdin.write(frame_bgr.tobytes())

    def close(self):
        if self.proc.stdin:
            self.proc.stdin.close()
        return_code = self.proc.wait()
        if return_code != 0:
            raise RuntimeError(f"FFmpeg writer exited with code {return_code}")


def mux_audio_to_video(
    silent_video_path: str,
    audio_path: str,
    out_path: str,
) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)


# ============================================================
# tensor / VAE decode
# ============================================================
def normalize_video_tensor_shape(video: torch.Tensor) -> torch.Tensor:
    if video.ndim != 4:
        raise ValueError(f"Expected 4D tensor after VAE decode, got shape={tuple(video.shape)}")

    if video.shape[0] in (1, 3):
        return video
    if video.shape[1] in (1, 3):
        return video.permute(1, 0, 2, 3).contiguous()

    raise ValueError(f"Cannot infer video tensor layout from shape={tuple(video.shape)}")


def video_tensor_to_bgr_frames(video: torch.Tensor) -> List[np.ndarray]:
    video = normalize_video_tensor_shape(video).detach().cpu().float()

    vmin = float(video.min())
    vmax = float(video.max())

    if vmin >= -1.01 and vmax <= 1.01:
        video = (video.clamp(-1.0, 1.0) + 1.0) * 127.5
    elif vmin >= -0.01 and vmax <= 1.01:
        video = video.clamp(0.0, 1.0) * 255.0
    else:
        video = video.clamp(0.0, 255.0)

    video = video.round().byte()
    video = video.permute(1, 2, 3, 0).numpy()

    frames = []
    for frame_rgb in video:
        if frame_rgb.shape[-1] == 1:
            frame_rgb = np.repeat(frame_rgb, 3, axis=-1)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frames.append(frame_bgr)
    return frames


@torch.no_grad()
def decode_one_clip_frames(
    vae: Wan2_2_VAE,
    latent_cpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> List[np.ndarray]:
    latent = latent_cpu.to(device=device, dtype=dtype)

    videos = vae.decode([latent])
    if not isinstance(videos, (list, tuple)) or len(videos) == 0:
        raise RuntimeError("VAE decode returned empty result.")

    video = videos[0]
    if not torch.is_tensor(video):
        raise RuntimeError("VAE decode result[0] is not a tensor.")

    frames = video_tensor_to_bgr_frames(video)

    del latent, videos, video
    torch_gc()
    return frames


# ============================================================
# main decode pipeline
# ============================================================
class DecodePipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = choose_device(args.device)
        self.dtype = choose_dtype(args.dtype, self.device)

        ensure_exists(args.latent_path, "latent_path")
        ensure_exists(args.vae_path, "vae_path")

        self.latent_result = load_latent_result(args.latent_path)
        self.meta = self.latent_result["meta"]
        self.clips = self.latent_result["clips"]

        self.face_box_source = str(self.meta.get("face_box_source", "landmark"))
        self.preprocess_schedule_version = str(
            self.meta.get(
                "preprocess_schedule_version",
                self.meta.get("schedule_version", "linear_legacy"),
            )
        )
        self.loop_start_mode = str(self.meta.get("loop_start_mode", "head"))

        self.video_path = resolve_path_with_meta(
            args.video_path,
            self.meta.get("video_path"),
            "video_path",
            required=True,
        )
        self.audio_path = resolve_path_with_meta(
            args.audio_path,
            self.meta.get("audio_path"),
            "audio_path",
            required=True,
        )
        self.preprocessed_path = resolve_path_with_meta(
            args.preprocessed_path,
            self.meta.get("preprocessed_path"),
            "preprocessed_path",
            required=False,
        )

        self.frame_num = int(
            args.frame_num if args.frame_num is not None else self.meta.get("frame_num", 41)
        )
        self.motion_frame = int(
            args.motion_frame if args.motion_frame is not None else self.meta.get("motion_frame", 5)
        )
        self.output_fps = float(
            args.fps if args.fps is not None else self.meta.get("target_fps", 25.0)
        )

        if not self.preprocessed_path and self.video_path:
            inferred = default_preprocessed_path(
                video_path=self.video_path,
                face_box_source=self.face_box_source,
                frame_num=self.frame_num,
                motion_frame=self.motion_frame,
                margin=int(self.meta.get("preprocess_margin", 32)),
                yolov8_weight_path=self.meta.get("yolov8_weight_path", None),
                yolov8_conf=float(self.meta.get("yolov8_conf", 0.5)),
                yolov8_iou=float(self.meta.get("yolov8_iou", 0.45)),
                yolov8_imgsz=int(self.meta.get("yolov8_imgsz", 640)),
                yolov8_max_det=int(self.meta.get("yolov8_max_det", 300)),
                yolov8_half=bool(self.meta.get("yolov8_half", False)),
                yolov8_face_box_scale=float(self.meta.get("yolov8_face_box_scale", 1.25)),
                schedule_version=self.preprocess_schedule_version,
                loop_start_mode=self.loop_start_mode,
            )
            if os.path.exists(inferred):
                self.preprocessed_path = inferred

        ensure_exists(self.video_path, "video_path")
        ensure_exists(self.audio_path, "audio_path")

        self.total_output_frames = infer_total_output_frames(self.meta, self.clips)

        self.preprocessed_records: Optional[List[Dict[str, Any]]] = None
        if self.preprocessed_path and os.path.exists(self.preprocessed_path):
            try:
                self.preprocessed_records = load_preprocessed_records(self.preprocessed_path)
            except Exception as e:
                logging.warning(f"Failed to load preprocessed records for fallback: {e}")

        self.reader = IndexedVideoReader(
            self.video_path,
            enable_cache=not args.disable_source_cache,
        )
        self.width = self.reader.width
        self.height = self.reader.height
        self.source_total_frames = self.reader.total_frames

        self.vae = Wan2_2_VAE(
            vae_pth=args.vae_path,
            device=self.device,
        )

        logging.info(
            f"Resolved decode settings:\n"
            f"  video_path={self.video_path}\n"
            f"  audio_path={self.audio_path}\n"
            f"  preprocessed_path={self.preprocessed_path}\n"
            f"  face_box_source={self.face_box_source}\n"
            f"  preprocess_schedule_version={self.preprocess_schedule_version}\n"
            f"  loop_start_mode={self.loop_start_mode}\n"
            f"  frame_num={self.frame_num}\n"
            f"  motion_frame={self.motion_frame}\n"
            f"  output_fps={self.output_fps}\n"
            f"  total_output_frames={self.total_output_frames}\n"
            f"  source_total_frames={self.source_total_frames}"
        )

    def resolve_preprocessed_record(self, clip: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        record_index = int(clip.get("record_index", 0))
        return get_record_from_preprocessed(self.preprocessed_records, record_index)

    def resolve_clip_box(self, clip: Dict[str, Any]) -> Tuple[int, int, int, int]:
        box = clip.get("bbox", None)

        if box is None:
            record = self.resolve_preprocessed_record(clip)
            if record is not None:
                box = record.get("bbox", None)

        if box is None:
            raise ValueError("Cannot resolve bbox for clip.")

        box = normalize_box_tuple(box)
        box = clamp_box_to_frame(box, self.width, self.height)
        return box

    def resolve_source_frame_indices(self, clip: Dict[str, Any]) -> List[int]:
        source_frame_indices = normalize_int_list(
            clip.get("source_frame_indices", None),
            "clip.source_frame_indices",
        )

        if source_frame_indices is None:
            record = self.resolve_preprocessed_record(clip)
            if record is not None:
                source_frame_indices = normalize_int_list(
                    record.get("source_frame_indices", None),
                    "record.source_frame_indices",
                )

        if source_frame_indices is None:
            if "source_start_idx" in clip:
                source_start_idx = int(clip["source_start_idx"])
            else:
                record = self.resolve_preprocessed_record(clip)
                if record is not None:
                    source_start_idx = int(
                        record.get("source_start_idx", record.get("start_idx", 0))
                    )
                else:
                    source_start_idx = int(clip.get("preprocess_start_idx", clip.get("frame_start", 0)))

            source_frame_indices = list(range(source_start_idx, source_start_idx + self.frame_num))

        if len(source_frame_indices) < self.frame_num:
            raise ValueError(
                f"source_frame_indices is shorter than frame_num: "
                f"len={len(source_frame_indices)}, frame_num={self.frame_num}, "
                f"clip_index={clip.get('clip_index', 'unknown')}"
            )

        for idx, source_idx in enumerate(source_frame_indices):
            if source_idx < 0 or source_idx >= self.source_total_frames:
                raise RuntimeError(
                    f"source_idx out of range in source_frame_indices[{idx}]: "
                    f"source_idx={source_idx}, source_total_frames={self.source_total_frames}, "
                    f"clip_index={clip.get('clip_index', 'unknown')}"
                )

        return source_frame_indices

    def resolve_source_start_idx(self, clip: Dict[str, Any]) -> int:
        source_frame_indices = self.resolve_source_frame_indices(clip)
        return int(source_frame_indices[0])

    def compose_one_output_frame(
        self,
        base_frame_bgr: np.ndarray,
        decoded_frame_bgr: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = box

        restored_patch = inverse_letterbox_to_box_size(decoded_frame_bgr, box)
        ref_crop = base_frame_bgr[y_min:y_max, x_min:x_max]

        corrected_patch = color_transfer_lab(
            restored_patch,
            ref_crop,
            strength=self.args.color_correction_strength,
        )

        out_frame = paste_patch_with_blend(
            base_frame_bgr=base_frame_bgr,
            patch_bgr=corrected_patch,
            box=box,
        )
        return out_frame

    def run(self) -> None:
        os.makedirs(os.path.dirname(self.args.out_path) or ".", exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            silent_video_path = os.path.join(tmpdir, "decode_silent.mp4")

            writer = FFmpegRawVideoWriter(
                out_path=silent_video_path,
                width=self.width,
                height=self.height,
                fps=self.output_fps,
                codec=self.args.video_codec,
                preset=self.args.video_preset,
                quality=self.args.video_quality,
            )

            expected_global_frame = 0
            written_source_indices: List[int] = []

            try:
                for clip_idx, clip in enumerate(self.clips):
                    box = self.resolve_clip_box(clip)
                    source_frame_indices = self.resolve_source_frame_indices(clip)

                    latent = clip["latent"]
                    if not torch.is_tensor(latent):
                        raise ValueError(f"clip[{clip_idx}] latent is not a tensor.")

                    decoded_frames = decode_one_clip_frames(
                        vae=self.vae,
                        latent_cpu=latent,
                        device=self.device,
                        dtype=self.dtype,
                    )

                    frame_start = int(clip["frame_start"])
                    valid_frame_count = int(
                        clip.get(
                            "valid_frame_count",
                            int(clip["valid_frame_end"]) - frame_start + 1,
                        )
                    )

                    local_start = 0 if clip_idx == 0 else self.motion_frame
                    local_end = min(valid_frame_count, len(decoded_frames))

                    if local_end > len(source_frame_indices):
                        raise RuntimeError(
                            f"source_frame_indices cannot cover decoded valid frames: "
                            f"local_end={local_end}, len(source_frame_indices)={len(source_frame_indices)}, "
                            f"clip_idx={clip_idx}"
                        )

                    if local_end < local_start:
                        logging.warning(
                            f"clip[{clip_idx}] has no valid frames after overlap removal: "
                            f"local_start={local_start}, local_end={local_end}"
                        )
                        del decoded_frames
                        torch_gc()
                        continue

                    kept_source_indices = source_frame_indices[local_start:local_end]
                    source_preview = (
                        f"{kept_source_indices[:3]}...{kept_source_indices[-3:]}"
                        if len(kept_source_indices) > 6
                        else str(kept_source_indices)
                    )

                    logging.info(
                        f"[clip {clip_idx + 1}/{len(self.clips)}] "
                        f"frame_start={frame_start}, valid_frame_count={valid_frame_count}, "
                        f"bbox={box}, face_box_source={self.face_box_source}, "
                        f"schedule={clip.get('schedule_version', self.preprocess_schedule_version)}, "
                        f"is_reversed={clip.get('is_reversed', False)}, "
                        f"keep_local=[{local_start}, {local_end}), "
                        f"source_indices={source_preview}"
                    )

                    for local_idx in range(local_start, local_end):
                        global_idx = frame_start + local_idx
                        source_idx = int(source_frame_indices[local_idx])

                        if global_idx != expected_global_frame:
                            raise RuntimeError(
                                f"Non-contiguous global timeline detected at clip[{clip_idx}]. "
                                f"expected_global_frame={expected_global_frame}, got global_idx={global_idx}. "
                                f"This indicates decode/inference timeline mismatch."
                            )

                        if source_idx < 0 or source_idx >= self.source_total_frames:
                            raise RuntimeError(
                                f"source_idx out of range: source_idx={source_idx}, "
                                f"source_total_frames={self.source_total_frames}, clip_idx={clip_idx}, "
                                f"local_idx={local_idx}"
                            )

                        base_frame = self.reader.read_frame(source_idx)
                        out_frame = self.compose_one_output_frame(
                            base_frame_bgr=base_frame,
                            decoded_frame_bgr=decoded_frames[local_idx],
                            box=box,
                        )
                        writer.write(out_frame)

                        written_source_indices.append(source_idx)
                        expected_global_frame += 1

                        if expected_global_frame % 20 == 0 or expected_global_frame == self.total_output_frames:
                            logging.info(
                                f"Writing frames: {expected_global_frame}/{self.total_output_frames}"
                            )

                    del decoded_frames
                    torch_gc()

                    if expected_global_frame >= self.total_output_frames:
                        break

                if expected_global_frame != self.total_output_frames:
                    raise RuntimeError(
                        f"Output frame count mismatch: wrote {expected_global_frame}, "
                        f"but expected {self.total_output_frames}. "
                        f"This usually means timeline stitching is inconsistent."
                    )

            finally:
                try:
                    writer.close()
                finally:
                    self.reader.close()

            mux_audio_to_video(
                silent_video_path=silent_video_path,
                audio_path=self.audio_path,
                out_path=self.args.out_path,
            )

        meta_out = {
            "latent_path": self.args.latent_path,
            "video_path": self.video_path,
            "audio_path": self.audio_path,
            "preprocessed_path": self.preprocessed_path,
            "vae_path": self.args.vae_path,
            "out_path": self.args.out_path,
            "frame_num": self.frame_num,
            "motion_frame": self.motion_frame,
            "total_output_frames": self.total_output_frames,
            "fps": self.output_fps,
            "color_correction_strength": self.args.color_correction_strength,
            "video_codec": self.args.video_codec,
            "video_quality": self.args.video_quality,
            "video_preset": self.args.video_preset,
            "face_box_source": self.face_box_source,
            "preprocess_schedule_version": self.preprocess_schedule_version,
            "loop_start_mode": self.loop_start_mode,
            "source_total_frames": self.source_total_frames,
            "written_source_indices": written_source_indices,
            "source_meta": self.meta,
        }

        meta_path = str(Path(self.args.out_path).with_suffix(".json"))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False, indent=2, default=str)

        logging.info(f"Saved final video to: {self.args.out_path}")
        logging.info(f"Saved decode meta to: {meta_path}")

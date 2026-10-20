from __future__ import annotations

import os
import cv2
import json
import torch
import inspect
import hashlib
import numpy as np

from math import gcd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from einops import rearrange

from chanjing_avatar.face.landmark import LandmarkRunner
from chanjing_avatar.models.vae import Wan2_2_VAE


PINGPONG_LOOP_SCHEDULE_VERSION = "pingpong_loop_v1"


def _stable_hash(payload: Dict[str, Any], length: int = 12) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def _normalize_path_for_hash(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path).resolve())


def _parse_cuda_device_id(device: str) -> int:
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise ValueError(
            "face_box_source='landmark' currently requires a CUDA preprocess_device "
            f"(got device={device!r})."
        )
    if ":" not in device:
        return 0
    suffix = device.split(":")[-1]
    if not suffix.isdigit():
        raise ValueError(f"Invalid CUDA device string: {device}")
    return int(suffix)


def build_yolov8_face_box_cache_path(
    video_path: str,
    weight_path: Optional[str],
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    max_det: int,
    half: bool,
) -> str:
    video = Path(video_path)
    payload = {
        "weight_path": _normalize_path_for_hash(weight_path),
        "conf_threshold": float(conf_threshold),
        "iou_threshold": float(iou_threshold),
        "imgsz": int(imgsz),
        "max_det": int(max_det),
        "half": bool(half),
    }
    tag = _stable_hash(payload)
    return str(video.with_name(f"{video.stem}_yolov8_face_boxes_{tag}.npy"))


def _create_yolov8_detector(
    weight_path: Optional[str],
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    device: str,
    max_det: int,
    half: bool,
) -> Any:
    from chanjing_avatar.face.yolov8 import Yolov8sFaceDetector
    kwargs = {
        "weight_path": weight_path,
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "imgsz": imgsz,
        "device": device,
        "max_det": max_det,
        "half": half,
    }

    try:
        sig = inspect.signature(Yolov8sFaceDetector.__init__)
        supported = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported}
    except Exception:
        filtered_kwargs = kwargs

    return Yolov8sFaceDetector(**filtered_kwargs)


def ensure_landmarks(lmk_weight_path, video_path: str, landmark_path: str, device_id: int = 0) -> str:
    """
    若 landmark 文件不存在，则按项目现有逻辑提取一份。
    当前实现依赖 CUDA onnx provider。
    """
    if os.path.exists(landmark_path):
        return landmark_path

    runner = LandmarkRunner(
        ckpt_path=lmk_weight_path,
        onnx_provider="cuda",
        device_id=device_id,
    )
    runner.warmup()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Empty video: {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    lmk = None
    landmarks = []

    for _ in range(10):
        lmk = runner.run(frame, lmk)

    while True:
        lmk203 = runner.run(frame, lmk)
        landmarks.append(lmk203)

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lmk = lmk203.copy()

    cap.release()

    landmarks = np.asarray(landmarks, dtype=np.float32)
    np.save(landmark_path, landmarks)
    return landmark_path


def temporal_smooth_landmarks(landmarks: np.ndarray, win: int = 5) -> np.ndarray:
    kernel = np.ones(win, dtype=np.float32) / win
    out = landmarks.copy()
    for i in range(landmarks.shape[1]):
        for d in range(2):
            out[:, i, d] = np.convolve(out[:, i, d], kernel, mode="same")
    return out


def ffprobe_video_info(video_path: str) -> Tuple[int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, total_frames


def segment_bbox_robust(
    landmarks_seg: np.ndarray,
    q_low: float = 0.03,
    q_high: float = 0.97,
) -> Tuple[float, float, float, float]:
    xs = landmarks_seg[:, :, 0].reshape(-1)
    ys = landmarks_seg[:, :, 1].reshape(-1)
    xmin = np.quantile(xs, q_low)
    xmax = np.quantile(xs, q_high)
    ymin = np.quantile(ys, q_low)
    ymax = np.quantile(ys, q_high)
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _finalize_square_box(
    cx: float,
    cy: float,
    side: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    if side <= 1:
        raise RuntimeError(f"Invalid square box side: {side}")

    x_min = cx - side / 2.0
    x_max = cx + side / 2.0
    y_min = cy - side / 2.0
    y_max = cy + side / 2.0

    if side <= width:
        x_min = max(0.0, min(x_min, float(width) - side))
        x_max = x_min + side
    else:
        x_min, x_max = 0.0, float(width)

    if side <= height:
        y_min = max(0.0, min(y_min, float(height) - side))
        y_max = y_min + side
    else:
        y_min, y_max = 0.0, float(height)

    x_min = max(0.0, min(x_min, float(width)))
    x_max = max(0.0, min(x_max, float(width)))
    y_min = max(0.0, min(y_min, float(height)))
    y_max = max(0.0, min(y_max, float(height)))

    x_min = int(x_min)
    y_min = int(y_min)
    x_max = int(x_max)
    y_max = int(y_max)

    if x_max <= x_min or y_max <= y_min:
        raise RuntimeError(
            f"Invalid finalized box: {(x_min, y_min, x_max, y_max)}, "
            f"image_size=({width}, {height})"
        )
    return x_min, y_min, x_max, y_max


def design_global_box_from_landmarks(
    landmarks: np.ndarray,
    width: int,
    height: int,
    margin: int = 32,
) -> Tuple[int, int, int, int]:
    landmarks = temporal_smooth_landmarks(landmarks, win=5)
    pxmin, pymin, pxmax, pymax = segment_bbox_robust(landmarks)

    w = pxmax - pxmin
    h = pymax - pymin
    side = max(w, h) * 2.0 + 2 * margin

    cx = (pxmin + pxmax) / 2.0
    cy = (pymin + pymax) / 2.0
    return _finalize_square_box(cx=cx, cy=cy, side=side, width=width, height=height)


def compute_iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, 0.0, None)
    inter_h = np.clip(y2 - y1, 0.0, None)
    inter = inter_w * inter_h

    area1 = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    area2 = np.clip((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6, None)

    union = area1 + area2 - inter
    return inter / np.clip(union, 1e-6, None)


def select_primary_face_box(
    boxes: np.ndarray,
    scores: np.ndarray,
    prev_box: Optional[np.ndarray] = None,
    iou_weight: float = 0.5,
) -> Optional[np.ndarray]:
    if boxes is None or len(boxes) == 0:
        return None

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if prev_box is None or not np.all(np.isfinite(prev_box)):
        best_idx = int(np.argmax(scores))
        return boxes[best_idx].astype(np.float32)

    ious = compute_iou_xyxy(prev_box.astype(np.float32), boxes)
    combined = scores + iou_weight * ious
    best_idx = int(np.argmax(combined))
    return boxes[best_idx].astype(np.float32)


def fill_missing_face_boxes(face_boxes: np.ndarray) -> np.ndarray:
    face_boxes = np.asarray(face_boxes, dtype=np.float32)
    if face_boxes.ndim != 2 or face_boxes.shape[1] != 4:
        raise ValueError(f"face_boxes shape must be [T,4], got {face_boxes.shape}")

    valid = np.all(np.isfinite(face_boxes), axis=1)
    if not np.any(valid):
        raise RuntimeError("All yolov8 face boxes are invalid (all NaN).")

    out = face_boxes.copy()
    t = np.arange(len(face_boxes), dtype=np.float32)
    valid_t = t[valid]

    for d in range(4):
        valid_v = out[valid, d]
        out[:, d] = np.interp(t, valid_t, valid_v)

    return out


def temporal_smooth_boxes(face_boxes: np.ndarray, win: int = 5) -> np.ndarray:
    kernel = np.ones(win, dtype=np.float32) / win
    out = face_boxes.copy()
    for d in range(4):
        out[:, d] = np.convolve(out[:, d], kernel, mode="same")
    return out


def segment_face_boxes_robust(
    face_boxes: np.ndarray,
    q_low: float = 0.03,
    q_high: float = 0.97,
) -> Tuple[float, float, float, float]:
    x1 = np.quantile(face_boxes[:, 0], q_low)
    y1 = np.quantile(face_boxes[:, 1], q_low)
    x2 = np.quantile(face_boxes[:, 2], q_high)
    y2 = np.quantile(face_boxes[:, 3], q_high)
    return float(x1), float(y1), float(x2), float(y2)


def design_global_box_from_face_boxes(
    face_boxes: np.ndarray,
    width: int,
    height: int,
    margin: int = 32,
    q_low: float = 0.03,
    q_high: float = 0.97,
    face_box_scale: float = 1.25,
) -> Tuple[int, int, int, int]:
    boxes = fill_missing_face_boxes(face_boxes)
    boxes = temporal_smooth_boxes(boxes, win=5)

    pxmin, pymin, pxmax, pymax = segment_face_boxes_robust(
        boxes,
        q_low=q_low,
        q_high=q_high,
    )

    w = pxmax - pxmin
    h = pymax - pymin
    if w <= 1 or h <= 1:
        raise RuntimeError(
            f"Invalid aggregated yolov8 face box: {(pxmin, pymin, pxmax, pymax)}"
        )

    side = max(w, h) * float(face_box_scale) + 2 * margin
    cx = (pxmin + pxmax) / 2.0
    cy = (pymin + pymax) / 2.0
    return _finalize_square_box(cx=cx, cy=cy, side=side, width=width, height=height)


def pick_target_size_from_box(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x_min, y_min, x_max, y_max = box
    frame_w = int(x_max - x_min)
    frame_h = int(y_max - y_min)

    candidates = [(512, 512), (448, 576), (576, 448)]
    orig_ratio = frame_w / max(frame_h, 1)

    target_w, target_h = min(
        candidates,
        key=lambda wh: abs((wh[0] / wh[1]) - orig_ratio)
    )
    return target_h, target_w


def crop_and_resize_frame(
    frame_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    x_min, y_min, x_max, y_max = box
    target_h, target_w = target_size

    crop = frame_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        raise RuntimeError(f"Invalid crop with box={box}")

    h, w = crop.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(crop, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(canvas).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    tensor = (tensor - 0.5) * 2.0
    return tensor


def select_reference_frame_indices(landmarks: np.ndarray) -> List[int]:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    total_frames = len(landmarks)

    upper_idx = np.arange(85, 96)
    lower_idx = np.arange(97, 108)[::-1]
    eps = 1e-6

    pair_weights = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=np.float32)
    pair_weights = pair_weights / pair_weights.sum()

    raw_open_scores = np.full(total_frames, np.nan, dtype=np.float32)

    for t in range(total_frames):
        pts = landmarks[t]
        if pts.shape[0] <= 107:
            continue

        upper = pts[upper_idx]
        lower = pts[lower_idx]

        if not np.all(np.isfinite(upper)) or not np.all(np.isfinite(lower)):
            continue
        if upper[-1, 0] <= upper[0, 0]:
            continue
        if lower[-1, 0] <= lower[0, 0]:
            continue

        left_corner = 0.5 * (upper[0] + lower[0])
        right_corner = 0.5 * (upper[-1] + lower[-1])

        mouth_axis = right_corner - left_corner
        mouth_width = float(np.linalg.norm(mouth_axis))
        if mouth_width < 5.0:
            continue

        u = mouth_axis / (mouth_width + eps)
        n = np.array([-u[1], u[0]], dtype=np.float32)

        center_vec = lower.mean(axis=0) - upper.mean(axis=0)
        if float(np.dot(center_vec, n)) < 0:
            n = -n

        gaps = (lower - upper) @ n
        gaps = np.clip(gaps, 0.0, None)

        weighted_gap = float(np.sum(gaps * pair_weights))
        center_gap = float(np.mean(gaps[4:7]))
        raw_open = (0.7 * weighted_gap + 0.3 * center_gap) / (mouth_width + eps)
        raw_open_scores[t] = raw_open

    valid_mask = np.isfinite(raw_open_scores)
    if not np.any(valid_mask):
        raise RuntimeError("No valid mouth open scores.")

    filled_scores = raw_open_scores.copy()
    filled_scores[~valid_mask] = float(np.nanmedian(filled_scores))

    kernel = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
    padded = np.pad(filled_scores, (1, 1), mode="edge")
    smooth_open_scores = np.convolve(padded, kernel, mode="valid").astype(np.float32)

    valid_indices = np.where(valid_mask)[0]
    valid_scores = smooth_open_scores[valid_mask]
    if len(valid_indices) < 4:
        raise RuntimeError("Not enough valid frames.")

    is_local_min = np.zeros(total_frames, dtype=bool)
    is_local_max = np.zeros(total_frames, dtype=bool)
    if total_frames >= 3:
        is_local_min[1:-1] = (
            (smooth_open_scores[1:-1] <= smooth_open_scores[:-2]) &
            (smooth_open_scores[1:-1] <= smooth_open_scores[2:])
        )
        is_local_max[1:-1] = (
            (smooth_open_scores[1:-1] >= smooth_open_scores[:-2]) &
            (smooth_open_scores[1:-1] >= smooth_open_scores[2:])
        )

    def pick_from_candidates(candidates, target_value, num_needed, selected_all, min_separation=2):
        picked = []
        ordered = sorted(
            candidates,
            key=lambda i: (abs(float(smooth_open_scores[i]) - float(target_value)), i)
        )
        for idx in ordered:
            if any(abs(int(idx) - j) < min_separation for j in selected_all):
                continue
            picked.append(int(idx))
            selected_all.append(int(idx))
            if len(picked) == num_needed:
                break
        return picked

    def select_group(num_needed, mode, selected_all):
        assert mode in ["closed", "open"]
        if mode == "closed":
            configs = [
                {"q_lo": 0, "q_hi": 30, "q_target": 15, "use_local": True},
                {"q_lo": 0, "q_hi": 35, "q_target": 20, "use_local": True},
                {"q_lo": 0, "q_hi": 40, "q_target": 20, "use_local": False},
            ]
            local_mask = is_local_min
        else:
            configs = [
                {"q_lo": 70, "q_hi": 100, "q_target": 85, "use_local": True},
                {"q_lo": 65, "q_hi": 100, "q_target": 80, "use_local": True},
                {"q_lo": 60, "q_hi": 100, "q_target": 80, "use_local": False},
            ]
            local_mask = is_local_max

        picked = []
        for cfg in configs:
            q_lo = np.percentile(valid_scores, cfg["q_lo"])
            q_hi = np.percentile(valid_scores, cfg["q_hi"])
            q_target = np.percentile(valid_scores, cfg["q_target"])

            candidate_mask = (
                valid_mask &
                (smooth_open_scores >= q_lo) &
                (smooth_open_scores <= q_hi)
            )
            if cfg["use_local"]:
                candidate_mask = candidate_mask & local_mask

            candidates = np.where(candidate_mask)[0]
            if len(candidates) == 0:
                continue

            picked.extend(
                pick_from_candidates(
                    candidates,
                    q_target,
                    num_needed - len(picked),
                    selected_all,
                    min_separation=2,
                )
            )
            if len(picked) == num_needed:
                break
        return picked

    selected_all = []
    closed_idx = select_group(2, "closed", selected_all)
    open_idx = select_group(2, "open", selected_all)

    if len(closed_idx) < 2:
        target = np.percentile(valid_scores, 15)
        closed_idx.extend(
            pick_from_candidates(valid_indices, target, 2 - len(closed_idx), selected_all)
        )

    if len(open_idx) < 2:
        target = np.percentile(valid_scores, 85)
        open_idx.extend(
            pick_from_candidates(valid_indices, target, 2 - len(open_idx), selected_all)
        )

    if len(closed_idx) < 2 or len(open_idx) < 2:
        raise RuntimeError("Failed to select enough reference frames.")

    closed_idx = sorted(closed_idx)
    open_idx = sorted(open_idx)
    return [closed_idx[0], open_idx[0], closed_idx[1], open_idx[1]]


def read_selected_frames(
    video_path: str,
    indices: List[int],
    box: Tuple[int, int, int, int],
    target_size: Tuple[int, int],
) -> List[torch.Tensor]:
    wanted = set(indices)
    cache = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in wanted:
            cache[frame_id] = crop_and_resize_frame(frame, box, target_size)
            if len(cache) == len(indices):
                break
        frame_id += 1

    cap.release()

    if len(cache) != len(indices):
        missing = sorted(list(wanted - set(cache.keys())))
        raise RuntimeError(f"Failed to read reference frames: missing {missing}")

    return [cache[i] for i in indices]


def get_five_key_frame_indices(frame_num: int) -> List[int]:
    if frame_num < 5 or (frame_num - 1) % 4 != 0:
        raise ValueError(f"frame_num must satisfy 4n+1, got {frame_num}")
    step = (frame_num - 1) // 4
    return [0, step, 2 * step, 3 * step, 4 * step]


def build_latent_y_input_landmark(
    clip_frames: torch.Tensor,
    ref_frames: List[torch.Tensor],
    frame_num: int,
) -> torch.Tensor:
    if ref_frames is None or len(ref_frames) != 4:
        raise ValueError("landmark mode requires 4 reference frames.")

    key_indices = get_five_key_frame_indices(frame_num)
    key_frames = clip_frames[:, :, key_indices, :, :]

    frame0 = key_frames[:, :, 0:1, :, :].clone()
    frame1 = key_frames[:, :, 1:2, :, :].clone()
    frame2 = key_frames[:, :, 2:3, :, :].clone()
    frame3 = key_frames[:, :, 3:4, :, :].clone()
    frame4 = key_frames[:, :, 4:5, :, :].clone()

    close1 = ref_frames[0][None, :, None, :, :].clone()
    open1 = ref_frames[1][None, :, None, :, :].clone()
    close2 = ref_frames[2][None, :, None, :, :].clone()
    open2 = ref_frames[3][None, :, None, :, :].clone()

    pattern = torch.cat(
        [frame0, close1, frame1, open1, frame2, close2, frame3, open2, frame4],
        dim=2,
    )

    if frame_num == 81:
        y_frames = torch.cat([pattern] * 9, dim=2)
    elif frame_num == 41:
        y_frames = torch.cat([pattern[:, :, 0:5, :, :]] + [pattern] * 4, dim=2)
    else:
        y_frames = pattern.clone()
        while y_frames.shape[2] < frame_num:
            remain = frame_num - y_frames.shape[2]
            y_frames = torch.cat(
                [y_frames, pattern[:, :, :min(remain, pattern.shape[2]), :, :]],
                dim=2,
            )

    assert y_frames.shape[2] == frame_num
    return y_frames


def build_latent_y_input_yolov8(
    clip_frames: torch.Tensor,
    frame_num: int,
) -> torch.Tensor:
    key_indices = get_five_key_frame_indices(frame_num)
    key_frames = clip_frames[:, :, key_indices, :, :].clone()

    y_frames = key_frames.clone()
    while y_frames.shape[2] < frame_num:
        remain = frame_num - y_frames.shape[2]
        y_frames = torch.cat(
            [y_frames, key_frames[:, :, :min(remain, key_frames.shape[2]), :, :]],
            dim=2,
        )

    assert y_frames.shape[2] == frame_num
    return y_frames


def build_latent_y_input(
    clip_frames: torch.Tensor,
    frame_num: int,
    reference_mode: str = "landmark",
    ref_frames: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    if reference_mode == "landmark":
        return build_latent_y_input_landmark(
            clip_frames=clip_frames,
            ref_frames=ref_frames,
            frame_num=frame_num,
        )
    if reference_mode == "yolov8":
        return build_latent_y_input_yolov8(
            clip_frames=clip_frames,
            frame_num=frame_num,
        )
    raise ValueError(f"Unsupported reference_mode: {reference_mode}")


def ensure_yolov8_face_boxes(
    video_path: str,
    face_box_path: str,
    device: str = "cuda:0",
    weight_path: Optional[str] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.45,
    imgsz: int = 640,
    max_det: int = 300,
    half: bool = False,
) -> str:
    if weight_path is not None and not os.path.exists(weight_path):
        raise FileNotFoundError(f"yolov8 weight not found: {weight_path}")

    if os.path.exists(face_box_path):
        try:
            cached = np.load(face_box_path)
            if cached.ndim == 2 and cached.shape[1] == 4 and len(cached) > 0:
                return face_box_path
        except Exception:
            pass

    detector = _create_yolov8_detector(
        weight_path=weight_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        half=half,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    all_boxes = []
    prev_box = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        boxes, scores = detector.detect(frame_bgr)
        box = select_primary_face_box(
            boxes=boxes,
            scores=scores,
            prev_box=prev_box,
            iou_weight=0.5,
        )

        if box is None:
            all_boxes.append(np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float32))
        else:
            all_boxes.append(box.astype(np.float32))
            prev_box = box.astype(np.float32)

    cap.release()

    if len(all_boxes) == 0:
        raise RuntimeError(f"Empty video or no frames read: {video_path}")

    face_boxes = np.stack(all_boxes, axis=0).astype(np.float32)
    np.save(face_box_path, face_boxes)
    return face_box_path


def best_loop_plan(n: int, m: int, c: int) -> Dict[str, int]:
    h = m - c
    if h <= 0:
        raise ValueError(f"需要满足 m > c, got m={m}, c={c}")
    if n < m:
        raise ValueError(f"需要满足 n >= m, got n={n}, m={m}")

    L = max(n - m, m - 1)

    best_R = None
    best_K = None

    for R in range(L, n + 1):
        K = 2 * R // gcd(2 * R, h)

        if best_K is None:
            best_R = R
            best_K = K
        else:
            if K < best_K or (K == best_K and R > best_R):
                best_R = R
                best_K = K

    if best_R is None or best_K is None:
        raise RuntimeError(f"Failed to build loop plan: n={n}, m={m}, c={c}, L={L}")

    D = n - best_R

    return {
        "R": int(best_R),
        "discard": int(D),
        "segments": int(best_K),
        "step": int(h),
        "period": int(2 * best_R),
        "frame_num": int(m),
        "motion_frame": int(c),
    }


def pingpong_index(t: int, R: int) -> int:
    if R <= 0:
        raise ValueError(f"R must be positive, got {R}")

    x = int(t) % (2 * R)
    if x < R:
        return x
    return 2 * R - 1 - x


def resolve_loop_start_offset(
    total_frames: int,
    effective_span: int,
    mode: str = "head",
) -> int:
    if effective_span <= 0 or effective_span > total_frames:
        raise ValueError(
            f"Invalid effective_span={effective_span}, total_frames={total_frames}"
        )

    discard = total_frames - effective_span

    if mode == "head":
        return 0
    if mode == "center":
        return discard // 2

    raise ValueError(f"Unsupported loop_start_mode: {mode}")


def build_pingpong_source_frame_indices(
    virtual_start_idx: int,
    frame_num: int,
    effective_span: int,
    loop_start_offset: int = 0,
) -> List[int]:
    indices = []
    for i in range(frame_num):
        local_idx = pingpong_index(virtual_start_idx + i, effective_span)
        source_idx = loop_start_offset + local_idx
        indices.append(int(source_idx))
    return indices


def validate_source_frame_indices(
    source_frame_indices: List[int],
    total_frames: int,
    expected_len: int,
) -> None:
    if len(source_frame_indices) != expected_len:
        raise RuntimeError(
            f"source_frame_indices length mismatch: "
            f"expected={expected_len}, got={len(source_frame_indices)}"
        )

    for idx in source_frame_indices:
        if idx < 0 or idx >= total_frames:
            raise RuntimeError(
                f"source frame index out of range: idx={idx}, total_frames={total_frames}"
            )


def is_pingpong_reversed(source_frame_indices: List[int]) -> bool:
    if len(source_frame_indices) < 2:
        return False

    diffs = np.diff(np.asarray(source_frame_indices, dtype=np.int64))
    neg = int(np.sum(diffs < 0))
    pos = int(np.sum(diffs > 0))
    return neg > pos


class IndexedVideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        self.last_index = -1

    def read_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = int(frame_idx)
        if frame_idx < 0:
            raise IndexError(f"frame_idx must be >= 0, got {frame_idx}")

        if frame_idx == self.last_index + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {self.video_path}")

        self.last_index = frame_idx
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def read_clip_frames_by_indices(
    reader: IndexedVideoReader,
    source_frame_indices: List[int],
    box: Tuple[int, int, int, int],
    target_size: Tuple[int, int],
) -> List[torch.Tensor]:
    frames = []
    for idx in source_frame_indices:
        frame_bgr = reader.read_frame(idx)
        frame_tensor = crop_and_resize_frame(
            frame_bgr=frame_bgr,
            box=box,
            target_size=target_size,
        )
        frames.append(frame_tensor)
    return frames


def preprocess_video(
    video_path: str,
    out_pt: str,
    vae_path: str,
    device: str = "cuda:0",
    frame_num: int = 41,
    motion_frame: int = 5,
    margin: int = 32,
    face_box_source: str = "landmark",
    lmk_weight_path: Optional[str] = None,
    landmark_path: Optional[str] = None,
    yolov8_weight_path: Optional[str] = None,
    yolov8_conf: float = 0.5,
    yolov8_iou: float = 0.45,
    yolov8_imgsz: int = 640,
    yolov8_max_det: int = 300,
    yolov8_half: bool = False,
    yolov8_face_box_scale: float = 1.25,
    loop_start_mode: str = "head",
):
    if face_box_source not in {"landmark", "yolov8"}:
        raise ValueError(f"Unsupported face_box_source: {face_box_source}")
    if frame_num < 5 or (frame_num - 1) % 4 != 0:
        raise ValueError(f"frame_num must satisfy 4n+1, got {frame_num}")
    if motion_frame < 0 or motion_frame >= frame_num:
        raise ValueError(
            f"motion_frame must satisfy 0 <= motion_frame < frame_num, got {motion_frame}"
        )
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if not (0.0 <= yolov8_conf <= 1.0):
        raise ValueError(f"yolov8_conf must be in [0,1], got {yolov8_conf}")
    if not (0.0 <= yolov8_iou <= 1.0):
        raise ValueError(f"yolov8_iou must be in [0,1], got {yolov8_iou}")
    if yolov8_imgsz <= 0:
        raise ValueError(f"yolov8_imgsz must be > 0, got {yolov8_imgsz}")
    if yolov8_max_det <= 0:
        raise ValueError(f"yolov8_max_det must be > 0, got {yolov8_max_det}")
    if yolov8_face_box_scale <= 0:
        raise ValueError(
            f"yolov8_face_box_scale must be > 0, got {yolov8_face_box_scale}"
        )

    width, height, total_frames = ffprobe_video_info(video_path)
    if total_frames < frame_num:
        raise RuntimeError(
            f"Video is too short for preprocessing: "
            f"total_frames={total_frames}, frame_num={frame_num}"
        )

    loop_plan = best_loop_plan(
        n=total_frames,
        m=frame_num,
        c=motion_frame,
    )
    effective_span = int(loop_plan["R"])
    discard = int(loop_plan["discard"])
    segment_count = int(loop_plan["segments"])
    stride = int(loop_plan["step"])
    loop_period = int(loop_plan["period"])

    loop_start_offset = resolve_loop_start_offset(
        total_frames=total_frames,
        effective_span=effective_span,
        mode=loop_start_mode,
    )
    effective_start_idx = int(loop_start_offset)
    effective_end_idx = int(loop_start_offset + effective_span - 1)

    if effective_start_idx < 0 or effective_end_idx >= total_frames:
        raise RuntimeError(
            f"Invalid effective range: "
            f"[{effective_start_idx}, {effective_end_idx}], total_frames={total_frames}"
        )

    effective_slice = slice(effective_start_idx, effective_end_idx + 1)

    resolved_landmark_path = None
    face_box_path = None
    ref_indices = None
    ref_indices_local = None
    ref_frames = None

    if face_box_source == "landmark":
        resolved_landmark_path = landmark_path or os.path.splitext(video_path)[0] + ".npy"
        device_id = _parse_cuda_device_id(device)
        ensure_landmarks(
            lmk_weight_path,
            video_path,
            resolved_landmark_path,
            device_id=device_id,
        )

        landmarks = np.load(resolved_landmark_path).astype(np.float32)
        if len(landmarks) != total_frames:
            raise RuntimeError(
                f"Landmark frame count mismatch: "
                f"len(landmarks)={len(landmarks)}, total_frames={total_frames}"
            )

        effective_landmarks = landmarks[effective_slice]
        if len(effective_landmarks) != effective_span:
            raise RuntimeError(
                f"Effective landmark length mismatch: "
                f"len={len(effective_landmarks)}, expected={effective_span}"
            )

        box = design_global_box_from_landmarks(
            effective_landmarks,
            width,
            height,
            margin=margin,
        )
        target_h, target_w = pick_target_size_from_box(box)

        ref_indices_local = select_reference_frame_indices(effective_landmarks)
        ref_indices = [int(loop_start_offset + i) for i in ref_indices_local]

        ref_frames = read_selected_frames(
            video_path=video_path,
            indices=ref_indices,
            box=box,
            target_size=(target_h, target_w),
        )
        reference_mode = "landmark"

    else:
        face_box_path = build_yolov8_face_box_cache_path(
            video_path=video_path,
            weight_path=yolov8_weight_path,
            conf_threshold=yolov8_conf,
            iou_threshold=yolov8_iou,
            imgsz=yolov8_imgsz,
            max_det=yolov8_max_det,
            half=yolov8_half,
        )
        ensure_yolov8_face_boxes(
            video_path=video_path,
            face_box_path=face_box_path,
            device=device,
            weight_path=yolov8_weight_path,
            conf_threshold=yolov8_conf,
            iou_threshold=yolov8_iou,
            imgsz=yolov8_imgsz,
            max_det=yolov8_max_det,
            half=yolov8_half,
        )

        face_boxes = np.load(face_box_path).astype(np.float32)
        if len(face_boxes) != total_frames:
            raise RuntimeError(
                f"Face boxes frame count mismatch: "
                f"len(face_boxes)={len(face_boxes)}, total_frames={total_frames}"
            )

        effective_face_boxes = face_boxes[effective_slice]
        if len(effective_face_boxes) != effective_span:
            raise RuntimeError(
                f"Effective face box length mismatch: "
                f"len={len(effective_face_boxes)}, expected={effective_span}"
            )

        box = design_global_box_from_face_boxes(
            face_boxes=effective_face_boxes,
            width=width,
            height=height,
            margin=margin,
            face_box_scale=yolov8_face_box_scale,
        )
        target_h, target_w = pick_target_size_from_box(box)

        ref_frames = None
        ref_indices = None
        ref_indices_local = None
        reference_mode = "yolov8"

    vae = Wan2_2_VAE(
        vae_pth=vae_path,
        device=torch.device(device),
    )

    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    if ref_frames is not None:
        ref_frames = [x.to(device=device, dtype=dtype) for x in ref_frames]

    records = []

    reader = IndexedVideoReader(video_path)
    try:
        with torch.no_grad():
            for clip_index in range(segment_count):
                virtual_start_idx = int(clip_index * stride)
                virtual_end_idx = int(virtual_start_idx + frame_num - 1)

                source_frame_indices = build_pingpong_source_frame_indices(
                    virtual_start_idx=virtual_start_idx,
                    frame_num=frame_num,
                    effective_span=effective_span,
                    loop_start_offset=loop_start_offset,
                )
                validate_source_frame_indices(
                    source_frame_indices=source_frame_indices,
                    total_frames=total_frames,
                    expected_len=frame_num,
                )

                clip_frames = read_clip_frames_by_indices(
                    reader=reader,
                    source_frame_indices=source_frame_indices,
                    box=box,
                    target_size=(target_h, target_w),
                )

                clip = torch.stack(clip_frames, dim=0)
                clip = rearrange(clip, "t c h w -> 1 c t h w")
                clip = clip.to(device=device, dtype=dtype)

                latent = vae.encode(clip)[0].detach().cpu()

                y_input = build_latent_y_input(
                    clip_frames=clip,
                    frame_num=frame_num,
                    reference_mode=reference_mode,
                    ref_frames=ref_frames,
                )
                latent_y = vae.encode(y_input)[0].detach().cpu()

                record_loop_plan = {
                    "R": int(effective_span),
                    "discard": int(discard),
                    "segments": int(segment_count),
                    "step": int(stride),
                    "period": int(loop_period),
                    "frame_num": int(frame_num),
                    "motion_frame": int(motion_frame),
                    "loop_start_mode": str(loop_start_mode),
                    "loop_start_offset": int(loop_start_offset),
                    "effective_start_idx": int(effective_start_idx),
                    "effective_end_idx": int(effective_end_idx),
                    "total_source_frames": int(total_frames),
                }

                records.append(
                    {
                        "schedule_version": PINGPONG_LOOP_SCHEDULE_VERSION,
                        "bbox": box,

                        "start_idx": int(virtual_start_idx),
                        "end_idx": int(virtual_end_idx),

                        "virtual_start_idx": int(virtual_start_idx),
                        "virtual_end_idx": int(virtual_end_idx),

                        "source_start_idx": int(source_frame_indices[0]),
                        "source_end_idx": int(source_frame_indices[-1]),
                        "source_frame_indices": [int(x) for x in source_frame_indices],
                        "total_source_frames": int(total_frames),

                        "clip_index_in_cycle": int(clip_index),
                        "cycle_segment_count": int(segment_count),
                        "loop_period": int(loop_period),
                        "loop_start_offset": int(loop_start_offset),
                        "effective_start_idx": int(effective_start_idx),
                        "effective_end_idx": int(effective_end_idx),
                        "effective_span": int(effective_span),
                        "discard": int(discard),
                        "stride": int(stride),
                        "is_reversed": bool(is_pingpong_reversed(source_frame_indices)),
                        "loop_plan": record_loop_plan,

                        "latent": latent,
                        "latent_y": latent_y,
                    }
                )

                del clip, clip_frames, y_input, latent, latent_y
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()

    finally:
        reader.close()

    if len(records) == 0:
        raise RuntimeError(
            f"No valid clips were generated. "
            f"total_frames={total_frames}, frame_num={frame_num}, motion_frame={motion_frame}"
        )

    loop_close_remainder = int((segment_count * stride) % loop_period)
    if loop_close_remainder != 0:
        raise RuntimeError(
            f"Loop plan is not closed: "
            f"segments={segment_count}, stride={stride}, period={loop_period}, "
            f"remainder={loop_close_remainder}"
        )

    os.makedirs(os.path.dirname(out_pt) or ".", exist_ok=True)
    torch.save(records, out_pt)

    print(f"Saved {len(records)} clips to {out_pt}")
    print(f"schedule_version={PINGPONG_LOOP_SCHEDULE_VERSION}")
    print(f"face_box_source={face_box_source}")
    if resolved_landmark_path is not None:
        print(f"landmark_path={resolved_landmark_path}")
    if face_box_path is not None:
        print(f"face_box_path={face_box_path}")
    print(
        f"loop_plan={loop_plan}, loop_start_mode={loop_start_mode}, "
        f"loop_start_offset={loop_start_offset}, "
        f"effective_range=[{effective_start_idx}, {effective_end_idx}], "
        f"box={box}, target_size=({target_h}, {target_w}), "
        f"ref_indices={ref_indices}, ref_indices_local={ref_indices_local}, "
        f"total_frames={total_frames}"
    )

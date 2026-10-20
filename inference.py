"""Command-line inference for Chanjing-Avatar V2V 1.3B Flash."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model_1_3b_flash.json"
EXPECTED_CONFIG = {
    "dim": 1536,
    "in_dim": 96,
    "ffn_dim": 8960,
    "out_dim": 48,
    "freq_dim": 256,
    "num_heads": 12,
    "num_layers": 30,
    "patch_size": [1, 1, 1],
    "vae_stride": [4, 16, 16],
}
MODEL_VARIANTS = {
    "original": {
        "directory": "original",
        "sample_steps": 14,
        "sample_shift": 7.0,
        "audio_guide_scale": 7.0,
    },
    "distilled": {
        "directory": "cfg-distilled",
        "sample_steps": 6,
        "sample_shift": 7.0,
        "audio_guide_scale": 1.0,
    },
}


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )


def _default_model_root() -> Path:
    return Path(os.environ.get("CHANJING_AVATAR_MODEL_DIR", PROJECT_ROOT / "models"))


def collect_model_files(path_value: str) -> list[str]:
    """Resolve a file, shard directory, or comma-separated checkpoint list."""
    resolved: list[str] = []
    for raw_part in (part.strip() for part in path_value.split(",")):
        if not raw_part:
            continue
        path = Path(raw_part).expanduser()
        if path.is_file():
            resolved.append(str(path.resolve()))
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Model path does not exist: {path}")
        files: list[Path] = []
        for pattern in ("model-*.safetensors", "*.safetensors", "*.pt", "*.pth"):
            files = sorted(path.glob(pattern))
            if files:
                break
        if not files:
            raise FileNotFoundError(f"No checkpoint files found in: {path}")
        resolved.extend(str(file.resolve()) for file in files)
    if not resolved:
        raise ValueError("--model_path did not resolve to any checkpoint files")
    return list(dict.fromkeys(resolved))


def validate_model_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    mismatches = {
        key: (EXPECTED_CONFIG[key], config.get(key))
        for key in EXPECTED_CONFIG
        if config.get(key) != EXPECTED_CONFIG[key]
    }
    if mismatches:
        details = ", ".join(
            f"{key}: expected {expected!r}, got {actual!r}"
            for key, (expected, actual) in mismatches.items()
        )
        raise ValueError(f"Incompatible Flash model config ({details})")
    return config


def validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    required_paths = {
        "video": args.video_path,
        "audio": args.audio_path,
        "VAE": args.vae_path,
        "audio encoder": args.wav2vec_dir,
    }
    if args.face_box_source == "landmark":
        required_paths["landmark detector"] = args.lmk_weight_path
        if not args.preprocess_device.startswith("cuda"):
            errors.append("landmark preprocessing currently requires a CUDA device")
    else:
        required_paths["YOLO face detector"] = args.yolov8_weight_path
    for label, value in required_paths.items():
        if not value or not Path(value).expanduser().exists():
            errors.append(f"{label} path does not exist: {value}")
    try:
        validate_model_config(args.dit_config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    try:
        collect_model_files(args.model_path)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    if shutil.which("ffmpeg") is None:
        errors.append("ffmpeg is not available on PATH")
    if args.frame_num < 5 or (args.frame_num - 1) % 4 != 0:
        errors.append(f"frame_num must satisfy 4n+1, got {args.frame_num}")
    if not 1 <= args.motion_frame < args.frame_num:
        errors.append("motion_frame must be at least 1 and smaller than frame_num")
    if args.audio_radius != 2:
        errors.append("audio_radius must be 2 for the released Flash architecture")
    if not 0.0 <= args.color_correction_strength <= 1.0:
        errors.append("color_correction_strength must be in [0, 1]")
    if args.sample_steps <= 0:
        errors.append("sample_steps must be greater than zero")
    if not 0.0 <= args.yolov8_conf <= 1.0 or not 0.0 <= args.yolov8_iou <= 1.0:
        errors.append("yolov8_conf and yolov8_iou must be in [0, 1]")
    if args.yolov8_imgsz <= 0 or args.yolov8_max_det <= 0 or args.yolov8_face_box_scale <= 0:
        errors.append("YOLO image size, max detections, and face-box scale must be positive")
    return errors


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _preprocess_signature(args: argparse.Namespace) -> str:
    payload = {
        "video": str(Path(args.video_path).resolve()),
        "frame_num": args.frame_num,
        "motion_frame": args.motion_frame,
        "preprocess_margin": args.preprocess_margin,
        "face_box_source": args.face_box_source,
        "lmk_weight_path": args.lmk_weight_path,
        "yolov8_weight_path": args.yolov8_weight_path,
        "yolov8_conf": args.yolov8_conf,
        "yolov8_iou": args.yolov8_iou,
        "yolov8_imgsz": args.yolov8_imgsz,
        "yolov8_max_det": args.yolov8_max_det,
        "yolov8_half": args.yolov8_half,
        "yolov8_face_box_scale": args.yolov8_face_box_scale,
        "loop_start_mode": args.loop_start_mode,
    }
    return _hash_payload(payload)


def _inference_signature(
    args: argparse.Namespace,
    model_files: Sequence[str],
    preprocess_tag: str,
) -> str:
    payload = {
        "preprocess_tag": preprocess_tag,
        "audio": str(Path(args.audio_path).resolve()),
        "model_variant": args.model_variant,
        "model": list(model_files),
        "sample_steps": args.sample_steps,
        "sample_shift": args.sample_shift,
        "audio_guide_scale": args.audio_guide_scale,
        "seed": args.seed,
    }
    return _hash_payload(payload)


def build_parser() -> argparse.ArgumentParser:
    model_root = _default_model_root()
    parser = argparse.ArgumentParser(
        description="Chanjing-Avatar V2V 1.3B Flash: preprocess, infer, and decode"
    )
    parser.add_argument("--video_path", required=True, help="Source portrait video")
    parser.add_argument("--audio_path", required=True, help="Driving audio")
    parser.add_argument(
        "--model_variant",
        choices=sorted(MODEL_VARIANTS),
        default="original",
        help="Select the original or audio-CFG-distilled inference defaults",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Checkpoint file, shard directory, or comma-separated checkpoint files",
    )
    parser.add_argument("--vae_path", default=str(model_root / "Wan2.2_VAE.pth"))
    parser.add_argument("--wav2vec_dir", default=str(model_root / "chinese-wav2vec2-base"))
    parser.add_argument("--dit_config_path", default=str(DEFAULT_CONFIG))
    parser.add_argument("--lmk_weight_path", default=str(model_root / "landmark.onnx"))
    parser.add_argument("--yolov8_weight_path", default=str(model_root / "face_yolov8s.pt"))
    parser.add_argument("--output_path", help="Output video; defaults beside the source video")
    parser.add_argument("--work_dir", default="outputs/cache", help="Intermediate tensor directory")

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audio_device", default="cpu")
    parser.add_argument("--preprocess_device", default="cuda:0")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO")

    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--motion_frame", type=int, default=9)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--audio_guide_scale", type=float, default=None)
    parser.add_argument("--audio_radius", type=int, default=2)
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--preprocess_margin", type=int, default=32)
    parser.add_argument("--loop_start_mode", choices=["head", "random"], default="head")

    parser.add_argument("--face_box_source", choices=["landmark", "yolov8"], default="landmark")
    parser.add_argument("--yolov8_conf", type=float, default=0.5)
    parser.add_argument("--yolov8_iou", type=float, default=0.45)
    parser.add_argument("--yolov8_imgsz", type=int, default=640)
    parser.add_argument("--yolov8_max_det", type=int, default=300)
    parser.add_argument("--yolov8_half", action="store_true")
    parser.add_argument("--yolov8_face_box_scale", type=float, default=1.25)

    parser.add_argument("--color_correction_strength", type=float, default=1.0)
    parser.add_argument("--video_codec", default="h264_nvenc")
    parser.add_argument("--video_preset", default="medium")
    parser.add_argument("--video_quality", type=int, default=10)
    parser.add_argument("--disable_source_cache", action="store_true")
    parser.add_argument("--overwrite_intermediate", action="store_true")
    parser.add_argument("--overwrite_output", action="store_true")
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate configuration and local assets without loading models",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    preset = MODEL_VARIANTS[args.model_variant]
    model_root = _default_model_root()
    if args.model_path is None:
        args.model_path = str(
            model_root
            / "Chanjing-Avatar-V2V-1.3B-Flash"
            / preset["directory"]
        )
    for name in ("sample_steps", "sample_shift", "audio_guide_scale"):
        if getattr(args, name) is None:
            setattr(args, name, preset[name])
    return args


def run(args: argparse.Namespace) -> Path:
    errors = validate_args(args)
    if errors:
        raise RuntimeError("Preflight validation failed:\n- " + "\n- ".join(errors))
    model_files = collect_model_files(args.model_path)
    if args.validate_only:
        logging.info("Validation passed; no model was loaded.")
        return Path(args.output_path) if args.output_path else Path(args.video_path)

    import torch
    from chanjing_avatar.decode import DecodePipeline
    from chanjing_avatar.inference import LatentOnlyInferenceEngine, set_seed
    from chanjing_avatar.preprocess import preprocess_video

    set_seed(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    preprocess_tag = _preprocess_signature(args)
    inference_tag = _inference_signature(args, model_files, preprocess_tag)
    preprocessed_path = work_dir / f"preprocessed_{preprocess_tag}.pt"
    latent_path = work_dir / f"latents_{inference_tag}.pt"
    output_path = Path(args.output_path) if args.output_path else Path(args.video_path).with_name(
        f"{Path(args.video_path).stem}_chanjing.mp4"
    )
    if output_path.exists() and not args.overwrite_output:
        raise FileExistsError(f"Output exists: {output_path}; pass --overwrite_output to replace it")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not preprocessed_path.exists() or args.overwrite_intermediate:
        preprocess_video(
            video_path=args.video_path,
            out_pt=str(preprocessed_path),
            vae_path=args.vae_path,
            device=args.preprocess_device,
            frame_num=args.frame_num,
            motion_frame=args.motion_frame,
            margin=args.preprocess_margin,
            face_box_source=args.face_box_source,
            lmk_weight_path=args.lmk_weight_path,
            landmark_path=str(work_dir / f"landmarks_{preprocess_tag}.npy"),
            yolov8_weight_path=args.yolov8_weight_path,
            yolov8_conf=args.yolov8_conf,
            yolov8_iou=args.yolov8_iou,
            yolov8_imgsz=args.yolov8_imgsz,
            yolov8_max_det=args.yolov8_max_det,
            yolov8_half=args.yolov8_half,
            yolov8_face_box_scale=args.yolov8_face_box_scale,
            loop_start_mode=args.loop_start_mode,
        )

    if not latent_path.exists() or args.overwrite_intermediate:
        engine_args = argparse.Namespace(**vars(args))
        engine_args.preprocessed_path = str(preprocessed_path)
        engine_args.model_path = ",".join(model_files)
        engine_args.out_path = str(latent_path)
        engine_args.overwrite_preprocess = False
        result = LatentOnlyInferenceEngine(engine_args).run()
        torch.save(result, latent_path)
        with latent_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(result["meta"], handle, indent=2)

    decode_args = argparse.Namespace(
        latent_path=str(latent_path),
        out_path=str(output_path),
        video_path=args.video_path,
        audio_path=args.audio_path,
        preprocessed_path=str(preprocessed_path),
        vae_path=args.vae_path,
        device=args.device,
        dtype=args.dtype,
        frame_num=None,
        motion_frame=None,
        fps=None,
        color_correction_strength=args.color_correction_strength,
        video_codec=args.video_codec,
        video_preset=args.video_preset,
        video_quality=args.video_quality,
        disable_source_cache=args.disable_source_cache,
        log_level=args.log_level,
    )
    DecodePipeline(decode_args).run()
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        output_path = run(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        logging.error("%s", exc)
        return 1
    if not args.validate_only:
        logging.info("Output saved to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

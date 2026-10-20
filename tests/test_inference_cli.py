from pathlib import Path

import inference


def test_collects_safetensor_shards_before_other_formats(tmp_path: Path):
    (tmp_path / "model-00002-of-00002.safetensors").touch()
    (tmp_path / "model-00001-of-00002.safetensors").touch()
    (tmp_path / "ignored.pth").touch()
    files = inference.collect_model_files(str(tmp_path))
    assert [Path(file).name for file in files] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]


def test_public_cli_requires_input_media():
    parser = inference.build_parser()
    try:
        parser.parse_args([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("CLI accepted missing input paths")


def test_cfg_distilled_variant_uses_six_step_defaults():
    args = inference.parse_args(
        ["--video_path", "source.mp4", "--audio_path", "driving.wav", "--model_variant", "distilled"]
    )
    assert args.model_path.endswith("cfg-distilled")
    assert args.sample_steps == 6
    assert args.sample_shift == 7.0
    assert args.audio_guide_scale == 1.0


def test_original_variant_keeps_explicit_sampling_override():
    args = inference.parse_args(
        ["--video_path", "source.mp4", "--audio_path", "driving.wav", "--sample_steps", "10"]
    )
    assert args.model_path.endswith("original")
    assert args.sample_steps == 10
    assert args.audio_guide_scale == 7.0


def test_model_variants_share_preprocess_cache_but_not_latent_cache():
    original = inference.parse_args(["--video_path", "source.mp4", "--audio_path", "driving.wav"])
    distilled = inference.parse_args(
        ["--video_path", "source.mp4", "--audio_path", "driving.wav", "--model_variant", "distilled"]
    )
    original_preprocess = inference._preprocess_signature(original)
    distilled_preprocess = inference._preprocess_signature(distilled)
    assert original_preprocess == distilled_preprocess
    assert inference._inference_signature(original, ["original.safetensors"], original_preprocess) != (
        inference._inference_signature(distilled, ["distilled.safetensors"], distilled_preprocess)
    )

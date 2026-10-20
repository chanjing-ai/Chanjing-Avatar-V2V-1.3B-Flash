import json
from pathlib import Path

import inference


def test_released_config_matches_runtime_contract():
    config = inference.validate_model_config(str(inference.DEFAULT_CONFIG))
    assert config["in_dim"] == 96
    assert config["out_dim"] == 48
    assert config["vae_stride"] == [4, 16, 16]


def test_incompatible_config_is_rejected(tmp_path: Path):
    path = tmp_path / "config.json"
    config = dict(inference.EXPECTED_CONFIG)
    config["out_dim"] = 16
    path.write_text(json.dumps(config), encoding="utf-8")
    try:
        inference.validate_model_config(str(path))
    except ValueError as exc:
        assert "out_dim" in str(exc)
    else:
        raise AssertionError("incompatible config was accepted")

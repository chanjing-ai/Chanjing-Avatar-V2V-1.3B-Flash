from pathlib import Path

import torch
from safetensors.torch import save_file

from training.train import ChanjingAvatarDataset, find_samples, load_base_weights
from training.preprocess import has_complete_samples


def _write_sample(root: Path, index: int = 1):
    suffix = f"{index:03d}.pth"
    torch.save(torch.zeros(48, 11, 4, 4), root / f"latents_{suffix}")
    torch.save(torch.zeros(41, 5, 12, 768), root / f"audio_{suffix}")
    torch.save(torch.zeros(48, 11, 4, 4), root / f"latents_y_{suffix}")


def test_dataset_discovers_only_complete_triplets(tmp_path: Path):
    _write_sample(tmp_path)
    torch.save(torch.zeros(1), tmp_path / "latents_999.pth")
    assert len(find_samples(tmp_path)) == 1
    sample = ChanjingAvatarDataset(tmp_path)[0]
    assert tuple(sample["latents"].shape) == (48, 11, 4, 4)
    assert tuple(sample["audio"].shape) == (41, 5, 12, 768)
    assert has_complete_samples(tmp_path)


def test_loaded_parameters_are_frozen_and_new_parameters_remain_trainable(tmp_path: Path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    checkpoint = tmp_path / "base.safetensors"
    save_file(
        {
            "model.0.weight": model[0].weight.detach().clone(),
            "model.0.bias": model[0].bias.detach().clone(),
        },
        checkpoint,
    )
    report = load_base_weights(model, [checkpoint], trainable="unloaded")
    assert report["loaded"] == 2
    assert not model[0].weight.requires_grad
    assert model[1].weight.requires_grad

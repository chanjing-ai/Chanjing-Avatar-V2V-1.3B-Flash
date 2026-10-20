"""Training entrypoint for Chanjing-Avatar V2V 1.3B Flash."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import lightning as pl
import torch
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, Dataset

from chanjing_avatar.models.dit import WanModelAudioProject
from chanjing_avatar.schedulers.flow_match import FlowMatchScheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_samples(data_dir: str | Path) -> list[tuple[Path, Path, Path]]:
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    samples: list[tuple[Path, Path, Path]] = []
    for latent in sorted(root.rglob("latents_*.pth")):
        suffix = latent.name.removeprefix("latents_")
        audio = latent.with_name(f"audio_{suffix}")
        condition = latent.with_name(f"latents_y_{suffix}")
        if audio.is_file() and condition.is_file():
            samples.append((latent, audio, condition))
    if not samples:
        raise ValueError(
            f"No complete samples under {root}; expected latents_NNN.pth, "
            "audio_NNN.pth, and latents_y_NNN.pth in the same directory"
        )
    return samples


class ChanjingAvatarDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        self.samples = find_samples(data_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        latent_path, audio_path, condition_path = self.samples[index]
        latents = torch.load(latent_path, map_location="cpu", weights_only=True)
        audio = torch.load(audio_path, map_location="cpu", weights_only=True)
        condition = torch.load(condition_path, map_location="cpu", weights_only=True)
        for label, value in (("latents", latents), ("audio", audio), ("latents_y", condition)):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{label} must be a tensor: {latent_path.parent}")
        if latents.shape != condition.shape:
            raise ValueError(
                f"Latent and condition shapes differ at {latent_path.parent}: "
                f"{tuple(latents.shape)} != {tuple(condition.shape)}"
            )
        if latents.ndim != 4 or latents.shape[0] != 48:
            raise ValueError(f"Expected latent shape [48,T,H,W], got {tuple(latents.shape)}")
        if audio.ndim != 4 or tuple(audio.shape[1:]) != (5, 12, 768):
            raise ValueError(f"Expected audio shape [T,5,12,768], got {tuple(audio.shape)}")
        return {"latents": latents, "audio": audio, "y": condition}


def resolve_checkpoint_files(value: str | None) -> list[Path]:
    if not value:
        return []
    files: list[Path] = []
    for raw in (item.strip() for item in value.split(",")):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            shards = sorted(path.glob("model-*.safetensors")) or sorted(path.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"No safetensors checkpoints in: {path}")
            files.extend(shards)
        else:
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    return list(dict.fromkeys(file.resolve() for file in files))


def _strip_prefix(key: str) -> str:
    for prefix in ("model.model.", "module.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def load_base_weights(
    model: torch.nn.Module,
    files: Sequence[Path],
    trainable: str,
) -> dict[str, Any]:
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    unexpected: list[str] = []
    for file in files:
        state = load_file(str(file), device="cpu")
        for raw_key, value in state.items():
            key = _strip_prefix(raw_key)
            if key not in target:
                unexpected.append(key)
            elif target[key].shape != value.shape:
                skipped_shape.append(key)
            else:
                compatible[key] = value
    result = model.load_state_dict(compatible, strict=False)
    loaded = set(compatible)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable == "all" or name not in loaded
    return {
        "loaded": len(loaded),
        "missing": len(result.missing_keys),
        "unexpected": len(set(unexpected)),
        "shape_mismatch": len(set(skipped_shape)),
    }


class ChanjingAvatarTrainingModule(pl.LightningModule):
    def __init__(
        self,
        config_path: str,
        base_model: str | None,
        learning_rate: float = 1e-5,
        trainable: str = "unloaded",
        init_from_scratch: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_model"])
        with Path(config_path).open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        self.model = WanModelAudioProject(**config)
        files = resolve_checkpoint_files(base_model)
        if not files and not init_from_scratch:
            raise ValueError("Provide --base_model or explicitly pass --init_from_scratch")
        self.load_report = (
            load_base_weights(self.model, files, trainable=trainable)
            if files
            else {"loaded": 0, "missing": len(self.model.state_dict()), "unexpected": 0, "shape_mismatch": 0}
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.learning_rate = learning_rate

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        latents = batch["latents"]
        audio = batch["audio"]
        condition = batch["y"]
        if random.random() < 0.15:
            audio = torch.zeros_like(audio)
        keep_heads = random.choices([0, 1, 2], weights=[0.1, 0.2, 0.7], k=1)[0]
        crop = random.choices([0, 1, 2], weights=[0.1, 0.3, 0.6], k=1)[0]
        spatial = slice(crop, -crop) if crop else slice(None)
        clean_core = latents[:, :, keep_heads:, spatial, spatial].clone()
        noise = torch.randn_like(clean_core)
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (1,)).item()
        timestep = self.scheduler.timesteps[timestep_id].to(latents.device, latents.dtype).reshape(1)
        noisy_core = self.scheduler.add_noise(clean_core, noise, timestep)
        target = self.scheduler.training_target(clean_core, noise, timestep)
        noisy_latents = latents.clone()
        noisy_latents[:, :, keep_heads:, spatial, spatial] = noisy_core
        prediction = self.model(
            x=noisy_latents,
            timestep=timestep,
            context=audio,
            y=condition,
            use_gradient_checkpointing=True,
        )
        prediction = prediction[:, :, keep_heads:, spatial, spatial]
        if prediction.shape != target.shape:
            raise RuntimeError(f"Prediction shape {prediction.shape} != target shape {target.shape}")
        flow_loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
        weight = self.scheduler.training_weight(timestep).to(flow_loss.device)
        loss = flow_loss * weight
        self.log_dict({"train_loss": loss, "flow_loss": flow_loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("No trainable parameters remain; use --trainable all")
        return torch.optim.AdamW(parameters, lr=self.learning_rate)


class SafetensorsCheckpoint(pl.Callback):
    def __init__(self, output_dir: str, every_n_steps: int):
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps

    def _save(self, trainer: pl.Trainer, module: ChanjingAvatarTrainingModule, name: str) -> None:
        if not trainer.is_global_zero:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainable_names = {name for name, param in module.model.named_parameters() if param.requires_grad}
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in module.model.state_dict().items()
            if key in trainable_names
        }
        save_file(state, str(self.output_dir / name))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self.every_n_steps > 0 and trainer.global_step % self.every_n_steps == 0:
            self._save(trainer, pl_module, f"step-{trainer.global_step:08d}.safetensors")

    def on_train_end(self, trainer, pl_module) -> None:
        self._save(trainer, pl_module, "final.safetensors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Chanjing-Avatar V2V 1.3B Flash")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--base_model", help="Safetensors file, shard directory, or comma-separated files")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "model_1_3b_flash.json"))
    parser.add_argument("--output_dir", default="outputs/training")
    parser.add_argument("--experiment_name", default="chanjing-avatar-flash")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1, help="Stop after this many optimizer steps; -1 disables")
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--devices", default="1", help="Lightning device count or 'auto'")
    parser.add_argument("--trainable", choices=["unloaded", "all"], default="unloaded")
    parser.add_argument("--init_from_scratch", action="store_true")
    parser.add_argument(
        "--strategy",
        default="auto",
        choices=["auto", "ddp"],
    )
    parser.add_argument("--validate_only", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> int:
    if not Path(args.config).is_file():
        raise FileNotFoundError(f"Model config does not exist: {args.config}")
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected = {"dim": 1536, "in_dim": 96, "out_dim": 48, "num_layers": 30, "vae_stride": [4, 16, 16]}
    mismatches = [key for key, value in expected.items() if config.get(key) != value]
    if mismatches:
        raise ValueError(f"Incompatible Flash model config keys: {', '.join(mismatches)}")
    sample_count = len(find_samples(args.data_dir))
    files = resolve_checkpoint_files(args.base_model)
    if not files and not args.init_from_scratch:
        raise ValueError("Provide --base_model or explicitly pass --init_from_scratch")
    if args.save_steps < 1 or args.accumulate_grad_batches < 1 or args.num_workers < 0:
        raise ValueError("save_steps/accumulate_grad_batches must be positive and num_workers non-negative")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    if args.devices != "auto" and (not args.devices.isdigit() or int(args.devices) < 1):
        raise ValueError("devices must be 'auto' or a positive integer")
    return sample_count


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sample_count = validate_args(args)
    if args.validate_only:
        print(f"Validation passed: {sample_count} training sample(s); no model was loaded.")
        return 0
    dataset = ChanjingAvatarDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    module = ChanjingAvatarTrainingModule(
        config_path=args.config,
        base_model=args.base_model,
        learning_rate=args.learning_rate,
        trainable=args.trainable,
        init_from_scratch=args.init_from_scratch,
    )
    print(f"Base weight load report: {module.load_report}")
    logger = pl.pytorch.loggers.TensorBoardLogger(
        save_dir=args.output_dir,
        name="tensorboard",
        version=args.experiment_name,
    )
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator="gpu",
        devices="auto" if args.devices == "auto" else int(args.devices),
        precision="bf16-mixed",
        strategy=args.strategy,
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=1,
        enable_checkpointing=False,
        callbacks=[SafetensorsCheckpoint(args.output_dir, args.save_steps)],
        logger=logger,
    )
    trainer.fit(module, dataloader)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

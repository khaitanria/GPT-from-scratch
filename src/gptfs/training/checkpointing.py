from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from gptfs.training.training_config import TrainingConfig

CONFIG_FILENAME = "config.json"
VOCAB_FILENAME = "vocab.json"
CHECKPOINTS_DIRNAME = "checkpoints"
LATEST_CHECKPOINT_FILENAME = "latest.pt"


def ensure_run_dirs(run_dir: Path) -> Path:
    """
    Ensure <run_dir>/checkpoints exists and return the checkpoints directory path.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / CHECKPOINTS_DIRNAME
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir


def write_config_json(config: TrainingConfig) -> Path:
    """
    Write config.json into the run directory.
    """
    run_dir = config.run_dir
    ensure_run_dirs(run_dir)

    config_path = run_dir / CONFIG_FILENAME
    payload = config.to_dict()
    _write_json(path=config_path, payload=payload)
    return config_path


def write_vocab_json(
    run_dir: Path,
    stoi: dict[str, int],
    itos: list[str],
) -> Path:
    """
    Write vocab.json into the run directory.
    """
    ensure_run_dirs(run_dir)

    vocab_path = run_dir / VOCAB_FILENAME
    payload: dict[str, Any] = {"stoi": stoi, "itos": itos}
    _write_json(path=vocab_path, payload=payload)
    return vocab_path


def latest_checkpoint_path(run_dir: Path) -> Path:
    """
    Return <run_dir>/checkpoints/latest.pt.
    """
    return run_dir / CHECKPOINTS_DIRNAME / LATEST_CHECKPOINT_FILENAME


def step_checkpoint_path(run_dir: Path, step: int) -> Path:
    """
    Return <run_dir>/checkpoints/step_<step>.pt with zero padding.
    """
    return run_dir / CHECKPOINTS_DIRNAME / f"step_{step:06d}.pt"


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    step: int,
    tokens_processed: int,
    epoch: int,
    epoch_fraction: float,
    save_rng_state: bool = True,
) -> None:
    """
    Save model/optimizer state plus progress metadata to a .pt checkpoint.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    progress = {
        "step": step,
        "tokens_processed": tokens_processed,
        "epoch": epoch,
        "epoch_fraction": epoch_fraction,
    }

    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "progress": progress,
        # IMPORTANT: torch.__version__ is a TorchVersion object in newer PyTorch versions.
        # Convert to a plain str so torch.load(weights_only=True) can load safely.
        "pytorch_version": str(torch.__version__),
    }

    if save_rng_state:
        payload["rng_state"] = _get_rng_state()

    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint dictionary from disk.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    loaded: dict[str, Any] = torch.load(checkpoint_path, map_location=map_location)
    return loaded


def restore_model_and_optimizer(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
) -> dict[str, Any]:
    """
    Restore model + optimizer state from a loaded checkpoint dict and return progress.
    """
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    progress: dict[str, Any] = checkpoint["progress"]
    return progress


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text + "\n", encoding="utf-8")


def _get_rng_state() -> dict[str, Any]:
    rng: dict[str, Any] = {"cpu": torch.get_rng_state()}

    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()

    return rng

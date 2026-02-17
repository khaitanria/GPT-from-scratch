from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from gptfs.training.checkpointing import (
    load_checkpoint,
    restore_model_and_optimizer,
    save_checkpoint,
    step_checkpoint_path,
    write_config_json,
    write_vocab_json,
)
from gptfs.training.training_config import TrainingConfig


def test_write_config_json_writes_expected_file() -> None:
    """
    Verifies config.json is written into the run directory.
    """
    with TemporaryDirectory() as tmp:
        cfg = TrainingConfig(runs_dir=Path(tmp), run_name="run")

        config_path = write_config_json(cfg)

        assert config_path.exists()
        assert config_path.name == "config.json"
        assert config_path.parent == cfg.run_dir

        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["run_name"] == "run"
        assert data["runs_dir"] == str(Path(tmp))
        assert data["run_dir"] == str(Path(tmp) / "run")


def test_write_vocab_json_writes_expected_file() -> None:
    """
    Verifies vocab.json is written into the run directory.
    """
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        stoi = {"a": 0, "b": 1}
        itos = ["a", "b"]

        vocab_path = write_vocab_json(run_dir=run_dir, stoi=stoi, itos=itos)

        assert vocab_path.exists()
        assert vocab_path.name == "vocab.json"
        assert vocab_path.parent == run_dir

        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        assert data["stoi"] == stoi
        assert data["itos"] == itos


def test_save_and_load_checkpoint_roundtrip_restores_state() -> None:
    """
    Verifies checkpoint save/load restores model parameters, optimizer state, and progress.
    """
    torch.manual_seed(0)

    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        ckpt_path = step_checkpoint_path(run_dir=run_dir, step=5)

        model = nn.Linear(4, 3, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

        x = torch.randn(2, 4)
        target = torch.randn(2, 3)
        loss = torch.mean((model(x) - target) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        saved_params = [p.detach().clone() for p in model.parameters()]

        save_checkpoint(
            checkpoint_path=ckpt_path,
            model=model,
            optimizer=optimizer,
            step=5,
            tokens_processed=1234,
            epoch=0,
            epoch_fraction=0.25,
            save_rng_state=True,
        )

        loaded = load_checkpoint(ckpt_path, map_location="cpu")

        model2 = nn.Linear(4, 3, bias=False)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3, weight_decay=0.1)

        progress = restore_model_and_optimizer(loaded, model=model2, optimizer=optimizer2)

        restored_params = [p.detach().clone() for p in model2.parameters()]
        for a, b in zip(saved_params, restored_params, strict=True):
            assert torch.allclose(a, b)

        assert progress["step"] == 5
        assert progress["tokens_processed"] == 1234
        assert progress["epoch"] == 0
        assert progress["epoch_fraction"] == 0.25

        opt_state = optimizer2.state_dict()
        assert "state" in opt_state
        assert "param_groups" in opt_state
        assert len(opt_state["param_groups"]) == 1

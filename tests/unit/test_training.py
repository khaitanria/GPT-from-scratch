from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from gptfs.training import TrainingConfig, train
from gptfs.training.checkpointing import load_checkpoint


def test_training_smoke_run_writes_artifacts_and_checkpoint() -> None:
    """
    Verifies a tiny CPU training run:
    - runs end-to-end without error
    - produces finite losses
    - writes config.json, vocab.json, and checkpoints/latest.pt
    - returns a TrainingResult with correct paths
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Ensure enough tokens for batching and validation split.
        text = ("FROM FAIRYLAND\n" * 200) + ("SONNET\n" * 200)
        (raw_dir / "dataset.txt").write_text(text, encoding="utf-8")

        runs_dir = tmp_path / "runs"

        cfg = TrainingConfig(
            raw_data_dir=raw_dir,
            runs_dir=runs_dir,
            run_name="smoke",
            corpus_files=("dataset.txt",),
            train_split=0.9,
            device="cpu",
            seed=123,
            context_length=16,
            model_dim=32,
            num_blocks=1,
            num_heads=4,
            batch_size=4,
            learning_rate=1e-3,
            weight_decay=0.0,
            max_steps=4,
            eval_interval=2,
            eval_steps=2,
            save_interval=2,
            save_step_checkpoints=False,
        )

        result = train(cfg)

        assert result.run_dir.exists()
        assert result.config_path.exists()
        assert result.vocab_path.exists()
        assert result.latest_checkpoint_path.exists()

        assert result.steps_completed == cfg.max_steps
        assert result.tokens_processed == cfg.max_steps * cfg.batch_size * cfg.context_length

        assert torch.isfinite(torch.tensor(result.final_train_loss))
        if result.final_val_loss is not None:
            assert torch.isfinite(torch.tensor(result.final_val_loss))

        checkpoint = load_checkpoint(result.latest_checkpoint_path, map_location="cpu")
        assert checkpoint["progress"]["step"] == cfg.max_steps

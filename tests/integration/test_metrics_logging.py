from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from gptfs.training import TrainingConfig, train


def test_training_writes_metrics_jsonl() -> None:
    """
    Verifies a tiny run writes metrics.jsonl with expected keys and sane values.
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Long enough to allow train/val split and val evaluation.
        text = ("SONNET\n" * 400) + ("LINE\n" * 400)
        (raw_dir / "dataset.txt").write_text(text, encoding="utf-8")

        runs_dir = tmp_path / "runs"

        cfg = TrainingConfig(
            raw_data_dir=raw_dir,
            runs_dir=runs_dir,
            run_name="metrics_smoke",
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
            max_steps=2,
            eval_interval=1,
            eval_steps=1,
            save_interval=2,
            save_step_checkpoints=False,
        )

        result = train(cfg)

        metrics_path = result.run_dir / "metrics.jsonl"
        assert metrics_path.exists()

        lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1

        rec = json.loads(lines[-1])

        assert rec["run_name"] == "metrics_smoke"
        assert rec["step"] == cfg.max_steps
        assert rec["tokens_processed"] == cfg.max_steps * cfg.batch_size * cfg.context_length

        assert isinstance(rec["train_loss"], float)
        assert math.isfinite(rec["train_loss"])
        assert isinstance(rec["train_ppl"], float)
        assert rec["train_ppl"] > 0.0

        # val_loss can be None in edge cases; accept either.
        if rec["val_loss"] is not None:
            assert isinstance(rec["val_loss"], float)
            assert math.isfinite(rec["val_loss"])
            assert rec["val_ppl"] is not None
            assert rec["val_ppl"] > 0.0

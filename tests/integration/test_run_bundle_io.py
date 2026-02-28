from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from gptfs.inference.run_bundle_io import read_config, read_vocab


def test_read_config_reads_json_object() -> None:
    """
    Verifies read_config() loads a JSON object from config.json.
    """
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "context_length": 8,
            "model_dim": 24,
            "num_blocks": 1,
            "num_heads": 3,
        }
        (run_dir / "config.json").write_text(json.dumps(payload), encoding="utf-8")

        config = read_config(run_dir)

        assert config["context_length"] == 8
        assert config["model_dim"] == 24
        assert config["num_blocks"] == 1
        assert config["num_heads"] == 3


def test_read_vocab_reads_stoi_and_itos() -> None:
    """
    Verifies read_vocab() returns (stoi, itos) with the expected types.
    """
    with TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "stoi": {"\n": 0, "a": 1, "b": 2},
            "itos": ["\n", "a", "b"],
        }
        (run_dir / "vocab.json").write_text(json.dumps(payload), encoding="utf-8")

        stoi, itos = read_vocab(run_dir)

        assert stoi["\n"] == 0
        assert stoi["a"] == 1
        assert itos[0] == "\n"
        assert itos[2] == "b"

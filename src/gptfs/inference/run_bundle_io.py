from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_config(run_dir: Path) -> dict[str, Any]:
    """
    Read config.json from a run bundle directory.
    """
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json at: {config_path}")

    raw = config_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise TypeError(f"config.json must contain a JSON object. Found: {type(data)}")

    return data


def read_vocab(run_dir: Path) -> tuple[dict[str, int], list[str]]:
    """
    Read vocab.json from a run bundle directory.

    Returns:
      stoi: dict[str, int]
      itos: list[str]
    """
    run_dir = Path(run_dir)
    vocab_path = run_dir / "vocab.json"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing vocab.json at: {vocab_path}")

    raw = vocab_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise TypeError(f"vocab.json must contain a JSON object. Found: {type(data)}")

    stoi_raw = data.get("stoi")
    itos_raw = data.get("itos")

    if not isinstance(stoi_raw, dict):
        raise TypeError(f"vocab.json 'stoi' must be a dict. Found: {type(stoi_raw)}")
    if not isinstance(itos_raw, list):
        raise TypeError(f"vocab.json 'itos' must be a list. Found: {type(itos_raw)}")

    stoi: dict[str, int] = {}
    for k, v in stoi_raw.items():
        if not isinstance(k, str) or not isinstance(v, int):
            raise TypeError("vocab.json 'stoi' must map str -> int.")
        stoi[k] = v

    itos: list[str] = []
    for ch in itos_raw:
        if not isinstance(ch, str):
            raise TypeError("vocab.json 'itos' must be a list[str].")
        itos.append(ch)

    return stoi, itos

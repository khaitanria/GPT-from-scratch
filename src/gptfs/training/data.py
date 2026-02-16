from __future__ import annotations

from pathlib import Path

import torch


def load_text_corpus(
    raw_data_dir: Path,
    file_names: tuple[str, ...] | None = None,
    encoding: str = "utf-8",
) -> str:
    """
    Load and concatenate text files into a single corpus string.
    """
    if not raw_data_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist: {raw_data_dir}. "
            "Expected something like data/raw/shakespeare/."
        )
    if not raw_data_dir.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {raw_data_dir}")

    if file_names is None:
        paths = sorted(raw_data_dir.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(
                f"No .txt files found under: {raw_data_dir}. "
                "Place dataset.txt (and optionally others) in this directory."
            )
    else:
        paths = [raw_data_dir / name for name in file_names]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing expected dataset files in {raw_data_dir}: {missing}")

    parts: list[str] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Expected a file but found: {path}")
        parts.append(path.read_text(encoding=encoding))

    return "\n\n".join(parts)


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    """
    Build a deterministic character-level vocabulary.
    """
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = list(chars)
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """
    Encode a corpus string into token ids (dtype long).
    """
    ids = [stoi[ch] for ch in text]
    return torch.tensor(ids, dtype=torch.long)


def decode(ids: torch.Tensor, itos: list[str]) -> str:
    """
    Decode token ids back into a string.
    """
    return "".join(itos[int(i)] for i in ids)


def split_data(
    ids: torch.Tensor,
    train_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Deterministically split a token-id tensor into train and validation tensors.
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1). Got: {train_frac}")

    n = int(len(ids) * train_frac)
    train_ids = ids[:n]
    val_ids = ids[n:]
    return train_ids, val_ids


def get_batch(
    data_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch (x, y) for next-token prediction.

    x: [batch_size, context_length]
    y: [batch_size, context_length] where y is x shifted by 1
    """
    if data_ids.dtype != torch.long:
        raise TypeError(f"data_ids must be torch.long. Got: {data_ids.dtype}")

    if context_length <= 0:
        raise ValueError(f"context_length must be > 0. Got: {context_length}")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0. Got: {batch_size}")

    max_start = len(data_ids) - context_length - 1
    if max_start < 0:
        raise ValueError(
            "Not enough tokens to sample a batch. "
            f"Need at least context_length+1 tokens. context_length={context_length}, "
            f"len(data_ids)={len(data_ids)}"
        )

    starts = torch.randint(
        low=0,
        high=max_start + 1,
        size=(batch_size,),
        generator=generator,
    )

    x = torch.stack([data_ids[int(s) : int(s) + context_length] for s in starts])
    y = torch.stack([data_ids[int(s) + 1 : int(s) + context_length + 1] for s in starts])
    return x, y

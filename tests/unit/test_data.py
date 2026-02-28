from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from gptfs.training.data import (
    build_vocab,
    decode,
    encode,
    get_batch,
    load_text_corpus,
    split_data,
)


def test_build_vocab_is_deterministic() -> None:
    """
    Verifies vocab ordering is deterministic (sorted unique characters).
    """
    text = "bca"
    stoi, itos = build_vocab(text)

    assert itos == ["a", "b", "c"]
    assert stoi["a"] == 0
    assert stoi["b"] == 1
    assert stoi["c"] == 2


def test_encode_decode_roundtrip() -> None:
    """
    Verifies encoding then decoding reproduces the original text.
    """
    text = "hello\n"
    stoi, itos = build_vocab(text)

    ids = encode(text, stoi)
    out = decode(ids, itos)

    assert out == text
    assert ids.dtype == torch.long


def test_split_data_sizes_are_consistent() -> None:
    """
    Verifies train/val split sizes follow train_frac deterministically.
    """
    ids = torch.arange(100, dtype=torch.long)
    train_ids, val_ids = split_data(ids, train_frac=0.9)

    assert len(train_ids) == 90
    assert len(val_ids) == 10


def test_get_batch_shapes_and_shift_property() -> None:
    """
    Verifies get_batch returns correct shapes and y is a one step shift of x.
    """
    data_ids = torch.arange(50, dtype=torch.long)
    gen = torch.Generator().manual_seed(0)

    batch_size = 4
    context_length = 8
    x, y = get_batch(
        data_ids=data_ids,
        batch_size=batch_size,
        context_length=context_length,
        generator=gen,
    )

    assert x.shape == (batch_size, context_length)
    assert y.shape == (batch_size, context_length)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    assert torch.all(y[:, :-1] == x[:, 1:])
    assert torch.all(y[:, -1] == x[:, -1] + 1)


def test_load_text_corpus_loads_specified_files_only() -> None:
    """
    Verifies corpus loader respects explicit file_names selection.
    """
    with TemporaryDirectory() as tmp_dir:
        raw_dir = Path(tmp_dir)
        (raw_dir / "dataset.txt").write_text("SONNETS", encoding="utf-8")
        (raw_dir / "dataset2.txt").write_text("PLAYS", encoding="utf-8")

        corpus = load_text_corpus(
            raw_data_dir=raw_dir,
            file_names=("dataset.txt",),
        )

        assert corpus == "SONNETS"

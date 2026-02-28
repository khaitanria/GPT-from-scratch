from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gptfs.model import GPT
from gptfs.training.checkpointing import (
    latest_checkpoint_path,
    save_checkpoint,
    step_checkpoint_path,
    write_config_json,
    write_vocab_json,
)
from gptfs.training.data import build_vocab, encode, get_batch, load_text_corpus, split_data
from gptfs.training.training_config import TrainingConfig
from gptfs.training.training_result import TrainingResult


def train(config: TrainingConfig) -> TrainingResult:
    """
    Training entrypoint. Writes artifacts under artifacts/runs/<run_name>/
    and returns a TrainingResult.
    """
    device = _resolve_device(config.device)
    _set_global_seed(config.seed, device=device)

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.jsonl"

    text = load_text_corpus(raw_data_dir=config.raw_data_dir, file_names=config.corpus_files)
    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    train_ids, val_ids = split_data(ids, train_frac=config.train_split)

    config_path = _maybe_write_config(config)
    vocab_path = _maybe_write_vocab(run_dir=run_dir, stoi=stoi, itos=itos)

    model = GPT(
        vocab_size=len(itos),
        context_length=config.context_length,
        model_dim=config.model_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    latest_path = latest_checkpoint_path(run_dir)
    step, tokens_processed = 0, 0

    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device)
        progress = _restore_from_checkpoint(checkpoint=checkpoint, model=model, optimizer=optimizer)
        step = int(progress["step"])
        tokens_processed = int(progress["tokens_processed"])
        _maybe_restore_rng_state(checkpoint=checkpoint, device=device)
        print(f"[resume] run={config.run_name} step={step} tokens_processed={tokens_processed}")

    final_train_loss: float | None = None
    final_val_loss: float | None = None

    model.train()
    while step < config.max_steps:
        x, y = get_batch(
            data_ids=train_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            generator=None,
        )
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        tokens_processed += config.batch_size * config.context_length

        if step % config.eval_interval == 0 or step == config.max_steps:
            final_train_loss = _estimate_loss(
                model=model,
                data_ids=train_ids,
                batch_size=config.batch_size,
                context_length=config.context_length,
                eval_steps=config.eval_steps,
                device=device,
            )
            final_val_loss = _estimate_loss_if_possible(
                model=model,
                data_ids=val_ids,
                batch_size=config.batch_size,
                context_length=config.context_length,
                eval_steps=config.eval_steps,
                device=device,
            )

            epoch, epoch_fraction = _compute_epoch_progress(
                tokens_processed=tokens_processed,
                train_tokens=len(train_ids),
            )
            lr = float(optimizer.param_groups[0]["lr"])

            _append_metrics_jsonl(
                metrics_path=metrics_path,
                record={
                    "run_name": config.run_name,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "step": step,
                    "tokens_processed": tokens_processed,
                    "epoch": epoch,
                    "epoch_fraction": epoch_fraction,
                    "train_loss": float(final_train_loss),
                    "val_loss": None if final_val_loss is None else float(final_val_loss),
                    "train_ppl": _safe_exp(float(final_train_loss)),
                    "val_ppl": None if final_val_loss is None else _safe_exp(float(final_val_loss)),
                    "learning_rate": lr,
                },
            )

            if final_val_loss is None:
                print(f"[eval] step={step} train_loss={final_train_loss:.4f} val_loss=NA")
            else:
                print(
                    f"[eval] step={step} train_loss={final_train_loss:.4f} "
                    f"val_loss={final_val_loss:.4f}"
                )

        if step % config.save_interval == 0 or step == config.max_steps:
            epoch, epoch_fraction = _compute_epoch_progress(
                tokens_processed=tokens_processed,
                train_tokens=len(train_ids),
            )
            save_checkpoint(
                checkpoint_path=latest_path,
                model=model,
                optimizer=optimizer,
                step=step,
                tokens_processed=tokens_processed,
                epoch=epoch,
                epoch_fraction=epoch_fraction,
                save_rng_state=True,
            )
            if config.save_step_checkpoints:
                step_path = step_checkpoint_path(run_dir=run_dir, step=step)
                save_checkpoint(
                    checkpoint_path=step_path,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    tokens_processed=tokens_processed,
                    epoch=epoch,
                    epoch_fraction=epoch_fraction,
                    save_rng_state=True,
                )

    if final_train_loss is None:
        final_train_loss = _estimate_loss(
            model=model,
            data_ids=train_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            eval_steps=max(1, config.eval_steps),
            device=device,
        )
        final_val_loss = _estimate_loss_if_possible(
            model=model,
            data_ids=val_ids,
            batch_size=config.batch_size,
            context_length=config.context_length,
            eval_steps=max(1, config.eval_steps),
            device=device,
        )

    return TrainingResult(
        run_dir=run_dir,
        config_path=config_path,
        vocab_path=vocab_path,
        latest_checkpoint_path=latest_path,
        steps_completed=step,
        tokens_processed=tokens_processed,
        final_train_loss=float(final_train_loss),
        final_val_loss=None if final_val_loss is None else float(final_val_loss),
    )


def _maybe_write_config(config: TrainingConfig) -> Path:
    config_path = config.run_dir / "config.json"
    if config_path.exists():
        return config_path
    return write_config_json(config)


def _maybe_write_vocab(run_dir: Path, stoi: dict[str, int], itos: list[str]) -> Path:
    vocab_path = run_dir / "vocab.json"
    if vocab_path.exists():
        return vocab_path
    return write_vocab_json(run_dir=run_dir, stoi=stoi, itos=itos)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _set_global_seed(seed: int, *, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _restore_from_checkpoint(
    checkpoint: dict[str, object],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    model.load_state_dict(checkpoint["model_state_dict"])  # type: ignore[index]
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # type: ignore[index]
    progress = checkpoint["progress"]  # type: ignore[index]
    return progress  # type: ignore[return-value]


def _maybe_restore_rng_state(checkpoint: dict[str, object], *, device: torch.device) -> None:
    rng_state = checkpoint.get("rng_state")
    if rng_state is None:
        return

    cpu_state = rng_state.get("cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)

    if device.type == "cuda":
        cuda_state = rng_state.get("cuda")
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _estimate_loss(
    *,
    model: torch.nn.Module,
    data_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    eval_steps: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()

    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y = get_batch(
                data_ids=data_ids,
                batch_size=batch_size,
                context_length=context_length,
                generator=None,
            )
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(float(loss.item()))

    if was_training:
        model.train()

    return sum(losses) / len(losses)


def _estimate_loss_if_possible(
    *,
    model: torch.nn.Module,
    data_ids: torch.Tensor,
    batch_size: int,
    context_length: int,
    eval_steps: int,
    device: torch.device,
) -> float | None:
    if len(data_ids) < context_length + 2:
        return None
    return _estimate_loss(
        model=model,
        data_ids=data_ids,
        batch_size=batch_size,
        context_length=context_length,
        eval_steps=eval_steps,
        device=device,
    )


def _compute_epoch_progress(*, tokens_processed: int, train_tokens: int) -> tuple[int, float]:
    denom = max(1, int(train_tokens))
    epoch_float = tokens_processed / denom
    epoch = int(epoch_float)
    epoch_fraction = float(epoch_float - epoch)
    return epoch, epoch_fraction


def _append_metrics_jsonl(*, metrics_path: Path, record: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True)
    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _safe_exp(x: float) -> float:
    try:
        return float(math.exp(x))
    except OverflowError:
        return float("inf")

# GPT-from-scratch (PyTorch)

Production-style GPT implementation built from scratch in PyTorch.

I wanted to understand GPTs beyond “I can call a library”.
That meant implementing the core building blocks myself, structuring the code like a real project, and writing tests that validate behavior and invariants.

> Status: model, inference, and training are implemented. Training produces reproducible run bundles (checkpoints + config + vocab + metrics) that can be loaded later for inference.

---

## Model architecture

This project does not use Hugging Face model classes or prebuilt Transformer modules for the core architecture. The following components are implemented directly:

- Single-head causal self-attention
- Multi-head self-attention
- Feed-forward network
- Transformer block (with residual connections and layer norm)
- Decoder-only GPT
- Autoregressive generation for inference

---

## Project structure and tooling

This repository is structured as a production-style project so that the implementation is easy to navigate, maintain, test, and extend. This repo intentionally uses:

- **src layout packaging** (`src/gptfs/`) to prevent accidental imports from the repo root and to mirror real package installs
- **modular one class per file model components** (`gptfs.model.*`) to keep responsibilities small and testing focused
- **clear separation of concerns**:
  - `gptfs.model/` for architecture
  - `gptfs.training/` for data, checkpointing, and training loop
  - `gptfs.inference/` for generation and bundle reading
- **Poetry dependency management** with `poetry.lock` committed for reproducible environments
- **Ruff formatting + linting** to enforce consistent style and catch common issues early
- **unit + integration test suite** (`tests/unit`, `tests/integration`) to validate both individual modules and end-to-end behavior
- **reproducible run bundles** under `artifacts/runs/<run_name>/` containing:
  - `config.json` (architecture + training config)
  - `vocab.json` (stoi/itos)
  - `metrics.jsonl` (loss + perplexity over time)
  - `checkpoints/latest.pt` and optional step checkpoints
- **artifact and data hygiene**: training outputs and raw datasets are gitignored to preserve repo cleanliness
- **CI-ready development workflow**: tests and linting are runnable via `poetry run ...`, designed to work consistently across machines

---

## Notes and constraints

* `artifacts/` and `data/` are not committed to git.
* This is a character level language model.

---

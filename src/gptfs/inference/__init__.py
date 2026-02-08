"""
Inference utilities for the GPT-from-scratch project.

This package exposes a stable public inference API (token generation).
"""

from __future__ import annotations

from .generate import generate

__all__ = ["generate"]
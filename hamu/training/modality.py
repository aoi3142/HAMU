"""Small modality helpers shared by runtime-selected Trainer modules."""

from __future__ import annotations

import os


def current_modality() -> str:
    return os.getenv("HAMU_MODALITY", "vision").lower()


def is_llm_modality() -> bool:
    return current_modality() == "llm"

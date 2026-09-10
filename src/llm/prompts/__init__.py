"""Prompt template loader for Cloud and Device LLM clients."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    path = _PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def format_prompt(name: str, **kwargs: str) -> str:
    template = load_prompt(name)
    return template.format(**kwargs)

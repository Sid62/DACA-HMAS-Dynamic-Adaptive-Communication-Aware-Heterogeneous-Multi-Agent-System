"""Gated execution trace logging — enable with DACA_DEBUG=1."""

from __future__ import annotations

import json
import os
from typing import Any

DEBUG: bool = os.environ.get("DACA_DEBUG", "").lower() in ("1", "true", "yes")


def dlog(tag: str, message: str, **data: Any) -> None:
    if not DEBUG:
        return
    if data:
        payload = " ".join(f"{k}={_fmt(v)}" for k, v in data.items())
        print(f"[DACA_DEBUG:{tag}] {message} | {payload}")
    else:
        print(f"[DACA_DEBUG:{tag}] {message}")


def _fmt(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return repr(value)
    return repr(value)

"""Minimal flat-YAML config loader (no PyYAML dependency)."""

from __future__ import annotations

from pathlib import Path


def load_config(path: str | Path) -> dict:
    cfg = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        try:
            cfg[k] = int(v)
        except ValueError:
            try:
                cfg[k] = float(v)
            except ValueError:
                cfg[k] = v
    return cfg

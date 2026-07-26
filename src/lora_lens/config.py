"""YAML config loading with dotted CLI overrides and attribute access."""

from __future__ import annotations

import yaml


class Cfg(dict):
    """Dict with recursive attribute access: cfg.model.name == cfg["model"]["name"]."""

    def __getattr__(self, key):
        try:
            val = self[key]
        except KeyError as e:
            raise AttributeError(f"Missing config key: {key!r}") from e
        if isinstance(val, dict) and not isinstance(val, Cfg):
            val = Cfg(val)
            self[key] = val
        return val

    def __setattr__(self, key, value):
        self[key] = value


def _set_dotted(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def load_config(path: str, overrides: tuple[str, ...] = ()) -> Cfg:
    """Load a YAML config; overrides are 'dotted.key=value' strings (YAML-parsed values)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        key, sep, raw = ov.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return Cfg(cfg)


def _plain(obj):
    """Recursively convert Cfg back to plain dicts (safe_dump rejects dict subclasses)."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def save_config(cfg: Cfg, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_plain(cfg), f, sort_keys=False)

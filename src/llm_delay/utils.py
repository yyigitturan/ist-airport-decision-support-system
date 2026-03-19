from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ─────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ─────────────────────────────────────────────
# Dosya yardımcıları
# ─────────────────────────────────────────────

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


class _JsonEncoder(json.JSONEncoder):

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super().default(obj)


def dump_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, cls=_JsonEncoder)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# StandardScaler1D
# ─────────────────────────────────────────────

class StandardScaler1D:

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.mean_: float = 0.0
        self.std_:  float = 1.0

    def fit(self, values: np.ndarray) -> StandardScaler1D:
        if self.enabled:
            arr = np.asarray(values, dtype=np.float64)
            self.mean_ = float(arr.mean())
            self.std_  = float(arr.std())
            if self.std_ < 1e-8:
                self.std_ = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if not self.enabled:
            return arr.astype(np.float32)
        return ((arr - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        if not self.enabled:
            return arr.astype(np.float32)
        return (arr * self.std_ + self.mean_).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "mean":    self.mean_,
            "std":     self.std_,
        }

    def load_state_dict(self, d: dict) -> StandardScaler1D:
        self.enabled = bool(d["enabled"])
        self.mean_   = float(d["mean"])
        self.std_    = float(d["std"])
        return self

    def save(self, path: str | Path) -> None:
        dump_json(self.state_dict(), path)

    @classmethod
    def from_file(cls, path: str | Path) -> StandardScaler1D:
        d = load_json(path)
        return cls(enabled=d["enabled"]).load_state_dict(d)
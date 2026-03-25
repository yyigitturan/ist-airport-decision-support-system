from __future__ import annotations

import ast
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import TrainConfig
from .prompting import (
    TRAJ_PROMPT_1,
    TRAJ_PROMPT_2,
    TRAJ_PROMPT_3,
    TRAJ_PROMPT_4,
    build_prompt,
)


# ─────────────────────────────────────────────
# Parse yardımcıları
# ─────────────────────────────────────────────

def _to_list(x, field: str = "?", idx: int = -1) -> list:
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        try:
            result = ast.literal_eval(x)
            if not isinstance(result, list):
                warnings.warn(
                    f"[dataset] idx={idx} field={field}: "
                    f"literal_eval list değil ({type(result).__name__}), boş liste dönülüyor."
                )
                return []
            return result
        except Exception as e:
            warnings.warn(
                f"[dataset] idx={idx} field={field}: "
                f"literal_eval başarısız ({e}), boş liste dönülüyor."
            )
            return []
    if x is None:
        return []
    warnings.warn(
        f"[dataset] idx={idx} field={field}: "
        f"beklenmedik tip ({type(x).__name__}), boş liste dönülüyor."
    )
    return []


def _to_array(x, field: str = "focusing_emb", idx: int = -1) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        try:
            return np.asarray(ast.literal_eval(x), dtype=np.float32)
        except Exception as e:
            warnings.warn(
                f"[dataset] idx={idx} field={field}: "
                f"literal_eval başarısız ({e}), zeros(0) dönülüyor."
            )
            return np.zeros(0, dtype=np.float32)
    if x is None:
        warnings.warn(f"[dataset] idx={idx} field={field}: None geldi, zeros(0) dönülüyor.")
        return np.zeros(0, dtype=np.float32)
    warnings.warn(
        f"[dataset] idx={idx} field={field}: "
        f"beklenmedik tip ({type(x).__name__}), zeros(0) dönülüyor."
    )
    return np.zeros(0, dtype=np.float32)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class ScenarioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        cfg: TrainConfig,
        target_scaler=None,
    ):
        self.cfg = cfg

        df = df.reset_index(drop=True)
        n  = len(df)

        # ── Static trajectory prompt segmentleri — bir kez tokenize ──
        def _tok(text: str) -> torch.Tensor:
            return tokenizer(
                text,
                truncation=True,
                max_length=96,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]

        self.st1 = _tok(TRAJ_PROMPT_1)
        self.st2 = _tok(TRAJ_PROMPT_2)
        self.st3 = _tok(TRAJ_PROMPT_3)
        self.st4 = _tok(TRAJ_PROMPT_4)

        # ── Target — sütun bazlı, vektörize, tek seferde tensor ──
        targets_raw = df[cfg.target_col].astype(float).to_numpy()

        if target_scaler is not None:
            targets_scaled = target_scaler.transform(
                targets_raw.reshape(-1, 1).astype(np.float32)
            ).reshape(-1)
        else:
            targets_scaled = targets_raw.copy()

        self._targets_raw    = torch.from_numpy(targets_raw.astype(np.float32))
        self._targets_scaled = torch.from_numpy(targets_scaled.astype(np.float32))

        # ── Scenario ID — sütun bazlı ──
        if "scenario_id" in df.columns:
            self._scenario_ids: List[str] = df["scenario_id"].astype(str).tolist()
        else:
            self._scenario_ids = [str(i) for i in range(n)]

        # ── df.to_dict("records") — iloc döngüsü yok ──
        # iloc her çağrıda pandas index lookup yapıyor.
        # records ile bunu bir kez dict'e çeviriyoruz, sonra saf Python erişimi.
        records = df.to_dict("records")

        # ── Prompt tokenization — tüm dataset için bir kez ──
        print(f"  [dataset] {n} prompt tokenize ediliyor...", flush=True)
        self._prompt_ids: List[torch.Tensor] = []
        for i, row in enumerate(records):
            prompt = build_prompt(row, use_weather=cfg.use_weather)
            ids = tokenizer(
                prompt,
                truncation=True,
                max_length=cfg.total_prompt_max_len,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"][0]
            self._prompt_ids.append(ids)

        # ── Trajectory embeddings — init'te tensor olarak cache'le ──
        # from_numpy + ascontiguousarray: sıfır kopya, bellek düzeni garantili
        # __getitem__'da artık hiç dönüşüm yapılmıyor
        print(f"  [dataset] embedding'ler cache'leniyor...", flush=True)
        self._focusing:       List[torch.Tensor]       = []
        self._active_tensors: List[List[torch.Tensor]] = []
        self._prior_tensors:  List[List[torch.Tensor]] = []

        for i, row in enumerate(records):
            # focusing
            focusing = _to_array(row["focusing_emb"], field="focusing_emb", idx=i)
            if focusing.ndim != 1 or focusing.shape[0] != cfg.traj_dim:
                raise ValueError(
                    f"Bozuk focusing_emb — idx={i}, shape={focusing.shape}, "
                    f"beklenen traj_dim={cfg.traj_dim}"
                )
            self._focusing.append(
                torch.from_numpy(np.ascontiguousarray(focusing))
            )

            # active
            active_np = self._parse_embs(
                row["active_embs"], cfg.max_active, "active_embs", i
            )
            self._active_tensors.append(
                [torch.from_numpy(np.ascontiguousarray(a)) for a in active_np]
            )

            # prior
            prior_np = self._parse_embs(
                row["prior_embs"], cfg.max_prior, "prior_embs", i
            )
            self._prior_tensors.append(
                [torch.from_numpy(np.ascontiguousarray(p)) for p in prior_np]
            )

        # df artık saklanmıyor — bellek tasarrufu
        print(f"  [dataset] {n} sample hazır.", flush=True)

    def __len__(self) -> int:
        return len(self._prompt_ids)

    def _parse_embs(
        self,
        emb_list,
        max_len: int,
        field: str,
        idx: int,
    ) -> List[np.ndarray]:
        """Parse + validate + kırp. Bozuk öğede ValueError fırlatır."""
        raw     = _to_list(emb_list, field=field, idx=idx)
        clipped = raw[:max_len]

        validated = []
        for item_idx, e in enumerate(clipped):
            arr = np.asarray(e, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != self.cfg.traj_dim:
                raise ValueError(
                    f"Bozuk embedding — field={field}, sample_idx={idx}, "
                    f"item_idx={item_idx}, shape={arr.shape}, "
                    f"beklenen traj_dim={self.cfg.traj_dim}"
                )
            validated.append(arr)
        return validated

    def __getitem__(self, idx: int) -> Dict:
        # Saf index lookup — parse yok, tensor dönüşümü yok, pandas yok
        return {
            "prompt_ids":   self._prompt_ids[idx],
            "st1_ids":      self.st1,
            "st2_ids":      self.st2,
            "st3_ids":      self.st3,
            "st4_ids":      self.st4,
            "focusing_emb": self._focusing[idx],
            "active_embs":  self._active_tensors[idx],
            "prior_embs":   self._prior_tensors[idx],
            "target":       self._targets_scaled[idx],
            "target_raw":   self._targets_raw[idx],
            "scenario_id":  self._scenario_ids[idx],
        }


# ─────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────

def make_collate_fn(pad_id: int, st1: torch.Tensor, st2: torch.Tensor,
                    st3: torch.Tensor, st4: torch.Tensor):
    """
    st1..st4 dataset'te her sample için aynı — collate kapanışına alındı.
    Böylece batch içinde sample başına taşınmıyor, sadece bir kez expand ediliyor.
    """
    if pad_id is None:
        raise ValueError("pad_id None olamaz. tokenizer.pad_token_id ayarlanmalı.")

    def collate_fn(batch: List[Dict]) -> Dict:

        def pad_1d(t: torch.Tensor, max_len: int) -> torch.Tensor:
            out = torch.full((max_len,), pad_id, dtype=torch.long)
            out[: t.shape[0]] = t
            return out

        B = len(batch)

        # Prompt padding
        max_prompt = max(x["prompt_ids"].shape[0] for x in batch)
        prompt_ids = torch.stack([pad_1d(x["prompt_ids"], max_prompt) for x in batch])
        prompt_attention_mask = (prompt_ids != pad_id).long()

        # Static segmentler — kapanıştan al, batch boyutuna expand et (kopya yok)
        st1_ids = st1.unsqueeze(0).expand(B, -1)
        st2_ids = st2.unsqueeze(0).expand(B, -1)
        st3_ids = st3.unsqueeze(0).expand(B, -1)
        st4_ids = st4.unsqueeze(0).expand(B, -1)

        # Trajectory embeddings
        focusing = torch.stack([x["focusing_emb"] for x in batch])
        traj_dim = focusing.shape[-1]

        max_active_n = max(len(x["active_embs"]) for x in batch)
        max_prior_n  = max(len(x["prior_embs"])  for x in batch)

        active      = torch.zeros(B, max(max_active_n, 1), traj_dim, dtype=torch.float32)
        prior       = torch.zeros(B, max(max_prior_n,  1), traj_dim, dtype=torch.float32)
        active_mask = torch.zeros(B, max(max_active_n, 1), dtype=torch.bool)
        prior_mask  = torch.zeros(B, max(max_prior_n,  1), dtype=torch.bool)

        for i, x in enumerate(batch):
            if x["active_embs"]:
                aa = torch.stack(x["active_embs"])
                active[i, : aa.shape[0]]      = aa
                active_mask[i, : aa.shape[0]] = True
            if x["prior_embs"]:
                pp = torch.stack(x["prior_embs"])
                prior[i, : pp.shape[0]]      = pp
                prior_mask[i, : pp.shape[0]] = True

        return {
            "prompt_ids":            prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "st1_ids":               st1_ids,
            "st2_ids":               st2_ids,
            "st3_ids":               st3_ids,
            "st4_ids":               st4_ids,
            "focusing_emb":          focusing,
            "active_embs":           active,
            "prior_embs":            prior,
            "active_mask":           active_mask,
            "prior_mask":            prior_mask,
            "target":                torch.stack([x["target"]     for x in batch]),
            "target_raw":            torch.stack([x["target_raw"] for x in batch]),
            "scenario_ids":          [x["scenario_id"] for x in batch],
        }

    return collate_fn
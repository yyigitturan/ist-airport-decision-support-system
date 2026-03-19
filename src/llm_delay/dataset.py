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
    # TODO: build_full_prompt(row, use_weather, use_notam) eklenince
    # cfg.use_notam buradan geçirilecek. Şu an use_notam config'de var
    # ama prompting tarafında desteklenmiyor — sessiz ignore ediliyor.
)


# ─────────────────────────────────────────────
# Parse yardımcıları
# ─────────────────────────────────────────────

def _to_list(x, field: str = "?", idx: int = -1) -> list:
    """
    active/prior emb listelerini parse eder.

    Parse başarısız olursa boş liste döner ve uyarı basar.
    Sessiz veri kaybını görünür kılmak için warnings.warn kullanılır —
    eğitimde beklenmedik boş embedding kullanımı fark edilsin.
    """
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
    """
    focusing_emb'i np.ndarray'e çevirir.

    Parse başarısız olursa shape=(0,) array döner —
    çağıran tarafta shape validation bunu yakalayacak.
    """
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
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.target_scaler = target_scaler

        def _tok(text: str) -> torch.Tensor:
            return tokenizer(
                text,
                truncation=True,
                max_length=96,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"][0]

        # Static trajectory prompt segmentleri — init'te bir kez tokenize et
        self.st1 = _tok(TRAJ_PROMPT_1)
        self.st2 = _tok(TRAJ_PROMPT_2)
        self.st3 = _tok(TRAJ_PROMPT_3)
        self.st4 = _tok(TRAJ_PROMPT_4)

    def __len__(self) -> int:
        return len(self.df)

    def _clip_embs(
        self,
        emb_list,
        max_len: int,
        field: str,
        idx: int,
    ) -> List[np.ndarray]:
        """
        Embedding listesini parse et, kırp ve her öğeyi doğrula.
        Bozuk öğe bulunursa ValueError fırlatır — hangi field/idx/item olduğu açıkça belirtilir.
        """
        raw = _to_list(emb_list, field=field, idx=idx)
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
        row = self.df.iloc[idx]

        # ── Text prompt ──────────────────────────────
        # use_notam cfg'de var ama build_prompt henüz desteklemiyor.
        # build_full_prompt(row, use_weather, use_notam) eklenince burası güncellenecek.
        prompt = build_prompt(row, use_weather=self.cfg.use_weather)

        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.cfg.total_prompt_max_len,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"][0]

        # ── Trajectory embeddings ────────────────────
        focusing = _to_array(row["focusing_emb"], field="focusing_emb", idx=idx)
        if focusing.ndim != 1 or focusing.shape[0] != self.cfg.traj_dim:
            raise ValueError(
                f"Bozuk focusing_emb — idx={idx}, shape={focusing.shape}, "
                f"beklenen traj_dim={self.cfg.traj_dim}"
            )

        active = self._clip_embs(
            row["active_embs"], self.cfg.max_active, field="active_embs", idx=idx
        )
        prior = self._clip_embs(
            row["prior_embs"], self.cfg.max_prior, field="prior_embs", idx=idx
        )

        # ── Target ───────────────────────────────────
        y_raw = float(row[self.cfg.target_col])
        if self.target_scaler is None:
            y_scaled = y_raw
        else:
            y_scaled = float(self.target_scaler.transform(
                np.array([y_raw], dtype=np.float32)
            )[0])

        return {
            "prompt_ids":  prompt_ids,
            "st1_ids":     self.st1,
            "st2_ids":     self.st2,
            "st3_ids":     self.st3,
            "st4_ids":     self.st4,
            "focusing_emb": torch.tensor(focusing, dtype=torch.float32),
            "active_embs":  [torch.tensor(x, dtype=torch.float32) for x in active],
            "prior_embs":   [torch.tensor(x, dtype=torch.float32) for x in prior],
            "target":       torch.tensor(y_scaled, dtype=torch.float32),
            "target_raw":   torch.tensor(y_raw,    dtype=torch.float32),
            "scenario_id":  str(row.get("scenario_id", idx)),
        }


# ─────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────

def make_collate_fn(pad_id: int):
    
    if pad_id is None:
        raise ValueError("pad_id None olamaz. tokenizer.pad_token_id ayarlanmalı.")
    
    def collate_fn(batch: List[Dict]) -> Dict:

        def pad_1d(t: torch.Tensor, max_len: int) -> torch.Tensor:
            out = torch.full((max_len,), pad_id, dtype=torch.long)
            out[: t.shape[0]] = t
            return out

        # ── Text prompt padding ──────────────────────
        max_prompt = max(x["prompt_ids"].shape[0] for x in batch)
        prompt_ids = torch.stack([pad_1d(x["prompt_ids"], max_prompt) for x in batch])
        prompt_attention_mask = (prompt_ids != pad_id).long()

        # ── Static segments — sabit uzunluk, direkt stack ────
        st1_ids = torch.stack([x["st1_ids"] for x in batch])
        st2_ids = torch.stack([x["st2_ids"] for x in batch])
        st3_ids = torch.stack([x["st3_ids"] for x in batch])
        st4_ids = torch.stack([x["st4_ids"] for x in batch])

        # ── Trajectory embeddings ────────────────────
        focusing  = torch.stack([x["focusing_emb"] for x in batch])
        traj_dim  = focusing.shape[-1]
        B         = len(batch)

        # max_active_n / max_prior_n: boş liste durumunda 0 olabilir — zeros ile güvenli
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
            "prompt_ids":           prompt_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "st1_ids":              st1_ids,
            "st2_ids":              st2_ids,
            "st3_ids":              st3_ids,
            "st4_ids":              st4_ids,
            "focusing_emb":         focusing,
            "active_embs":          active,
            "prior_embs":           prior,
            "active_mask":          active_mask,
            "prior_mask":           prior_mask,
            "target":               torch.stack([x["target"]     for x in batch]),
            "target_raw":           torch.stack([x["target_raw"] for x in batch]),
            "scenario_ids":         [x["scenario_id"] for x in batch],
        }

    return collate_fn
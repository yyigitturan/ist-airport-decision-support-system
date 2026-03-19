from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

CKPT_PATH = "/Users/YGT/ist-airport-decision-support-system/src/modeling/atscc_checkpoints_v10_reg/atscc_best.pt"
TRAJ_PATH = "/Users/YGT/ist-airport-decision-support-system/data/gold/trajectories/trajectory_gold.parquet"
OUT_PATH  = "/Users/YGT/ist-airport-decision-support-system/data/gold/atscc_embeddings_final_all_fixed.parquet"

BATCH_SIZE = 256        # memory'e göre 64 / 128 / 256
WRITE_EVERY = 10_000    # kaç flight'ta bir parquet flush
DEVICE = "mps"          # "cpu" / "cuda" / "mps"


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ATSCCConfig:
    traj_path: str = TRAJ_PATH
    out_dir: str = ""

    n_flights: Optional[int] = None
    sample_seed: int = 42
    feature_cols: Tuple[str, ...] = (
        "e_m", "n_m", "u_m",
        "ux", "uy", "uz",
        "r", "sin_theta", "cos_theta",
        "delta_t", "gap_flag",
    )
    min_seq_len: int = 16
    max_seq_len: int = 256
    valid_ratio: float = 0.10
    flight_types: object = "all"

    d_model: int = 192
    n_head: int = 8
    n_layer: int = 6
    d_ff: int = 1024
    d_out: int = 256
    dropout: float = 0.40
    random_mask_prob: float = 0.30
    drop_path_rate: float = 0.0

    batch_size: int = 32
    accum_steps: int = 2
    num_workers: int = 0
    lr: float = 3e-6
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    epochs: int = 60
    patience: int = 12
    min_delta: float = 1e-4
    temperature: float = 0.10
    max_timesteps_per_batch: int = 8000

    warmup_epochs: int = 5
    eta_min: float = 1e-6

    device: str = DEVICE
    save_best_only: bool = True
    log_every: int = 1
    metric_token_cap: int = 20000


# ============================================================
# DEVICE
# ============================================================

def resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ============================================================
# MODEL
# ============================================================

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob).float()
        return x * keep / keep_prob


class TransformerLayerWithDropPath(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float,
        drop_path_rate: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class ATSCCEncoder(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        d_ff: int,
        d_out: int,
        dropout: float,
        random_mask_prob: float,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.random_mask_prob = random_mask_prob

        self.in_proj = nn.Linear(d_input, d_model, bias=True)
        self.in_ln = nn.LayerNorm(d_model)

        dpr = [float(r) for r in np.linspace(0, drop_path_rate, n_layer)]
        self.layers = nn.ModuleList([
            TransformerLayerWithDropPath(
                d_model=d_model,
                n_head=n_head,
                d_ff=d_ff,
                dropout=dropout,
                drop_path_rate=dpr[i],
            )
            for i in range(n_layer)
        ])

        self.final_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out, bias=True)
        self.out_ln = nn.LayerNorm(d_out)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # inference/export sırasında training=False olduğu için random masking yok
        if self.training and self.random_mask_prob > 0:
            rand_mask = (
                torch.rand(key_padding_mask.shape, device=x.device) < self.random_mask_prob
            ) & (~key_padding_mask)
            x = x.masked_fill(rand_mask.unsqueeze(-1), 0.0)
            merged_mask = key_padding_mask | rand_mask
        else:
            merged_mask = key_padding_mask

        h = F.normalize(self.in_ln(self.in_proj(x)), p=2, dim=-1)

        T = h.shape[1]
        causal_mask = torch.triu(
            torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1
        )

        for layer in self.layers:
            h = layer(h, attn_mask=causal_mask, key_padding_mask=merged_mask)

        h = self.final_ln(h)
        z = F.normalize(self.out_ln(self.out_proj(h)), p=2, dim=-1)
        return z


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    checkpoint_path: str,
    device: Optional[str] = None,
) -> Tuple[ATSCCEncoder, ATSCCConfig]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    valid_fields = ATSCCConfig.__dataclass_fields__.keys()
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_fields}
    cfg = ATSCCConfig(**cfg_dict)
    cfg.device = resolve_device(device)

    model = ATSCCEncoder(
        d_input=len(cfg.feature_cols),
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        d_ff=cfg.d_ff,
        d_out=cfg.d_out,
        dropout=cfg.dropout,
        random_mask_prob=cfg.random_mask_prob,
        drop_path_rate=getattr(cfg, "drop_path_rate", 0.0),
    ).to(cfg.device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"Model yüklendi — epoch={ckpt['epoch']}, "
        f"val_loss={ckpt['val_loss']:.6f}, device={cfg.device}"
    )
    return model, cfg


# ============================================================
# UTILS
# ============================================================

def classify_flight_type(g: pd.DataFrame) -> str:
    """
    Training'deki mantığı koruyoruz ki clipping kuralı tutarlı olsun.
    """
    if "distance_km" not in g.columns:
        return "unknown"

    dist = pd.to_numeric(g["distance_km"], errors="coerce").to_numpy(dtype=np.float32)
    finite = dist[np.isfinite(dist)]
    if len(finite) < 4:
        return "unknown"

    n = max(1, len(finite) // 5)
    start_dist = float(np.nanmean(finite[:n]))
    end_dist   = float(np.nanmean(finite[-n:]))

    if not (np.isfinite(start_dist) and np.isfinite(end_dist)):
        return "unknown"

    delta = end_dist - start_dist
    if delta < -5.0:
        return "arrival"
    if delta > 5.0:
        return "departure"
    return "unknown"


def extract_hex_from_flight_id(flight_id: str) -> str:
    """
    flight_id örnek: 4bb061_0  -> hex_icao = 4BB061
    """
    if not isinstance(flight_id, str):
        return ""
    return flight_id.split("_")[0].upper().strip()


def load_raw_trajectories(
    traj_path: str,
    feature_cols: Tuple[str, ...],
) -> Dict[str, pd.DataFrame]:
    """
    FULL RAW trajectory yükler.
    Burada clipping yok.
    """
    t0 = time.time()
    print("=" * 60)
    print("RAW TRAJECTORY YÜKLENİYOR")
    print("=" * 60)

    needed = {"flight_id", "point_timestamp"} | set(feature_cols) | {"distance_km"}
    traj = pd.read_parquet(traj_path)

    missing = [c for c in needed if c not in traj.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    traj["flight_id"] = traj["flight_id"].astype(str)
    traj["point_timestamp"] = pd.to_datetime(traj["point_timestamp"], utc=True, errors="coerce")

    for c in feature_cols:
        traj[c] = pd.to_numeric(traj[c], errors="coerce").astype("float32")

    if "distance_km" in traj.columns:
        traj["distance_km"] = pd.to_numeric(traj["distance_km"], errors="coerce").astype("float32")

    drop_cols = ["flight_id", "point_timestamp"] + list(feature_cols)
    traj = traj.dropna(subset=drop_cols).copy()
    traj = traj.sort_values(["flight_id", "point_timestamp"], kind="mergesort").reset_index(drop=True)

    groups: Dict[str, pd.DataFrame] = {
        fid: g.reset_index(drop=True)
        for fid, g in traj.groupby("flight_id", sort=False)
    }

    print(f"  Kullanılabilir raw flight: {len(groups):,}")
    print(f"  Yükleme süresi           : {time.time() - t0:.1f}s")
    print()
    return groups


def build_export_rows_for_batch(
    model: ATSCCEncoder,
    cfg: ATSCCConfig,
    batch_items: List[Tuple[str, pd.DataFrame]],
) -> List[Dict[str, Any]]:
    """
    Bir batch flight için:
    - raw metadata
    - clipped inference sequence
    - final timestep embedding
    üretir.
    """
    if not batch_items:
        return []

    seqs = []
    meta = []

    for fid, g_raw in batch_items:
        if g_raw.empty:
            continue

        raw_start = g_raw["point_timestamp"].iloc[0]
        raw_end   = g_raw["point_timestamp"].iloc[-1]
        raw_len   = len(g_raw)

        flight_type = classify_flight_type(g_raw)

        # training mantığıyla uyumlu clipping
        if raw_len > cfg.max_seq_len:
            if flight_type == "departure":
                g_use = g_raw.iloc[:cfg.max_seq_len].reset_index(drop=True)
            else:
                g_use = g_raw.iloc[-cfg.max_seq_len:].reset_index(drop=True)
        else:
            g_use = g_raw

        used_len = len(g_use)
        if used_len < 1:
            continue

        x = g_use.loc[:, cfg.feature_cols].to_numpy(dtype=np.float32)
        seqs.append(x)
        meta.append({
            "flight_id": fid,
            "hex_icao": extract_hex_from_flight_id(fid),
            "traj_start": raw_start,
            "traj_end": raw_end,
            "traj_len_raw": raw_len,
            "traj_len_used": used_len,
            "flight_type": flight_type,
        })

    if not seqs:
        return []

    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    feat_dim = seqs[0].shape[1]
    B = len(seqs)

    x_pad = torch.zeros((B, max_len, feat_dim), dtype=torch.float32)
    kpm   = torch.ones((B, max_len), dtype=torch.bool)

    for i, (seq, L) in enumerate(zip(seqs, lengths)):
        x_pad[i, :L, :] = torch.from_numpy(seq)
        kpm[i, :L] = False

    x_pad = x_pad.to(cfg.device)
    kpm = kpm.to(cfg.device)

    with torch.no_grad():
        z = model(x_pad, kpm)
        last_idx = torch.tensor([L - 1 for L in lengths], device=cfg.device)
        bidx = torch.arange(B, device=cfg.device)
        embs = z[bidx, last_idx].cpu().numpy().astype(np.float32)

    rows: List[Dict[str, Any]] = []
    for m, emb in zip(meta, embs):
        row = dict(m)
        for j, v in enumerate(emb):
            row[f"atscc_emb_{j:03d}"] = float(v)
        rows.append(row)

    return rows


def export_full_embeddings(
    model: ATSCCEncoder,
    cfg: ATSCCConfig,
    flight_groups: Dict[str, pd.DataFrame],
    out_path: str,
    batch_size: int = BATCH_SIZE,
    write_every: int = WRITE_EVERY,
) -> None:
    out_path = str(out_path)
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    flight_ids = sorted(flight_groups.keys())
    total = len(flight_ids)

    print("=" * 60)
    print("FULL EMBEDDING EXPORT BAŞLADI")
    print("=" * 60)
    print(f"  Total flights : {total:,}")
    print(f"  Emb dim       : {cfg.d_out}")
    print(f"  Device        : {cfg.device}")
    print(f"  Out           : {out_path}")
    print()

    writer: Optional[pq.ParquetWriter] = None
    buffer_rows: List[Dict[str, Any]] = []
    written = 0
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch_fids = flight_ids[start:start + batch_size]
        batch_items = [(fid, flight_groups[fid]) for fid in batch_fids]

        rows = build_export_rows_for_batch(model, cfg, batch_items)
        buffer_rows.extend(rows)

        if len(buffer_rows) >= write_every or (start + batch_size >= total):
            df = pd.DataFrame(buffer_rows)

            # dtype düzenleme
            if not df.empty:
                df["flight_id"] = df["flight_id"].astype("string")
                df["hex_icao"] = df["hex_icao"].astype("string")
                df["flight_type"] = df["flight_type"].astype("string")
                df["traj_start"] = pd.to_datetime(df["traj_start"], utc=True)
                df["traj_end"] = pd.to_datetime(df["traj_end"], utc=True)
                df["traj_len_raw"] = pd.to_numeric(df["traj_len_raw"], downcast="integer")
                df["traj_len_used"] = pd.to_numeric(df["traj_len_used"], downcast="integer")

            table = pa.Table.from_pandas(df, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(
                    out_path,
                    table.schema,
                    compression="zstd",
                    compression_level=3,
                )

            writer.write_table(table)
            written += len(df)
            buffer_rows.clear()

            elapsed = time.time() - t0
            print(f"  ✓ Written: {written:,}/{total:,}  ({elapsed:.1f}s)")

        if cfg.device == "mps":
            torch.mps.empty_cache()

    if writer is not None:
        writer.close()

    print()
    print("=" * 60)
    print("EXPORT TAMAMLANDI")
    print("=" * 60)
    print(f"  Saved rows : {written:,}")
    print(f"  File       : {out_path}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    model, cfg = load_model(CKPT_PATH, device=DEVICE)

    raw_groups = load_raw_trajectories(
        traj_path=TRAJ_PATH,
        feature_cols=cfg.feature_cols,
    )

    export_full_embeddings(
        model=model,
        cfg=cfg,
        flight_groups=raw_groups,
        out_path=OUT_PATH,
        batch_size=BATCH_SIZE,
        write_every=WRITE_EVERY,
    )


if __name__ == "__main__":
    main()
from __future__ import annotations

import gc
import json
import os
import pickle
import shutil
import hashlib
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ScenarioConfig:
    ckpt_path: str = "/Users/YGT/ist-airport-decision-support-system/src/modeling/atscc_checkpoints_v10_reg/atscc_best.pt"
    flights_path: str = "/Users/YGT/ist-airport-decision-support-system/data/gold/ist_flights_labeled_v6.parquet"
    full_emb_path: str = "/Users/YGT/ist-airport-decision-support-system/data/gold/atscc_embeddings_final_all_fixed.parquet"
    traj_path: str = "/Users/YGT/ist-airport-decision-support-system/data/gold/trajectories/trajectory_gold.parquet"

    output_root: str = "/Users/YGT/ist-airport-decision-support-system/data/gold/scenario_runs"
    run_name: str = "paper_faithful_labeled_v6"

    only_plausible: bool = True
    n_obs_max: int = 5
    entry_time_col: str = "actual_entry_time"
    landing_time_col: str = "actual_landing_time"
    label_col: str = "post_terminal_duration_min"

    active_batch_size: int = 32
    checkpoint_every_flights: int = 1000
    chunk_rows_soft_limit: int = 5000

    prefix_cache_maxsize: int = 50_000
    resume: bool = True
    overwrite_if_exists: bool = False

    # Smoke test için None yerine örn 200 yaz
    limit_flights: Optional[int] = None

    # Chunk compression
    parquet_compression: str = "zstd"


CFG = ScenarioConfig()


# ============================================================
# LOGGING
# ============================================================

def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("scenario_builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


# ============================================================
# PATHS / RUN LAYOUT
# ============================================================

def utc_run_id() -> str:
    return pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")


def prepare_run_dirs(cfg: ScenarioConfig) -> dict:
    base = Path(cfg.output_root) / cfg.run_name
    if cfg.resume:
        run_dir = base
    else:
        run_dir = base.parent / f"{cfg.run_name}_{utc_run_id()}"

    chunks_dir = run_dir / "chunks"
    meta_dir = run_dir / "meta"
    logs_dir = run_dir / "logs"

    if run_dir.exists() and not cfg.resume:
        if cfg.overwrite_if_exists:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(f"Run dir already exists: {run_dir}")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "chunks_dir": chunks_dir,
        "meta_dir": meta_dir,
        "logs_dir": logs_dir,
        "state_path": meta_dir / "state.pkl",
        "manifest_path": meta_dir / "manifest.json",
        "chunk_index_path": meta_dir / "chunk_index.parquet",
        "config_path": meta_dir / "config.json",
        "log_path": logs_dir / "run.log",
    }


# ============================================================
# FINGERPRINTS
# ============================================================

def file_fingerprint(path: str) -> dict:
    p = Path(path)
    st = p.stat()
    return {
        "path": str(p),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def config_fingerprint(cfg: ScenarioConfig) -> str:
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# ============================================================
# STATE
# ============================================================

def save_state(state_path: Path, state: dict) -> None:
    with open(state_path, "wb") as f:
        pickle.dump(state, f)


def load_state(state_path: Path) -> Optional[dict]:
    if not state_path.exists():
        return None
    with open(state_path, "rb") as f:
        return pickle.load(f)


# ============================================================
# GLOBAL CACHE
# ============================================================

_PREFIX_CACHE: Dict[Tuple[str, int], Optional[np.ndarray]] = {}


def cache_key(flight_id: str, t: pd.Timestamp) -> Tuple[str, int]:
    return (str(flight_id), int(pd.Timestamp(t).value))


def cache_set(cfg: ScenarioConfig, key: Tuple[str, int], val: Optional[np.ndarray]) -> None:
    if len(_PREFIX_CACHE) >= cfg.prefix_cache_maxsize:
        evict_n = max(1, cfg.prefix_cache_maxsize // 10)
        for k in list(_PREFIX_CACHE.keys())[:evict_n]:
            del _PREFIX_CACHE[k]
    _PREFIX_CACHE[key] = val


# ============================================================
# MODEL
# ============================================================

class DropPath(torch.nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) < keep_prob).float()
        return x * keep / keep_prob


class TransformerLayer(torch.nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
            torch.nn.Dropout(dropout),
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.attn(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.ff(self.norm2(x)))
        return x


class ATSCCEncoder(torch.nn.Module):
    def __init__(self, cfg_model):
        super().__init__()
        self.random_mask_prob = cfg_model.random_mask_prob
        n_layer = cfg_model.n_layer
        dpr = [float(r) for r in np.linspace(0, getattr(cfg_model, "drop_path_rate", 0.0), n_layer)]
        self.in_proj = torch.nn.Linear(len(cfg_model.feature_cols), cfg_model.d_model, bias=True)
        self.in_ln = torch.nn.LayerNorm(cfg_model.d_model)
        self.layers = torch.nn.ModuleList([
            TransformerLayer(cfg_model.d_model, cfg_model.n_head, cfg_model.d_ff, cfg_model.dropout, dpr[i])
            for i in range(n_layer)
        ])
        self.final_ln = torch.nn.LayerNorm(cfg_model.d_model)
        self.out_proj = torch.nn.Linear(cfg_model.d_model, cfg_model.d_out, bias=True)
        self.out_ln = torch.nn.LayerNorm(cfg_model.d_out)

    def forward(self, x, key_padding_mask):
        import torch.nn.functional as F

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = F.normalize(self.in_ln(self.in_proj(x)), p=2, dim=-1)
        T = h.shape[1]
        causal_mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            h = layer(h, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        h = self.final_ln(h)
        return F.normalize(self.out_ln(self.out_proj(h)), p=2, dim=-1)


def load_model(cfg: ScenarioConfig, logger: logging.Logger):
    logger.info("Model yükleniyor...")
    ckpt = torch.load(cfg.ckpt_path, map_location="cpu", weights_only=False)

    class ModelCfg:
        pass

    model_cfg = ModelCfg()
    for k, v in ckpt["config"].items():
        setattr(model_cfg, k, v)

    if torch.backends.mps.is_available():
        model_cfg.device = "mps"
    elif torch.cuda.is_available():
        model_cfg.device = "cuda"
    else:
        model_cfg.device = "cpu"

    model = ATSCCEncoder(model_cfg).to(model_cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(
        "Model hazır | device=%s | epoch=%s | val_loss=%.6f",
        model_cfg.device,
        ckpt.get("epoch"),
        ckpt.get("val_loss", np.nan),
    )
    return model, model_cfg


# ============================================================
# DATA HELPERS
# ============================================================

def load_flights(cfg: ScenarioConfig, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Labeled flights yükleniyor...")
    flights_df = pd.read_parquet(cfg.flights_path)

    flights_df[cfg.entry_time_col] = pd.to_datetime(flights_df[cfg.entry_time_col], utc=True, errors="coerce")
    flights_df[cfg.landing_time_col] = pd.to_datetime(flights_df[cfg.landing_time_col], utc=True, errors="coerce")

    mask = (
        flights_df[cfg.entry_time_col].notna()
        & flights_df[cfg.landing_time_col].notna()
        & flights_df[cfg.label_col].notna()
    )
    if cfg.only_plausible and "is_plausible_label" in flights_df.columns:
        mask &= flights_df["is_plausible_label"].fillna(False)
    if "is_matched" in flights_df.columns:
        mask &= flights_df["is_matched"].fillna(False)

    flights_df = flights_df.loc[mask].copy()

    bad_order = flights_df[cfg.landing_time_col] <= flights_df[cfg.entry_time_col]
    if bad_order.any():
        flights_df = flights_df.loc[~bad_order].copy()

    if cfg.limit_flights is not None:
        flights_df = flights_df.head(cfg.limit_flights).copy()

    logger.info("Modeling flights: %s", f"{len(flights_df):,}")
    return flights_df


def load_full_embeddings(cfg: ScenarioConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, List[str]]:
    logger.info("Full embedding tablosu yükleniyor...")
    full_emb_df = pd.read_parquet(cfg.full_emb_path)
    full_emb_df["flight_id"] = full_emb_df["flight_id"].astype(str)
    full_emb_df["hex_icao"] = full_emb_df["hex_icao"].astype(str).str.upper().str.strip()
    full_emb_df["traj_start"] = pd.to_datetime(full_emb_df["traj_start"], utc=True)
    full_emb_df["traj_end"] = pd.to_datetime(full_emb_df["traj_end"], utc=True)
    emb_cols = [c for c in full_emb_df.columns if c.startswith("atscc_emb_")]
    logger.info("Embedding rows=%s | dim=%s", f"{len(full_emb_df):,}", len(emb_cols))
    return full_emb_df, emb_cols


def load_trajectory_groups(cfg: ScenarioConfig, model_cfg, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    logger.info("Trajectory yükleniyor...")
    needed_cols = ["flight_id", "point_timestamp"] + list(model_cfg.feature_cols)
    traj = pd.read_parquet(cfg.traj_path, columns=needed_cols)
    traj["flight_id"] = traj["flight_id"].astype(str)
    traj["point_timestamp"] = pd.to_datetime(traj["point_timestamp"], utc=True)

    for c in model_cfg.feature_cols:
        traj[c] = pd.to_numeric(traj[c], errors="coerce").astype("float32")

    traj = traj.dropna(subset=["flight_id", "point_timestamp"] + list(model_cfg.feature_cols))
    traj = traj.sort_values(["flight_id", "point_timestamp"]).reset_index(drop=True)

    flight_groups = {
        fid: g.reset_index(drop=True)
        for fid, g in traj.groupby("flight_id", sort=False)
    }
    logger.info("Trajectory groups: %s", f"{len(flight_groups):,}")

    del traj
    gc.collect()
    return flight_groups


# ============================================================
# SCENARIO LOGIC
# ============================================================

def find_focusing_flight_id(
    full_emb_df: pd.DataFrame,
    hex_icao: str,
    entry_time: pd.Timestamp,
) -> Optional[str]:
    cand = full_emb_df[
        (full_emb_df["hex_icao"] == hex_icao)
        & (full_emb_df["traj_start"] <= entry_time)
        & (full_emb_df["traj_end"] >= entry_time)
    ]

    if cand.empty:
        return None

    return str(cand.sort_values("traj_start", ascending=False).iloc[0]["flight_id"])


def pick_observation_times(
    g: pd.DataFrame,
    entry_time: pd.Timestamp,
    landing_time: pd.Timestamp,
    n_max: int,
) -> List[pd.Timestamp]:
    steps = g[
        (g["point_timestamp"] >= entry_time)
        & (g["point_timestamp"] < landing_time)
    ]["point_timestamp"].sort_values()

    if len(steps) == 0:
        return [pd.Timestamp(entry_time)]

    n = min(n_max, len(steps))
    idx = np.unique(np.linspace(0, len(steps) - 1, num=n, dtype=int))
    return [pd.Timestamp(steps.iloc[i]) for i in idx]


def get_active_flight_ids(
    full_emb_df: pd.DataFrame,
    focusing_fid: str,
    t: pd.Timestamp,
) -> List[str]:
    mask = (
        (full_emb_df["traj_start"] <= t)
        & (full_emb_df["traj_end"] >= t)
        & (full_emb_df["flight_id"] != str(focusing_fid))
    )
    return full_emb_df.loc[mask, "flight_id"].astype(str).tolist()


def get_prior_embeddings(
    full_emb_df: pd.DataFrame,
    emb_cols: List[str],
    t: pd.Timestamp,
) -> Tuple[List[str], np.ndarray]:
    active_now = full_emb_df[
        (full_emb_df["traj_start"] <= t)
        & (full_emb_df["traj_end"] >= t)
    ]

    if active_now.empty:
        return [], np.empty((0, len(emb_cols)), dtype=np.float32)

    earliest = active_now["traj_start"].min()
    active_ids = set(active_now["flight_id"].astype(str).tolist())

    prior = full_emb_df[
        (full_emb_df["traj_start"] <= earliest)
        & (full_emb_df["traj_end"] >= earliest)
        & (full_emb_df["traj_end"] < t)
        & (~full_emb_df["flight_id"].isin(active_ids))
    ].sort_values("traj_end")

    if prior.empty:
        return [], np.empty((0, len(emb_cols)), dtype=np.float32)

    return (
        prior["flight_id"].astype(str).tolist(),
        prior[emb_cols].to_numpy(dtype=np.float32),
    )


@torch.no_grad()
def batch_prefix_embeddings(
    cfg: ScenarioConfig,
    model,
    model_cfg,
    flight_groups: Dict[str, pd.DataFrame],
    flight_ids: List[str],
    cutoff_time: pd.Timestamp,
    batch_size: int,
) -> Dict[str, Optional[np.ndarray]]:
    results: Dict[str, Optional[np.ndarray]] = {}
    to_compute: List[str] = []

    for fid in flight_ids:
        key = cache_key(fid, cutoff_time)
        if key in _PREFIX_CACHE:
            results[fid] = _PREFIX_CACHE[key]
        else:
            to_compute.append(fid)

    if not to_compute:
        return results

    for batch_start in range(0, len(to_compute), batch_size):
        batch_fids = to_compute[batch_start: batch_start + batch_size]
        seqs = []

        for fid in batch_fids:
            g = flight_groups.get(str(fid))
            if g is None or g.empty:
                seqs.append(None)
                continue

            g_cut = g[g["point_timestamp"] <= cutoff_time]
            if len(g_cut) == 0:
                seqs.append(None)
                continue

            if len(g_cut) > model_cfg.max_seq_len:
                g_cut = g_cut.iloc[-model_cfg.max_seq_len:].reset_index(drop=True)

            x = g_cut[list(model_cfg.feature_cols)].to_numpy(dtype=np.float32)
            seqs.append(x)

        valid_idx = [i for i, s in enumerate(seqs) if s is not None]
        if not valid_idx:
            for fid in batch_fids:
                results[fid] = None
                cache_set(cfg, cache_key(fid, cutoff_time), None)
            continue

        valid_seqs = [seqs[i] for i in valid_idx]
        lengths = [s.shape[0] for s in valid_seqs]
        max_len = max(lengths)
        feat_dim = valid_seqs[0].shape[1]
        B = len(valid_seqs)

        x_pad = torch.zeros((B, max_len, feat_dim), dtype=torch.float32)
        kpm = torch.ones((B, max_len), dtype=torch.bool)

        for bi, (seq, L) in enumerate(zip(valid_seqs, lengths)):
            x_pad[bi, :L, :] = torch.tensor(seq)
            kpm[bi, :L] = False

        x_pad = x_pad.to(model_cfg.device)
        kpm = kpm.to(model_cfg.device)

        z = model(x_pad, kpm)
        last_idx = torch.tensor([L - 1 for L in lengths], device=model_cfg.device)
        bidx = torch.arange(B, device=model_cfg.device)
        embs = z[bidx, last_idx].cpu().numpy()

        vi = 0
        for i, fid in enumerate(batch_fids):
            key = cache_key(fid, cutoff_time)
            if seqs[i] is None:
                results[fid] = None
                cache_set(cfg, key, None)
            else:
                emb = embs[vi]
                results[fid] = emb
                cache_set(cfg, key, emb)
                vi += 1

    return results


# ============================================================
# CHUNK WRITING
# ============================================================

def existing_chunk_files(chunks_dir: Path) -> List[Path]:
    return sorted(chunks_dir.glob("chunk_*.parquet"))


def next_chunk_path(chunks_dir: Path) -> Path:
    files = existing_chunk_files(chunks_dir)
    next_id = len(files)
    return chunks_dir / f"chunk_{next_id:05d}.parquet"


def flush_records_to_chunk(
    records: List[dict],
    chunks_dir: Path,
    chunk_index_rows: List[dict],
    compression: str,
) -> Optional[Path]:
    if not records:
        return None

    chunk_path = next_chunk_path(chunks_dir)
    df = pd.DataFrame(records)
    df.to_parquet(chunk_path, index=False, compression=compression)

    chunk_index_rows.append({
        "chunk_file": str(chunk_path),
        "n_rows": int(len(df)),
    })
    return chunk_path


# ============================================================
# MAIN BUILD
# ============================================================

def build_all_scenarios(
    cfg: ScenarioConfig,
    paths: dict,
    logger: logging.Logger,
    model,
    model_cfg,
    flight_groups: Dict[str, pd.DataFrame],
    flights_df: pd.DataFrame,
    full_emb_df: pd.DataFrame,
    emb_cols: List[str],
) -> dict:
    state = load_state(paths["state_path"])
    done_indices = set(state["done_indices"]) if state else set()
    chunk_index_rows: List[dict] = []

    if paths["chunk_index_path"].exists():
        prev_chunk_index = pd.read_parquet(paths["chunk_index_path"])
        chunk_index_rows = prev_chunk_index.to_dict("records")

    records: List[dict] = []
    n_total = len(flights_df)
    scenarios_written = int(sum(r["n_rows"] for r in chunk_index_rows))

    logger.info("Scenario üretimi başlıyor | total_flights=%s | resumed_done=%s", f"{n_total:,}", f"{len(done_indices):,}")

    model.eval()

    for i, (_, row) in enumerate(flights_df.iterrows()):
        if i in done_indices:
            continue

        hex_icao = str(row["hex_icao"]).upper().strip()
        entry_time = row[cfg.entry_time_col]
        landing_time = row[cfg.landing_time_col]

        if pd.isna(entry_time) or pd.isna(landing_time):
            done_indices.add(i)
            continue

        entry_time = pd.Timestamp(entry_time)
        landing_time = pd.Timestamp(landing_time)
        if landing_time.tzinfo is None:
            landing_time = landing_time.tz_localize("UTC")
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize("UTC")

        if landing_time <= entry_time:
            done_indices.add(i)
            continue

        focusing_fid = find_focusing_flight_id(full_emb_df, hex_icao, entry_time)
        if focusing_fid is None:
            done_indices.add(i)
            continue

        g = flight_groups.get(focusing_fid)
        if g is None or g.empty:
            done_indices.add(i)
            continue

        obs_times = pick_observation_times(g, entry_time, landing_time, cfg.n_obs_max)

        for scen_idx, t in enumerate(obs_times):
            foc_result = batch_prefix_embeddings(
                cfg, model, model_cfg, flight_groups, [focusing_fid], t, batch_size=1
            )
            focusing_emb = foc_result.get(focusing_fid)
            if focusing_emb is None:
                continue

            active_ids = get_active_flight_ids(full_emb_df, focusing_fid, t)
            active_results = batch_prefix_embeddings(
                cfg, model, model_cfg, flight_groups, active_ids, t, batch_size=cfg.active_batch_size
            )

            valid_active_ids = [fid for fid in active_ids if active_results.get(fid) is not None]
            valid_active_embs = [active_results[fid] for fid in valid_active_ids]
            active_embs = (
                np.stack(valid_active_embs, axis=0).astype(np.float32)
                if valid_active_embs
                else np.empty((0, len(emb_cols)), dtype=np.float32)
            )

            prior_ids, prior_embs = get_prior_embeddings(full_emb_df, emb_cols, t)

            label_min = row.get(cfg.label_col, np.nan)
            label_s = label_min * 60.0 if pd.notna(label_min) else np.nan

            records.append({
                "scenario_id": f"{focusing_fid}__{int(pd.Timestamp(t).value)}",
                "focusing_flight_id": focusing_fid,
                "hex_icao": hex_icao,
                "entry_time": entry_time,
                "obs_time": t,
                "scen_idx": scen_idx,

                "focusing_emb": focusing_emb.tolist(),
                "active_embs": active_embs.tolist(),
                "prior_embs": prior_embs.tolist(),

                "active_flight_ids": valid_active_ids,
                "prior_flight_ids": prior_ids,

                "n_active": len(valid_active_ids),
                "n_prior": len(prior_ids),

                "label_post_terminal_s": label_s,
                "label_post_terminal_min": label_min,

                "airline_name_english": row.get("airline_name_english", ""),
                "callsign_code_iata": row.get("callsign_code_iata", ""),
                "callsign_code_icao": row.get("callsign_code_icao", ""),
                "haul": row.get("haul", ""),
                "dep_code_iata": row.get("dep_code_iata", ""),
                "dep_name_english": row.get("dep_name_english", ""),
                "dest_code_iata": row.get("dest_code_iata", ""),
                "dest_name_english": row.get("dest_name_english", ""),
                "distance": row.get("distance", np.nan),
                "aircraft_type": row.get("aircraft_type", ""),
                "wake_turbulence_cat": row.get("wake_turbulence_cat", ""),
                "arr_sched_time_utc": row.get("arr_sched_time_utc", pd.NaT),
                "date": row.get("date", ""),
                "day_of_week": row.get("day_of_week", ""),
            })

            if len(records) >= cfg.chunk_rows_soft_limit:
                chunk_path = flush_records_to_chunk(
                    records, paths["chunks_dir"], chunk_index_rows, cfg.parquet_compression
                )
                scenarios_written += len(records)
                records.clear()
                pd.DataFrame(chunk_index_rows).to_parquet(paths["chunk_index_path"], index=False)
                save_state(paths["state_path"], {"done_indices": sorted(done_indices)})
                gc.collect()
                logger.info(
                    "Chunk yazıldı | file=%s | total_scenarios=%s | cache=%s",
                    chunk_path.name if chunk_path else "-",
                    f"{scenarios_written:,}",
                    f"{len(_PREFIX_CACHE):,}",
                )

        done_indices.add(i)

        if (i + 1) % cfg.checkpoint_every_flights == 0:
            if records:
                chunk_path = flush_records_to_chunk(
                    records, paths["chunks_dir"], chunk_index_rows, cfg.parquet_compression
                )
                scenarios_written += len(records)
                records.clear()
            else:
                chunk_path = None

            pd.DataFrame(chunk_index_rows).to_parquet(paths["chunk_index_path"], index=False)
            save_state(paths["state_path"], {"done_indices": sorted(done_indices)})
            gc.collect()

            logger.info(
                "[%s/%s] checkpoint | chunk=%s | scenarios=%s | cache=%s",
                f"{i+1:,}",
                f"{n_total:,}",
                chunk_path.name if chunk_path else "-",
                f"{scenarios_written:,}",
                f"{len(_PREFIX_CACHE):,}",
            )
        elif (i + 1) % 100 == 0:
            logger.info(
                "[%s/%s] in-memory-records=%s | cache=%s",
                f"{i+1:,}",
                f"{n_total:,}",
                f"{len(records):,}",
                f"{len(_PREFIX_CACHE):,}",
            )

    if records:
        flush_records_to_chunk(records, paths["chunks_dir"], chunk_index_rows, cfg.parquet_compression)
        scenarios_written += len(records)
        records.clear()

    pd.DataFrame(chunk_index_rows).to_parquet(paths["chunk_index_path"], index=False)
    save_state(paths["state_path"], {"done_indices": sorted(done_indices)})

    summary = {
        "run_dir": str(paths["run_dir"]),
        "n_flights_total": int(n_total),
        "n_flights_done": int(len(done_indices)),
        "n_chunks": int(len(chunk_index_rows)),
        "n_scenarios_total": int(scenarios_written),
        "chunk_index_path": str(paths["chunk_index_path"]),
    }

    with open(paths["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("Bitti | chunks=%s | scenarios=%s", f"{summary['n_chunks']:,}", f"{summary['n_scenarios_total']:,}")
    return summary


# ============================================================
# ENTRYPOINT
# ============================================================

def main():
    paths = prepare_run_dirs(CFG)
    logger = setup_logger(paths["log_path"])

    manifest_exists = paths["manifest_path"].exists()
    state_exists = paths["state_path"].exists()

    if CFG.resume and (manifest_exists or state_exists):
        logger.info("Resume mode | run_dir=%s", paths["run_dir"])
    else:
        logger.info("Fresh run | run_dir=%s", paths["run_dir"])

    cfg_dump = {
        "config": asdict(CFG),
        "config_hash": config_fingerprint(CFG),
        "inputs": {
            "ckpt": file_fingerprint(CFG.ckpt_path),
            "flights": file_fingerprint(CFG.flights_path),
            "full_emb": file_fingerprint(CFG.full_emb_path),
            "traj": file_fingerprint(CFG.traj_path),
        },
    }
    with open(paths["config_path"], "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, ensure_ascii=False, indent=2)

    t0 = pd.Timestamp.utcnow()

    model, model_cfg = load_model(CFG, logger)
    flights_df = load_flights(CFG, logger)
    full_emb_df, emb_cols = load_full_embeddings(CFG, logger)
    flight_groups = load_trajectory_groups(CFG, model_cfg, logger)

    summary = build_all_scenarios(
        cfg=CFG,
        paths=paths,
        logger=logger,
        model=model,
        model_cfg=model_cfg,
        flight_groups=flight_groups,
        flights_df=flights_df,
        full_emb_df=full_emb_df,
        emb_cols=emb_cols,
    )

    elapsed_min = (pd.Timestamp.utcnow() - t0).total_seconds() / 60.0
    logger.info("Toplam süre: %.1f dakika", elapsed_min)
    logger.info("Manifest: %s", paths["manifest_path"])
    logger.info("Chunk index: %s", paths["chunk_index_path"])
    logger.info("Toplam senaryo: %s", f"{summary['n_scenarios_total']:,}")


if __name__ == "__main__":
    main()
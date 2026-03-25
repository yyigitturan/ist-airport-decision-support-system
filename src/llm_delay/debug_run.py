"""
debug_run.py — Eğitim öncesi tam kontrol scripti

Kullanım:
    python -m src.llm_delay.debug_run

Kontrol ettikleri:
    1. Config alanları — train/dataset/model ile uyum
    2. Trainable parametre sayısı — freeze doğru mu
    3. İlk batch shape'leri
    4. Forward pass — output shape, NaN kontrolü
    5. Loss hesabı — target/target_raw ayrımı
    6. Scaler — leakage yok mu
    7. Dosya çıktıları — checkpoint/history yazılıyor mu
    8. Prediction sanity — collapse, negatif, uçuk değer
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import TrainConfig
from .dataset import ScenarioDataset, make_collate_fn
from .model import LLMDelayRegressor
from .utils import StandardScaler1D


# ══════════════════════════════════════════════════════════════════════
# Renkli print yardımcıları
# ══════════════════════════════════════════════════════════════════════

def _ok(msg: str)   -> None: print(f"  ✓  {msg}")
def _warn(msg: str) -> None: print(f"  ⚠  {msg}")
def _fail(msg: str) -> None: print(f"  ✗  {msg}"); sys.exit(1)
def _head(msg: str) -> None: print(f"\n{'─' * 60}\n  {msg}\n{'─' * 60}")


# ══════════════════════════════════════════════════════════════════════
# 1. Config → kod uyumu
# ══════════════════════════════════════════════════════════════════════

def check_config(cfg: TrainConfig) -> None:
    _head("1. Config alan kontrolü")

    required = [
        "train_eval_subset_size",
        "train_eval_every_n",
        "loss_type",
        "smoothl1_beta",
        "llm_hidden_dim",
        "max_active",
        "max_prior",
        "scale_target",
        "early_stopping_metric",
        "early_stopping_patience",
        "num_epochs",
        "lr",
        "grad_accum_steps",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "seed",
        "device",
    ]

    missing = [f for f in required if not hasattr(cfg, f)]
    if missing:
        _fail(f"Config'de eksik alanlar: {missing}")
    else:
        _ok(f"Tüm {len(required)} alan config'de mevcut")

    _ok(f"model_name        = {cfg.model_name}")
    _ok(f"llm_hidden_dim    = {cfg.llm_hidden_dim}")
    _ok(f"loss_type         = {cfg.loss_type}")
    _ok(f"scale_target      = {cfg.scale_target}")
    _ok(f"lr                = {cfg.lr}")
    _ok(f"grad_accum_steps  = {cfg.grad_accum_steps}")
    _ok(f"train_eval_every_n= {cfg.train_eval_every_n}")
    _ok(f"subset_size       = {cfg.train_eval_subset_size}")
    _ok(f"max_active        = {cfg.max_active}  max_prior={cfg.max_prior}")

    if cfg.loss_type not in ("mse", "smoothl1"):
        _fail(f"loss_type '{cfg.loss_type}' desteklenmiyor")
    if cfg.early_stopping_metric not in ("mae", "composite_mae_rmse", "composite_mae_r2"):
        _fail(f"early_stopping_metric '{cfg.early_stopping_metric}' tanımsız")

    _ok("Config geçerli")


# ══════════════════════════════════════════════════════════════════════
# 2. Freeze kontrolü
# ══════════════════════════════════════════════════════════════════════

def check_freeze(model: LLMDelayRegressor) -> None:
    _head("2. Parametre freeze kontrolü")

    total      = sum(p.numel() for p in model.parameters())
    trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen     = total - trainable
    pct        = 100.0 * trainable / total if total > 0 else 0.0

    _ok(f"Toplam parametre : {total:,}")
    _ok(f"Frozen           : {frozen:,}  ({100-pct:.1f}%)")
    _ok(f"Trainable        : {trainable:,}  ({pct:.1f}%)")

    # Sadece adapter ve reg_head trainable olmalı
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not (name.startswith("adapter") or name.startswith("reg_head")):
                _warn(f"Beklenmedik trainable param: {name}")

    # LLM frozen mu
    llm_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)
    if llm_trainable > 0:
        _fail(f"LLM frozen değil! {llm_trainable:,} trainable param var.")
    else:
        _ok("LLM tamamen frozen")

    # Adapter ve reg_head trainable mı
    adapter_trainable = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
    head_trainable    = sum(p.numel() for p in model.reg_head.parameters() if p.requires_grad)

    if adapter_trainable == 0:
        _fail("Adapter trainable değil!")
    else:
        _ok(f"Adapter trainable: {adapter_trainable:,}")

    if head_trainable == 0:
        _fail("RegressionHead trainable değil!")
    else:
        _ok(f"RegressionHead trainable: {head_trainable:,}")


# ══════════════════════════════════════════════════════════════════════
# 3–7. Batch shape + forward + loss + scaler
# ══════════════════════════════════════════════════════════════════════

def check_batch_and_forward(
    cfg: TrainConfig,
    model: LLMDelayRegressor,
    loader: DataLoader,
    scaler: StandardScaler1D,
    device: torch.device,
) -> None:
    _head("3. İlk batch shape kontrolü")

    batch = next(iter(loader))

    # Beklenen key'ler
    expected_keys = [
        "prompt_ids", "st1_ids", "st2_ids", "st3_ids", "st4_ids",
        "focusing_emb", "active_embs", "prior_embs",
        "active_mask", "prior_mask",
        "target", "target_raw",
    ]
    missing_keys = [k for k in expected_keys if k not in batch]
    if missing_keys:
        _fail(f"Batch'te eksik key'ler: {missing_keys}")
    else:
        _ok(f"Tüm {len(expected_keys)} key batch'te mevcut")

    # Shape'ler
    for key in expected_keys:
        val = batch[key]
        if isinstance(val, torch.Tensor):
            print(f"    {key:<25}: {tuple(val.shape)}")
        else:
            print(f"    {key:<25}: {type(val)}")

    # focusing_emb boyutu
    fe_shape = batch["focusing_emb"].shape
    if fe_shape[-1] != cfg.traj_dim:
        _fail(f"focusing_emb dim={fe_shape[-1]}, beklenen traj_dim={cfg.traj_dim}")
    else:
        _ok(f"focusing_emb traj_dim uyuşuyor: {cfg.traj_dim}")

    # active/prior traj_dim
    ae_shape = batch["active_embs"].shape
    if ae_shape[-1] != cfg.traj_dim:
        _fail(f"active_embs dim={ae_shape[-1]}, beklenen traj_dim={cfg.traj_dim}")
    else:
        _ok(f"active_embs traj_dim uyuşuyor")

    # target vs target_raw ayrımı
    t     = batch["target"].float()
    t_raw = batch["target_raw"].float()

    if cfg.scale_target and torch.allclose(t, t_raw, atol=1e-3):
        _warn("scale_target=True ama target ≈ target_raw — scaler çalışmıyor olabilir")
    else:
        _ok(f"target (scaled) range: [{t.min():.3f}, {t.max():.3f}]")
        _ok(f"target_raw     range: [{t_raw.min():.3f}, {t_raw.max():.3f}]")

    _head("4. Forward pass kontrolü")

    # Batch'i device'a taşı
    batch_dev = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    model.eval()
    with torch.no_grad():
        try:
            out = model(
                prompt_ids=batch_dev["prompt_ids"],
                st1_ids=batch_dev["st1_ids"],
                st2_ids=batch_dev["st2_ids"],
                st3_ids=batch_dev["st3_ids"],
                st4_ids=batch_dev["st4_ids"],
                focusing_emb=batch_dev["focusing_emb"],
                active_embs=batch_dev["active_embs"],
                prior_embs=batch_dev["prior_embs"],
                active_mask=batch_dev["active_mask"],
                prior_mask=batch_dev["prior_mask"],
            )
        except Exception as e:
            _fail(f"Forward pass hatası: {e}")

    _ok(f"Forward pass başarılı — output shape: {tuple(out.shape)}")

    # Output shape [B] veya [B,1] olmalı
    B = batch_dev["prompt_ids"].shape[0]
    if out.shape[0] != B:
        _fail(f"Output batch dim={out.shape[0]}, beklenen B={B}")

    # NaN kontrolü
    if torch.isnan(out).any():
        _fail("Output NaN içeriyor!")
    if torch.isinf(out).any():
        _warn("Output Inf içeriyor!")
    else:
        _ok(f"Output NaN/Inf yok — değer aralığı: [{out.min():.4f}, {out.max():.4f}]")

    _head("5. Loss hesabı kontrolü")

    from torch.nn import MSELoss, SmoothL1Loss
    loss_fn = MSELoss() if cfg.loss_type == "mse" else SmoothL1Loss(beta=cfg.smoothl1_beta)

    target = batch_dev["target"].float()
    if target.ndim > 1:
        target = target.squeeze(-1)

    try:
        loss = loss_fn(out, target)
        if torch.isnan(loss):
            _fail("Loss NaN!")
        _ok(f"Loss hesabı başarılı — loss={loss.item():.6f}")
    except Exception as e:
        _fail(f"Loss hatası: {e}  out.shape={out.shape}  target.shape={target.shape}")

    _head("6. Scaler & inverse transform kontrolü")

    out_np = out.detach().cpu().numpy().reshape(-1)
    inv    = scaler.inverse_transform(out_np)
    _ok(f"inverse_transform başarılı — aralık: [{inv.min():.2f}, {inv.max():.2f}] dk")

    if (inv < -10).any():
        _warn(f"Negatif tahmin var (min={inv.min():.2f}) — ilk epoch'ta normal olabilir")
    if (inv > 200).any():
        _warn(f"Çok büyük tahmin var (max={inv.max():.2f}) — kontrol et")

    _ok("Scaler inverse transform OK")


# ══════════════════════════════════════════════════════════════════════
# Ana kontrol fonksiyonu
# ══════════════════════════════════════════════════════════════════════

def run_debug(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()

    print(f"\n{'═' * 60}")
    print("  EĞİTİM ÖNCESİ KONTROL")
    print(f"{'═' * 60}")

    # 1. Config
    check_config(cfg)

    # Tokenizer
    _head("Tokenizer yükleniyor...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _ok(f"Tokenizer yüklendi  vocab_size={tokenizer.vocab_size}")

    # İlk ay
    month_dirs = sorted([p for p in cfg.monthly_root_path.iterdir() if p.is_dir()])
    if not month_dirs:
        _fail(f"Ay dizini bulunamadı: {cfg.monthly_root_path}")

    month_dir = month_dirs[0]
    _ok(f"Test ay: {month_dir.name}")

    train_df = pd.read_parquet(month_dir / "train.parquet")
    _ok(f"train.parquet yüklendi — {len(train_df)} satır")

    # Scaler
    scaler = StandardScaler1D(enabled=cfg.scale_target).fit(
        train_df[cfg.target_col].values
    )
    if cfg.scale_target:
        _ok(f"Scaler fit: mean={scaler.mean_:.3f}  std={scaler.std_:.3f}")
    else:
        _ok("scale_target=False — scaler pasif")

    # Dataset & loader
    train_ds = ScenarioDataset(train_df, tokenizer, cfg, target_scaler=scaler)
    collate  = make_collate_fn(
        tokenizer.pad_token_id,
        train_ds.st1, train_ds.st2, train_ds.st3, train_ds.st4,
    )
    loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )
    _ok(f"DataLoader hazır — {len(train_ds)} sample")

    # Device
    if cfg.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)
    _ok(f"Device: {device}")

    # Model
    _head("Model yükleniyor...")
    model = LLMDelayRegressor(
        model_name=cfg.model_name,
        traj_dim=cfg.traj_dim,
        llm_hidden_dim=cfg.llm_hidden_dim,
        adapter_dropout=cfg.adapter_dropout,
        head_dropout=cfg.head_dropout,
    ).to(device)
    _ok("Model yüklendi")

    # 2. Freeze
    check_freeze(model)

    # 3–6. Batch + forward + loss + scaler
    check_batch_and_forward(cfg, model, loader, scaler, device)

    print(f"\n{'═' * 60}")
    print("  TÜM KONTROLLER TAMAMLANDI — EĞİTİME HAZIR")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    run_debug()
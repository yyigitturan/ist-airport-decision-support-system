"""
train.py — LLM-based Flight Delay Regression · Training Pipeline

Her epoch sonunda loglanır:
  - train_loss  : loss fonksiyonundan gelen ortalama (scaled space, cfg.loss_type baz alınır)
  - train_mae, train_mse, train_r2 : cfg.train_eval_every_n'e göre hesaplanır.
      Epoch 1 ve son epoch dahil. Diğer epoch'larda "—" görünür — bu kasıtlı.
      Her epoch farklı rastgele subset kullanılır (seed + epoch).
  - val_mae, val_mse, val_r2, val_rmse : her epoch hesaplanır.
"""

from __future__ import annotations

import dataclasses
import gc
import math
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss, SmoothL1Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import TrainConfig
from .dataset import ScenarioDataset, make_collate_fn
from .evaluate import regression_metrics_full, print_metrics
from .model import LLMDelayRegressor
from .utils import StandardScaler1D, dump_json, ensure_dir, set_seed


# ══════════════════════════════════════════════════════════════════════
# Yardımcılar
# ══════════════════════════════════════════════════════════════════════

def _build_loss_fn(cfg: TrainConfig) -> torch.nn.Module:
    """cfg.loss_type'a göre loss döndür. Yeni loss eklemek için buraya ekle."""
    if cfg.loss_type == "mse":
        return MSELoss()
    if cfg.loss_type == "smoothl1":
        return SmoothL1Loss(beta=cfg.smoothl1_beta)
    raise ValueError(f"Desteklenmeyen loss_type: '{cfg.loss_type}'. Geçerli: 'mse', 'smoothl1'")


def _resolve_device(cfg: TrainConfig) -> torch.device:
    if cfg.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if cfg.device == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if cfg.device == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    return torch.device("cpu")


def _move_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _flush_optimizer(
    optimizer: AdamW,
    scheduler,
    model: LLMDelayRegressor,
    max_grad_norm: float = 1.0,
) -> None:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _set_train_modes(model: LLMDelayRegressor) -> None:
    """Frozen backbone eval modda. Adapter + reg_head train modunda."""
    model.train()
    model.llm.eval()
    model.adapter.train()
    model.reg_head.train()


def _should_eval_train(epoch: int, total: int, every_n: int) -> bool:
    """Epoch 1, her every_n epoch ve son epoch'ta True döner."""
    return epoch == 1 or epoch % every_n == 0 or epoch == total


def _early_stopping_score(val_mf, metric: str) -> float:
    """
    Early stopping score — minimize edilir.
    "mae" | "composite_mae_rmse" | "composite_mae_r2"
    """
    mae  = val_mf.mae
    rmse = val_mf.rmse
    r2   = val_mf.r2 if (isinstance(val_mf.r2, float) and val_mf.r2 == val_mf.r2) else 0.0
    if metric == "composite_mae_rmse":
        return mae + 0.5 * rmse
    if metric == "composite_mae_r2":
        return mae - 0.2 * r2
    return mae


def _f(v: float, decimals: int = 4) -> str:
    """NaN-safe format."""
    return f"{v:.{decimals}f}" if (v == v) else "  —  "


def _model_forward(model: LLMDelayRegressor, batch: dict) -> torch.Tensor:
    """Model forward — prompt_attention_mask model tarafından kullanılmıyor, geçilmez."""
    return model(
        prompt_ids=batch["prompt_ids"],
        st1_ids=batch["st1_ids"],
        st2_ids=batch["st2_ids"],
        st3_ids=batch["st3_ids"],
        st4_ids=batch["st4_ids"],
        focusing_emb=batch["focusing_emb"],
        active_embs=batch["active_embs"],
        prior_embs=batch["prior_embs"],
        active_mask=batch["active_mask"],
        prior_mask=batch["prior_mask"],
    )


# ══════════════════════════════════════════════════════════════════════
# Subset loader — train eval için hızlı yaklaşım
# ══════════════════════════════════════════════════════════════════════

def _make_subset_loader(
    dataset: ScenarioDataset,
    collate_fn,
    subset_size: int,
    eval_batch_size: int,
    num_workers: int,
    seed: int,
    epoch: int = 0,
) -> DataLoader:
    """
    Train setinden rastgele subset seç, DataLoader döndür.
    Tüm train seti yerine subset kullanmak train eval süresini ~10x kısaltır.
    subset_size = 0 → tüm dataset kullanılır (pahalı, dikkatli ol).

    seed + epoch ile her epoch farklı subset seçilir — aynı örneklere
    tekrar tekrar bakarak train metric'i yanıltmaktan kaçınılır.
    """
    n = len(dataset)
    if subset_size <= 0 or subset_size >= n:
        return DataLoader(
            dataset, batch_size=eval_batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn,
        )

    rng     = np.random.default_rng(seed + epoch)   # epoch'a göre farklı subset
    indices = rng.choice(n, size=min(subset_size, n), replace=False).tolist()
    subset  = Subset(dataset, indices)
    return DataLoader(
        subset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )


# ══════════════════════════════════════════════════════════════════════
# Inference loop
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_epoch(
    model: LLMDelayRegressor,
    loader: DataLoader,
    scaler: StandardScaler1D,
    device: torch.device,
    desc: str = "eval",
) -> tuple[List[float], List[float]]:
    """
    model.eval() moduna alır, inference yapar, inverse transform uygular.
    Döndükten sonra çağıran _set_train_modes() ile train moduna geçer.
    """
    model.eval()
    preds, trues = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        batch  = _move_to_device(batch, device)
        y_hat  = scaler.inverse_transform(
            _model_forward(model, batch).detach().cpu().numpy().reshape(-1)
        )
        y_true = batch["target_raw"].detach().cpu().numpy().reshape(-1)
        preds.extend(y_hat.tolist())
        trues.extend(y_true.tolist())

    return preds, trues


# ══════════════════════════════════════════════════════════════════════
# Epoch log
# ══════════════════════════════════════════════════════════════════════

def _print_epoch(
    epoch_num: int,
    total_epochs: int,
    train_loss: float,
    tr_mae: float,
    tr_mse: float,
    tr_r2: float,
    val_mf,
) -> None:
    """
    Standart epoch özet satırı.
    train_mae/mse/r2 sadece _should_eval_train() True döndüğünde dolu gelir,
    diğer epoch'larda "—" görünür.
    """
    sep = " | "
    print(
        f"  epoch {epoch_num}/{total_epochs}{sep}"
        f"train_loss={_f(train_loss)}{sep}"
        f"tr_mae={_f(tr_mae)}  tr_mse={_f(tr_mse)}  tr_r2={_f(tr_r2)}{sep}"
        f"val_mae={_f(val_mf.mae)}  val_mse={_f(val_mf.mse)}  val_r2={_f(val_mf.r2)}{sep}"
        f"val_rmse={_f(val_mf.rmse)}{sep}"
        f"p90={_f(val_mf.mae_p90)}  p99={_f(val_mf.mae_p99)}{sep}"
        f"w2min={val_mf.within_2min_pct:.1f}%{sep}"
        f"health={val_mf.model_health}"
    )


# ══════════════════════════════════════════════════════════════════════
# Tek ay eğitimi
# ══════════════════════════════════════════════════════════════════════

def train_one_month(cfg: TrainConfig, month_dir: Path, tokenizer) -> Dict:

    train_df = pd.read_parquet(month_dir / "train.parquet")
    valid_df = pd.read_parquet(month_dir / "valid.parquet")
    test_df  = pd.read_parquet(month_dir / "test.parquet")

    # Scaler sadece train split üzerinde fit edilir
    scaler = StandardScaler1D(enabled=cfg.scale_target).fit(
        train_df[cfg.target_col].values
    )

    train_ds = ScenarioDataset(train_df, tokenizer, cfg, target_scaler=scaler)
    valid_ds = ScenarioDataset(valid_df, tokenizer, cfg, target_scaler=scaler)
    test_ds  = ScenarioDataset(test_df,  tokenizer, cfg, target_scaler=scaler)

    # st1..st4 closure'a alınır — dataset.py güncel imzası
    collate = make_collate_fn(
        tokenizer.pad_token_id,
        train_ds.st1, train_ds.st2, train_ds.st3, train_ds.st4,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=collate,
    )

    device = _resolve_device(cfg)
    model  = LLMDelayRegressor(
        model_name=cfg.model_name,
        traj_dim=cfg.traj_dim,
        llm_hidden_dim=cfg.llm_hidden_dim,
        adapter_dropout=cfg.adapter_dropout,
        head_dropout=cfg.head_dropout,
    ).to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps     = max(1, steps_per_epoch * cfg.num_epochs)
    warmup_steps    = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    loss_fn   = _build_loss_fn(cfg)
    es_metric = cfg.early_stopping_metric

    best_score = float("inf")
    best_epoch = 0
    patience   = 0
    history: List[Dict] = []

    month_out = cfg.output_root_path / month_dir.name
    ensure_dir(month_out)

    subset_info = (
        f"subset={cfg.train_eval_subset_size}"
        if cfg.train_eval_subset_size > 0
        else "full_train"
    )

    print(f"\n{'─' * 72}")
    print(f"  {month_dir.name}  |  train={len(train_ds)}  valid={len(valid_ds)}  test={len(test_ds)}")
    print(f"  device={device}  |  model={cfg.model_name}")
    print(f"  loss={cfg.loss_type}  scale_target={cfg.scale_target}  lr={cfg.lr}")
    print(f"  steps/epoch={steps_per_epoch}  total_steps={total_steps}")
    print(f"  early_stopping={es_metric}  patience={cfg.early_stopping_patience}")
    print(f"  train_eval every {cfg.train_eval_every_n} epochs  ({subset_info})")
    print(f"{'─' * 72}\n")

    # ══════════════════════════════════════════════════════════════════
    # Train loop
    # ══════════════════════════════════════════════════════════════════
    for epoch in range(cfg.num_epochs):
        epoch_num = epoch + 1

        # ── Forward + backward ──────────────────────────────────────
        _set_train_modes(model)
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{month_dir.name}] epoch {epoch_num}/{cfg.num_epochs}",
        )

        for step, batch in enumerate(pbar, start=1):
            batch    = _move_to_device(batch, device)
            raw_loss = loss_fn(_model_forward(model, batch), batch["target"])
            (raw_loss / cfg.grad_accum_steps).backward()

            running_loss += raw_loss.item()
            steps += 1

            if step % cfg.grad_accum_steps == 0:
                _flush_optimizer(optimizer, scheduler, model)

            pbar.set_postfix(loss=f"{running_loss / steps:.4f}")

        if steps % cfg.grad_accum_steps != 0:
            _flush_optimizer(optimizer, scheduler, model)

        train_loss_avg = running_loss / max(steps, 1)

        # ── Train metrics — subset üzerinden, seçili epoch'larda ─────
        # Tüm seti eval etmek MPS'te çok pahalı.
        # Subset boyutu cfg.train_eval_subset_size ile kontrol edilir.
        # Ara epoch'larda NaN kalır — bu davranış kasıtlıdır.
        tr_mae = float("nan")
        tr_mse = float("nan")
        tr_r2  = float("nan")

        if _should_eval_train(epoch_num, cfg.num_epochs, cfg.train_eval_every_n):
            # Her epoch yeni subset — seed + epoch ile farklı örnekler seçilir
            train_eval_loader = _make_subset_loader(
                dataset=train_ds,
                collate_fn=collate,
                subset_size=cfg.train_eval_subset_size,
                eval_batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
                seed=cfg.seed,
                epoch=epoch_num,
            )
            tr_pred, tr_true = predict_epoch(
                model, train_eval_loader, scaler, device, desc="train-eval"
            )
            # predict_epoch model.eval() yaptı — train moduna geri al
            _set_train_modes(model)

            tr_mf  = regression_metrics_full(tr_true, tr_pred, prefix="train")
            tr_mae = tr_mf.mae
            tr_mse = tr_mf.mse
            tr_r2  = tr_mf.r2

        # ── Validation — her epoch ────────────────────────────────────
        val_pred, val_true = predict_epoch(
            model, valid_loader, scaler, device, desc="val"
        )
        # predict_epoch model.eval() yaptı — sonraki epoch _set_train_modes ile düzelir
        val_mf = regression_metrics_full(val_true, val_pred, prefix="val")

        # ── History ──────────────────────────────────────────────────
        history.append({
            "epoch":               epoch_num,
            "train_loss":          train_loss_avg,
            "train_mae":           tr_mae,
            "train_mse":           tr_mse,
            "train_r2":            tr_r2,
            "val_mae":             val_mf.mae,
            "val_mse":             val_mf.mse,
            "val_r2":              val_mf.r2,
            "val_rmse":            val_mf.rmse,
            "val_mae_p90":         val_mf.mae_p90,
            "val_mae_p99":         val_mf.mae_p99,
            "val_within_2min_pct": val_mf.within_2min_pct,
            "val_within_5min_pct": val_mf.within_5min_pct,
            "val_spearman_r":      val_mf.spearman_r,
            "val_ccc":             val_mf.ccc,
            "val_mean_bias":       val_mf.mean_bias,
            "val_health":          val_mf.model_health,
        })

        # ── Epoch özet ───────────────────────────────────────────────
        _print_epoch(
            epoch_num, cfg.num_epochs,
            train_loss_avg,
            tr_mae, tr_mse, tr_r2,
            val_mf,
        )

        # Aktif diagnostic uyarıları
        active_flags = [
            k for k, v in {
                "high_bias":   val_mf.high_bias_flag,
                "heavy_tail":  val_mf.heavy_tail_flag,
                "calibration": val_mf.calibration_flag,
                "hetero":      val_mf.heteroscedasticity_flag,
                "nonlinear":   val_mf.nonlinear_relationship_flag,
            }.items() if v
        ]
        if active_flags:
            print(f"    ⚠ val diagnostics: {', '.join(active_flags)}")

        # ── Checkpoint & early stopping ───────────────────────────────
        current_score = _early_stopping_score(val_mf, es_metric)

        if current_score < best_score:
            best_score = current_score
            best_epoch = epoch_num
            patience   = 0

            torch.save(model.state_dict(), month_out / "best_model.pt")
            dump_json(dataclasses.asdict(cfg),   month_out / "best_config.json")
            dump_json(val_mf.to_dict(flat=True), month_out / "best_val_metrics.json")
            dump_json(
                {"best_epoch": best_epoch, "best_score": best_score, "metric": es_metric},
                month_out / "best_epoch.json",
            )
            scaler.save(month_out / "scaler.json")
            print(
                f"  ✓ checkpoint  "
                f"val_mae={val_mf.mae:.4f}  val_mse={val_mf.mse:.4f}  "
                f"val_r2={val_mf.r2:.4f}  p90={val_mf.mae_p90:.4f}  epoch={best_epoch}"
            )
        else:
            patience += 1
            print(f"  patience {patience}/{cfg.early_stopping_patience}")
            if patience >= cfg.early_stopping_patience:
                print(f"  ⏹ early stopping — epoch {epoch_num}")
                break

    # ══════════════════════════════════════════════════════════════════
    # Test — best checkpoint yüklenir
    # ══════════════════════════════════════════════════════════════════
    try:
        state = torch.load(
            month_out / "best_model.pt", map_location=device, weights_only=True
        )
    except TypeError:
        state = torch.load(month_out / "best_model.pt", map_location=device)

    model.load_state_dict(state)

    test_pred, test_true = predict_epoch(
        model, test_loader, scaler, device, desc="test"
    )
    test_mf = regression_metrics_full(test_true, test_pred, prefix="test")

    # Test özet — train/val ile karşılaştırılabilir format
    print(f"\n{'─' * 72}")
    print(f"  TEST [{month_dir.name}]  (best epoch={best_epoch})")
    print(f"{'─' * 72}")
    print(
        f"  test_mae={test_mf.mae:.4f}  "
        f"test_mse={test_mf.mse:.4f}  "
        f"test_r2={test_mf.r2:.4f}  "
        f"test_rmse={test_mf.rmse:.4f}"
    )
    print(
        f"  p90={test_mf.mae_p90:.4f}  "
        f"p99={test_mf.mae_p99:.4f}  "
        f"within_2min={test_mf.within_2min_pct:.1f}%  "
        f"within_5min={test_mf.within_5min_pct:.1f}%"
    )
    print(
        f"  spearman_r={test_mf.spearman_r:.4f}  "
        f"ccc={test_mf.ccc:.4f}  "
        f"mean_bias={test_mf.mean_bias:.4f}  "
        f"health={test_mf.model_health}"
    )
    print(f"{'─' * 72}\n")

    # Detaylı analiz
    print_metrics(test_mf, title=f"TEST FULL — {month_dir.name}")

    # Dosyalar
    pd.DataFrame(history).to_csv(month_out / "history.csv", index=False)
    pd.DataFrame({"y_true": test_true, "y_pred": test_pred}).to_csv(
        month_out / "test_predictions.csv", index=False
    )
    dump_json(test_mf.to_dict(flat=True), month_out / "test_metrics.json")

    return {
        "month":           month_dir.name,
        "best_epoch":      best_epoch,
        "mae":             test_mf.mae,
        "mse":             test_mf.mse,
        "rmse":            test_mf.rmse,
        "r2":              test_mf.r2,
        "mae_p90":         test_mf.mae_p90,
        "mae_p99":         test_mf.mae_p99,
        "mape":            test_mf.mape,
        "smape":           test_mf.smape,
        "spearman_r":      test_mf.spearman_r,
        "ccc":             test_mf.ccc,
        "mean_bias":       test_mf.mean_bias,
        "within_2min_pct": test_mf.within_2min_pct,
        "within_5min_pct": test_mf.within_5min_pct,
        "model_health":    test_mf.model_health,
        "health_score":    test_mf.health_score,
    }


# ══════════════════════════════════════════════════════════════════════
# Tüm aylar
# ══════════════════════════════════════════════════════════════════════

def train_all_months(cfg: TrainConfig) -> pd.DataFrame:
    set_seed(cfg.seed)
    ensure_dir(cfg.output_root_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    month_dirs = sorted([p for p in cfg.monthly_root_path.iterdir() if p.is_dir()])
    results: List[Dict] = []

    for month_dir in month_dirs:
        print(f"\n{'=' * 72}")
        print(f"  TRAINING MONTH: {month_dir.name}")
        print(f"{'=' * 72}")

        try:
            res = train_one_month(cfg, month_dir, tokenizer)
            results.append(res)
        except Exception:
            print(f"[ERROR] {month_dir.name}:")
            traceback.print_exc()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    results_df = pd.DataFrame(results)
    results_df.to_csv(cfg.output_root_path / "all_month_results.csv", index=False)

    if len(results_df) > 0:
        numeric_cols = [
            "mae", "mse", "rmse", "r2",
            "mae_p90", "mae_p99",
            "spearman_r", "ccc",
            "within_2min_pct", "within_5min_pct",
            "health_score",
        ]
        summary: Dict = {}
        for col in numeric_cols:
            if col in results_df.columns:
                summary[f"mean_{col}"] = float(results_df[col].mean())

        dump_json(summary, cfg.output_root_path / "summary.json")

        print(f"\n{'═' * 72}")
        print("  ÖZET — TÜM AYLAR")
        print(f"{'═' * 72}")
        print(
            f"  mean_mae={summary.get('mean_mae', float('nan')):.4f}  "
            f"mean_mse={summary.get('mean_mse', float('nan')):.4f}  "
            f"mean_r2={summary.get('mean_r2', float('nan')):.4f}  "
            f"mean_rmse={summary.get('mean_rmse', float('nan')):.4f}"
        )
        print(
            f"  mean_p90={summary.get('mean_mae_p90', float('nan')):.4f}  "
            f"mean_within_2min={summary.get('mean_within_2min_pct', float('nan')):.1f}%"
        )
        print(f"{'═' * 72}\n")

    return results_df
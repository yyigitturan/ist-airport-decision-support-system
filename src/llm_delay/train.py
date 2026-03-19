from __future__ import annotations

import dataclasses
import gc
import math
import traceback
from pathlib import Path

import pandas as pd
import torch
from torch.nn import SmoothL1Loss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .config import TrainConfig
from .dataset import ScenarioDataset, make_collate_fn
from .evaluate import regression_metrics
from .model import LLMDelayRegressor
from .utils import StandardScaler1D, dump_json, ensure_dir, set_seed


# ─────────────────────────────────────────────
# Yardımcı
# ─────────────────────────────────────────────

def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _flush_optimizer(optimizer, scheduler, model, max_grad_norm: float = 1.0) -> None:
    """Grad clip + optimizer step + scheduler step + sıfırla."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _resolve_device(cfg: TrainConfig) -> torch.device:
    """
    "auto" → CUDA > MPS > CPU öncelik sırası.
    "cuda" / "mps" açık seçimde önce availability kontrol edilir, yoksa CPU'ya düşer.
    """
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


def _set_train_modes(model: LLMDelayRegressor) -> None:
    """
    Frozen backbone eval modda tutulur — içindeki dropout'lar devre dışı kalır.
    Sadece adapter ve regression head train modunda çalışır.
    """
    model.train()
    model.llm.eval()
    model.adapter.train()
    model.reg_head.train()


# ─────────────────────────────────────────────
# Eval loop
# ─────────────────────────────────────────────

@torch.no_grad()
def predict_epoch(model, loader, scaler, device):
    model.eval()
    preds, trues = [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch_to_device(batch, device)

        y_hat_scaled = model(
            prompt_ids=batch["prompt_ids"],
            prompt_attention_mask=batch["prompt_attention_mask"],
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

        y_hat  = scaler.inverse_transform(y_hat_scaled.detach().cpu().numpy().reshape(-1))
        y_true = batch["target_raw"].detach().cpu().numpy().reshape(-1)

        preds.extend(y_hat.tolist())
        trues.extend(y_true.tolist())

    return preds, trues


# ─────────────────────────────────────────────
# Tek ay train
# ─────────────────────────────────────────────

def train_one_month(cfg: TrainConfig, month_dir: Path, tokenizer) -> dict:
    train_df = pd.read_parquet(month_dir / "train.parquet")
    valid_df = pd.read_parquet(month_dir / "valid.parquet")
    test_df  = pd.read_parquet(month_dir / "test.parquet")

    # Scaler sadece train split üzerinde fit edilir
    scaler = StandardScaler1D(enabled=cfg.scale_target).fit(
        train_df[cfg.target_col].values
    )

    collate = make_collate_fn(tokenizer.pad_token_id)

    train_ds = ScenarioDataset(train_df, tokenizer, cfg, target_scaler=scaler)
    valid_ds = ScenarioDataset(valid_df, tokenizer, cfg, target_scaler=scaler)
    test_ds  = ScenarioDataset(test_df,  tokenizer, cfg, target_scaler=scaler)

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

    model = LLMDelayRegressor(
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

    # ceil tabanlı hesap — epoch sonu flush ile gerçek adım sayısına daha yakın
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps     = max(1, steps_per_epoch * cfg.num_epochs)
    warmup_steps    = int(total_steps * cfg.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = SmoothL1Loss()

    best_val_mae = float("inf")
    best_epoch   = 0
    patience     = 0
    history      = []

    month_out = cfg.output_root_path / month_dir.name
    ensure_dir(month_out)

    # ── Train loop ───────────────────────────────
    for epoch in range(cfg.num_epochs):
        _set_train_modes(model)
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"{month_dir.name} epoch {epoch + 1}/{cfg.num_epochs}",
        )

        for step, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, device)

            y_hat_scaled = model(
                prompt_ids=batch["prompt_ids"],
                prompt_attention_mask=batch["prompt_attention_mask"],
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

            raw_loss = loss_fn(y_hat_scaled, batch["target"])
            (raw_loss / cfg.grad_accum_steps).backward()

            running_loss += raw_loss.item()
            steps += 1

            if step % cfg.grad_accum_steps == 0:
                _flush_optimizer(optimizer, scheduler, model)

            pbar.set_postfix(train_loss=running_loss / steps)

        # Epoch sonu — kalan gradient'ları flush et
        if steps % cfg.grad_accum_steps != 0:
            _flush_optimizer(optimizer, scheduler, model)

        # ── Validation ───────────────────────────
        val_pred, val_true = predict_epoch(model, valid_loader, scaler, device)
        val_metrics = regression_metrics(val_true, val_pred)

        history.append({
            "epoch":      epoch + 1,
            "train_loss": running_loss / max(steps, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"[{month_dir.name}] epoch {epoch + 1} | "
            f"train_loss={running_loss / max(steps, 1):.4f} | "
            f"val_mae={val_metrics['mae']:.4f}"
        )

        # ── Checkpoint & early stopping ──────────
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch   = epoch + 1
            patience     = 0

            torch.save(model.state_dict(), month_out / "best_model.pt")
            dump_json(dataclasses.asdict(cfg),  month_out / "best_config.json")
            dump_json(val_metrics,              month_out / "best_val_metrics.json")
            dump_json({"best_epoch": best_epoch}, month_out / "best_epoch.json")
            scaler.save(month_out / "scaler.json")
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                print(f"Early stopping — epoch {epoch + 1}")
                break

    # ── Test ─────────────────────────────────────
    # weights_only=True: sadece state_dict tensor'ları yüklenir, güvenli.
    # Eski PyTorch sürümlerinde bu parametre yoksa sessizce kaldırılır.
    try:
        state = torch.load(
            month_out / "best_model.pt",
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # PyTorch < 1.13 weights_only desteklemiyor
        state = torch.load(month_out / "best_model.pt", map_location=device)

    model.load_state_dict(state)

    test_pred, test_true = predict_epoch(model, test_loader, scaler, device)
    test_metrics = regression_metrics(test_true, test_pred)

    pd.DataFrame(history).to_csv(month_out / "history.csv", index=False)
    pd.DataFrame({"y_true": test_true, "y_pred": test_pred}).to_csv(
        month_out / "test_predictions.csv", index=False
    )
    dump_json(test_metrics, month_out / "test_metrics.json")

    return {"month": month_dir.name, **test_metrics}


# ─────────────────────────────────────────────
# Tüm aylar
# ─────────────────────────────────────────────

def train_all_months(cfg: TrainConfig) -> pd.DataFrame:
    set_seed(cfg.seed)
    ensure_dir(cfg.output_root_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    month_dirs = sorted([p for p in cfg.monthly_root_path.iterdir() if p.is_dir()])
    results = []

    for month_dir in month_dirs:
        print("=" * 80)
        print(f"TRAINING MONTH: {month_dir.name}")
        print("=" * 80)

        try:
            res = train_one_month(cfg, month_dir, tokenizer)
            results.append(res)
        except Exception:
            # Tam stack trace — sadece mesaj değil
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
        summary = {
            "mean_mae":  float(results_df["mae"].mean()),
            "mean_rmse": float(results_df["rmse"].mean()),
            "mean_r2":   float(results_df["r2"].mean()),
        }
        dump_json(summary, cfg.output_root_path / "summary.json")

    return results_df
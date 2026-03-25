from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # ── Veri ────────────────────────────────────────────────────────
    monthly_root: str = "data/model/splits/monthly"
    output_root:  str = "artifacts/llm_delay_llama"
    target_col:   str = "label_post_terminal_min"
  
    # ── Model ────────────────────────────────────────────────────────
    # 0.5B → llm_hidden_dim=896
    # 1.5B → llm_hidden_dim=1536
    # 3B   → llm_hidden_dim=2048
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_hidden_dim:  int   = 896
    traj_dim:        int   = 256
    adapter_dropout: float = 0.15
    head_dropout:    float = 0.15

    # ── Prompt uzunlukları ───────────────────────────────────────────
    total_prompt_max_len:       int = 775
    flight_prompt_max_len:      int = 480
    weather_prompt_max_len:     int = 230
    static_traj_prompt_max_len: int = 65
    max_active:                 int = 21
    max_prior:                  int = 20

    # ── Özellik seçimi ───────────────────────────────────────────────
    use_weather: bool = True
    use_notam:   bool = False

    # ── Hedef ölçekleme ──────────────────────────────────────────────
    # True → StandardScaler ile normalize — R² için kritik
    scale_target: bool = True

    # ── Loss ─────────────────────────────────────────────────────────
    # "mse"      → MSELoss
    # "smoothl1" → SmoothL1Loss (Huber) — outlier robust, paper'da kullanılan
    loss_type:     str   = "mse"
    smoothl1_beta: float = 1.0

    # ── Optimizasyon ─────────────────────────────────────────────────
    lr:               float = 1e-5
    weight_decay:     float = 2e-5
    warmup_ratio:     float = 0.06
    grad_accum_steps: int   = 16

    # ── Batch ────────────────────────────────────────────────────────
    batch_size:      int = 1
    eval_batch_size: int = 1
    num_workers:     int = 0    # M4 Mac için güvenli

    # ── Epoch & early stopping ───────────────────────────────────────
    num_epochs:              int = 10
    early_stopping_patience: int = 3
    # "mae" | "composite_mae_rmse" | "composite_mae_r2"
    early_stopping_metric:   str = "mae"

    # ── Train metrik hesaplama ───────────────────────────────────────
    # Epoch 1, her train_eval_every_n epoch ve son epoch'ta hesaplanır.
    # Diğer epoch'larda log satırında "—" görünür — kasıtlı davranış.
    train_eval_every_n: int = 2

    # Train eval için subset boyutu.
    # Tüm train setini eval etmek MPS'te çok pahalı (~epoch kadar sürer).
    # 0 → tüm dataset (dikkat: yavaş)
    # 1500 → validation seti boyutuna yakın, hızlı ve temsili
    train_eval_subset_size: int = 1500

    # ── Tekrarlanabilirlik ───────────────────────────────────────────
    seed: int = 42

    # ── Cihaz ────────────────────────────────────────────────────────
    # "auto" | "cuda" | "mps" | "cpu"
    device: str = "auto"

    # ── Path property'ler ────────────────────────────────────────────
    @property
    def monthly_root_path(self) -> Path:
        return Path(self.monthly_root)

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)
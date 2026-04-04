from pathlib import Path
from transformers import AutoTokenizer
from .config import TrainConfig
from .train import train_one_month

if __name__ == "__main__":
    cfg = TrainConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        num_epochs=15,
        loss_type="smoothl1",
        output_root="artifacts/llm_delay_qwen",
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    month_dir = Path("data/model/splits/monthly/2025-12")

    results = train_one_month(
        cfg,
        month_dir,
        tokenizer,
    )

    print(results)
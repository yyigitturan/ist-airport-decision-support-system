from pathlib import Path

from transformers import AutoTokenizer

from .config import TrainConfig
from .train import train_one_month


if __name__ == "__main__":
    cfg = TrainConfig(
        monthly_root="data/model/splits/monthly",
        output_root="artifacts/llm_delay_qwen",
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        scale_target=False,      
        adapter_dropout=0.2,     
        head_dropout=0.2,        
        batch_size=1,
        eval_batch_size=1,
        grad_accum_steps=16,
        num_epochs=5,
        max_active=21,
        max_prior=20,
        use_weather=True,
        use_notam=False,
        device="auto",
    )


    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    month_dir = Path("/Users/YGT/ist-airport-decision-support-system/data/model/splits/monthly/2025-03")

    results = train_one_month(cfg, month_dir, tokenizer)
    print(results)
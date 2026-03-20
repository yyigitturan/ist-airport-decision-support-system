from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    monthly_root: str = "data/model/splits/monthly"
    output_root:  str = "artifacts/llm_delay_qwen"

    model_name:   str = "Qwen/Qwen2.5-0.5B-Instruct"
    target_col:   str = "label_post_terminal_min"

    scale_target: bool = False

    traj_dim:                   int = 256
    llm_hidden_dim:             int = 896
    total_prompt_max_len:       int = 775   
    flight_prompt_max_len:      int = 480
    weather_prompt_max_len:     int = 230
    static_traj_prompt_max_len: int = 65
    max_active:                 int = 21
    max_prior:                  int = 20

    batch_size:           int   = 1
    eval_batch_size:      int   = 1
    grad_accum_steps:     int   = 16
    num_epochs:           int   = 5
    lr:                   float = 1e-5
    weight_decay:         float = 2e-5
    warmup_ratio:         float = 0.06
    adapter_dropout:      float = 0.1
    head_dropout:         float = 0.1

    num_workers:              int  = 0
    seed:                     int  = 42
    use_weather:              bool = True
    use_notam:                bool = False
    early_stopping_patience:  int  = 2

    device: str = "auto"

    @property
    def monthly_root_path(self) -> Path:
        return Path(self.monthly_root)

    @property
    def output_root_path(self) -> Path:
        return Path(self.output_root)
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    max_seq_length: int = 1024


@dataclass
class GRPOConfig:
    num_generations: int = 8
    max_completion_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    learning_rate: float = 2e-5
    beta: float = 0.04
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    log_level: str = "info"
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 3
    eval_strategy: str = "no"
    bf16: bool = True
    output_dir: str = "grpo-gsm8k-output"
    report_to: str = "wandb"
    seed: int = 42


@dataclass
class DataConfig:
    dataset_name: str = "openai/gsm8k"
    dataset_split_train: str = "train"
    dataset_split_test: str = "test"
    max_samples: Optional[int] = 1000
    num_proc: int = 4


@dataclass
class ExperimentConfig:
    dataset_config_name: str = "main"  # GSM8K subset: "main" (numeric answers) or "socratic" (step-by-step reasoning style)
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb_project: str = "slm_model"
    wandb_run_name: Optional[str] = None

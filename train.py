"""Main training script for GRPO fine-tuning on GSM8K."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer

from config import ExperimentConfig
from data_utils import load_gsm8k_dataset
from rewards import (
    correctness_reward,
    format_reward,
    int_reward,
    xmlcount_reward,
)
from utils import print_trainable_parameters, set_seed, setup_wandb


def main() -> None:
    """Run GRPO training on GSM8K."""
    config = ExperimentConfig()

    set_seed(config.grpo.seed)
    setup_wandb(config.wandb_project, config.wandb_run_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = None
    if config.model.use_lora:
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=config.model.lora_r,
            lora_alpha=config.model.lora_alpha,
            lora_dropout=config.model.lora_dropout,
            target_modules=list(config.model.lora_target_modules),
            task_type="CAUSAL_LM",
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        dtype=getattr(torch, config.model.torch_dtype),
        attn_implementation=config.model.attn_implementation,
        device_map="auto"   
    )

    print_trainable_parameters(model)

    train_dataset, test_dataset = load_gsm8k_dataset(config)

    training_args = TRLGRPOConfig(
        output_dir=config.grpo.output_dir,
        num_generations=config.grpo.num_generations,
        max_completion_length=config.grpo.max_completion_length,
        temperature=config.grpo.temperature,
        top_p=config.grpo.top_p,
        learning_rate=config.grpo.learning_rate,
        beta=config.grpo.beta,
        num_train_epochs=config.grpo.num_train_epochs,
        per_device_train_batch_size=config.grpo.per_device_train_batch_size,
        gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
        warmup_ratio=config.grpo.warmup_ratio,
        weight_decay=config.grpo.weight_decay,
        max_grad_norm=config.grpo.max_grad_norm,
        log_level=config.grpo.log_level,
        logging_steps=config.grpo.logging_steps,
        save_steps=config.grpo.save_steps,
        save_total_limit=config.grpo.save_total_limit,
        eval_strategy=config.grpo.eval_strategy,
        bf16=config.grpo.bf16,
        use_cpu=False,
        report_to=config.grpo.report_to,
        seed=config.grpo.seed,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            correctness_reward,
            format_reward,
            int_reward,
            xmlcount_reward,
        ],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
    )

    trainer.train()

    trainer.save_model(config.grpo.output_dir)
    tokenizer.save_pretrained(config.grpo.output_dir)
    print(f"Model saved to {config.grpo.output_dir}")


if __name__ == "__main__":
    main()

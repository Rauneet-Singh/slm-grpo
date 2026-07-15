# slm-grpo

GRPO (Group Relative Policy Optimization) fine-tuning of a small language model (Qwen2.5-0.5B-Instruct) on the GSM8K math reasoning dataset.

---

## Project Structure

```
slm-grpo/
├── config.py          # All hyperparameters and settings as dataclasses
├── data_utils.py      # Dataset loading, prompt formatting, answer extraction
├── rewards.py         # Reward functions used by the GRPO trainer
├── train.py           # Main training entrypoint
├── utils.py           # Seed setting, wandb setup, parameter printing
├── evaluate.py        # Post-training evaluation on GSM8K test set
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## File-by-File Code Explanation

---

## `config.py`

This file defines all experiment settings using Python `dataclass` decorators. Each dataclass auto-generates `__init__`, `__repr__`, and `__eq__` methods from the field declarations.

### Imports

```python
from dataclasses import dataclass, field
```
- `dataclass`: Decorator that turns a class into a mutable data container with auto-generated constructor.
- `field`: Used to specify complex default values (like `default_factory`) for dataclass fields.

```python
from typing import Optional
```
- `Optional`: Type hint indicating a value can be the specified type or `None`.

### `ModelConfig` (lines 5-18)

```python
@dataclass
class ModelConfig:
```
Declares a dataclass for all model-related settings.

```python
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
```
The HuggingFace Hub model identifier. Qwen2.5-0.5B-Instruct is a 500-million parameter instruction-tuned language model from Alibaba.

```python
    torch_dtype: str = "bfloat16"
```
The precision format to load model weights in. Bfloat16 uses 16 bits per weight (vs 32 in float32), halving memory usage while maintaining training stability.

```python
    attn_implementation: str = "sdpa"
```
Selects PyTorch's native Scaled Dot Product Attention kernel. This is a fused, memory-efficient attention implementation faster than naive attention.

```python
    use_lora: bool = True
```
Whether to use LoRA (Low-Rank Adaptation). When enabled, only small adapter matrices are trained instead of the full model, reducing trainable parameters dramatically.

```python
    lora_r: int = 64
```
The rank `r` of the LoRA decomposition. Each weight matrix `W` is updated as `W + BA` where `B` is `(d, r)` and `A` is `(r, d)`. Higher rank = more capacity but more parameters.

```python
    lora_alpha: int = 64
```
The LoRA scaling factor. The actual adapter output is scaled by `alpha/r`. Here `64/64 = 1.0`, so no scaling.

```python
    lora_dropout: float = 0.0
```
Dropout probability applied to LoRA layers during training. `0.0` means no dropout is applied.

```python
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
```
Which linear layers in the transformer receive LoRA adapters:
- `q_proj`, `k_proj`, `v_proj`, `o_proj`: Query, Key, Value, and Output projections in multi-head attention.
- `gate_proj`, `up_proj`, `down_proj`: The three linear layers inside each SwiGLU MLP block.

```python
    max_seq_length: int = 1024
```
Maximum number of tokens in the input sequence (prompt + answer). Sequences longer than this are truncated.

---

### `GRPOConfig` (lines 21-43)

```python
@dataclass
class GRPOConfig:
```
Settings specific to the GRPO reinforcement learning algorithm.

```python
    num_generations: int = 8
```
GRPO generates this many completions per prompt. The group of completions is scored, and their relative rewards are used to compute the policy gradient advantage.

```python
    max_completion_length: int = 512
```
Maximum tokens to generate for each completion. Keeps generation bounded.

```python
    temperature: float = 0.5
```
Controls sampling randomness. Temperature divides the logits before softmax. Lower values (closer to 0) make the model more deterministic; higher values make it more random. 0.5 is moderately deterministic.

```python
    top_p: float = 0.95
```
Nucleus sampling threshold. Only tokens whose cumulative probability reaches 95% are considered for sampling. Filters out low-probability tail tokens.

```python
    learning_rate: float = 3e-8
```
The optimizer learning rate. Extremely small (3 x 10^-8) because RL fine-tuning is fragile -- large updates destroy the base model's capabilities.

```python
    beta: float = 0.04
```
The KL divergence penalty coefficient. During training, the loss includes `beta * KL(policy || reference)`, which penalizes the model for drifting too far from the original pretrained model. Prevents reward hacking.

```python
    num_train_epochs: int = 1
```
Number of complete passes over the training dataset.

```python
    per_device_train_batch_size: int = 2
```
Number of training examples processed per GPU per step.

```python
    gradient_accumulation_steps: int = 4
```
Gradients are accumulated over 4 micro-steps before a weight update. Effective batch size = `per_device_train_batch_size * gradient_accumulation_steps = 2 * 4 = 8`.

```python
    warmup_ratio: float = 0.1
```
The first 10% of total training steps use a linearly increasing learning rate from 0 to the target LR. Helps stabilize early training.

```python
    weight_decay: float = 0.01
```
L2 regularization coefficient. Slightly penalizes large weights to reduce overfitting.

```python
    max_grad_norm: float = 1.0
```
Gradient clipping threshold. If the total gradient norm exceeds 1.0, all gradients are scaled down proportionally. Prevents exploding gradients.

```python
    log_level: str = "info"
```
Sets the verbosity of the HuggingFace Transformers logger.

```python
    logging_steps: int = 1
```
Log training metrics every 1 step.

```python
    save_steps: int = 50
```
Save a model checkpoint every 50 training steps.

```python
    save_total_limit: int = 3
```
Only keep the 3 most recent checkpoints on disk. Older ones are deleted.

```python
    eval_strategy: str = "no"
```
No evaluation is run during training. Evaluation is handled separately by `evaluate.py`.

```python
    bf16: bool = True
```
Enable bfloat16 mixed-precision training. Gradients and activations are computed in bfloat16 to save memory and speed up computation on modern GPUs.

```python
    output_dir: str = "grpo-gsm8k-output"
```
Directory where checkpoints and the final model are saved.

```python
    report_to: str = "wandb"
```
Send training metrics to Weights & Biases for visualization.

```python
    seed: int = 42
```
Random seed for reproducibility. 42 is the classic default.

---

### `DataConfig` (lines 46-53)

```python
@dataclass
class DataConfig:
```
Settings for dataset loading and preprocessing.

```python
    dataset_name: str = "openai/gsm8k"
```
The HuggingFace dataset identifier for GSM8K -- a benchmark of 8,500 grade-school math word problems.

```python
    dataset_split_train: str = "train"
```
Name of the training split in the dataset.

```python
    dataset_split_test: str = "test"
```
Name of the test split in the dataset.

```python
    max_samples: Optional[int] = 1000
```
Cap the training set to 1000 examples. Useful for fast iteration and debugging. Set to `None` to use the full dataset.

```python
    num_proc: int = 4
```
Number of CPU worker processes for parallel dataset mapping.

---

### `ExperimentConfig` (lines 56-61)

```python
@dataclass
class ExperimentConfig:
```
Top-level config that composes all sub-configs into one object.

```python
    model: ModelConfig = field(default_factory=ModelConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    data: DataConfig = field(default_factory=DataConfig)
```
`default_factory` creates a new instance of each sub-config every time an `ExperimentConfig` is instantiated. This avoids the mutable default argument pitfall.

```python
    wandb_project: str = "grpo-gsm8k"
```
The Weights & Biases project name under which all runs are logged.

```python
    wandb_run_name: Optional[str] = None
```
Optional custom name for the wandb run. If `None`, wandb auto-generates a name.

---

## `data_utils.py`

Handles loading the GSM8K dataset, formatting prompts into chat messages, and extracting answers from model text.

### Imports

```python
"""Data loading and preprocessing for the GSM8K math dataset."""
```
Module docstring describing the file's purpose.

```python
import re
```
Regular expression library for pattern matching in text extraction.

```python
from typing import Optional
```
Type hint for nullable return values.

```python
from datasets import load_dataset
```
HuggingFace `datasets` library function to load datasets from the Hub.

```python
from config import ExperimentConfig
```
Imports the project's configuration class.

### Constants

```python
SYSTEM_PROMPT = (
    "You are a helpful math tutor. Think step by step and provide "
    "your final answer after ####. For example: #### 42"
)
```
The system message injected into every prompt. It instructs the model to think step-by-step and place the final answer after `####`. This format is used by the reward functions to extract answers.

```python
HASH_ANSWER_PATTERN = re.compile(r"####\s*([0-9\.\-]+)")
```
Pre-compiled regex pattern that matches:
- `####` -- literal four hashes
- `\s*` -- zero or more whitespace characters
- `([0-9\.\-]+)` -- capture group: one or more digits, dots, or hyphens (handles integers, decimals, negative numbers)

Pre-compiling avoids re-parsing the regex on every call.

### `extract_hash_answer` (lines 18-30)

```python
def extract_hash_answer(text: str) -> Optional[str]:
```
Function signature: takes a string, returns a string or `None`.

```python
    match = HASH_ANSWER_PATTERN.search(text)
```
Searches `text` for the first occurrence of the `#### <number>` pattern. `.search()` scans the entire string (vs `.match()` which only checks the start).

```python
    if match:
        return match.group(1).strip()
```
If a match is found, `group(1)` returns the first captured group (the number string). `.strip()` removes leading/trailing whitespace.

```python
    return None
```
If no match is found, returns `None` to signal failure.

### `extract_final_number` (lines 33-61)

```python
def extract_final_number(text: str) -> Optional[str]:
```
A more robust answer extractor that tries 4 different strategies in order.

```python
    hash_match = HASH_ANSWER_PATTERN.search(text)
    if hash_match:
        return hash_match.group(1).strip()
```
**Strategy 1**: Try the `####` marker. If found, return the number.

```python
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()
```
**Strategy 2**: Try `\boxed{...}` LaTeX notation. Common in math papers and some model outputs.

```python
    answer_match = re.search(r"the answer is[:\s]*([0-9\.\-]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
```
**Strategy 3**: Look for "the answer is: 42" style phrasing. `re.IGNORECASE` makes it case-insensitive.

```python
    numbers = re.findall(r"[-]?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
```
**Strategy 4**: Find all numbers in the text and return the last one. The assumption is that the final number stated is the answer.

```python
    return None
```
If none of the 4 strategies find a number, return `None`.

### `normalize_number` (lines 64-74)

```python
def normalize_number(num_str: str) -> float:
```
Converts a formatted number string into a clean float.

```python
    cleaned = num_str.replace(",", "").replace(" ", "").replace("$", "")
```
Removes commas (e.g. `"1,234"` becomes `"1234"`), spaces, and dollar signs.

```python
    return float(cleaned)
```
Converts the cleaned string to a Python float.

### `format_prompt` (lines 77-89)

```python
def format_prompt(question: str) -> list[dict]:
```
Takes a raw math question and returns a list of chat messages.

```python
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
```
Creates the standard chat format expected by the Qwen2.5 tokenizer:
- First message is the system prompt (instructions)
- Second message is the user's question

This list is later passed to `tokenizer.apply_chat_template()` to produce the final tokenized input.

### `load_gsm8k_dataset` (lines 92-144)

```python
def load_gsm8k_dataset(config: ExperimentConfig) -> tuple:
```
Main data loading function. Returns a `(train_dataset, test_dataset)` tuple.

```python
    dataset = load_dataset(
        config.data.dataset_name,
        split=[
            config.data.dataset_split_train,
            config.data.dataset_split_test,
        ],
    )
```
Loads both train and test splits from the HuggingFace Hub in a single call. Returns a DatasetDict or list of datasets.

```python
    train_dataset = dataset[0]
    test_dataset = dataset[1]
```
Index into the list to get each split separately.

```python
    if config.data.max_samples is not None:
        train_dataset = train_dataset.select(
            range(min(config.data.max_samples, len(train_dataset)))
        )
```
If `max_samples` is set (default: 1000), truncate the training set. `select()` returns a new dataset with only the specified indices.

```python
    def _preprocess(example: dict) -> dict:
        example["prompt"] = format_prompt(example["question"])
        return example
```
Inner function that adds a `"prompt"` column to each example by formatting the `"question"` field into chat messages.

```python
    train_dataset = train_dataset.map(
        _preprocess,
        num_proc=config.data.num_proc,
        desc="Formatting train prompts",
    )
    test_dataset = test_dataset.map(
        _preprocess,
        num_proc=config.data.num_proc,
        desc="Formatting test prompts",
    )
```
Applies `_preprocess` to every example in both datasets. `num_proc=4` parallelizes the operation across 4 CPU workers.

```python
    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in ("prompt", "answer")]
    )
    test_dataset = test_dataset.remove_columns(
        [c for c in test_dataset.column_names if c not in ("prompt", "answer")]
    )
```
Removes all columns except `"prompt"` and `"answer"`. This drops the raw `"question"` field and any other metadata (like `"id"`), keeping the dataset clean for the trainer.

```python
    return train_dataset, test_dataset
```
Returns the processed datasets.

---

## `rewards.py`

Defines the reward functions that the GRPO trainer uses to score model completions. Each function receives a batch of completions and returns a list of float rewards.

### Imports

```python
"""Reward functions for GRPO training."""
```
Module docstring.

```python
import re
```
Regex library for answer extraction.

### `correctness_reward` (lines 6-51)

```python
def correctness_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
```
The primary reward function. Takes a list of model outputs and arbitrary keyword arguments (which TRL populates from dataset columns).

```python
    answers = kwargs.get("answer", [""] * len(completions))
```
Extracts the `"answer"` key from kwargs (ground-truth answers from the dataset). If missing, defaults to a list of empty strings so the loop does not crash.

```python
    rewards = []
    for completion, ground_truth in zip(completions, answers):
```
Iterates over each completion alongside its corresponding ground truth.

```python
        if isinstance(completion, list):
            completion = completion[0]["content"]
```
Handles TRL's message format. Sometimes completions are passed as `[{"role": "assistant", "content": "..."}]` instead of plain strings. This extracts the content string.

```python
        if not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)
```
Ensures ground truth is a string (dataset might store it as int/float).

```python
        hash_match = re.search(r"####\s*([0-9\.\-]+)", completion)
        if not hash_match:
            rewards.append(0.0)
            continue
```
Searches for `#### <number>` in the model output. If not found, award 0.0 and move to the next example.

```python
        extracted = hash_match.group(1).strip().replace(",", "")
        ground_clean = ground_truth.strip().replace(",", "")
```
Extracts the number strings and cleans them (strips whitespace and commas for fair comparison).

```python
        if extracted == ground_clean:
            rewards.append(2.0)
```
First try exact string match. If identical, award 2.0 (the full correctness reward).

```python
        else:
            try:
                if abs(float(extracted) - float(ground_clean)) < 1e-6:
                    rewards.append(2.0)
                else:
                    rewards.append(0.0)
            except ValueError:
                rewards.append(0.0)
```
If strings do not match, try converting both to floats and checking numerical equality with a small tolerance (`1e-6`). This handles formatting differences like `"7"` vs `"7.0"`. If conversion fails (`ValueError`), award 0.0.

```python
    return rewards
```
Returns the list of correctness rewards for the batch.

### `format_reward` (lines 54-83)

```python
def format_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
```
Rewards the model for producing properly structured output.

```python
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"]
```
Same list-unpacking logic as before.

```python
        has_reasoning = (
            "<reasoning>" in completion and "</reasoning>" in completion
        )
        has_answer = "<answer>" in completion and "</answer>" in completion
```
Checks whether both opening and closing tags are present for `<reasoning>` and `<answer>`.

```python
        if has_reasoning and has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
```
Award 0.5 if both tag pairs are present, 0.0 otherwise. This encourages the model to separate its thinking from its answer.

```python
    return rewards
```

### `int_reward` (lines 86-117)

```python
def int_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
```
Rewards completions where the answer inside `<answer>` tags is a valid integer.

```python
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"]
```
Standard iteration and unpacking.

```python
        answer_match = re.search(r"<answer>([^<]+)</answer>", completion)
```
Extracts the text content between `<answer>` and `</answer>` tags. `[^<]+` ensures we do not match across tag boundaries.

```python
        if answer_match:
            content = answer_match.group(1).strip()
            try:
                int(content)
                rewards.append(0.5)
            except ValueError:
                rewards.append(0.0)
```
If answer tags are found, tries to convert their content to an integer. If successful, awards 0.5. If not a valid integer (e.g. `"3.5"` or `"hello"`), awards 0.0.

```python
        else:
            rewards.append(0.0)
    return rewards
```
If no `<answer>` tags found at all, awards 0.0.

### `xmlcount_reward` (lines 120-156)

```python
def xmlcount_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
```
Rewards structural completeness -- how many of the expected formatting elements are present.

```python
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"]
```
Standard iteration and unpacking.

```python
        count = 0
        if "<reasoning>" in completion:
            count += 1
        if "</reasoning>" in completion:
            count += 1
        if "<answer>" in completion:
            count += 1
        if "</answer>" in completion:
            count += 1
        if "####" in completion:
            count += 1
```
Counts how many of these 5 structural markers are present: `<reasoning>`, `</reasoning>`, `<answer>`, `</answer>`, `####`. Each found element increments the counter by 1.

```python
        reward = min(0.5, count * 0.1)
        rewards.append(reward)
```
Each element is worth 0.1, capped at 0.5 maximum. So finding all 5 gives 0.5, finding 3 gives 0.3, etc.

```python
    return rewards
```

---

## `train.py`

The main training script. Run this file to start GRPO training.

### Imports (lines 1-16)

```python
"""Main training script for GRPO fine-tuning on GSM8K."""
```
Module docstring.

```python
import torch
```
PyTorch -- needed for `torch.bfloat16` and `getattr(torch, ...)`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```
- `AutoModelForCausalLM`: Automatically loads the correct model class for any causal LM on HuggingFace Hub.
- `AutoTokenizer`: Automatically loads the matching tokenizer.

```python
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer
```
- `TRLGRPOConfig`: TRL's configuration class for GRPO training (aliased to avoid name conflict with our own `GRPOConfig`).
- `GRPOTrainer`: TRL's trainer that implements the GRPO algorithm.

```python
from config import ExperimentConfig
from data_utils import load_gsm8k_dataset
from rewards import (
    correctness_reward,
    format_reward,
    int_reward,
    xmlcount_reward,
)
from utils import print_trainable_parameters, set_seed, setup_wandb
```
Project imports: config, data loading, all 4 reward functions, and utilities.

### `main()` (lines 19-101)

```python
def main() -> None:
```
The main training function. No return value.

```python
    config = ExperimentConfig()
```
Creates an `ExperimentConfig` with all default values (no arguments means all defaults from the dataclasses).

```python
    set_seed(config.grpo.seed)
```
Seeds all random number generators (Python, NumPy, PyTorch) for reproducibility.

```python
    setup_wandb(config.wandb_project, config.wandb_run_name)
```
Sets wandb environment variables so TRL auto-logs to the correct project.

```python
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
```
Downloads and loads the Qwen2.5 tokenizer from HuggingFace Hub.

```python
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
```
Many causal LMs do not have a pad token. Setting `pad_token = eos_token` allows the tokenizer to pad shorter sequences to the same length for batching. The EOS token is used because it signals "end of text" and is a safe choice.

```python
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
```
Conditionally creates a LoRA configuration:
- `from peft import LoraConfig` -- imports `peft` library only when LoRA is enabled (lazy import saves time if LoRA is disabled).
- `r` -- rank of the low-rank decomposition matrices.
- `lora_alpha` -- scaling factor applied to adapter outputs.
- `lora_dropout` -- dropout rate within adapter layers.
- `target_modules` -- which linear layers to attach adapters to (converted from tuple to list since peft expects a list).
- `task_type="CAUSAL_LM"` -- tells peft this is a causal language modeling task (affects how weight merges work).

```python
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        dtype=getattr(torch, config.model.torch_dtype),
        attn_implementation=config.model.attn_implementation,
        device_map="auto"
    )
```
Loads the pretrained model:
- `getattr(torch, config.model.torch_dtype)` -- converts the string `"bfloat16"` to the actual `torch.bfloat16` dtype object.
- `attn_implementation="sdpa"` -- uses PyTorch's fused scaled dot product attention kernel.
- `device_map="auto"` -- automatically distributes model across available GPUs (or places on single GPU/CPU).

```python
    print_trainable_parameters(model)
```
Prints how many parameters are trainable vs total (with LoRA, typically <1%).

```python
    train_dataset, test_dataset = load_gsm8k_dataset(config)
```
Loads and preprocesses the GSM8K dataset.

```python
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
```
Maps all our config values into TRL's GRPOConfig:
- `use_cpu=False` -- forces GPU usage.
- `remove_unused_columns=False` -- keeps all dataset columns so reward functions can access them.

```python
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
```
Creates the GRPO trainer:
- `model` -- the base Qwen2.5 model.
- `processing_class` -- the tokenizer (used for encoding/decoding).
- `reward_funcs` -- list of 4 reward functions. During training, for each prompt, the model generates 8 completions, each is scored by all 4 functions, and the rewards are summed to produce a total reward per completion. The GRPO algorithm uses these relative rewards to update the policy.
- `args` -- the training configuration.
- `train_dataset` -- the preprocessed GSM8K training data.
- `peft_config` -- the LoRA config. If `None`, full fine-tuning is performed.

```python
    trainer.train()
```
Runs the full GRPO training loop. This involves:
1. Sampling prompts from the dataset
2. Generating 8 completions per prompt
3. Scoring each completion with all 4 reward functions
4. Computing group-relative advantages
5. Updating the policy using the GRPO loss

```python
    trainer.save_model(config.grpo.output_dir)
```
Saves the fine-tuned model weights (only the LoRA adapter if LoRA is used) to the output directory.

```python
    tokenizer.save_pretrained(config.grpo.output_dir)
```
Saves the tokenizer files (vocab, merges, special tokens) to the same directory.

```python
    print(f"Model saved to {config.grpo.output_dir}")
```
Confirmation message.

---

## `utils.py`

Utility functions used across the project.

### Imports

```python
"""Utility functions for the GRPO training project."""
```
Module docstring.

```python
import os
```
Operating system interface -- used to set environment variables for wandb.

```python
import random
```
Python's built-in random number generator.

```python
import numpy as np
```
NumPy -- used for numerical operations and has its own random state.

```python
import torch
```
PyTorch -- has its own random state for GPU operations.

```python
import wandb
```
Weights & Biases library (imported here but only env vars are set; TRL handles actual logging).

### `setup_wandb` (lines 11-20)

```python
def setup_wandb(project: str = "grpo-gsm8k", run_name: str | None = None) -> None:
```
Initializes wandb configuration via environment variables. `str | None` is Python 3.10+ union type syntax.

```python
    os.environ["WANDB_PROJECT"] = project
```
Sets the wandb project name. TRL reads this environment variable to know where to log.

```python
    if run_name:
        os.environ["WANDB_NAME"] = run_name
```
Optionally sets a custom run name for the wandb dashboard.

### `set_seed` (lines 23-33)

```python
def set_seed(seed: int = 42) -> None:
```
Sets the random seed across all libraries for full reproducibility.

```python
    random.seed(seed)
```
Seeds Python's built-in `random` module (used by many libraries internally).

```python
    np.random.seed(seed)
```
Seeds NumPy's random number generator.

```python
    torch.manual_seed(seed)
```
Seeds PyTorch's CPU random number generator.

```python
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```
Seeds all CUDA GPU random number generators. `manual_seed_all` handles multi-GPU setups.

### `print_trainable_parameters` (lines 36-49)

```python
def print_trainable_parameters(model: torch.nn.Module) -> None:
```
Inspects a model and prints its parameter counts.

```python
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
Counts trainable parameters: iterates all parameters, filters to those with `requires_grad=True` (which LoRA adapters have), sums their element counts (`.numel()` = number of elements).

```python
    total = sum(p.numel() for p in model.parameters())
```
Counts total parameters regardless of whether they are frozen or trainable.

```python
    trainable_pct = 100 * trainable / total if total > 0 else 0
```
Computes the percentage of trainable parameters. Safe division check for empty models.

```python
    print(
        f"Trainable params: {trainable:,} || "
        f"All params: {total:,} || "
        f"Trainable%: {trainable_pct:.2f}%"
    )
```
Prints formatted output with comma-separated numbers and 2-decimal percentage. Example output: `Trainable params: 12,582,912 || All params: 497,704,960 || Trainable%: 2.53%`

---

## `evaluate.py`

Post-training evaluation script. Loads the trained model and tests it on GSM8K.

### Imports

```python
"""Evaluation script for GRPO-trained models on GSM8K."""
```
Module docstring.

```python
import re
```
Regex for answer extraction.

```python
from typing import Optional
```
Type hint for nullable values.

```python
import torch
```
PyTorch for model inference.

```python
from datasets import load_dataset
```
To load the GSM8K test split.

```python
from tqdm import tqdm
```
Progress bar library for visual feedback during evaluation.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```
To load the trained model and tokenizer.

```python
from config import ExperimentConfig
from data_utils import SYSTEM_PROMPT
```
Project config and the system prompt constant.

### `extract_answer` (lines 15-39)

```python
def extract_answer(text: str) -> Optional[str]:
```
Extracts a numerical answer from model output text using 3 strategies.

```python
    hash_match = re.search(r"####\s*([0-9\.\-]+)", text)
    if hash_match:
        return hash_match.group(1).strip().replace(",", "")
```
**Strategy 1**: Look for `#### <number>` marker. Strip whitespace and remove commas from the result.

```python
    answer_match = re.search(r"<answer>([^<]+)</answer>", text)
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")
```
**Strategy 2**: Look for content inside `<answer>` tags. This handles the XML-style format the reward functions encourage.

```python
    numbers = re.findall(r"[-]?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]
```
**Strategy 3**: Find all numbers in the text and return the last one (fallback).

```python
    return None
```
If no answer found, return `None`.

### `evaluate` (lines 42-121)

```python
def evaluate(
    model_path: str,
    config: ExperimentConfig,
    max_samples: Optional[int] = None,
) -> float:
```
Evaluates a trained model. Returns accuracy as a float between 0 and 1.

```python
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
```
Loads the tokenizer from the saved checkpoint and sets the pad token.

```python
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
```
Loads the fine-tuned model in bfloat16 precision on the best available device.

```python
    model.eval()
```
Sets the model to evaluation mode. Disables dropout layers and other training-specific behaviors.

```python
    dataset = load_dataset(
        config.data.dataset_name,
        split=config.data.dataset_split_test,
    )
```
Loads the GSM8K test split.

```python
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
```
Caps evaluation to `max_samples` examples (default: 500 in `main()`).

```python
    correct = 0
    total = 0
```
Counters for accuracy calculation.

```python
    for example in tqdm(dataset, desc="Evaluating"):
```
Iterates over each test example with a progress bar.

```python
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
```
Creates the chat-format messages for this question.

```python
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
```
Applies the Qwen2.5 chat template to produce the formatted input string. `tokenize=False` returns text instead of token IDs. `add_generation_prompt=True` appends the assistant turn prefix so the model knows to start generating.

```python
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
```
Tokenizes the formatted text into PyTorch tensors and moves them to the same device as the model (GPU).

```python
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.model.max_seq_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
```
Generates a completion:
- `torch.no_grad()` -- disables gradient computation (saves memory during inference).
- `**inputs` -- unpacks `input_ids` and `attention_mask` as keyword arguments.
- `max_new_tokens=1024` -- maximum tokens to generate beyond the prompt.
- `temperature=0.7` -- slightly more random than training (0.5) to get a realistic assessment.
- `top_p=0.95` -- nucleus sampling.
- `do_sample=True` -- enables sampling (vs greedy decoding).

```python
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
```
Decodes only the generated tokens (not the prompt):
- `outputs[0]` -- first (and only) sequence in the batch.
- `[inputs["input_ids"].shape[1]:]` -- slices off the prompt tokens, keeping only the newly generated tokens.
- `skip_special_tokens=True` -- removes special tokens like `<eos>` from the output string.

```python
        predicted = extract_answer(generated)
```
Extracts the predicted answer from the generated text.

```python
        ground_truth = example["answer"].strip()
```
Gets the ground truth answer string from the dataset.

```python
        hash_match = re.search(r"####\s*([0-9\.\-]+)", ground_truth)
        if hash_match:
            ground_truth = hash_match.group(1).strip().replace(",", "")
```
Extracts just the number from the ground truth (GSM8K answers are in the format `"#### 42"`).

```python
        if predicted is not None and predicted == ground_truth:
            correct += 1
```
Compares prediction to ground truth. If they match, increment the correct counter.

```python
        total += 1
```
Increment total counter regardless of correctness.

```python
    accuracy = correct / total if total > 0 else 0.0
```
Compute accuracy as the ratio of correct predictions. Safe division for empty datasets.

```python
    print(f"\nResults: {correct}/{total} correct")
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy
```
Prints results and returns the accuracy float.

### `main()` (lines 124-133)

```python
def main() -> None:
```
Entry point for evaluation.

```python
    config = ExperimentConfig()
```
Creates default config.

```python
    model_path = config.grpo.output_dir
```
Uses the same output directory as training (`"grpo-gsm8k-output"`) to find the saved model.

```python
    accuracy = evaluate(model_path, config, max_samples=500)
```
Evaluates on up to 500 test examples.

```python
    print(f"\nFinal accuracy: {accuracy:.2%}")
```
Prints the final accuracy.

### Script execution guard

```python
if __name__ == "__main__":
    main()
```
Only runs `main()` when the script is executed directly (not when imported as a module).

---

## Summary

This project fine-tunes a 0.5B parameter language model using GRPO, a reinforcement learning algorithm that:

1. **Generates** multiple completions per prompt (8 by default)
2. **Scores** each completion using 4 reward functions (correctness + format + integer + structure)
3. **Updates** the model to favor high-reward completions while staying close to the base model (KL penalty)

The 4 reward functions and their maximum contributions per completion:
| Reward Function | Max Reward | What it checks |
|---|---|---|
| `correctness_reward` | 2.0 | Answer matches ground truth |
| `format_reward` | 0.5 | Has `<reasoning>` and `<answer>` tags |
| `int_reward` | 0.5 | Answer is a valid integer |
| `xmlcount_reward` | 0.5 | Structural completeness (5 markers) |
| **Total** | **3.5** | |

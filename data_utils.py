"""Data loading and preprocessing for the GSM8K math dataset."""

import re
from typing import Optional

from datasets import load_dataset

from config import ExperimentConfig

SYSTEM_PROMPT = (
    "You are a helpful math tutor. Think step by step and provide "
    "your final answer after ####. For example: #### 42"
)

HASH_ANSWER_PATTERN = re.compile(r"####\s*([0-9\.\-]+)")


def extract_hash_answer(text: str) -> Optional[str]:
    """Extract the answer following #### from model output.

    Args:
        text: The model output text.

    Returns:
        The extracted answer string, or None if not found.
    """
    match = HASH_ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return None


def extract_final_number(text: str) -> Optional[str]:
    """Extract the final numerical answer from text using multiple patterns.

    Tries several patterns in order: #### marker, boxed notation,
    'the answer is' phrasing, or the last number in the text.

    Args:
        text: The text to extract a number from.

    Returns:
        The extracted number as a string, or None if not found.
    """
    hash_match = HASH_ANSWER_PATTERN.search(text)
    if hash_match:
        return hash_match.group(1).strip()

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match.group(1).strip()

    answer_match = re.search(r"the answer is[:\s]*([0-9\.\-]+)", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()

    numbers = re.findall(r"[-]?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None


def normalize_number(num_str: str) -> float:
    """Normalize a number string by removing commas and whitespace.

    Args:
        num_str: The number string to normalize.

    Returns:
        The normalized number as a float.
    """
    cleaned = num_str.replace(",", "").replace(" ", "").replace("$", "")
    return float(cleaned)


def format_prompt(question: str) -> list[dict]:
    """Format a question into chat messages for the Qwen2.5 template.

    Args:
        question: The math question to format.

    Returns:
        A list of message dictionaries in chat format.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def load_gsm8k_dataset(
    config: ExperimentConfig,
) -> tuple:
    """Load and preprocess the GSM8K dataset.

    Loads the dataset, formats prompts using the Qwen2.5 chat template,
    and removes unused columns.

    Args:
        config: The experiment configuration.

    Returns:
        A tuple of (train_dataset, test_dataset).
    """
    dataset = load_dataset(
        config.data.dataset_name,
        split=[
            config.data.dataset_split_train,
            config.data.dataset_split_test,
        ],
    )

    train_dataset = dataset[0]
    test_dataset = dataset[1]

    if config.data.max_samples is not None:
        train_dataset = train_dataset.select(
            range(min(config.data.max_samples, len(train_dataset)))
        )

    def _preprocess(example: dict) -> dict:
        example["prompt"] = format_prompt(example["question"])
        return example

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

    train_dataset = train_dataset.remove_columns(
        [c for c in train_dataset.column_names if c not in ("prompt", "answer")]
    )
    test_dataset = test_dataset.remove_columns(
        [c for c in test_dataset.column_names if c not in ("prompt", "answer")]
    )

    return train_dataset, test_dataset

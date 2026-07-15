"""Evaluation script for GRPO-trained models on GSM8K."""

import re
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ExperimentConfig
from data_utils import SYSTEM_PROMPT


def extract_answer(text: str) -> Optional[str]:
    """Extract the numerical answer from model output.

    Tries #### marker first, then <answer> tags, then falls back
    to the last number in the text.

    Args:
        text: The model output text.

    Returns:
        The extracted answer as a string, or None if not found.
    """
    hash_match = re.search(r"####\s*([0-9\.\-]+)", text)
    if hash_match:
        return hash_match.group(1).strip().replace(",", "")

    answer_match = re.search(r"<answer>([^<]+)</answer>", text)
    if answer_match:
        return answer_match.group(1).strip().replace(",", "")

    numbers = re.findall(r"[-]?\d+\.?\d*", text)
    if numbers:
        return numbers[-1]

    return None


def evaluate(
    model_path: str,
    config: ExperimentConfig,
    max_samples: Optional[int] = None,
) -> float:
    """Evaluate a trained model on the GSM8K test set.

    Args:
        model_path: Path to the saved model checkpoint.
        config: The experiment configuration.
        max_samples: Maximum number of samples to evaluate on.

    Returns:
        The accuracy as a float between 0 and 1.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    dataset = load_dataset(
    config.data.dataset_name,
    config.data.dataset_config_name,  # "main" or "socratic"
    split=config.data.dataset_split_test,
)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    for example in tqdm(dataset, desc="Evaluating"):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]

        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.model.max_seq_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        predicted = extract_answer(generated)
        ground_truth = example["answer"].strip()

        hash_match = re.search(r"####\s*([0-9\.\-]+)", ground_truth)
        if hash_match:
            ground_truth = hash_match.group(1).strip().replace(",", "")

        if predicted is not None and predicted == ground_truth:
            correct += 1

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nResults: {correct}/{total} correct")
    print(f"Accuracy: {accuracy:.2%}")
    return accuracy


def main() -> None:
    """Run evaluation on the latest trained model."""
    config = ExperimentConfig()
    model_path = config.grpo.output_dir
    accuracy = evaluate(model_path, config, max_samples=500)
    print(f"\nFinal accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()

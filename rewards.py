"""Reward functions for GRPO training."""

import re


def correctness_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
    """Reward completions based on correctness of the extracted answer.

    Compares the extracted answer (after ####) against the ground truth
    from the dataset. Awards 2.0 for correct, 0.0 for incorrect.

    Args:
        completions: List of model completion strings.
        **kwargs: Must contain 'answer' key with ground truth answers.

    Returns:
        List of reward values.
    """
    answers = kwargs.get("answer", [""] * len(completions))
    rewards = []

    for completion, ground_truth in zip(completions, answers):
        if isinstance(completion, list):
            completion = completion[0]["content"]

        if not isinstance(ground_truth, str):
            ground_truth = str(ground_truth)

        hash_match = re.search(r"####\s*([0-9\.\-]+)", completion)
        if not hash_match:
            rewards.append(0.0)
            continue

        extracted = hash_match.group(1).strip().replace(",", "")
        ground_clean = ground_truth.strip().replace(",", "")

        if extracted == ground_clean:
            rewards.append(2.0)
        else:
            try:
                if abs(float(extracted) - float(ground_clean)) < 1e-6:
                    rewards.append(2.0)
                else:
                    rewards.append(0.0)
            except ValueError:
                rewards.append(0.0)

    return rewards


def format_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
    """Reward completions for proper formatting with thinking and answer tags.

    Checks that the output contains <reasoning>...</reasoning> and
    <answer>...</answer> tags. Awards 0.5 if present.

    Args:
        completions: List of model completion strings.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        List of reward values.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"]

        has_reasoning = (
            "<reasoning>" in completion and "</reasoning>" in completion
        )
        has_answer = "<answer>" in completion and "</answer>" in completion
        if has_reasoning and has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def int_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
    """Reward completions that provide a plain integer as the final answer.

    Extracts the content within <answer> tags and checks if it is a
    valid integer. Awards 0.5 if so.

    Args:
        completions: List of model completion strings.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        List of reward values.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"]

        answer_match = re.search(r"<answer>([^<]+)</answer>", completion)
        if answer_match:
            content = answer_match.group(1).strip()
            try:
                int(content)
                rewards.append(0.5)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def xmlcount_reward(
    completions: list[str],
    **kwargs,
) -> list[float]:
    """Reward completions for structural completeness of XML-style tags.

    Awards up to 0.5 based on how many of the expected structural
    elements are present: <reasoning>, </reasoning>, <answer>, </answer>,
    and #### marker.

    Args:
        completions: List of model completion strings.
        **kwargs: Additional keyword arguments (unused).

    Returns:
        List of reward values.
    """
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[0]["content"]

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

        reward = min(0.5, count * 0.1)
        rewards.append(reward)
    return rewards
PYEOF

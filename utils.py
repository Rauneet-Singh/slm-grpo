"""Utility functions for the GRPO training project."""

import os
import random

import numpy as np
import torch
import wandb


def setup_wandb(project: str = "grpo-gsm8k", run_name: str | None = None) -> None:
    """Initialize Weights & Biases logging.

    Args:
        project: The wandb project name.
        run_name: Optional name for the run.
    """
    os.environ["WANDB_PROJECT"] = project
    if run_name:
        os.environ["WANDB_NAME"] = run_name


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """Print the number and percentage of trainable parameters.

    Args:
        model: The PyTorch model to inspect.
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable / total if total > 0 else 0
    print(
        f"Trainable params: {trainable:,} || "
        f"All params: {total:,} || "
        f"Trainable%: {trainable_pct:.2f}%"
    )

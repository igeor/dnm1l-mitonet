"""Small, self-contained helpers used across the pipeline."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG involved in training for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(message: str, log_file: str | None = None) -> None:
    """Print a message and optionally append it to a log file."""
    print(message)
    if log_file:
        with open(log_file, "a") as f:
            f.write(message + "\n")


def bit_depth_scaling(array: np.ndarray, bit_depth: int) -> np.ndarray:
    """Clip to [0, 2**bit_depth - 1] and scale to a [0, 1] float array."""
    assert bit_depth > 0, "bit_depth must be positive"
    max_value = 2 ** bit_depth - 1
    array = np.clip(array, 0, max_value)
    return array.astype(np.float32) / max_value

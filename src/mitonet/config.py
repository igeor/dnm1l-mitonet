"""Experiment configuration.

A single `Config` covers both training strategies described in the paper:

- Fine-Tuning (FT / Base): train once at full resolution.
  -> image_sizes=[1024], num_epochs=[10]
- Curriculum-learning Fine-Tuning (CFT): train progressively from low to
  high resolution, reusing the previous stage's weights.
  -> image_sizes=[128, 256, 512, 1024], num_epochs=[4, 6, 8, 10]

See configs/fine_tuning.json and configs/curriculum_fine_tuning.json for the
exact settings used in the paper.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # Experiment identity
    alias: str = "experiment"

    # Paths (all relative to the repo root by default)
    data_root: str = "data"
    results_dir: str = "results"

    # Reproducibility / device
    seed: int = 42
    device: str = "cuda"

    # Classification setup
    num_classes: int = 3  # CONTROL, DNM1L, OTHERMIT
    class_names: List[str] = field(default_factory=lambda: ["CONTROL", "DNM1L", "OTHERMIT"])

    # Cross-validation
    num_splits: int = 5
    train_ratio: float = 0.8

    # Curriculum schedule: one entry per stage.
    # A plain fine-tuning run is the special case of a single stage.
    image_sizes: List[int] = field(default_factory=lambda: [1024])
    num_epochs: List[int] = field(default_factory=lambda: [10])

    # Optimization
    train_batch_size: int = 4
    eval_batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-3
    scheduler_step_size: int = 1
    scheduler_gamma: float = 0.9

    # Model
    model_name: str = "efficientnet_b0"
    dense_layers: List[int] = field(default_factory=lambda: [128])
    dropout: float = 0.35
    pretrained: bool = True
    freeze_backbone: bool = False

    def __post_init__(self):
        if len(self.image_sizes) != len(self.num_epochs):
            raise ValueError("image_sizes and num_epochs must have the same length")

    @property
    def experiment_dir(self) -> Path:
        return Path(self.results_dir) / self.alias

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path) as f:
            payload = json.load(f)
        return cls(**payload)

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

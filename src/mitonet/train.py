"""Training entrypoint: implements both FT and CFT as a single curriculum.

A plain fine-tuning run is a curriculum with one stage; a curriculum
fine-tuning run is a curriculum with several increasing image sizes, each
stage's model initialized from the previous stage's checkpoint. Which one
runs is controlled entirely by `Config.image_sizes` / `Config.num_epochs`
(see configs/fine_tuning.json and configs/curriculum_fine_tuning.json).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from .config import Config
from .dataset import MitoNetworkDataset, make_cv_splits
from .metrics import compute_metrics
from .model import EfficientNetClassifier
from .utils import log, set_seed

TRAIN_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomVerticalFlip(),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(degrees=(-45, 45)),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
])

EVAL_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _ensure_splits(config: Config) -> None:
    """Create the k-fold split CSVs next to metadata.csv if missing."""
    existing = glob.glob(os.path.join(config.data_root, "training_data_*.csv"))
    if len(existing) >= config.num_splits:
        return
    metadata_csv = os.path.join(config.data_root, "metadata.csv")
    make_cv_splits(metadata_csv, config.data_root, num_splits=config.num_splits, seed=config.seed)


def run_training(config: Config) -> dict:
    set_seed(config.seed)
    device = config.device if torch.cuda.is_available() else "cpu"

    experiment_dir = config.experiment_dir
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.to_json(experiment_dir / "config.json")
    log_file = str(experiment_dir / "log.txt")

    _ensure_splits(config)

    all_stage_summaries = {}

    for stage_idx, (image_size, epochs) in enumerate(zip(config.image_sizes, config.num_epochs)):
        prev_image_size = config.image_sizes[stage_idx - 1] if stage_idx > 0 else None
        stage_metrics = []

        for split_idx in range(config.num_splits):
            model = EfficientNetClassifier(
                model_name=config.model_name,
                num_classes=config.num_classes,
                image_size=image_size,
                pretrained=config.pretrained,
                freeze_backbone=config.freeze_backbone,
                dense_layers=config.dense_layers,
                dropout=config.dropout,
            ).to(device)

            # Warm-start from the previous curriculum stage's checkpoint.
            prev_ckpt = checkpoint_dir / f"split{split_idx}_res{prev_image_size}.pt"
            if prev_ckpt.exists():
                model.load_state_dict(torch.load(prev_ckpt, map_location=device))

            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            scheduler = None
            if config.scheduler_step_size:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
                )
            criterion = torch.nn.CrossEntropyLoss()

            train_dataset = MitoNetworkDataset(
                os.path.join(config.data_root, f"training_data_{split_idx}.csv"),
                class_names=config.class_names,
                transform=TRAIN_TRANSFORM,
            )
            test_dataset = MitoNetworkDataset(
                os.path.join(config.data_root, f"test_data_{split_idx}.csv"),
                class_names=config.class_names,
                transform=EVAL_TRANSFORM,
            )
            train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

            for epoch in range(epochs):
                epoch_loss = model.fit(train_loader, optimizer, criterion, device=device)
                if scheduler:
                    scheduler.step()
                log(
                    f"[res={image_size}] split {split_idx + 1}/{config.num_splits} "
                    f"epoch {epoch + 1}/{epochs} train_loss={epoch_loss:.3f}",
                    log_file,
                )

            torch.save(model.state_dict(), checkpoint_dir / f"split{split_idx}_res{image_size}.pt")

            real_probs, pred_probs = model.evaluate(test_loader, device=device)
            metrics = compute_metrics(real_probs, pred_probs, multi_class=config.num_classes > 2)
            stage_metrics.append(metrics)
            log(f"[res={image_size}] split {split_idx + 1} test metrics: {metrics}", log_file)

        summary = {
            key: (float(np.mean([m[key] for m in stage_metrics])), float(np.std([m[key] for m in stage_metrics])))
            for key in stage_metrics[0]
        }
        all_stage_summaries[image_size] = summary
        log(f"[res={image_size}] cross-validated summary (mean, std): {summary}\n", log_file)

    with open(experiment_dir / "summary.json", "w") as f:
        json.dump(all_stage_summaries, f, indent=2)

    return all_stage_summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON config file")
    args = parser.parse_args()

    config = Config.from_json(args.config)
    run_training(config)


if __name__ == "__main__":
    main()

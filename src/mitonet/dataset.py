"""Dataset and cross-validation split creation for the classification task.

Expects a `metadata.csv` with, at minimum, the columns:
    id     -- subject/sample identifier (used to prevent patient leakage)
    date   -- acquisition batch/date (combined with `id` to key a fold group)
    label  -- one of the class names in Config.class_names
    path   -- absolute or repo-relative path to the image file
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import PIL.Image as Image
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset


class MitoNetworkDataset(Dataset):
    """Loads mitochondrial-network images and one-hot-free integer labels."""

    def __init__(self, csv, class_names, transform=None):
        self.data = pd.read_csv(csv) if isinstance(csv, (str, Path)) else csv
        self.class_names = [c.lower() for c in class_names]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self._encode_label(row["label"])

    def _encode_label(self, label: str) -> int:
        label = str(label).lower()
        if label not in self.class_names:
            raise ValueError(f"Unknown label '{label}', expected one of {self.class_names}")
        return self.class_names.index(label)


def make_cv_splits(metadata_csv, output_dir, num_splits=5, seed=42):
    """Create stratified, group-aware (per-subject) k-fold train/test CSVs.

    Groups rows by `id`+`date` so that all images from the same acquisition
    of the same subject fall in the same fold, then stratifies folds by
    class label. Writes `training_data_{fold}.csv` / `test_data_{fold}.csv`
    into `output_dir` and returns the list of (train_path, test_path) pairs.
    """
    metadata_df = pd.read_csv(metadata_csv)
    metadata_df["id_date"] = metadata_df["id"].astype(str) + "_" + metadata_df["date"].astype(str)

    grouped = metadata_df[["id_date", "label"]].drop_duplicates()

    os.makedirs(output_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)

    split_paths = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(grouped, grouped["label"])):
        train_groups = grouped.iloc[train_idx]["id_date"]
        test_groups = grouped.iloc[test_idx]["id_date"]

        train_df = metadata_df[metadata_df["id_date"].isin(train_groups)]
        test_df = metadata_df[metadata_df["id_date"].isin(test_groups)]

        train_path = os.path.join(output_dir, f"training_data_{fold_idx}.csv")
        test_path = os.path.join(output_dir, f"test_data_{fold_idx}.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        split_paths.append((train_path, test_path))

        print(f"Fold {fold_idx}: train={len(train_df)} ({train_df['label'].value_counts().to_dict()}), "
              f"test={len(test_df)} ({test_df['label'].value_counts().to_dict()})")

    return split_paths

"""Dataset and cross-validation split creation for the classification task.

Expects a `metadata.csv` with, at minimum, the columns:
    id     -- subject/sample identifier (used to prevent patient leakage)
    date   -- acquisition batch/date (combined with `id` to key a fold group
              when group_by="id_date")
    label  -- one of the class names in Config.class_names
    path   -- absolute or repo-relative path to the image file

See `make_cv_splits` for the two available group-by strategies.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import numpy as np
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


def _write_fold(output_dir, fold_idx, train_df, test_df):
    train_path = os.path.join(output_dir, f"training_data_{fold_idx}.csv")
    test_path = os.path.join(output_dir, f"test_data_{fold_idx}.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Fold {fold_idx}: train={len(train_df)} ({train_df['label'].value_counts().to_dict()}), "
          f"test={len(test_df)} ({test_df['label'].value_counts().to_dict()})")
    return train_path, test_path


def _splits_by_id_date(metadata_df, output_dir, num_splits, seed):
    """Stratified k-fold, grouped by `id`+`date` (per-acquisition, not per-subject).

    All images from the same subject on the same acquisition date fall in the
    same fold, and folds are stratified by class label via sklearn's
    StratifiedKFold. The same subject CAN appear in both train and test of a
    given fold if they were imaged on different dates -- by design, not a bug:
    this key intentionally treats each acquisition/session as the unit of
    leakage control (the common practice when different sessions of the same
    subject are treated as independent samples), rather than the whole subject.
    """
    metadata_df = metadata_df.copy()
    metadata_df["id_date"] = metadata_df["id"].astype(str) + "_" + metadata_df["date"].astype(str)
    grouped = metadata_df[["id_date", "label"]].drop_duplicates()

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)

    split_paths = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(grouped, grouped["label"])):
        train_groups = grouped.iloc[train_idx]["id_date"]
        test_groups = grouped.iloc[test_idx]["id_date"]

        train_df = metadata_df[metadata_df["id_date"].isin(train_groups)]
        test_df = metadata_df[metadata_df["id_date"].isin(test_groups)]

        split_paths.append(_write_fold(output_dir, fold_idx, train_df, test_df))

    return split_paths


def _assign_test_ids_per_fold(ids, num_splits, rng):
    """Assign each id to exactly one fold's test set.

    If there are enough ids (>= num_splits), partition them into disjoint
    groups (no repetition). If there are fewer ids than num_splits (e.g. only
    3 CONTROL ids for 5 folds), cycle through a shuffled repetition of the ids
    so each id is used as test as evenly as possible (counts differ by at
    most 1) instead of being sampled independently per fold, which can be
    skewed by chance. The same id is always excluded from its own fold's
    train set, so there is never leakage within a single fold.
    """
    ids = np.array(ids)

    if len(ids) >= num_splits:
        rng.shuffle(ids)
        chunks = np.array_split(ids, num_splits)
        return [list(chunk) for chunk in chunks]

    n_cycles = -(-num_splits // len(ids))  # ceil division
    sequence = []
    for _ in range(n_cycles):
        cycle = ids.copy()
        rng.shuffle(cycle)
        sequence.extend(cycle)
    sequence = sequence[:num_splits]
    return [[test_id] for test_id in sequence]


def _splits_by_id(metadata_df, output_dir, num_splits, seed):
    """Subject-level k-fold, grouped by `id` only (strict, no cross-date leakage).

    Every image from a given subject -- across all acquisition dates -- lands
    in the same role (train or test) within a fold, so a subject can never
    appear in both. Test-id assignment is balanced per class label so that
    under-represented classes still get a test sample every fold.
    """
    rng = np.random.default_rng(seed)

    fold_test_ids = [[] for _ in range(num_splits)]
    for _, group_df in metadata_df.groupby("label"):
        label_ids = group_df["id"].unique()
        label_fold_test_ids = _assign_test_ids_per_fold(label_ids, num_splits, rng)
        for fold_idx in range(num_splits):
            fold_test_ids[fold_idx].extend(label_fold_test_ids[fold_idx])

    split_paths = []
    for fold_idx in range(num_splits):
        test_ids = set(fold_test_ids[fold_idx])
        train_df = metadata_df[~metadata_df["id"].isin(test_ids)]
        test_df = metadata_df[metadata_df["id"].isin(test_ids)]

        overlap = set(train_df["id"]) & set(test_df["id"])
        assert not overlap, f"Fold {fold_idx}: id leakage detected: {overlap}"

        split_paths.append(_write_fold(output_dir, fold_idx, train_df, test_df))

    test_usage = Counter(tid for fold in fold_test_ids for tid in fold)
    print("Test-set usage count per id across folds:")
    for id_, count in sorted(test_usage.items()):
        print(f"  {id_}: {count}")

    return split_paths


_STRATEGIES = {
    "id_date": _splits_by_id_date,
    "id": _splits_by_id,
}


def make_cv_splits(metadata_csv, output_dir, num_splits=5, seed=42, group_by="id_date"):
    """Create group-aware k-fold train/test CSVs.

    `group_by` selects how leakage is controlled:
      - "id_date" (default): groups by subject+acquisition-date, then
        stratifies by class label with sklearn's StratifiedKFold. The same
        subject can appear in both train and test of a fold if imaged on
        different dates -- treating each acquisition as the unit of leakage
        control, which is standard practice when sessions are independent.
      - "id": groups strictly by subject id, so a subject never appears in
        both train and test of the same fold regardless of acquisition date.
        Test-id assignment is balanced per class label.

    Writes `training_data_{fold}.csv` / `test_data_{fold}.csv` into
    `output_dir` and returns the list of (train_path, test_path) pairs.
    """
    if group_by not in _STRATEGIES:
        raise ValueError(f"Unknown group_by '{group_by}', expected one of {list(_STRATEGIES)}")

    metadata_df = pd.read_csv(metadata_csv)
    os.makedirs(output_dir, exist_ok=True)

    return _STRATEGIES[group_by](metadata_df, output_dir, num_splits, seed)

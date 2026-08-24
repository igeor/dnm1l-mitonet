#!/usr/bin/env python
"""Create the stratified, patient-grouped k-fold train/test CSVs.

Usage:
    python scripts/make_splits.py --data_root data --num_splits 5 --seed 42
    python scripts/make_splits.py --data_root data --group_by id
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitonet.dataset import make_cv_splits  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="data", help="Directory containing metadata.csv; splits are written here too.")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group_by",
        choices=["id_date", "id"],
        default="id_date",
        help=(
            "Leakage-control unit: 'id_date' (default) groups by subject+acquisition-date "
            "and stratifies with sklearn's StratifiedKFold -- the same subject can land in "
            "both train and test if imaged on different dates. 'id' groups strictly by "
            "subject, so a subject never appears in both train and test of the same fold."
        ),
    )
    args = parser.parse_args()

    metadata_csv = os.path.join(args.data_root, "metadata.csv")
    make_cv_splits(
        metadata_csv,
        args.data_root,
        num_splits=args.num_splits,
        seed=args.seed,
        group_by=args.group_by,
    )


if __name__ == "__main__":
    main()

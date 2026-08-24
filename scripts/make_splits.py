#!/usr/bin/env python
"""Create the stratified, patient-grouped k-fold train/test CSVs.

Usage:
    python scripts/make_splits.py --data_root data --num_splits 5 --seed 42
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
    args = parser.parse_args()

    metadata_csv = os.path.join(args.data_root, "metadata.csv")
    make_cv_splits(metadata_csv, args.data_root, num_splits=args.num_splits, seed=args.seed)


if __name__ == "__main__":
    main()

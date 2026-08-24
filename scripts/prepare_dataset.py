#!/usr/bin/env python
"""Build the classifier's `metadata.csv` dataset from raw .lif microscopy files.

This is an *optional* preprocessing step: it is only needed if you are
starting from raw Leica .lif acquisitions. If you already have a directory
of images with a metadata.csv (id,label,date,path), skip this script and
point `Config.data_root` at that directory directly.

For each .lif file, this script takes the deconvolved image series, computes
a max-intensity projection over z, and writes it out as an 8-bit RGB-style
TIFF (channel order as acquired), plus a `metadata.csv` row.

Label assignment is driven by a user-supplied JSON mapping so that no
patient/sample identifiers are hard-coded in this repository:

    {
      "DNM1L": ["<id_or_substring_1>", "<id_or_substring_2>"],
      "CONTROL": ["<id_or_substring_3>"],
      "OTHERMIT": ["<id_or_substring_4>", "..."]
    }

A filename is assigned the first label whose list contains a substring of
its basename; the subject id is taken as the token before the first
underscore, and the date/batch is taken from the parent directory name.

Usage:
    python scripts/prepare_dataset.py \\
        --images_dir /path/to/raw/lif/files \\
        --output_dir data \\
        --labels_map labels_map.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np
import tifffile
import torch
import torchvision
from readlif.reader import LifFile

try:
    from tqdm import tqdm
except ImportError:  # tqdm is a light convenience dependency; fall back to no progress bar
    def tqdm(iterable, **kwargs):
        return iterable


def load_lif_series(path: str, series_idx: int) -> torch.Tensor:
    """Load one series of a .lif file as a (z, c, h, w) tensor."""
    lif_file = LifFile(path)
    image = lif_file.image_list[series_idx]
    dims = image["dims_n"]
    z = dims.get(3, 1)
    c = image["channels"]
    w = dims.get(1, 0)
    h = dims.get(2, 0)

    tensor = torch.zeros((z, c, h, w), dtype=torch.float32)
    to_tensor = torchvision.transforms.ToTensor()
    for z_idx in range(z):
        for c_idx in range(c):
            frame = lif_file.get_image(series_idx).get_frame(z=z_idx, t=0, c=c_idx, m=0)
            tensor[z_idx, c_idx] = to_tensor(frame)
    return tensor


def assign_label(basename: str, labels_map: dict) -> str | None:
    for label, needles in labels_map.items():
        if any(needle in basename for needle in needles):
            return label
    return None


def process_dataset(images_dir: str, output_dir: str, labels_map: dict, deconv_series_idx: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    lif_paths = glob.glob(os.path.join(images_dir, "**", "*.lif"), recursive=True)
    for lif_path in tqdm(lif_paths, desc="Processing .lif files"):
        base_no_ext = os.path.splitext(os.path.basename(lif_path))[0]
        date = os.path.basename(os.path.dirname(lif_path))
        subject_id = base_no_ext.split("_")[0]

        label = assign_label(base_no_ext, labels_map)
        if label is None:
            print(f"Skipping {base_no_ext}: no label match in labels_map.")
            continue

        image_3d = load_lif_series(lif_path, deconv_series_idx)  # (z, c, h, w)
        max_proj = torch.max(image_3d, dim=0).values  # (c, h, w)
        max_proj = np.swapaxes(max_proj.numpy(), 0, 2)  # (w, h, c)
        max_proj = (max_proj * 255).astype(np.uint8)

        out_name = f"{base_no_ext}_{date}_{label}.tiff"
        out_path = os.path.join(output_dir, label, out_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tifffile.imwrite(out_path, max_proj)

        rows.append([subject_id, label, date, os.path.abspath(out_path)])

    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "date", "path"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", required=True, help="Directory containing raw .lif files (searched recursively).")
    parser.add_argument("--output_dir", required=True, help="Directory to write TIFFs and metadata.csv to.")
    parser.add_argument("--labels_map", required=True, help="Path to a JSON file mapping class label -> list of id substrings.")
    parser.add_argument("--deconv_series_idx", type=int, default=1, help="Series index of the deconvolved image within each .lif file.")
    args = parser.parse_args()

    with open(args.labels_map) as f:
        labels_map = json.load(f)

    process_dataset(args.images_dir, args.output_dir, labels_map, args.deconv_series_idx)


if __name__ == "__main__":
    main()

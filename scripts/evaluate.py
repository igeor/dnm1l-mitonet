#!/usr/bin/env python
"""Evaluate a trained checkpoint on its held-out test split, optionally
saving Grad-CAM visualizations for a handful of test images.

Usage:
    python scripts/evaluate.py --config configs/curriculum_fine_tuning.json \\
        --split 0 --image_size 1024 --num_gradcams 10
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mitonet.config import Config  # noqa: E402
from mitonet.dataset import MitoNetworkDataset  # noqa: E402
from mitonet.gradcam import generate_gradcam, overlay_heatmap  # noqa: E402
from mitonet.metrics import compute_metrics  # noqa: E402
from mitonet.model import EfficientNetClassifier  # noqa: E402
from mitonet.train import EVAL_TRANSFORM  # noqa: E402

# Layer to extract Grad-CAM activations/gradients from; the last
# convolutional block of the EfficientNetB0 feature extractor.
GRADCAM_LAYER = "model.features.8.0"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", type=int, required=True, help="Which cross-validation split to evaluate.")
    parser.add_argument("--image_size", type=int, required=True, help="Resolution the checkpoint was trained at.")
    parser.add_argument("--num_gradcams", type=int, default=0, help="Number of test images to render Grad-CAMs for.")
    args = parser.parse_args()

    config = Config.from_json(args.config)
    device = config.device if torch.cuda.is_available() else "cpu"

    checkpoint_path = config.experiment_dir / "checkpoints" / f"split{args.split}_res{args.image_size}.pt"
    model = EfficientNetClassifier(
        model_name=config.model_name,
        num_classes=config.num_classes,
        image_size=args.image_size,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
        dense_layers=config.dense_layers,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_dataset = MitoNetworkDataset(
        os.path.join(config.data_root, f"test_data_{args.split}.csv"),
        class_names=config.class_names,
        transform=EVAL_TRANSFORM,
    )
    test_loader = DataLoader(test_dataset, batch_size=config.eval_batch_size, shuffle=False)

    real_probs, pred_probs = model.evaluate(test_loader, device=device)
    metrics = compute_metrics(real_probs, pred_probs, multi_class=config.num_classes > 2)
    print(f"Split {args.split} (res={args.image_size}) test metrics: {metrics}")

    if args.num_gradcams > 0:
        gradcam_dir = config.experiment_dir / "gradcams" / f"split{args.split}"
        gradcam_dir.mkdir(parents=True, exist_ok=True)

        for param in model.parameters():
            param.requires_grad = True

        for idx in range(min(args.num_gradcams, len(test_dataset))):
            image_name = os.path.basename(test_dataset.data.iloc[idx]["path"])
            image, _ = test_dataset[idx]
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                pred_class = model(image).softmax(dim=1).argmax(dim=1).item()

            heatmap = generate_gradcam(image, model, GRADCAM_LAYER, class_idx=pred_class)

            image_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            overlay = overlay_heatmap(image_np, heatmap)

            out_path = gradcam_dir / f"{os.path.splitext(image_name)[0]}_pred{config.class_names[pred_class]}.png"
            import cv2  # local import: only needed for this optional visualization path

            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved {args.num_gradcams} Grad-CAM overlays to {gradcam_dir}")


if __name__ == "__main__":
    main()

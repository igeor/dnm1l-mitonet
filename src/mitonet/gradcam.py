"""Grad-CAM (Selvaraju et al., 2017) for model interpretability.

Used to visualize which regions of the mitochondrial network image drove a
given classification, as reported in the paper (Figure 3D, F).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def generate_gradcam(img_tensor: torch.Tensor, model: torch.nn.Module, target_layer: str, class_idx: int | None = None):
    """Compute a normalized Grad-CAM heatmap in [0, 1] for `class_idx`
    (or the predicted class if None) at `target_layer` of `model`."""
    model.eval()
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    activations, gradients = [], []

    def forward_hook(_module, _input, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    layer = dict(model.named_modules()).get(target_layer)
    if layer is None:
        raise ValueError(f"Target layer '{target_layer}' not found in model")

    fwd_handle = layer.register_forward_hook(forward_hook)
    bwd_handle = layer.register_full_backward_hook(backward_hook)

    try:
        output = model(img_tensor)
        if class_idx is None:
            class_idx = output.argmax().item()

        model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)

        activation = activations[0]
        gradient = gradients[0]

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=1, keepdim=True))

        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam
    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on a [0, 1] or uint8 HxWx3 image.

    Returns a uint8 HxWx3 image ready to save/display.
    """
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

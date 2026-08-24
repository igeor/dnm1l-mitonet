"""Classification metrics used for reporting and model selection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(real_probs, pred_probs, multi_class: bool = True):
    """Compute accuracy/precision/recall/F1/AUC from one-hot targets and
    predicted class probabilities (torch tensors on any device)."""
    real_probs = real_probs.cpu().numpy()
    pred_probs = pred_probs.cpu().numpy()

    real_labels = np.argmax(real_probs, axis=1)
    pred_labels = np.argmax(pred_probs, axis=1)

    average = "weighted" if multi_class else "binary"
    acc = accuracy_score(real_labels, pred_labels)
    prec = precision_score(real_labels, pred_labels, average=average)
    rec = recall_score(real_labels, pred_labels, average=average)
    f1 = f1_score(real_labels, pred_labels, average=average)

    if multi_class:
        auc = roc_auc_score(real_labels, pred_probs, average="weighted", multi_class="ovr")
    else:
        auc = roc_auc_score(real_labels, pred_probs[:, 1])

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

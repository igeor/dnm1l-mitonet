"""mitonet: deep learning classification of mitochondrial network morphology.

Companion code for:
    Musokhranova et al., "AI-Based Mitochondrial Network Characterization as a
    Biomarker for DNM1L-Related Disease Diagnosis", 2026.

This package implements the deep learning branch of the paper: an
EfficientNetB0 image classifier trained to discriminate DNM1L-patient,
other-mitochondrial-disease, and control fibroblasts from immunofluorescence
images of the mitochondrial network, trained either by direct fine-tuning
(FT) or by curriculum-learning fine-tuning (CFT).
"""

from .config import Config
from .dataset import MitoNetworkDataset, make_cv_splits
from .metrics import compute_metrics
from .model import EfficientNetClassifier

__all__ = [
    "Config",
    "MitoNetworkDataset",
    "make_cv_splits",
    "compute_metrics",
    "EfficientNetClassifier",
]

# dnm1l-mitonet

Deep learning code accompanying:

> U. Musokhranova, I. Georvasilis, S. Illescas, M. Saritzoglou, M. Frías,
> M. Roldán, A. Ardissone, I. Moroni, À. García-Cazorla, A. Oyarzábal\*.
> **AI-Based Mitochondrial Network Characterization as a Biomarker for
> DNM1L-Related Disease Diagnosis.** 2026.

Pathogenic variants in *DNM1L* impair Drp1, a key regulator of mitochondrial
fission, producing a hyperfused, elongated mitochondrial network in patient
fibroblasts. This repository implements the paper's deep learning branch: an
EfficientNetB0 classifier trained on immunofluorescence images of the
mitochondrial network to discriminate DNM1L patients from other
mitochondrial-disease patients (OTHERMIT) and healthy controls.

Two training strategies are implemented, selected purely by config:

- **FT (direct fine-tuning)** — the pretrained backbone is fine-tuned once,
  directly at full (1024×1024) resolution.
- **CFT (curriculum-learning fine-tuning)** — the same backbone is fine-tuned
  progressively, at increasing resolutions (128 → 256 → 512 → 1024), each
  stage warm-started from the previous one. This is the best-performing
  approach reported in the paper (77% accuracy, 0.92 AUC vs. 60% / 0.85 for
  direct fine-tuning).

This repository only covers the deep learning pipeline. The classical
machine-learning baselines reported in the paper (Random Forest, SVM,
Multinomial Logistic Regression on hand-engineered Mitochondria Analyzer
features) are not included here.

## Repository layout

```
configs/                     Experiment configs (FT and CFT, as used in the paper)
data/                         Dataset goes here (not included, see data/README.md)
scripts/
  prepare_dataset.py          Optional: raw .lif -> TIFF + metadata.csv
  make_splits.py               Build the 5-fold cross-validation splits
  evaluate.py                   Evaluate a checkpoint, optionally render Grad-CAMs
src/mitonet/
  config.py                     Config dataclass
  dataset.py                     MitoNetworkDataset + split creation
  model.py                        EfficientNetB0 classifier
  metrics.py                       accuracy / precision / recall / F1 / AUC
  gradcam.py                        Grad-CAM interpretability
  train.py                          Training loop (implements both FT and CFT)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

The dataset (patient and control fibroblast images) is not distributed with
this repository. Populate `data/` yourself before running anything — see
[`data/README.md`](data/README.md) for the expected layout and
`metadata.csv` schema.

## Usage

```bash
# 1. (optional) build metadata.csv from raw .lif microscopy files
python scripts/prepare_dataset.py \
    --images_dir /path/to/raw/lif --output_dir data --labels_map labels_map.json

# 2. create the 5-fold, patient-grouped, stratified cross-validation splits
python scripts/make_splits.py --data_root data

# 3. train — direct fine-tuning
python -m mitonet.train --config configs/fine_tuning.json

# 3'. train — curriculum-learning fine-tuning (best performing, paper default)
python -m mitonet.train --config configs/curriculum_fine_tuning.json

# 4. evaluate a checkpoint and render Grad-CAM overlays
python scripts/evaluate.py --config configs/curriculum_fine_tuning.json \
    --split 0 --image_size 1024 --num_gradcams 10
```

Checkpoints, logs, and a `summary.json` of cross-validated metrics per
curriculum stage are written to `results/<alias>/`.

## Method summary

- **Backbone**: EfficientNetB0 (ImageNet-pretrained), classifier head
  replaced with a 128-unit hidden layer + 3-way output (CONTROL / DNM1L /
  OTHERMIT).
- **Input**: 3-channel deconvolved immunofluorescence max-projections
  (TOMM20 network signal + tubulin + DAPI), resized on the fly per
  curriculum stage.
- **Cross-validation**: 5-fold, stratified by class and grouped by subject
  (`id` + `date`) so no subject appears in both train and test for a fold.
- **Optimization**: Adam, lr 2e-4, weight decay 1e-3, StepLR (γ=0.9, step 1),
  cross-entropy loss, batch size 4, dropout 0.35.
- **Interpretability**: Grad-CAM on the last EfficientNetB0 convolutional
  block.

Full details are in the paper's Methods section 4.3(b) and Supporting
Information (Tables S3–S4).

## License

MIT — see [`LICENSE`](LICENSE).

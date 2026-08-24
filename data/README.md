# Data

This directory is intentionally empty. The dataset used in the paper is
patient-derived and is not distributed with this repository.

To reproduce the pipeline, populate this directory with:

```
data/
├── metadata.csv
└── <label>/
    └── <image files referenced by metadata.csv>
```

`metadata.csv` must have (at least) these columns:

| column | description |
|---|---|
| `id` | subject/sample identifier |
| `date` | acquisition batch or date, used with `id` to key cross-validation groups |
| `label` | one of `CONTROL`, `DNM1L`, `OTHERMIT` |
| `path` | path to the image file (absolute, or resolvable from the working directory) |

Images are RGB(-like) 3-channel immunofluorescence max-projections of the
mitochondrial network (TOMM20 channel + tubulin + DAPI, see the paper's
Methods 4.2). Any image format readable by Pillow works.

If you are starting from raw Leica `.lif` acquisitions instead of a ready
`metadata.csv`, see `scripts/prepare_dataset.py`.

Once `metadata.csv` is in place, generate cross-validation splits with:

```bash
python scripts/make_splits.py --data_root data
```

which writes `training_data_{fold}.csv` / `test_data_{fold}.csv` here as
well (these are also regenerated automatically the first time you run
`mitonet.train` if they don't already exist).

# Model

The segmentation model and its Core ML export.

## Setup

Python 3.13 — not 3.14, which torch does not yet support.

```sh
cd model
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Weights

See "Provenance" below. Weights are committed; the training dataset is not.

## Re-exporting

```sh
.venv/bin/python export_coreml.py
.venv/bin/python -m pytest tests/
```

## Provenance

Filled in by Task 2.

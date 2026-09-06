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

Both checkpoints come from the archived v1 repo,
[willasbery/vision-assist](https://github.com/willasbery/vision-assist),
under `model/runs/segment/`. Single class: `sidewalk`. Trained on
[sidewalk-dlu6l](https://universe.roboflow.com/projects-5k1o6/sidewalk-dlu6l/dataset/1)
from Roboflow — 5,296 images, public domain, 3,707/1,059/530 train/val/test.

| File | Run | Base | imgsz | Params | Best epoch | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|
| `train11-yolov8n-seg-640.pt` | train11 | yolov8n-seg | 640 | 3,263,811 | 97/100 | 0.859 | 0.734 |
| `train16-yolo11n-seg-240.pt` | train16 | yolo11n-seg | 240 | 2,842,803 | 128/150 | 0.813 | 0.689 |

Metrics are validation-set, taken from the best epoch of each run rather than
the last, since `best.pt` is what was copied. They come from v1's own
`results.csv` and have not been re-measured here.

`train11` is what the v1 backend loaded in production, and is the default for
export despite the older architecture — it is the stronger model. `train16` is
kept so the comparison stays reproducible.
